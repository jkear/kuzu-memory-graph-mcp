#!/usr/bin/env python3
"""
Kuzu Memory Graph MCP Server

A high-performance LLM memory server using Kuzu graph database with MLX embeddings.
Follows MCP FastMCP patterns and uses unified entity model for simplicity.

Multi-primary database support - AI can switch between writable databases
using the switch_primary_database() tool without server restart.
"""

import asyncio
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

import kuzu
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession


class WritableDatabaseManager:
    """Manages multiple writable databases with connection pool pattern.

    This implements a hybrid connection pool where databases stay open
    and connections are created on-demand when switching. Ensures single
    source of truth for all database state.
    """

    def __init__(self):
        self.databases: dict[str, kuzu.Database] = {}  # Connection pool
        self.current_conn: Optional[kuzu.Connection] = None
        self.current_name: Optional[str] = None
        self.writable_databases: list[str] = []
        self.databases_dir: str = ""

    def initialize(
        self,
        writable_dbs: list[str],
        databases_dir: str,
        embedding_model: Any,
        tokenizer: Any,
    ) -> None:
        """Initialize the database manager and open all databases in pool."""
        self.writable_databases = writable_dbs
        self.databases_dir = databases_dir

        print(f"Initializing database manager with writable databases: {writable_dbs}", file=sys.stderr)

        # Open all databases in the pool
        for db_name in writable_dbs:
            db_path = os.path.join(databases_dir, f"{db_name}.kuzu")

            if not os.path.exists(db_path):
                print(f"Warning: Database file not found: {db_path}", file=sys.stderr)
                continue

            try:
                db = kuzu.Database(db_path)
                self.databases[db_name] = db
                print(f"✓ Opened database in pool: {db_name}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Failed to open database '{db_name}': {e}", file=sys.stderr)

    def switch_to(self, db_name: str) -> tuple[bool, str]:
        """Switch active primary database using connection pool.

        Args:
            db_name: Name of database to make primary (writable)

        Returns:
            Tuple of (success, message)
        """
        if db_name not in self.writable_databases:
            available = ", ".join(self.writable_databases)
            return (
                False,
                f"Database '{db_name}' not in writable databases list. Available: {available}",
            )

        if db_name not in self.databases:
            return False, f"Database '{db_name}' not available in pool"

        # If already the current database, just return success
        if self.current_name == db_name and self.current_conn:
            return True, f"Already using '{db_name}' as primary database"

        # Close current connection (but keep database open in pool)
        if self.current_conn:
            try:
                self.current_conn.close()
                print(f"Closed previous connection to '{self.current_name}'", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Error closing connection: {e}", file=sys.stderr)

        # Create new connection from pooled database
        try:
            db = self.databases[db_name]
            self.current_conn = kuzu.Connection(db)
            self.current_name = db_name

            # Ensure schema exists
            self._ensure_schema()
            self._load_vector_extension()

            print(f"✓ Switched to '{db_name}' as primary (writable)", file=sys.stderr)
            return (
                True,
                f"Successfully switched to '{db_name}' as primary (writable) database",
            )

        except Exception as e:
            self.current_conn = None
            self.current_name = None
            return False, f"Failed to create connection for '{db_name}': {str(e)}"

    def _ensure_schema(self) -> None:
        """Ensure Entity and RELATED_TO tables exist in current database."""
        if not self.current_conn:
            return

        try:
            self.current_conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Entity (
                    name STRING PRIMARY KEY,
                    type STRING,
                    observations STRING[],
                    embedding FLOAT[384],
                    created_date DATE DEFAULT current_date(),
                    updated_date DATE DEFAULT current_date()
                )
            """)

            self.current_conn.execute("""
                CREATE REL TABLE IF NOT EXISTS RELATED_TO (
                    FROM Entity TO Entity,
                    relationship_type STRING,
                    confidence FLOAT DEFAULT 1.0,
                    created_date DATE DEFAULT current_date()
                )
            """)

            print(f"✓ Schema verified for '{self.current_name}'", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Schema creation error: {e}", file=sys.stderr)

    def _load_vector_extension(self) -> None:
        """Load vector extension for current database."""
        if not self.current_conn:
            return

        try:
            self.current_conn.execute("INSTALL vector;")
            self.current_conn.execute("LOAD EXTENSION vector;")
            print(f"✓ Vector extension loaded for '{self.current_name}'", file=sys.stderr)
        except Exception as e:
            # Extension might already be loaded
            print(f"Vector extension note: {e}", file=sys.stderr)

    def _is_connection_healthy(self, conn: Optional[kuzu.Connection]) -> bool:
        """Check if a database connection is healthy and usable."""
        if not conn:
            return False

        try:
            # Simple test query to verify connection is alive
            result = conn.execute("RETURN 1 as test")
            return result.has_next()
        except Exception as e:
            print(f"Warning: Connection health check failed: {e}", file=sys.stderr)
            return False

    def _is_safe_database_path(self, db_path: str) -> bool:
        """Validate that database path is safe and within expected directory."""
        try:
            from pathlib import Path

            abs_db_path = Path(db_path).resolve()
            abs_dir_path = Path(self.databases_dir).resolve()

            # Ensure database is within the configured directory
            try:
                abs_db_path.relative_to(abs_dir_path)
                return True
            except ValueError:
                print(f"Warning: Database path '{db_path}' is outside of databases directory '{self.databases_dir}'", file=sys.stderr)
                return False
        except Exception as e:
            print(f"Warning: Path validation error: {e}", file=sys.stderr)
            return False

    def get_connection(self) -> Optional[kuzu.Connection]:
        """Get current primary database connection with health check."""
        if not self.current_conn:
            print("Warning: No current connection available", file=sys.stderr)
            return None

        # Health check - auto-recover if needed
        if not self._is_connection_healthy(self.current_conn):
            print(f"Connection to '{self.current_name}' is unhealthy, attempting recovery...", file=sys.stderr)

            if self.current_name and self.current_name in self.databases:
                try:
                    # Recreate connection from pooled database
                    old_conn = self.current_conn
                    self.current_conn = kuzu.Connection(self.databases[self.current_name])

                    # Close old connection
                    try:
                        old_conn.close()
                    except:
                        pass

                    # Verify new connection
                    if self._is_connection_healthy(self.current_conn):
                        print(f"✓ Successfully recovered connection to '{self.current_name}'", file=sys.stderr)
                    else:
                        print(f"✗ Failed to recover connection to '{self.current_name}'", file=sys.stderr)
                        self.current_conn = None
                        self.current_name = None
                except Exception as e:
                    print(f"✗ Connection recovery failed: {e}", file=sys.stderr)
                    self.current_conn = None
                    self.current_name = None
            else:
                print("✗ Cannot recover connection - no current database set", file=sys.stderr)
                self.current_conn = None

        return self.current_conn

    def get_current_name(self) -> Optional[str]:
        """Get current primary database name."""
        return self.current_name

    def is_writable(self, db_name: str) -> bool:
        """Check if database is in writable list."""
        return db_name in self.writable_databases

    def get_available_databases(self) -> list[str]:
        """Get list of available databases in pool."""
        return list(self.databases.keys())

    def cleanup(self) -> None:
        """Close all connections and databases in pool."""
        print("Starting database manager cleanup...", file=sys.stderr)

        # Close current connection
        if self.current_conn:
            try:
                self.current_conn.close()
                print(f"✓ Closed connection to '{self.current_name}'", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Error closing connection: {e}", file=sys.stderr)

        # Close all databases in pool
        for db_name, db in self.databases.items():
            try:
                db.close()
                print(f"✓ Closed database in pool: {db_name}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Error closing database '{db_name}': {e}", file=sys.stderr)

        self.databases.clear()
        self.current_conn = None
        self.current_name = None
        print("Database manager cleanup complete.", file=sys.stderr)


# Global database manager instance
db_manager = WritableDatabaseManager()


@dataclass
class AppContext:
    """Application context with database manager and MLX model.

    All database access goes through db_manager to ensure single source of truth.
    """

    db_manager: WritableDatabaseManager
    embedding_model: Any
    tokenizer: Any
    databases_dir: str


def generate_embedding(model: Any, tokenizer: Any, text: str) -> list[float]:
    """Generate 384-dimensional embedding using MLX."""
    if not text or not text.strip():
        return [0.0] * 384

    import mlx.core as mx

    inputs = tokenizer.encode(text.strip(), return_tensors="mlx")
    outputs = model(inputs)
    return outputs.text_embeds.tolist()


def batch_generate_embeddings(
    model: Any, tokenizer: Any, texts: list[str]
) -> list[list[float]]:
    """Generate embeddings for multiple texts efficiently."""
    if not texts:
        return []

    import mlx.core as mx

    # Filter empty texts and prepare batch
    valid_texts = [
        (text.strip() if text else "") for text in texts if text and text.strip()
    ]

    if not valid_texts:
        return [[0.0] * 384] * len(texts)

    inputs = tokenizer.batch_encode_plus(
        valid_texts, return_tensors="mlx", padding=True, truncation=True, max_length=512
    )
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])

    # Map results back to original order
    embeddings = outputs.text_embeds.tolist()
    result = []
    valid_idx = 0
    for text in texts:
        if text and text.strip():
            result.append(embeddings[valid_idx])
            valid_idx += 1
        else:
            result.append([0.0] * 384)

    return result


def discover_databases(databases_dir: str) -> dict[str, dict[str, str]]:
    """Discover all Kuzu databases in directory."""
    import glob
    from pathlib import Path

    pattern = os.path.join(databases_dir, "*.kuzu")
    db_paths = glob.glob(pattern)

    databases = {}
    for db_path in db_paths:
        path_obj = Path(db_path)
        db_name = path_obj.stem

        databases[db_name] = {
            "path": db_path,
            "name": db_name,
            "description": f"Database: {db_name}",
        }

    return databases


def get_primary_db_name(db_path: str) -> str:
    """Extract database name from path."""
    from pathlib import Path

    return Path(db_path).stem


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with database manager and model initialization."""
    # Get configuration from environment
    db_path = os.getenv("KUZU_MEMORY_DB_PATH", "./DBMS/memory.kuzu")
    databases_dir = os.getenv("KUZU_DATABASES_DIR", "./DBMS")

    # Get writable databases list from environment
    writable_dbs_str = os.getenv("KUZU_WRITABLE_DATABASES", "")
    if writable_dbs_str:
        writable_databases = [db.strip() for db in writable_dbs_str.split(",")]
        print(
            f"Multi-primary mode: Writable databases: {writable_databases}",
            file=sys.stderr,
        )
    else:
        # Default: only primary database is writable
        writable_databases = [get_primary_db_name(db_path)]
        print(
            f"Single-primary mode: Only '{writable_databases[0]}' is writable",
            file=sys.stderr,
        )

    print(f"Initializing database manager...", file=sys.stderr)
    print(f"Databases directory: {databases_dir}", file=sys.stderr)

    # Initialize MLX embeddings model
    print("Loading MLX embeddings model...", file=sys.stderr)
    try:
        from mlx_embeddings.utils import load

        embedding_model, tokenizer = load("mlx-community/all-MiniLM-L6-v2-4bit")
        print("MLX model loaded successfully (dimension: 384)", file=sys.stderr)
    except Exception as e:
        print(f"Failed to load MLX model: {e}", file=sys.stderr)
        # Fallback to sentence transformers
        print("Falling back to sentence transformers...", file=sys.stderr)
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        tokenizer = None

        def fallback_generate_embedding(text: str) -> list[float]:
            if not text or not text.strip():
                return [0.0] * 384
            embedding = embedding_model.encode(text.strip(), convert_to_numpy=True)
            return embedding.tolist()

        def fallback_batch_generate_embeddings(texts: list[str]) -> list[list[float]]:
            if not texts:
                return []
            valid_texts = [text.strip() for text in texts if text and text.strip()]
            if not valid_texts:
                return [[0.0] * 384] * len(texts)

            embeddings = embedding_model.encode(valid_texts, convert_to_numpy=True)

            result = []
            valid_idx = 0
            for text in texts:
                if text and text.strip():
                    result.append(embeddings[valid_idx].tolist())
                    valid_idx += 1
                else:
                    result.append([0.0] * 384)
            return result

        # Monkey patch for fallback
        globals()["generate_embedding"] = (
            lambda model, tokenizer, text: fallback_generate_embedding(text)
        )
        globals()["batch_generate_embeddings"] = (
            lambda model, tokenizer, texts: fallback_batch_generate_embeddings(texts)
        )

    # Initialize database manager with connection pool
    db_manager.initialize(
        writable_dbs=writable_databases,
        databases_dir=databases_dir,
        embedding_model=embedding_model,
        tokenizer=tokenizer,
    )

    # Switch to initial primary database
    initial_db = writable_databases[0] if writable_databases else get_primary_db_name(db_path)
    success, msg = db_manager.switch_to(initial_db)
    if not success:
        print(f"ERROR: Failed to initialize primary database '{initial_db}': {msg}", file=sys.stderr)
        raise RuntimeError(f"Failed to initialize primary database: {msg}")

    # Discover available databases for resource listing
    discovered_dbs = discover_databases(databases_dir)
    available_in_pool = db_manager.get_available_databases()

    print("=" * 60, file=sys.stderr)
    print("Kuzu Memory Graph MCP Server Ready!", file=sys.stderr)
    print(f"Primary database: {db_manager.get_current_name()} (READ-WRITE)", file=sys.stderr)
    print(f"Writable databases: {writable_databases}", file=sys.stderr)
    print(f"Databases in pool: {available_in_pool}", file=sys.stderr)
    print(f"Discovered databases: {list(discovered_dbs.keys())}", file=sys.stderr)
    if len(writable_databases) > 1:
        print(
            f"✨ Multi-primary mode enabled! Use switch_primary_database() to switch.",
            file=sys.stderr,
        )
    print("=" * 60, file=sys.stderr)

    try:
        yield AppContext(
            db_manager=db_manager,
            embedding_model=embedding_model,
            tokenizer=tokenizer,
            databases_dir=databases_dir,
        )
    finally:
        # Cleanup database manager (closes all connections and databases)
        print("Cleaning up database manager...", file=sys.stderr)
        db_manager.cleanup()
        print("Database manager cleanup complete.", file=sys.stderr)


# Create FastMCP server with lifespan management
mcp = FastMCP("kuzu-memory-graph", lifespan=app_lifespan)


# === MCP Resources ===


@mcp.resource("kuzu://databases/list")
async def list_databases(ctx: Context[ServerSession, AppContext]) -> str:
    """List all available Kuzu databases."""
    app_ctx = ctx.request_context.lifespan_context

    databases = discover_databases(app_ctx.databases_dir)
    current_primary = app_ctx.db_manager.get_current_name()
    writable_databases = app_ctx.db_manager.writable_databases
    available_in_pool = app_ctx.db_manager.get_available_databases()

    db_list = []
    for name, info in databases.items():
        db_list.append(
            {
                "name": name,
                "description": info["description"],
                "is_primary": name == current_primary,
                "is_writable": name in writable_databases,
                "is_in_pool": name in available_in_pool,
            }
        )

    db_list.sort(key=lambda x: (not x["is_primary"], x["name"]))

    import json

    return json.dumps(
        {
            "databases": db_list,
            "count": len(db_list),
            "current_primary": current_primary,
            "writable_databases": writable_databases,
            "databases_in_pool": available_in_pool
        },
        indent=2,
    )


# === MCP Tools ===


@mcp.tool()
async def switch_primary_database(
    ctx: Context[ServerSession, AppContext], database: str
) -> dict[str, Any]:
    """Switch the active primary (writable) database.

    This allows dynamically changing which database accepts write operations
    without restarting the server. Only works for databases listed in the
    KUZU_WRITABLE_DATABASES environment variable.

    After switching, all create_entity, create_relationship, and add_observations
    calls will write to the new primary database.

    Args:
        database: Name of database to make primary (writable). Must be in KUZU_WRITABLE_DATABASES list.

    Returns:
        Status of the switch operation including:
        - success: Whether switch was successful
        - message: Descriptive message
        - previous_primary: Name of previous primary database
        - current_primary: Name of new primary database
        - writable_databases: List of all databases that can be made primary

    Example:
        # Switch to prompt_engineering database
        switch_primary_database(database="prompt_engineering")

        # Now you can create entities in prompt_engineering
        create_entity(database="prompt_engineering", name="Chain-of-Thought", ...)
    """
    app_ctx = ctx.request_context.lifespan_context

    previous_primary = app_ctx.db_manager.get_current_name()

    # Attempt to switch
    success, message = app_ctx.db_manager.switch_to(database)

    return {
        "success": success,
        "message": message,
        "previous_primary": previous_primary,
        "current_primary": app_ctx.db_manager.get_current_name(),
        "writable_databases": app_ctx.db_manager.writable_databases,
        "switched": success and (previous_primary != database),
    }


@mcp.tool()
async def create_entity(
    ctx: Context[ServerSession, AppContext],
    database: str,
    name: str,
    entity_type: str,
    observations: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Create a new entity in the knowledge graph.

    Args:
        database: Database name (query kuzu://databases/list to see available)
        name: Unique name for the entity
        entity_type: Type/category of the entity (person, concept, document, etc.)
        observations: List of observations/facts about the entity
    """
    app_ctx = ctx.request_context.lifespan_context

    # Get current connection and primary database from manager
    conn = app_ctx.db_manager.get_connection()
    current_primary = app_ctx.db_manager.get_current_name()

    if not conn or not current_primary:
        return {
            "status": "error",
            "message": "No active database connection. Please contact administrator.",
        }

    # Check if trying to write to non-primary database
    if database != current_primary:
        # Check if target database is in writable list
        if app_ctx.db_manager.is_writable(database):
            return {
                "status": "error",
                "message": f"Cannot write to '{database}' - it's not the current primary database. Use switch_primary_database(database='{database}') first to make it writable.",
                "database": database,
                "current_primary": current_primary,
                "suggestion": f"switch_primary_database(database='{database}')",
            }
        else:
            return {
                "status": "error",
                "message": f"Cannot write to '{database}' - only databases in KUZU_WRITABLE_DATABASES can be written to. Current primary: '{current_primary}'.",
                "database": database,
                "current_primary": current_primary,
                "writable_databases": app_ctx.db_manager.writable_databases,
            }

    # Prepare text for embedding
    text_to_embed = " ".join(observations or [name])

    # Generate embedding using MLX
    embedding = generate_embedding(
        app_ctx.embedding_model, app_ctx.tokenizer, text_to_embed
    )

    try:
        result = conn.execute(
            """
            CREATE (e:Entity {
                name: $name,
                type: $entity_type,
                observations: $observations,
                embedding: $embedding,
                created_date: current_date(),
                updated_date: current_date()
            })
            RETURN e.name as name, e.type as type
        """,
            {
                "name": name,
                "entity_type": entity_type,
                "observations": observations or [],
                "embedding": embedding,
            },
        )

        return {
            "status": "created",
            "database": database,
            "name": name,
            "type": entity_type,
            "observations_count": len(observations or []),
            "embedded": True,
        }
    except Exception as e:
        if "duplicate" in str(e).lower():
            return {
                "status": "exists",
                "database": database,
                "name": name,
                "type": entity_type,
                "message": "Entity already exists",
            }
        raise e


@mcp.tool()
async def create_relationship(
    ctx: Context[ServerSession, AppContext],
    database: str,
    from_entity: str,
    to_entity: str,
    relationship_type: str,
    confidence: float = 1.0,
) -> dict[str, Any]:
    """Create a relationship between two entities.

    Args:
        database: Database name (query kuzu://databases/list to see available)
        from_entity: Name of the source entity
        to_entity: Name of the target entity
        relationship_type: Type of relationship (WORKS_WITH, KNOWS, RELATED_TO, etc.)
        confidence: Confidence score for the relationship (0.0-1.0)
    """
    app_ctx = ctx.request_context.lifespan_context

    # Get current connection and primary database from manager
    conn = app_ctx.db_manager.get_connection()
    current_primary = app_ctx.db_manager.get_current_name()

    if not conn or not current_primary:
        return {
            "status": "error",
            "message": "No active database connection. Please contact administrator.",
        }

    # Check if trying to write to non-primary database
    if database != current_primary:
        if app_ctx.db_manager.is_writable(database):
            return {
                "status": "error",
                "message": f"Cannot write to '{database}' - it's not the current primary database. Use switch_primary_database(database='{database}') first.",
                "database": database,
                "current_primary": current_primary,
                "suggestion": f"switch_primary_database(database='{database}')",
            }
        else:
            return {
                "status": "error",
                "message": f"Cannot write to '{database}' - only databases in KUZU_WRITABLE_DATABASES can be written to.",
                "database": database,
                "current_primary": current_primary,
                "writable_databases": app_ctx.db_manager.writable_databases,
            }

    try:
        result = conn.execute(
            """
            MATCH (from:Entity {name: $from_name})
            MATCH (to:Entity {name: $to_name})
            CREATE (from)-[r:RELATED_TO {
                relationship_type: $relationship_type,
                confidence: $confidence,
                created_date: current_date()
            }]->(to)
            RETURN from.name as from_name, to.name as to_name, r.relationship_type as type
        """,
            {
                "from_name": from_entity,
                "to_name": to_entity,
                "relationship_type": relationship_type,
                "confidence": confidence,
            },
        )

        return {
            "status": "created",
            "database": database,
            "from": from_entity,
            "to": to_entity,
            "relationship_type": relationship_type,
            "confidence": confidence,
        }
    except Exception as e:
        if "does not exist" in str(e).lower():
            return {
                "status": "error",
                "database": database,
                "message": f"One or both entities not found: {from_entity}, {to_entity}",
            }
        raise e


@mcp.tool()
async def add_observations(
    ctx: Context[ServerSession, AppContext],
    database: str,
    entity_name: str,
    observations: list[str],
) -> dict[str, Any]:
    """Add new observations to an existing entity.

    Args:
        database: Database name (query kuzu://databases/list to see available)
        entity_name: Name of the entity to update
        observations: List of new observations to add
    """
    app_ctx = ctx.request_context.lifespan_context

    # Get current connection and primary database from manager
    conn = app_ctx.db_manager.get_connection()
    current_primary = app_ctx.db_manager.get_current_name()

    if not conn or not current_primary:
        return {
            "status": "error",
            "message": "No active database connection. Please contact administrator.",
        }

    # Check if trying to write to non-primary database
    if database != current_primary:
        if app_ctx.db_manager.is_writable(database):
            return {
                "status": "error",
                "message": f"Cannot write to '{database}' - it's not the current primary database. Use switch_primary_database(database='{database}') first.",
                "database": database,
                "current_primary": current_primary,
                "suggestion": f"switch_primary_database(database='{database}')",
            }
        else:
            return {
                "status": "error",
                "message": f"Cannot write to '{database}' - only databases in KUZU_WRITABLE_DATABASES can be written to.",
                "database": database,
                "current_primary": current_primary,
                "writable_databases": app_ctx.db_manager.writable_databases,
            }

    # Get current entity
    result = conn.execute(
        """
        MATCH (e:Entity {name: $name})
        RETURN e.observations as current_obs, e.type as type
    """,
        {"name": entity_name},
    )

    if not result.has_next():  # type: ignore
        return {
            "status": "error",
            "database": database,
            "message": f"Entity '{entity_name}' not found",
        }

    current_obs, entity_type = result.get_next()  # type: ignore
    current_obs = list(current_obs) if current_obs else []  # type: ignore

    # Filter out existing observations
    new_obs = [obs for obs in observations if obs not in current_obs]

    if not new_obs:
        return {
            "status": "no_change",
            "database": database,
            "entity_name": entity_name,
            "message": "All observations already exist",
        }

    # Update observations and regenerate embedding
    updated_obs = current_obs + new_obs
    text_to_embed = " ".join(updated_obs)
    embedding = generate_embedding(
        app_ctx.embedding_model, app_ctx.tokenizer, text_to_embed
    )

    conn.execute(
        """
        MATCH (e:Entity {name: $name})
        SET e.observations = $observations,
            e.embedding = $embedding,
            e.updated_date = current_date()
    """,
        {"name": entity_name, "observations": updated_obs, "embedding": embedding},
    )

    return {
        "status": "updated",
        "database": database,
        "entity_name": entity_name,
        "added_observations": new_obs,
        "total_observations": len(updated_obs),
        "reembedded": True,
    }


@mcp.tool()
async def search_entities(
    ctx: Context[ServerSession, AppContext], database: str, query: str, limit: int = 10
) -> dict[str, Any]:
    """Search entities using text-based search.

    Args:
        database: Database name (query kuzu://databases/list to see available)
        query: Search query string
        limit: Maximum number of results
    """
    app_ctx = ctx.request_context.lifespan_context

    # For read operations, we can work with any database, but need to switch context
    # Check if database is available and switch if needed
    if not app_ctx.db_manager.is_writable(database) and database != app_ctx.db_manager.get_current_name():
        # For read-only databases, we need to create a temporary connection
        try:
            db_path = os.path.join(app_ctx.databases_dir, f"{database}.kuzu")
            if not os.path.exists(db_path):
                return {"status": "error", "message": f"Database file not found: {db_path}", "database": database}

            temp_db = kuzu.Database(db_path)
            temp_conn = kuzu.Connection(temp_db)
        except Exception as e:
            return {"status": "error", "message": f"Failed to connect to database '{database}': {str(e)}", "database": database}
    else:
        # Use current connection from manager
        temp_conn = app_ctx.db_manager.get_connection()
        if not temp_conn:
            return {"status": "error", "message": "No active database connection", "database": database}

    try:
        result = temp_conn.execute(
            """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($query)
                OR toLower(e.type) CONTAINS toLower($query)
                OR ANY(obs IN e.observations WHERE toLower(obs) CONTAINS toLower($query))
            RETURN e.name, e.type, e.observations, e.created_date
            ORDER BY e.updated_date DESC
            LIMIT $limit
        """,
            {"query": query.lower(), "limit": limit},
        )

        entities = []
        while result.has_next():  # type: ignore
            row = result.get_next()  # type: ignore
            entities.append(
                {
                    "name": row[0],  # type: ignore
                    "type": row[1],  # type: ignore
                    "observations": list(row[2]) if row[2] else [],  # type: ignore
                    "created_date": str(row[3]),  # type: ignore
                }
            )

        return {
            "query": query,
            "database": database,
            "entities": entities,
            "count": len(entities),
        }
    except Exception as e:
        return {"status": "error", "message": f"Search failed: {str(e)}", "database": database}
    finally:
        # Clean up temporary connection if created
        if 'temp_db' in locals():
            try:
                temp_conn.close()
                temp_db.close()
            except:
                pass


@mcp.tool()
async def semantic_search(
    ctx: Context[ServerSession, AppContext],
    database: str,
    query: str,
    limit: int = 10,
    threshold: float = 0.3,
) -> dict[str, Any]:
    """Search entities using semantic similarity.

    Args:
        database: Database name (query kuzu://databases/list to see available)
        query: Search query for semantic matching
        limit: Maximum number of results
        threshold: Minimum similarity threshold (0.0-1.0)
    """
    app_ctx = ctx.request_context.lifespan_context

    # For read operations, handle database connections like search_entities
    if not app_ctx.db_manager.is_writable(database) and database != app_ctx.db_manager.get_current_name():
        try:
            db_path = os.path.join(app_ctx.databases_dir, f"{database}.kuzu")
            if not os.path.exists(db_path):
                return {"status": "error", "message": f"Database file not found: {db_path}", "database": database}

            temp_db = kuzu.Database(db_path)
            temp_conn = kuzu.Connection(temp_db)
        except Exception as e:
            return {"status": "error", "message": f"Failed to connect to database '{database}': {str(e)}", "database": database}
    else:
        temp_conn = app_ctx.db_manager.get_connection()
        if not temp_conn:
            return {"status": "error", "message": "No active database connection", "database": database}

    try:
        # Generate query embedding
        query_embedding = generate_embedding(
            app_ctx.embedding_model, app_ctx.tokenizer, query
        )

        # Perform vector similarity search
        try:
            result = temp_conn.execute(
                """
                MATCH (e:Entity)
                WHERE e.embedding IS NOT NULL
                WITH e, array_cosine_similarity(e.embedding, $query_embedding) as similarity
                WHERE similarity >= $threshold
                RETURN e.name, e.type, e.observations, similarity
                ORDER BY similarity DESC
                LIMIT $limit
            """,
                {
                    "query_embedding": query_embedding,
                    "threshold": threshold,
                    "limit": limit,
                },
            )
        except Exception:
            # Fallback to manual similarity calculation
            result = temp_conn.execute("""
                MATCH (e:Entity)
                WHERE e.embedding IS NOT NULL
                RETURN e.name, e.type, e.observations, e.embedding
            """)

            entities = []
            while result.has_next():  # type: ignore
                row = result.get_next()  # type: ignore
                entities.append(
                    {
                        "name": row[0],  # type: ignore
                        "type": row[1],  # type: ignore
                        "observations": list(row[2]) if row[2] else [],  # type: ignore
                        "embedding": row[3],  # type: ignore
                    }
                )

            # Calculate similarities manually
            import numpy as np
            query_vec = np.array(query_embedding)
            similarities = []

            for entity in entities:
                entity_vec = np.array(entity["embedding"])
                if np.linalg.norm(query_vec) > 0 and np.linalg.norm(entity_vec) > 0:
                    similarity = np.dot(query_vec, entity_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(entity_vec)
                    )
                    if similarity >= threshold:
                        similarities.append(
                            {
                                "name": entity["name"],
                                "type": entity["type"],
                                "observations": entity["observations"],
                                "similarity": float(similarity),
                            }
                        )

            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            similarities = similarities[:limit]

            return {
                "query": query,
                "database": database,
                "entities": similarities,
                "count": len(similarities),
                "threshold": threshold,
                "method": "manual_calculation",
            }

        entities = []
        while result.has_next():  # type: ignore
            row = result.get_next()  # type: ignore
            entities.append(
                {
                    "name": row[0],  # type: ignore
                    "type": row[1],  # type: ignore
                    "observations": list(row[2]) if row[2] else [],  # type: ignore
                    "similarity": float(row[3]),  # type: ignore
                }
            )

        return {
            "query": query,
            "database": database,
            "entities": entities,
            "count": len(entities),
            "threshold": threshold,
            "method": "vector_similarity",
        }
    except Exception as e:
        return {"status": "error", "message": f"Semantic search failed: {str(e)}", "database": database}
    finally:
        if 'temp_db' in locals():
            try:
                temp_conn.close()
                temp_db.close()
            except:
                pass


@mcp.tool()
async def get_related_entities(
    ctx: Context[ServerSession, AppContext],
    database: str,
    entity_name: str,
    max_depth: int = 2,
    limit: int = 20,
) -> dict[str, Any]:
    """Get entities related to the specified entity through relationship traversal.

    Args:
        database: Database name (query kuzu://databases/list to see available)
        entity_name: Name of the entity to find relations for
        max_depth: Maximum relationship depth to traverse
        limit: Maximum number of results
    """
    app_ctx = ctx.request_context.lifespan_context

    # Handle database connection like other read tools
    if not app_ctx.db_manager.is_writable(database) and database != app_ctx.db_manager.get_current_name():
        try:
            db_path = os.path.join(app_ctx.databases_dir, f"{database}.kuzu")
            if not os.path.exists(db_path):
                return {"status": "error", "message": f"Database file not found: {db_path}", "database": database}

            temp_db = kuzu.Database(db_path)
            temp_conn = kuzu.Connection(temp_db)
        except Exception as e:
            return {"status": "error", "message": f"Failed to connect to database '{database}': {str(e)}", "database": database}
    else:
        temp_conn = app_ctx.db_manager.get_connection()
        if not temp_conn:
            return {"status": "error", "message": "No active database connection", "database": database}

    try:
        result = temp_conn.execute(
            f"""
            MATCH path = (start:Entity {{name: $name}})-[:RELATED_TO*1..{max_depth}]-(related:Entity)
            WHERE related.name <> $name
            RETURN DISTINCT related.name, related.type, related.observations,
                    length(path) as distance,
                    [r IN relationships(path) | r.relationship_type] as relationship_path,
                    [r IN relationships(path) | r.confidence] as confidence_path
            ORDER BY distance, related.name
            LIMIT $limit
        """,
            {"name": entity_name, "limit": limit},
        )

        entities = []
        while result.has_next():  # type: ignore
            row = result.get_next()  # type: ignore
            entities.append(
                {
                    "name": row[0],  # type: ignore
                    "type": row[1],  # type: ignore
                    "observations": list(row[2]) if row[2] else [],  # type: ignore
                    "distance": row[3],  # type: ignore
                    "relationship_path": row[4],  # type: ignore
                    "confidence_path": row[5],  # type: ignore
                }
            )

        return {
            "entity_name": entity_name,
            "database": database,
            "entities": entities,
            "count": len(entities),
            "max_depth": max_depth,
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to get related entities: {str(e)}", "database": database}
    finally:
        if 'temp_db' in locals():
            try:
                temp_conn.close()
                temp_db.close()
            except:
                pass


@mcp.tool()
async def get_graph_summary(
    ctx: Context[ServerSession, AppContext], database: Optional[str] = None
) -> dict[str, Any]:
    """Get a summary of the knowledge graph(s).

    Args:
        database: Specific database to summarize, or None for all databases
    """
    app_ctx = ctx.request_context.lifespan_context

    # If no database specified, summarize all discovered databases
    if database is None:
        discovered_dbs = discover_databases(app_ctx.databases_dir)
        summaries = {}

        for db_name in discovered_dbs.keys():
            try:
                db_path = os.path.join(app_ctx.databases_dir, f"{db_name}.kuzu")
                if not os.path.exists(db_path):
                    summaries[db_name] = {"error": "Database file not found"}
                    continue

                temp_db = kuzu.Database(db_path)
                temp_conn = kuzu.Connection(temp_db)

                try:
                    entity_count = temp_conn.execute(
                        "MATCH (e:Entity) RETURN COUNT(e)"
                    ).get_next()[0]  # type: ignore
                    relation_count = temp_conn.execute(
                        "MATCH ()-[r]->() RETURN COUNT(r)"
                    ).get_next()[0]  # type: ignore

                    summaries[db_name] = {
                        "entities": entity_count,
                        "relationships": relation_count,
                    }
                except Exception as e:
                    summaries[db_name] = {"error": str(e)}
                finally:
                    temp_conn.close()
                    temp_db.close()
            except Exception as e:
                summaries[db_name] = {"error": str(e)}

        return {
            "scope": "all_databases",
            "databases": summaries,
            "total_databases": len(discovered_dbs),
        }

    # If database specified, provide detailed summary
    try:
        if not app_ctx.db_manager.is_writable(database) and database != app_ctx.db_manager.get_current_name():
            db_path = os.path.join(app_ctx.databases_dir, f"{database}.kuzu")
            if not os.path.exists(db_path):
                return {"status": "error", "message": f"Database file not found: {db_path}", "database": database}

            temp_db = kuzu.Database(db_path)
            temp_conn = kuzu.Connection(temp_db)
            cleanup_temp = True
        else:
            temp_conn = app_ctx.db_manager.get_connection()
            if not temp_conn:
                return {"status": "error", "message": "No active database connection", "database": database}
            cleanup_temp = False

        # Get counts
        entity_count = temp_conn.execute("MATCH (e:Entity) RETURN COUNT(e)").get_next()[0]  # type: ignore
        relation_count = temp_conn.execute("MATCH ()-[r]->() RETURN COUNT(r)").get_next()[0]  # type: ignore

        # Get entity types
        result = temp_conn.execute("""
            MATCH (e:Entity)
            RETURN e.type, COUNT(e)
            ORDER BY COUNT(e) DESC
        """)

        entity_types = []
        while result.has_next():  # type: ignore
            row = result.get_next()  # type: ignore
            entity_types.append({"type": row[0], "count": row[1]})  # type: ignore

        # Get relationship types
        result = temp_conn.execute("""
            MATCH ()-[r:RELATED_TO]->()
            RETURN r.relationship_type, COUNT(r)
            ORDER BY COUNT(r) DESC
        """)

        relationship_types = []
        while result.has_next():  # type: ignore
            row = result.get_next()  # type: ignore
            relationship_types.append({"type": row[0], "count": row[1]})  # type: ignore

        return {
            "scope": "single_database",
            "database": database,
            "stats": {
                "entities": entity_count,
                "relationships": relation_count,
                "entity_types": len(entity_types),
                "relationship_types": len(relationship_types),
            },
            "entity_types": entity_types,
            "relationship_types": relationship_types,
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to get graph summary: {str(e)}", "database": database}
    finally:
        if cleanup_temp and 'temp_db' in locals():
            try:
                temp_conn.close()
                temp_db.close()
            except:
                pass


def main():
    """Entry point for the MCP server."""
    print("Starting Kuzu Memory Graph MCP Server...", file=sys.stderr)
    mcp.run()


if __name__ == "__main__":
    main()
