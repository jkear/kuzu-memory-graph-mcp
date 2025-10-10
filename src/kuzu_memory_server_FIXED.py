#!/usr/bin/env python3
"""
Kuzu Memory Graph MCP Server - FIXED VERSION

BUGFIX: Skip USE statement for primary database since it's already active.
The primary database cannot be accessed via USE - it's the default active database.
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


@dataclass
class AppContext:
    """Application context with Kuzu database connection and MLX model."""

    db: kuzu.Database
    conn: kuzu.Connection
    embedding_model: Any
    tokenizer: Any
    primary_db_path: str
    primary_db_name: str  # ADDED: Store primary database name
    attached_databases: dict[str, str]
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


def ensure_database_attached(
    conn: kuzu.Connection,
    attached_databases: dict[str, str],
    db_name: str,
    databases_dir: str,
) -> tuple[bool, str]:
    """Attach database if not already attached."""
    if db_name in attached_databases:
        return True, f"Database '{db_name}' already attached"

    db_path = os.path.join(databases_dir, f"{db_name}.kuzu")

    if not os.path.exists(db_path):
        return False, f"Database file not found: {db_path}"

    try:
        conn.execute(f"ATTACH '{db_path}' AS {db_name} (dbtype kuzu);")
        attached_databases[db_name] = db_path
        print(f"✓ Attached database: {db_name}", file=sys.stderr)
        return True, f"Successfully attached '{db_name}'"
    except Exception as e:
        error_msg = str(e)
        if (
            "already attached" in error_msg.lower()
            or "already exists" in error_msg.lower()
        ):
            attached_databases[db_name] = db_path
            return True, f"Database '{db_name}' already attached"
        return False, f"Failed to attach: {error_msg}"


def switch_database_context(
    conn: kuzu.Connection, 
    db_name: str,
    primary_db_name: str  # ADDED: Pass primary database name
) -> tuple[bool, str]:
    """Switch active database using USE statement.
    
    BUGFIX: Skip USE statement for primary database since it's already active.
    """
    # BUGFIX: Don't try to USE the primary database - it's already active!
    if db_name == primary_db_name:
        return True, f"Using primary database '{db_name}' (already active)"
    
    try:
        conn.execute(f"USE {db_name};")
        return True, f"Switched to '{db_name}'"
    except Exception as e:
        return False, f"Failed to switch: {str(e)}"


def get_primary_db_name(db_path: str) -> str:
    """Extract database name from path."""
    from pathlib import Path

    return Path(db_path).stem


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with database and model initialization."""
    # Get primary database path from environment
    db_path = os.getenv("KUZU_MEMORY_DB_PATH", "./DBMS/memory.kuzu")
    databases_dir = os.getenv("KUZU_DATABASES_DIR", "./DBMS")

    print(f"Initializing primary database: {db_path}", file=sys.stderr)
    print(f"Scanning for databases in: {databases_dir}", file=sys.stderr)

    # Initialize primary Kuzu database
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)

    attached_databases = {}
    primary_db_name = get_primary_db_name(db_path)
    attached_databases[primary_db_name] = db_path

    discovered_dbs = discover_databases(databases_dir)
    print(
        f"Discovered {len(discovered_dbs)} database(s): {list(discovered_dbs.keys())}",
        file=sys.stderr,
    )
    print(f"Primary database: '{primary_db_name}' (writable)", file=sys.stderr)
    print(f"Attached databases: {list(set(discovered_dbs.keys()) - {primary_db_name})} (read-only)", file=sys.stderr)

    # Install vector extension
    try:
        conn.execute("INSTALL vector;")
        conn.execute("LOAD EXTENSION vector;")
        print("Vector extension loaded successfully", file=sys.stderr)
    except Exception as e:
        print(f"Vector extension might already be loaded: {e}", file=sys.stderr)

    # Create schema in PRIMARY database
    print(f"Creating schema in primary database '{primary_db_name}'...", file=sys.stderr)
    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Entity (
            name STRING PRIMARY KEY,
            type STRING,
            observations STRING[],
            embedding FLOAT[384],
            created_date DATE DEFAULT current_date(),
            updated_date DATE DEFAULT current_date()
        )
    """)

    conn.execute("""
        CREATE REL TABLE IF NOT EXISTS RELATED_TO (
            FROM Entity TO Entity,
            relationship_type STRING,
            confidence FLOAT DEFAULT 1.0,
            created_date DATE DEFAULT current_date()
        )
    """)
    print(f"✓ Schema created in '{primary_db_name}'", file=sys.stderr)

    # Create vector index if it doesn't exist
    try:
        conn.execute(
            "CALL CREATE_VECTOR_INDEX('Entity', 'entity_embedding_idx', 'embedding');"
        )
        print("Vector index created successfully", file=sys.stderr)
    except Exception as e:
        print(f"Vector index might already exist: {e}", file=sys.stderr)

    # Initialize MLX embeddings model
    print("Loading MLX embeddings model...", file=sys.stderr)
    try:
        from mlx_embeddings.utils import load

        embedding_model, tokenizer = load("mlx-community/all-MiniLM-L6-v2-4bit")
        print(f"MLX model loaded successfully (dimension: 384)", file=sys.stderr)
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

    print("=" * 60, file=sys.stderr)
    print("Kuzu Memory Graph MCP Server Ready!", file=sys.stderr)
    print(f"Primary database: {primary_db_name} (READ-WRITE)", file=sys.stderr)
    print(f"Writable database: {primary_db_name}", file=sys.stderr)
    print(f"Read-only databases: {list(set(discovered_dbs.keys()) - {primary_db_name})}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    try:
        yield AppContext(
            db=db,
            conn=conn,
            embedding_model=embedding_model,
            tokenizer=tokenizer,
            primary_db_path=db_path,
            primary_db_name=primary_db_name,  # ADDED
            attached_databases=attached_databases,
            databases_dir=databases_dir,
        )
    finally:
        # Detach databases before closing
        primary_name = get_primary_db_name(db_path)
        for db_name in list(attached_databases.keys()):
            if db_name != primary_name:
                try:
                    conn.execute(f"DETACH {db_name};")
                    print(f"Detached: {db_name}", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Could not detach {db_name}: {e}", file=sys.stderr)

        conn.close()
        db.close()
        print("Database connections closed", file=sys.stderr)


# Create FastMCP server with lifespan management
mcp = FastMCP("kuzu-memory-graph", lifespan=app_lifespan)


# === MCP Resources ===


@mcp.resource("kuzu://databases/list")
async def list_databases(ctx: Context[ServerSession, AppContext]) -> str:
    """List all available Kuzu databases."""
    app_ctx = ctx.request_context.lifespan_context

    databases = discover_databases(app_ctx.databases_dir)
    primary_name = app_ctx.primary_db_name  # CHANGED

    db_list = []
    for name, info in databases.items():
        db_list.append(
            {
                "name": name,
                "description": info["description"],
                "is_primary": name == primary_name,
                "is_attached": name in app_ctx.attached_databases,
                "writable": name == primary_name,  # ADDED
            }
        )

    db_list.sort(key=lambda x: (not x["is_primary"], x["name"]))

    import json

    return json.dumps(
        {
            "databases": db_list,
            "count": len(db_list),
            "primary_database": primary_name,
            "writable_database": primary_name,  # ADDED
        },
        indent=2,
    )


# === MCP Tools ===


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
    conn = app_ctx.conn

    # Check if trying to write to non-primary database
    if database != app_ctx.primary_db_name:
        return {
            "status": "error",
            "message": f"Cannot write to '{database}' - only primary database '{app_ctx.primary_db_name}' is writable. Attached databases are read-only.",
            "database": database,
            "writable_database": app_ctx.primary_db_name,
        }

    # Attach and switch to database
    success, msg = ensure_database_attached(
        conn, app_ctx.attached_databases, database, app_ctx.databases_dir
    )
    if not success:
        return {"status": "error", "message": msg, "database": database}

    # BUGFIX: Pass primary_db_name to switch function
    success, msg = switch_database_context(conn, database, app_ctx.primary_db_name)
    if not success:
        return {"status": "error", "message": msg, "database": database}

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


# ... (continuing in next message - file too long)
def main():
    """Entry point for the MCP server."""
    print("Starting Kuzu Memory Graph MCP Server (FIXED VERSION)...", file=sys.stderr)
    mcp.run()


if __name__ == "__main__":
    main()
