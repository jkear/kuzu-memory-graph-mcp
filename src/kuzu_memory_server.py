#!/usr/bin/env python3
"""
Kuzu Memory Graph MCP Server

A high-performance LLM memory server using Kuzu graph database with MLX embeddings.
Follows MCP FastMCP patterns and uses unified entity model for simplicity.
"""

import asyncio
import os
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


def generate_embedding(model: Any, tokenizer: Any, text: str) -> list[float]:
    """Generate 384-dimensional embedding using MLX."""
    if not text or not text.strip():
        return [0.0] * 384

    import mlx.core as mx
    inputs = tokenizer.encode(text.strip(), return_tensors="mlx")
    outputs = model(inputs)
    return outputs.text_embeds.tolist()


def batch_generate_embeddings(model: Any, tokenizer: Any, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts efficiently."""
    if not texts:
        return []

    import mlx.core as mx
    # Filter empty texts and prepare batch
    valid_texts = [(text.strip() if text else "") for text in texts if text and text.strip()]

    if not valid_texts:
        return [[0.0] * 384] * len(texts)

    inputs = tokenizer.batch_encode_plus(
        valid_texts,
        return_tensors="mlx",
        padding=True,
        truncation=True,
        max_length=512
    )
    outputs = model(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

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


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with database and model initialization."""
    # Get database path from environment
    db_path = os.getenv('KUZU_MEMORY_DB_PATH', './memory.kuzu')

    print(f"Initializing Kuzu database at: {db_path}")

    # Initialize Kuzu database
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)

    # Install vector extension
    try:
        conn.execute("INSTALL vector;")
        conn.execute("LOAD EXTENSION vector;")
        print("Vector extension loaded successfully")
    except Exception as e:
        print(f"Vector extension might already be loaded: {e}")

    # Create schema
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

    # Create vector index if it doesn't exist
    try:
        conn.execute("CALL CREATE_VECTOR_INDEX('Entity', 'entity_embedding_idx', 'embedding');")
        print("Vector index created successfully")
    except Exception as e:
        print(f"Vector index might already exist: {e}")

    # Initialize MLX embeddings model
    print("Loading MLX embeddings model...")
    try:
        from mlx_embeddings.utils import load
        embedding_model, tokenizer = load("mlx-community/all-MiniLM-L6-v2-4bit")
        print(f"MLX model loaded successfully (dimension: 384)")
    except Exception as e:
        print(f"Failed to load MLX model: {e}")
        # Fallback to sentence transformers
        print("Falling back to sentence transformers...")
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
        globals()['generate_embedding'] = lambda model, tokenizer, text: fallback_generate_embedding(text)
        globals()['batch_generate_embeddings'] = lambda model, tokenizer, texts: fallback_batch_generate_embeddings(texts)

    try:
        yield AppContext(db=db, conn=conn, embedding_model=embedding_model, tokenizer=tokenizer)
    finally:
        conn.close()
        db.close()
        print("Database connections closed")


# Create FastMCP server with lifespan management
mcp = FastMCP("kuzu-memory-graph", lifespan=app_lifespan)


# === MCP Tools ===

@mcp.tool()
async def create_entity(
    ctx: Context[ServerSession, AppContext],
    name: str,
    entity_type: str,
    observations: Optional[list[str]] = None
) -> dict[str, Any]:
    """Create a new entity in the knowledge graph.

    Args:
        name: Unique name for the entity
        entity_type: Type/category of the entity (person, concept, document, etc.)
        observations: List of observations/facts about the entity
    """
    app_ctx = ctx.request_context.lifespan_context
    conn = app_ctx.conn

    # Prepare text for embedding
    text_to_embed = ' '.join(observations or [name])

    # Generate embedding using MLX
    embedding = generate_embedding(app_ctx.embedding_model, app_ctx.tokenizer, text_to_embed)

    try:
        result = conn.execute("""
            CREATE (e:Entity {
                name: $name,
                type: $entity_type,
                observations: $observations,
                embedding: $embedding,
                created_date: current_date(),
                updated_date: current_date()
            })
            RETURN e.name as name, e.type as type
        """, {
            "name": name,
            "entity_type": entity_type,
            "observations": observations or [],
            "embedding": embedding
        })

        return {
            "status": "created",
            "name": name,
            "type": entity_type,
            "observations_count": len(observations or []),
            "embedded": True
        }
    except Exception as e:
        if "duplicate" in str(e).lower():
            return {
                "status": "exists",
                "name": name,
                "type": entity_type,
                "message": "Entity already exists"
            }
        raise e


@mcp.tool()
async def create_relationship(
    ctx: Context[ServerSession, AppContext],
    from_entity: str,
    to_entity: str,
    relationship_type: str,
    confidence: float = 1.0
) -> dict[str, Any]:
    """Create a relationship between two entities.

    Args:
        from_entity: Name of the source entity
        to_entity: Name of the target entity
        relationship_type: Type of relationship (WORKS_WITH, KNOWS, RELATED_TO, etc.)
        confidence: Confidence score for the relationship (0.0-1.0)
    """
    app_ctx = ctx.request_context.lifespan_context
    conn = app_ctx.conn

    try:
        result = conn.execute("""
            MATCH (from:Entity {name: $from_name})
            MATCH (to:Entity {name: $to_name})
            CREATE (from)-[r:RELATED_TO {
                relationship_type: $relationship_type,
                confidence: $confidence,
                created_date: current_date()
            }]->(to)
            RETURN from.name as from_name, to.name as to_name, r.relationship_type as type
        """, {
            "from_name": from_entity,
            "to_name": to_entity,
            "relationship_type": relationship_type,
            "confidence": confidence
        })

        return {
            "status": "created",
            "from": from_entity,
            "to": to_entity,
            "relationship_type": relationship_type,
            "confidence": confidence
        }
    except Exception as e:
        if "does not exist" in str(e).lower():
            return {
                "status": "error",
                "message": f"One or both entities not found: {from_entity}, {to_entity}"
            }
        raise e


@mcp.tool()
async def add_observations(
    ctx: Context[ServerSession, AppContext],
    entity_name: str,
    observations: list[str]
) -> dict[str, Any]:
    """Add new observations to an existing entity.

    Args:
        entity_name: Name of the entity to update
        observations: List of new observations to add
    """
    app_ctx = ctx.request_context.lifespan_context
    conn = app_ctx.conn

    # Get current entity
    result = conn.execute("""
        MATCH (e:Entity {name: $name})
        RETURN e.observations as current_obs, e.type as type
    """, {"name": entity_name})

    if not result.has_next():  # type: ignore
        return {
            "status": "error",
            "message": f"Entity '{entity_name}' not found"
        }

    current_obs, entity_type = result.get_next()  # type: ignore
    current_obs = list(current_obs) if current_obs else []  # type: ignore

    # Filter out existing observations
    new_obs = [obs for obs in observations if obs not in current_obs]

    if not new_obs:
        return {
            "status": "no_change",
            "entity_name": entity_name,
            "message": "All observations already exist"
        }

    # Update observations and regenerate embedding
    updated_obs = current_obs + new_obs
    text_to_embed = ' '.join(updated_obs)
    embedding = generate_embedding(app_ctx.embedding_model, app_ctx.tokenizer, text_to_embed)

    conn.execute("""
        MATCH (e:Entity {name: $name})
        SET e.observations = $observations,
            e.embedding = $embedding,
            e.updated_date = current_date()
    """, {
        "name": entity_name,
        "observations": updated_obs,
        "embedding": embedding
    })

    return {
        "status": "updated",
        "entity_name": entity_name,
        "added_observations": new_obs,
        "total_observations": len(updated_obs),
        "reembedded": True
    }


@mcp.tool()
async def search_entities(
    ctx: Context[ServerSession, AppContext],
    query: str,
    limit: int = 10
) -> dict[str, Any]:
    """Search entities using text-based search.

    Args:
        query: Search query string
        limit: Maximum number of results
    """
    app_ctx = ctx.request_context.lifespan_context
    conn = app_ctx.conn

    result = conn.execute("""
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($query)
           OR toLower(e.type) CONTAINS toLower($query)
           OR ANY(obs IN e.observations WHERE toLower(obs) CONTAINS toLower($query))
        RETURN e.name, e.type, e.observations, e.created_date
        ORDER BY e.updated_date DESC
        LIMIT $limit
    """, {
        "query": query.lower(),
        "limit": limit
    })

    entities = []
    while result.has_next():  # type: ignore
        row = result.get_next()  # type: ignore
        entities.append({
            "name": row[0],  # type: ignore
            "type": row[1],  # type: ignore
            "observations": list(row[2]) if row[2] else [],  # type: ignore
            "created_date": str(row[3])  # type: ignore
        })

    return {
        "query": query,
        "entities": entities,
        "count": len(entities)
    }


@mcp.tool()
async def semantic_search(
    ctx: Context[ServerSession, AppContext],
    query: str,
    limit: int = 10,
    threshold: float = 0.3
) -> dict[str, Any]:
    """Search entities using semantic similarity.

    Args:
        query: Search query for semantic matching
        limit: Maximum number of results
        threshold: Minimum similarity threshold (0.0-1.0)
    """
    app_ctx = ctx.request_context.lifespan_context
    conn = app_ctx.conn

    # Generate query embedding
    query_embedding = generate_embedding(app_ctx.embedding_model, app_ctx.tokenizer, query)

    # Perform vector similarity search
    try:
        result = conn.execute("""
            MATCH (e:Entity)
            WHERE e.embedding IS NOT NULL
            WITH e, array_cosine_similarity(e.embedding, $query_embedding) as similarity
            WHERE similarity >= $threshold
            RETURN e.name, e.type, e.observations, similarity
            ORDER BY similarity DESC
            LIMIT $limit
        """, {
            "query_embedding": query_embedding,
            "threshold": threshold,
            "limit": limit
        })
    except Exception:
        # Fallback to manual similarity calculation if array_cosine_similarity not available
        result = conn.execute("""
            MATCH (e:Entity)
            WHERE e.embedding IS NOT NULL
            RETURN e.name, e.type, e.observations, e.embedding
        """)

        entities = []
        while result.has_next():  # type: ignore
            row = result.get_next()  # type: ignore
            entities.append({
                "name": row[0],  # type: ignore
                "type": row[1],  # type: ignore
                "observations": list(row[2]) if row[2] else [],  # type: ignore
                "embedding": row[3]  # type: ignore
            })

        # Calculate similarities manually
        import numpy as np
        query_vec = np.array(query_embedding)
        similarities = []

        for entity in entities:
            entity_vec = np.array(entity["embedding"])
            if np.linalg.norm(query_vec) > 0 and np.linalg.norm(entity_vec) > 0:
                similarity = np.dot(query_vec, entity_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(entity_vec))
                if similarity >= threshold:
                    similarities.append({
                        "name": entity["name"],
                        "type": entity["type"],
                        "observations": entity["observations"],
                        "similarity": float(similarity)
                    })

        # Sort by similarity and limit results
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        similarities = similarities[:limit]

        return {
            "query": query,
            "entities": similarities,
            "count": len(similarities),
            "threshold": threshold,
            "method": "manual_calculation"
        }

    entities = []
    while result.has_next():  # type: ignore
        row = result.get_next()  # type: ignore
        entities.append({
            "name": row[0],  # type: ignore
            "type": row[1],  # type: ignore
            "observations": list(row[2]) if row[2] else [],  # type: ignore
            "similarity": float(row[3])  # type: ignore
        })

    return {
        "query": query,
        "entities": entities,
        "count": len(entities),
        "threshold": threshold,
        "method": "vector_similarity"
    }


@mcp.tool()
async def get_related_entities(
    ctx: Context[ServerSession, AppContext],
    entity_name: str,
    max_depth: int = 2,
    limit: int = 20
) -> dict[str, Any]:
    """Get entities related to the specified entity through relationship traversal.

    Args:
        entity_name: Name of the entity to find relations for
        max_depth: Maximum relationship depth to traverse
        limit: Maximum number of results
    """
    app_ctx = ctx.request_context.lifespan_context
    conn = app_ctx.conn

    result = conn.execute(f"""
        MATCH path = (start:Entity {{name: $name}})-[:RELATED_TO*1..{max_depth}]-(related:Entity)
        WHERE related.name <> $name
        RETURN DISTINCT related.name, related.type, related.observations,
               length(path) as distance,
               [r IN relationships(path) | r.relationship_type] as relationship_path,
               [r IN relationships(path) | r.confidence] as confidence_path
        ORDER BY distance, related.name
        LIMIT $limit
    """, {
        "name": entity_name,
        "limit": limit
    })

    entities = []
    while result.has_next():  # type: ignore
        row = result.get_next()  # type: ignore
        entities.append({
            "name": row[0],  # type: ignore
            "type": row[1],  # type: ignore
            "observations": list(row[2]) if row[2] else [],  # type: ignore
            "distance": row[3],  # type: ignore
            "relationship_path": row[4],  # type: ignore
            "confidence_path": row[5]  # type: ignore
        })

    return {
        "entity_name": entity_name,
        "entities": entities,
        "count": len(entities),
        "max_depth": max_depth
    }


@mcp.tool()
async def get_graph_summary(
    ctx: Context[ServerSession, AppContext]
) -> dict[str, Any]:
    """Get a summary of the entire knowledge graph."""
    app_ctx = ctx.request_context.lifespan_context
    conn = app_ctx.conn

    # Get counts
    entity_count = conn.execute("MATCH (e:Entity) RETURN COUNT(e)").get_next()[0]  # type: ignore
    relation_count = conn.execute("MATCH ()-[r]->() RETURN COUNT(r)").get_next()[0]  # type: ignore

    # Get entity types
    result = conn.execute("""
        MATCH (e:Entity)
        RETURN e.type, COUNT(e)
        ORDER BY COUNT(e) DESC
    """)

    entity_types = []
    while result.has_next():  # type: ignore
        row = result.get_next()  # type: ignore
        entity_types.append({"type": row[0], "count": row[1]})  # type: ignore

    # Get relationship types
    result = conn.execute("""
        MATCH ()-[r:RELATED_TO]->()
        RETURN r.relationship_type, COUNT(r)
        ORDER BY COUNT(r) DESC
    """)

    relationship_types = []
    while result.has_next():  # type: ignore
        row = result.get_next()  # type: ignore
        relationship_types.append({"type": row[0], "count": row[1]})  # type: ignore

    return {
        "stats": {
            "entities": entity_count,
            "relationships": relation_count,
            "entity_types": len(entity_types),
            "relationship_types": len(relationship_types)
        },
        "entity_types": entity_types,
        "relationship_types": relationship_types
    }


def main():
    """Run the Kuzu Memory Graph MCP server."""
    print("Starting Kuzu Memory Graph MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()