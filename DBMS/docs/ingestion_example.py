#!/usr/bin/env python3
"""
Example script for ingesting prompt engineering data into KuzuDB
with MLX embeddings for Apple Silicon optimization
"""

import kuzu
import json
from pathlib import Path
from typing import List, Dict, Any

def setup_database(db_path: str = "./prompt_engineering.kuzu") -> kuzu.Connection:
    """Initialize KuzuDB with vector extension"""
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)

    # Install vector extension for semantic search
    conn.execute("INSTALL vector;")
    conn.execute("LOAD EXTENSION vector;")

    return conn

def create_schema(conn: kuzu.Connection):
    """Create the database schema from Cypher file"""
    schema_file = Path("prompt_engineering_schema.cypher")
    if not schema_file.exists():
        raise FileNotFoundError("Schema file not found")

    with open(schema_file, 'r') as f:
        schema_content = f.read()

    # Execute schema statements
    for statement in schema_content.split(';'):
        statement = statement.strip()
        if statement and not statement.startswith('--'):
            try:
                conn.execute(statement)
                print(f"✓ Executed: {statement[:50]}...")
            except Exception as e:
                if "already exists" not in str(e):
                    print(f"⚠️  Warning: {e}")

def load_neo4j_data(file_path: str = "neo4j_extract.json") -> Dict[str, Any]:
    """Load the Neo4j extracted data"""
    with open(file_path, 'r') as f:
        return json.load(f)

def ingest_communities(conn: kuzu.Connection, data: Dict[str, Any]):
    """Ingest community data from Neo4j extraction"""
    prompt_data = data['domains']['prompt_engineering']

    # Ingest root community
    root = prompt_data['root_community']
    conn.execute("""
        CREATE (:Community {
            identifier: $id,
            type: $type,
            observations: $obs,
            created_date: current_date()
        })
    """, {
        'id': root['identifier'],
        'type': root['type'],
        'obs': root['observations']
    })
    print(f"✓ Created root community: {root['identifier']}")

    # Ingest subcommunities
    for subcomm in prompt_data['subcommunities']:
        conn.execute("""
            CREATE (:Community {
                identifier: $id,
                type: $type,
                observations: $obs,
                created_date: current_date()
            })
        """, {
            'id': subcomm['identifier'],
            'type': subcomm['type'],
            'obs': subcomm['observations']
        })
        print(f"✓ Created subcommunity: {subcomm['identifier']}")

    # Ingest key memories
    for memory in prompt_data['key_memories']:
        conn.execute("""
            CREATE (:Community {
                identifier: $id,
                type: $type,
                observations: $obs,
                created_date: current_date()
            })
        """, {
            'id': memory['identifier'],
            'type': memory['type'],
            'obs': memory['observations']
        })
        print(f"✓ Created key memory: {memory['identifier']}")

def create_hierarchy_relationships(conn: kuzu.Connection, data: Dict[str, Any]):
    """Create hierarchical relationships between communities"""
    prompt_data = data['domains']['prompt_engineering']

    root_id = prompt_data['root_community']['identifier']

    # Connect subcommunities to root
    for subcomm in prompt_data['subcommunities']:
        conn.execute("""
            MATCH (parent:Community {identifier: $root})
            MATCH (child:Community {identifier: $child})
            CREATE (child)-[:SUBCOMMUNITY]->(parent)
        """, {
            'root': root_id,
            'child': subcomm['identifier']
        })
        print(f"✓ Connected {subcomm['identifier']} -> {root_id}")

    # Connect key memories to relevant communities
    for memory in prompt_data['key_memories']:
        # For Query Templates, connect to Text-Based Techniques
        if memory['identifier'] == 'Query Templates':
            target = 'Text-Based Techniques'
            conn.execute("""
                MATCH (parent:Community {identifier: $target})
                MATCH (child:Community {identifier: $child})
                CREATE (child)-[:SUBCOMMUNITY]->(parent)
            """, {
                'target': target,
                'child': memory['identifier']
            })
            print(f"✓ Connected {memory['identifier']} -> {target}")

def add_ml_embeddings(conn: kuzu.Connection):
    """
    Add MLX embeddings to communities (Apple Silicon optimized)
    Note: Requires mlx-embeddings package
    """
    try:
        from mlx_embeddings.utils import load

        # Load 4-bit quantized model for efficiency
        model_name = "mlx-community/all-MiniLM-L6-v2-4bit"
        model, tokenizer = load(model_name)

        def get_embedding(text: str) -> List[float]:
            """Generate 384-dimensional embedding using MLX"""
            inputs = tokenizer.encode(text, return_tensors="mlx")
            outputs = model(inputs)
            return outputs.text_embeds.tolist()

        # Get all communities without embeddings
        result = conn.execute("""
            MATCH (c:Community)
            WHERE c.embedding IS NULL
            RETURN c.identifier, c.observations
        """)

        communities = []
        while result.has_next():
            row = result.get_next()
            communities.append((row[0], row[1]))

        print(f"🔄 Processing {len(communities)} communities for embeddings...")

        # Process in batches for efficiency
        batch_size = 8
        for i in range(0, len(communities), batch_size):
            batch = communities[i:i + batch_size]
            texts = []

            for identifier, observations in batch:
                # Combine observations for embedding
                if observations:
                    text = ' '.join(observations)
                else:
                    text = identifier
                texts.append(text)

            # Generate batch embeddings
            try:
                # Generate embeddings one by one if batch fails
                embeddings = []
                for text in texts:
                    embedding = get_embedding(text)
                    embeddings.append(embedding)

                # Store embeddings
                for (identifier, _), embedding in zip(batch, embeddings):
                    conn.execute("""
                        MATCH (c:Community {identifier: $id})
                        SET c.embedding = $embedding
                    """, {
                        'id': identifier,
                        'embedding': embedding
                    })

                print(f"✓ Processed batch {i//batch_size + 1}/{(len(communities)-1)//batch_size + 1}")

            except Exception as e:
                print(f"⚠️  Batch processing failed: {e}")
                continue

    except ImportError:
        print("⚠️  MLX embeddings not available. Install with: pip install mlx-embeddings")
        print("   Embeddings will be added manually later if needed.")

def validate_database(conn: kuzu.Connection):
    """Validate the database setup and content"""
    print("\n🔍 Database Validation:")

    # Check communities
    result = conn.execute("MATCH (c:Community) RETURN count(c) as count")
    count = result.get_next()[0]
    print(f"✓ Communities: {count}")

    # Check relationships
    result = conn.execute("MATCH ()-[r:SUBCOMMUNITY]->() RETURN count(r) as count")
    count = result.get_next()[0]
    print(f"✓ Subcommunity relationships: {count}")

    # Check embeddings
    result = conn.execute("MATCH (c:Community) WHERE c.embedding IS NOT NULL RETURN count(c) as count")
    count = result.get_next()[0]
    print(f"✓ Communities with embeddings: {count}")

    # Sample queries
    print("\n📊 Sample Data:")
    result = conn.execute("MATCH (c:Community) RETURN c.identifier, c.type LIMIT 5")
    while result.has_next():
        row = result.get_next()
        print(f"  - {row[0]} ({row[1]})")

def run_sample_queries(conn: kuzu.Connection):
    """Run example queries to demonstrate functionality"""
    print("\n🚀 Sample Queries:")

    # 1. Hierarchy traversal
    print("\n1. Community Hierarchy:")
    result = conn.execute("""
        MATCH (root:Community {identifier: 'Prompt Engineering Research'})
        <-[:SUBCOMMUNITY]-(subcommunities:Community)
        RETURN subcommunities.identifier as subcommunity
        ORDER BY subcommunities.identifier
    """)
    while result.has_next():
        row = result.get_next()
        print(f"   - {row[0]}")

    # 2. Community details
    print("\n2. Text-Based Techniques Community:")
    result = conn.execute("""
        MATCH (c:Community {identifier: 'Text-Based Techniques'})
        RETURN c.observations as observations
    """)
    if result.has_next():
        obs = result.get_next()[0]
        for i, observation in enumerate(obs, 1):
            print(f"   {i}. {observation}")

    # 3. Vector similarity (if embeddings available)
    print("\n3. Vector Similarity Search (if embeddings available):")
    result = conn.execute("""
        MATCH (c:Community)
        WHERE c.embedding IS NOT NULL
        RETURN count(c) as communities_with_embeddings
    """)
    if result.has_next() and result.get_next()[0] > 0:
        print("   Vector similarity search is available!")
        print("   Example query: Find communities similar to 'reasoning techniques'")
    else:
        print("   No embeddings found. Run add_ml_embeddings() to enable.")

def main():
    """Main ingestion pipeline"""
    print("🚀 Starting Prompt Engineering Database Ingestion")

    try:
        # Setup database
        print("\n📁 Setting up database...")
        conn = setup_database()

        # Create schema
        print("\n🏗️  Creating schema...")
        create_schema(conn)

        # Load data
        print("\n📊 Loading Neo4j data...")
        data = load_neo4j_data()

        # Ingest data
        print("\n💾 Ingesting communities...")
        ingest_communities(conn, data)

        print("\n🔗 Creating relationships...")
        create_hierarchy_relationships(conn, data)

        # Add embeddings
        print("\n🧠 Adding MLX embeddings...")
        add_ml_embeddings(conn)

        # Validate
        print("\n✅ Validating database...")
        validate_database(conn)

        # Sample queries
        run_sample_queries(conn)

        print("\n🎉 Ingestion complete!")
        print(f"📍 Database location: ./prompt_engineering.kuzu")
        print("💡 Use conn.close() when finished with the database")

    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        raise

if __name__ == "__main__":
    main()