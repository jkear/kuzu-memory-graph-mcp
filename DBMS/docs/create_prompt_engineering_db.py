#!/usr/bin/env python3
"""
Create the Prompt Engineering KuzuDB from Neo4j extracted data
"""

import kuzu
import json
from pathlib import Path

def create_database():
    """Create the prompt engineering database"""
    print("🚀 Creating Prompt Engineering KuzuDB")

    # Create database
    db = kuzu.Database("./prompt_engineering.kuzu")
    conn = kuzu.Connection(db)

    print("✓ Database created")

    # Install vector extension
    try:
        conn.execute("INSTALL vector;")
        conn.execute("LOAD EXTENSION vector;")
        print("✓ Vector extension loaded")
    except Exception as e:
        if "already loaded" in str(e):
            print("✓ Vector extension already loaded")
        else:
            print(f"⚠️  Vector extension warning: {e}")

    # Create Community table
    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Community(
            identifier STRING PRIMARY KEY,
            type STRING,
            observations STRING[],
            created_date DATE DEFAULT current_date(),
            embedding FLOAT[384]
        );
    """)
    print("✓ Community table created")

    # Create Technique table
    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Technique(
            name STRING PRIMARY KEY,
            description STRING,
            category STRING,
            observations STRING[],
            embedding FLOAT[384]
        );
    """)
    print("✓ Technique table created")

    # Create UseCase table
    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS UseCase(
            name STRING PRIMARY KEY,
            description STRING,
            domain STRING,
            embedding FLOAT[384]
        );
    """)
    print("✓ UseCase table created")

    # Create SUBCOMMUNITY relationship
    conn.execute("""
        CREATE REL TABLE IF NOT EXISTS SUBCOMMUNITY(
            FROM Community TO Community,
            MANY_ONE
        );
    """)
    print("✓ SUBCOMMUNITY relationship created")

    # Create CONTAINS relationship
    conn.execute("""
        CREATE REL TABLE IF NOT EXISTS CONTAINS(
            FROM Community TO Technique,
            context STRING
        );
    """)
    print("✓ CONTAINS relationship created")

    # Create BEST_FOR relationship
    conn.execute("""
        CREATE REL TABLE IF NOT EXISTS BEST_FOR(
            FROM Technique TO UseCase,
            evidence STRING[]
        );
    """)
    print("✓ BEST_FOR relationship created")

    return conn

def load_data(conn):
    """Load Neo4j extracted data"""
    with open("neo4j_extract.json", 'r') as f:
        return json.load(f)

def ingest_communities(conn, data):
    """Ingest community data"""
    prompt_data = data['domains']['prompt_engineering']

    # Root community
    root = prompt_data['root_community']
    conn.execute("""
        CREATE (:Community {
            identifier: $id,
            type: $type,
            observations: $obs,
            created_date: current_date()
        });
    """, {
        'id': root['identifier'],
        'type': root['type'],
        'obs': root['observations']
    })
    print(f"✓ Created root community: {root['identifier']}")

    # Subcommunities
    for subcomm in prompt_data['subcommunities']:
        conn.execute("""
            CREATE (:Community {
                identifier: $id,
                type: $type,
                observations: $obs,
                created_date: current_date()
            });
        """, {
            'id': subcomm['identifier'],
            'type': subcomm['type'],
            'obs': subcomm['observations']
        })
        print(f"✓ Created subcommunity: {subcomm['identifier']}")

    # Key memories
    for memory in prompt_data['key_memories']:
        conn.execute("""
            CREATE (:Community {
                identifier: $id,
                type: $type,
                observations: $obs,
                created_date: current_date()
            });
        """, {
            'id': memory['identifier'],
            'type': memory['type'],
            'obs': memory['observations']
        })
        print(f"✓ Created key memory: {memory['identifier']}")

def create_relationships(conn, data):
    """Create hierarchical relationships"""
    prompt_data = data['domains']['prompt_engineering']
    root_id = prompt_data['root_community']['identifier']

    # Connect subcommunities to root
    for subcomm in prompt_data['subcommunities']:
        conn.execute("""
            MATCH (parent:Community {identifier: $root})
            MATCH (child:Community {identifier: $child})
            CREATE (child)-[:SUBCOMMUNITY]->(parent);
        """, {
            'root': root_id,
            'child': subcomm['identifier']
        })
        print(f"✓ Connected {subcomm['identifier']} -> {root_id}")

    # Connect key memories
    for memory in prompt_data['key_memories']:
        if memory['identifier'] == 'Query Templates':
            target = 'Text-Based Techniques'
            conn.execute("""
                MATCH (parent:Community {identifier: $target})
                MATCH (child:Community {identifier: $child})
                CREATE (child)-[:SUBCOMMUNITY]->(parent);
            """, {
                'target': target,
                'child': memory['identifier']
            })
            print(f"✓ Connected {memory['identifier']} -> {target}")

def verify_database(conn):
    """Verify database contents"""
    print("\n🔍 Database Verification:")

    # Count communities
    result = conn.execute("MATCH (c:Community) RETURN count(c) as count;")
    count = result.get_next()[0]
    print(f"✓ Communities: {count}")

    # Count relationships
    result = conn.execute("MATCH ()-[r:SUBCOMMUNITY]->() RETURN count(r) as count;")
    count = result.get_next()[0]
    print(f"✓ Subcommunity relationships: {count}")

    # Show communities
    print("\n📊 Communities in database:")
    result = conn.execute("MATCH (c:Community) RETURN c.identifier, c.type ORDER BY c.identifier;")
    while result.has_next():
        row = result.get_next()
        print(f"   - {row[0]} ({row[1]})")

    # Show hierarchy
    print("\n🏗️  Community Hierarchy:")
    result = conn.execute("""
        MATCH (root:Community {identifier: 'Prompt Engineering Research'})
        <-[:SUBCOMMUNITY]-(subcommunities:Community)
        RETURN subcommunities.identifier as subcommunity
        ORDER BY subcommunities.identifier;
    """)
    while result.has_next():
        row = result.get_next()
        print(f"   └── {row[0]}")

def main():
    """Main function"""
    try:
        # Create database and schema
        conn = create_database()

        # Load data
        print("\n📊 Loading Neo4j data...")
        data = load_data(conn)

        # Ingest data
        print("\n💾 Ingesting communities...")
        ingest_communities(conn, data)

        print("\n🔗 Creating relationships...")
        create_relationships(conn, data)

        # Verify
        print("\n✅ Verifying database...")
        verify_database(conn)

        print("\n🎉 Database creation complete!")
        print(f"📍 Database location: ./prompt_engineering.kuzu")
        print("\n💡 Next steps:")
        print("   1. Add MLX embeddings: python add_embeddings.py")
        print("   2. Run sample queries: python sample_queries.py")

        conn.close()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()