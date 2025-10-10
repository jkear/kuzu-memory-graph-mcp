#!/usr/bin/env python3
"""
Simple demonstration queries for the Prompt Engineering KuzuDB
"""

import kuzu

def demo_database():
    """Demonstrate the prompt engineering database"""
    print("🎯 Prompt Engineering KuzuDB Demo")
    print("=" * 50)

    # Connect to database
    db = kuzu.Database("./prompt_engineering.kuzu")
    conn = kuzu.Connection(db)

    # 1. Basic Community Overview
    print("\n1. 📊 COMMUNITY OVERVIEW")
    result = conn.execute("MATCH (c:Community) RETURN c.identifier, c.type")
    communities = []
    while result.has_next():
        row = result.get_next()
        communities.append(row)

    communities.sort(key=lambda x: x[0])  # Sort alphabetically
    for name, type_val in communities:
        print(f"   • {name} ({type_val})")

    # 2. Root Community Details
    print("\n2. 🌳 ROOT COMMUNITY")
    result = conn.execute("""
        MATCH (c:Community {identifier: 'Prompt Engineering Research'})
        RETURN c.observations
    """)
    if result.has_next():
        obs = result.get_next()[0]
        for i, observation in enumerate(obs, 1):
            print(f"   {i}. {observation}")

    # 3. Direct Subcommunities
    print("\n3. 🏗️  SUBCOMMUNITIES")
    result = conn.execute("""
        MATCH (root:Community {identifier: 'Prompt Engineering Research'})
        <-[:SUBCOMMUNITY]-(sub:Community)
        RETURN sub.identifier, sub.type
    """)
    subcommunities = []
    while result.has_next():
        row = result.get_next()
        subcommunities.append(row)

    subcommunities.sort(key=lambda x: x[0])
    for name, type_val in subcommunities:
        print(f"   └── {name} ({type_val})")

    # 4. Text-Based Techniques Details
    print("\n4. 📝 TEXT-BASED TECHNIQUES")
    result = conn.execute("""
        MATCH (c:Community {identifier: 'Text-Based Techniques'})
        RETURN c.observations
    """)
    if result.has_next():
        obs = result.get_next()[0]
        for i, observation in enumerate(obs, 1):
            print(f"   {i}. {observation}")

    # 5. Query Templates Documentation
    print("\n5. 📚 QUERY TEMPLATES")
    result = conn.execute("""
        MATCH (c:Community {identifier: 'Query Templates'})
        RETURN c.observations
    """)
    if result.has_next():
        obs = result.get_next()[0]
        for i, observation in enumerate(obs, 1):
            print(f"   {i}. {observation}")

    # 6. Statistics
    print("\n6. 📈 DATABASE STATISTICS")

    # Count communities
    result = conn.execute("MATCH (c:Community) RETURN count(c)")
    community_count = result.get_next()[0]
    print(f"   • Total Communities: {community_count}")

    # Count relationships
    result = conn.execute("MATCH ()-[r:SUBCOMMUNITY]->() RETURN count(r)")
    rel_count = result.get_next()[0]
    print(f"   • Subcommunity Relationships: {rel_count}")

    # Count by type
    result = conn.execute("MATCH (c:Community) RETURN c.type, count(c)")
    type_counts = {}
    while result.has_next():
        row = result.get_next()
        type_counts[row[0]] = row[1]

    for type_val, count in type_counts.items():
        print(f"   • {type_val}: {count}")

    # 7. Full Hierarchy
    print("\n7. 🌲 COMPLETE HIERARCHY")
    result = conn.execute("""
        MATCH (root:Community {identifier: 'Prompt Engineering Research'})
        RETURN root.identifier as root
    """)
    root_name = result.get_next()[0]
    print(f"   📁 {root_name}")

    result = conn.execute("""
        MATCH (root:Community {identifier: 'Prompt Engineering Research'})
        <-[:SUBCOMMUNITY]-(sub:Community)
        RETURN sub.identifier as sub
    """)
    subcommunities = []
    while result.has_next():
        subcommunities.append(result.get_next()[0])

    subcommunities.sort()
    for sub in subcommunities:
        print(f"      └── 📂 {sub}")

    # 8. Vector Capabilities Info
    print("\n8. 🧠 VECTOR SIMILARITY CAPABILITIES")
    print("   • Vector Extension: Installed and loaded")
    print("   • Embedding Dimension: 384 (MLX compatible)")
    print("   • Available Functions:")
    print("     - array_cosine_similarity()")
    print("     - array_distance()")
    print("     - array_dot_product()")
    print("   • Status: Ready for MLX embeddings")

    print("\n" + "=" * 50)
    print("✅ Demo completed successfully!")
    print("\n💡 Next steps:")
    print("   1. Add MLX embeddings for semantic search")
    print("   2. Populate Technique and UseCase tables")
    print("   3. Create CONTAINS and BEST_FOR relationships")
    print("   4. Implement vector similarity queries")

    conn.close()

if __name__ == "__main__":
    demo_database()