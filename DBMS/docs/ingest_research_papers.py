#!/usr/bin/env python3
"""
Research Papers Database Ingestion Script
Migrates Neo4j research papers domain to KuzuDB

Usage:
    python ingest_research_papers.py
"""

import json
import kuzu
from datetime import datetime

def load_neo4j_data():
    """Load the Neo4j extracted data"""
    with open('neo4j_extract.json', 'r') as f:
        return json.load(f)

def create_database_schema(conn):
    """Create the KuzuDB schema for research papers"""

    print("Creating node tables...")

    # Create Node Tables
    node_tables = [
        """
        CREATE NODE TABLE IF NOT EXISTS ResearchPaper(
            identifier STRING PRIMARY KEY,
            title STRING,
            authors STRING[],
            year INT16,
            arxiv_id STRING,
            observations STRING[]
        )
        """,
        """
        CREATE NODE TABLE IF NOT EXISTS Concept(
            name STRING PRIMARY KEY,
            domain STRING,
            description STRING
        )
        """,
        """
        CREATE NODE TABLE IF NOT EXISTS Dataset(
            name STRING PRIMARY KEY,
            purpose STRING,
            observations STRING[]
        )
        """,
        """
        CREATE NODE TABLE IF NOT EXISTS Framework(
            name STRING PRIMARY KEY,
            type STRING,
            observations STRING[]
        )
        """
    ]

    for table in node_tables:
        conn.execute(table)

    print("Creating relationship tables...")

    # Create Relationship Tables
    rel_tables = [
        """
        CREATE REL TABLE IF NOT EXISTS PROPOSES(
            FROM ResearchPaper TO Concept,
            contribution STRING
        )
        """,
        """
        CREATE REL TABLE IF NOT EXISTS USES_DATASET(
            FROM ResearchPaper TO Dataset,
            methodology STRING
        )
        """,
        """
        CREATE REL TABLE IF NOT EXISTS IMPLEMENTS(
            FROM ResearchPaper TO Framework
        )
        """,
        """
        CREATE REL TABLE IF NOT EXISTS BUILDS_ON(
            FROM ResearchPaper TO ResearchPaper,
            relationship STRING
        )
        """
    ]

    for table in rel_tables:
        conn.execute(table)

    print("Schema creation completed!")

def ingest_research_papers(conn, data):
    """Ingest research papers from the extracted data"""

    research_domain = data['domains']['research_papers']

    print("Ingesting research papers...")

    # Ingest root community as a research paper with unique identifier
    root_community = research_domain['root_community']
    conn.execute("""
        CREATE (:ResearchPaper {
            identifier: $identifier,
            title: $title,
            authors: $authors,
            year: $year,
            arxiv_id: $arxiv_id,
            observations: $observations
        })
    """, {
        'identifier': f"community-{root_community['identifier']}",  # Make unique
        'title': f"Research Papers Community: {root_community['identifier']}",
        'authors': ['Knowledge Graph System'],
        'year': 2025,
        'arxiv_id': None,
        'observations': root_community['observations']
    })

    # Ingest individual papers
    papers = research_domain.get('papers', [])
    for paper in papers:
        if paper['observations']:  # Only ingest papers with observations
            conn.execute("""
                CREATE (:ResearchPaper {
                    identifier: $identifier,
                    title: $title,
                    authors: $authors,
                    year: $year,
                    arxiv_id: $arxiv_id,
                    observations: $observations
                })
            """, {
                'identifier': paper['identifier'],
                'title': paper['identifier'],  # Use identifier as title for now
                'authors': ['Various Authors'],  # Default if not specified
                'year': 2024,  # Default year
                'arxiv_id': None,
                'observations': paper['observations']
            })

            # Ingest related memories as separate papers if they're research citations
            related_memories = paper.get('related_memories', [])
            for memory in related_memories:
                if memory['type'] == 'Research Citation':
                    # Extract title from observations
                    title = next((obs.split(': ')[1] if ': ' in obs else obs
                                for obs in memory['observations'] if obs.startswith('Title:')),
                                memory['identifier'])

                    # Extract year from observations
                    year = None
                    for obs in memory['observations']:
                        if obs.startswith('Year:'):
                            try:
                                year = int(obs.split(': ')[1])
                            except:
                                year = 2024

                    # Extract ArXiv ID
                    arxiv_id = None
                    for obs in memory['observations']:
                        if obs.startswith('ArXiv:'):
                            arxiv_id = obs.split(': ')[1]

                    conn.execute("""
                        CREATE (:ResearchPaper {
                            identifier: $identifier,
                            title: $title,
                            authors: $authors,
                            year: $year,
                            arxiv_id: $arxiv_id,
                            observations: $observations
                        })
                    """, {
                        'identifier': memory['identifier'],
                        'title': title,
                        'authors': [memory['observations'][0].split(': ')[1]] if memory['observations'] and memory['observations'][0].startswith('Author:') else ['Unknown'],
                        'year': year if year else 2024,
                        'arxiv_id': arxiv_id,
                        'observations': memory['observations']
                    })

    print(f"Ingested {len(papers) + 1} research papers")

def ingest_concepts(conn, data):
    """Ingest concepts from related memories"""

    research_domain = data['domains']['research_papers']
    related_memories = research_domain.get('related_memories', [])

    print("Ingesting concepts...")

    for memory in related_memories:
        if memory['type'] in ['Theoretical Framework', 'Technical Method']:
            conn.execute("""
                CREATE (:Concept {
                    name: $name,
                    domain: $domain,
                    description: $description
                })
            """, {
                'name': memory['identifier'],
                'domain': memory['type'],
                'description': ' '.join(memory['observations'])
            })

    print("Concepts ingestion completed!")

def ingest_datasets(conn, data):
    """Ingest datasets from related memories"""

    research_domain = data['domains']['research_papers']

    print("Ingesting datasets...")

    # Look for datasets in paper related memories
    papers = research_domain.get('papers', [])
    for paper in papers:
        related_memories = paper.get('related_memories', [])
        for memory in related_memories:
            if memory['type'] == 'Research Resource':
                conn.execute("""
                    CREATE (:Dataset {
                        name: $name,
                        purpose: $purpose,
                        observations: $observations
                    })
                """, {
                    'name': memory['identifier'],
                    'purpose': memory['observations'][1] if len(memory['observations']) > 1 else 'Research dataset',
                    'observations': memory['observations']
                })

    print("Datasets ingestion completed!")

def ingest_frameworks(conn, data):
    """Ingest frameworks from related memories"""

    research_domain = data['domains']['research_papers']
    related_memories = research_domain.get('related_memories', [])

    print("Ingesting frameworks...")

    for memory in related_memories:
        if memory['type'] == 'Technical Method':
            conn.execute("""
                CREATE (:Framework {
                    name: $name,
                    type: $type,
                    observations: $observations
                })
            """, {
                'name': memory['identifier'],
                'type': memory['type'],
                'observations': memory['observations']
            })

    print("Frameworks ingestion completed!")

def create_relationships(conn, data):
    """Create relationships between entities"""

    research_domain = data['domains']['research_papers']

    print("Creating relationships...")

    # Create relationships between papers and their related memories
    papers = research_domain.get('papers', [])
    for paper in papers:
        related_memories = paper.get('related_memories', [])
        for memory in related_memories:
            if memory['type'] == 'Research Citation':
                # Create BUILDS_ON relationship
                try:
                    conn.execute("""
                        MATCH (p1:ResearchPaper {identifier: $paper_id})
                        MATCH (p2:ResearchPaper {identifier: $memory_id})
                        CREATE (p1)-[:BUILDS_ON {relationship: $relationship}]->(p2)
                    """, {
                        'paper_id': paper['identifier'],
                        'memory_id': memory['identifier'],
                        'relationship': 'Cites or responds to'
                    })
                except Exception as e:
                    print(f"Could not create relationship: {e}")

            elif memory['type'] == 'Research Resource':
                # Create USES_DATASET relationship
                try:
                    conn.execute("""
                        MATCH (p:ResearchPaper {identifier: $paper_id})
                        MATCH (d:Dataset {name: $memory_id})
                        CREATE (p)-[:USES_DATASET {methodology: $methodology}]->(d)
                    """, {
                        'paper_id': paper['identifier'],
                        'memory_id': memory['identifier'],
                        'methodology': 'Uses dataset for research validation'
                    })
                except Exception as e:
                    print(f"Could not create dataset relationship: {e}")

    print("Relationship creation completed!")

def validate_ingestion(conn):
    """Validate the ingestion by running some test queries"""

    print("\n=== Ingestion Validation ===")

    # Count nodes
    paper_count = conn.execute("MATCH (p:ResearchPaper) RETURN COUNT(p) AS count").get_next()[0]
    concept_count = conn.execute("MATCH (c:Concept) RETURN COUNT(c) AS count").get_next()[0]
    dataset_count = conn.execute("MATCH (d:Dataset) RETURN COUNT(d) AS count").get_next()[0]
    framework_count = conn.execute("MATCH (f:Framework) RETURN COUNT(f) AS count").get_next()[0]

    print(f"Research Papers: {paper_count}")
    print(f"Concepts: {concept_count}")
    print(f"Datasets: {dataset_count}")
    print(f"Frameworks: {framework_count}")

    # Count relationships
    builds_on_count = conn.execute("MATCH ()-[r:BUILDS_ON]->() RETURN COUNT(r) AS count").get_next()[0]
    uses_dataset_count = conn.execute("MATCH ()-[r:USES_DATASET]->() RETURN COUNT(r) AS count").get_next()[0]

    print(f"BUILDS_ON relationships: {builds_on_count}")
    print(f"USES_DATASET relationships: {uses_dataset_count}")

    # Sample data
    print("\n=== Sample Research Papers ===")
    result = conn.execute("MATCH (p:ResearchPaper) RETURN p.identifier, p.title LIMIT 5")
    while result.has_next():
        row = result.get_next()
        print(f"- {row[0]}: {row[1]}")

    print("\n=== Sample Concepts ===")
    result = conn.execute("MATCH (c:Concept) RETURN c.name, c.domain LIMIT 3")
    while result.has_next():
        row = result.get_next()
        print(f"- {row[0]} ({row[1]})")

def main():
    """Main ingestion function"""

    print("Starting Research Papers Database Ingestion...")

    # Load data
    data = load_neo4j_data()
    print(f"Loaded Neo4j data with {data['extraction_metadata']['valuable_nodes_extracted']} valuable nodes")

    # Create database
    db_path = "./research_papers.kuzu"
    print(f"Creating KuzuDB at: {db_path}")
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)

    try:
        # Create schema
        create_database_schema(conn)

        # Ingest data
        ingest_research_papers(conn, data)
        ingest_concepts(conn, data)
        ingest_datasets(conn, data)
        ingest_frameworks(conn, data)

        # Create relationships
        create_relationships(conn, data)

        # Validate
        validate_ingestion(conn)

        print(f"\n✅ Research Papers database successfully created at: {db_path}")
        print("Ready for querying with KuzuDB!")

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        raise
    finally:
        conn.close()
        db.close()

if __name__ == "__main__":
    main()