# Research Papers Database - KuzuDB Schema

## Overview

This document defines the KuzuDB schema for the Research Papers domain, extracted from Neo4j knowledge graph. This schema organizes academic research papers, their associated concepts, datasets, and frameworks with rich relationships.

## Domain Description

Academic research papers on AI, LLMs, and machine learning with their associated concepts, datasets, and methodological frameworks. The domain includes 7 research papers with various relationships and theoretical frameworks.

## Schema Design

### Node Tables

#### ResearchPaper
```sql
CREATE NODE TABLE IF NOT EXISTS ResearchPaper(
    identifier STRING PRIMARY KEY,
    title STRING,
    authors STRING[],
    year INT16,
    arxiv_id STRING,
    observations STRING[]
);
```

**Purpose**: Stores academic research papers with metadata
**Key Features**:
- Primary identifier for unique paper identification
- Array support for multiple authors
- Year field for temporal analysis
- ArXiv ID for paper reference
- Observations array for research findings and contributions

#### Concept
```sql
CREATE NODE TABLE IF NOT EXISTS Concept(
    name STRING PRIMARY KEY,
    domain STRING,
    description STRING
);
```

**Purpose**: Theoretical concepts and frameworks introduced in papers
**Key Features**:
- Unique concept names
- Domain categorization (AI, ML, Reasoning, etc.)
- Rich descriptions for concept understanding

#### Dataset
```sql
CREATE NODE TABLE IF NOT EXISTS Dataset(
    name STRING PRIMARY KEY,
    purpose STRING,
    observations STRING[]
);
```

**Purpose**: Research datasets used or created in papers
**Key Features**:
- Dataset identification
- Purpose description
- Observations array for dataset characteristics and findings

#### Framework
```sql
CREATE NODE TABLE IF NOT EXISTS Framework(
    name STRING PRIMARY KEY,
    type STRING,
    observations STRING[]
);
```

**Purpose**: Methodological and theoretical frameworks
**Key Features**:
- Framework name and type
- Observations for framework details and applications

### Relationship Tables

#### PROPOSES
```sql
CREATE REL TABLE IF NOT EXISTS PROPOSES(
    FROM ResearchPaper TO Concept,
    contribution STRING
);
```

**Purpose**: Links papers to concepts they introduce or propose
**Direction**: ResearchPaper → Concept
**Attributes**: Contribution description

#### USES_DATASET
```sql
CREATE REL TABLE IF NOT EXISTS USES_DATASET(
    FROM ResearchPaper TO Dataset,
    methodology STRING
);
```

**Purpose**: Connects papers to datasets they use or create
**Direction**: ResearchPaper → Dataset
**Attributes**: Methodology description

#### IMPLEMENTS
```sql
CREATE REL TABLE IF NOT EXISTS IMPLEMENTS(
    FROM ResearchPaper TO Framework
);
```

**Purpose**: Links papers to frameworks they implement
**Direction**: ResearchPaper → Framework
**Attributes**: None (simple implementation relationship)

#### BUILDS_ON
```sql
CREATE REL TABLE IF NOT EXISTS BUILDS_ON(
    FROM ResearchPaper TO ResearchPaper,
    relationship STRING
);
```

**Purpose**: Academic citation and building relationships
**Direction**: ResearchPaper → ResearchPaper
**Attributes**: Type of relationship (citation, extension, critique)

## Complete Schema DDL

```sql
-- Create Node Tables
CREATE NODE TABLE IF NOT EXISTS ResearchPaper(
    identifier STRING PRIMARY KEY,
    title STRING,
    authors STRING[],
    year INT16,
    arxiv_id STRING,
    observations STRING[]
);

CREATE NODE TABLE IF NOT EXISTS Concept(
    name STRING PRIMARY KEY,
    domain STRING,
    description STRING
);

CREATE NODE TABLE IF NOT EXISTS Dataset(
    name STRING PRIMARY KEY,
    purpose STRING,
    observations STRING[]
);

CREATE NODE TABLE IF NOT EXISTS Framework(
    name STRING PRIMARY KEY,
    type STRING,
    observations STRING[]
);

-- Create Relationship Tables
CREATE REL TABLE IF NOT EXISTS PROPOSES(
    FROM ResearchPaper TO Concept,
    contribution STRING
);

CREATE REL TABLE IF NOT EXISTS USES_DATASET(
    FROM ResearchPaper TO Dataset,
    methodology STRING
);

CREATE REL TABLE IF NOT EXISTS IMPLEMENTS(
    FROM ResearchPaper TO Framework
);

CREATE REL TABLE IF NOT EXISTS BUILDS_ON(
    FROM ResearchPaper TO ResearchPaper,
    relationship STRING
);
```

## Sample Data Ingestion

### Research Papers
```sql
-- The Illusion of Thinking paper
CREATE (:ResearchPaper {
    identifier: 'The Illusion of Thinking',
    title: 'The Illusion of Thinking: Analyzing Large Reasoning Models',
    authors: ['Apple Research Team'],
    year: 2025,
    arxiv_id: '2504.xxxxx',
    observations: [
        'Research paper analyzing Large Reasoning Models (LRMs) through controlled puzzle environments',
        'Published by Apple researchers in 2025',
        'Reveals fundamental limitations in current reasoning models',
        'Introduces three distinct complexity regimes for model performance'
    ]
});

-- Kambhampati Paper (as related research)
CREATE (:ResearchPaper {
    identifier: 'Kambhampati Paper',
    title: 'Stop Anthropomorphizing Intermediate Tokens as Reasoning/Thinking Traces!',
    authors: ['Subbarao Kambhampati'],
    year: 2025,
    arxiv_id: '2504.09762',
    observations: [
        'Main critique: Anthropomorphization of token generation',
        'Questions the interpretation of intermediate tokens as reasoning traces'
    ]
});
```

### Concepts
```sql
-- Psychometric AI Evaluation Framework
CREATE (:Concept {
    name: 'Psychometric AI Evaluation',
    domain: 'AI Assessment',
    description: 'Applies psychometric principles to AI assessment with three-stage framework: construct identification, measurement, validation'
});

-- Large Reasoning Models
CREATE (:Concept {
    name: 'Large Reasoning Models',
    domain: 'AI Architecture',
    description: 'AI models designed for complex reasoning tasks with intermediate token generation'
});
```

### Datasets
```sql
CREATE (:Dataset {
    name: 'GSM-NoOp Dataset',
    purpose: 'Probes LLM reasoning limitations through irrelevant information injection',
    observations: [
        'Developed by Mirzadeh et al. (2024)',
        'Key finding: Models consistently incorporate irrelevant numbers into calculations',
        'Impact: Some models show over 65% drop in accuracy'
    ]
});
```

### Frameworks
```sql
CREATE (:Framework {
    name: 'Traditional GraphRAG',
    type: 'Technical Method',
    observations: [
        'Graph-based RAG using entity knowledge graphs with query-focused summarization',
        'Uses LLM to extract and describe entities and relationships',
        'Form of breadth-first search using community structure'
    ]
});
```

### Relationships
```sql
-- Paper proposes concepts
MATCH (paper:ResearchPaper {identifier: 'The Illusion of Thinking'})
MATCH (concept:Concept {name: 'Large Reasoning Models'})
CREATE (paper)-[:PROPOSES {contribution: 'Introduces and analyzes LRMs through controlled experiments'}]->(concept);

-- Paper uses datasets
MATCH (paper:ResearchPaper {identifier: 'The Illusion of Thinking'})
MATCH (dataset:Dataset {name: 'GSM-NoOp Dataset'})
CREATE (paper)-[:USES_DATASET {methodology: 'Uses dataset to demonstrate reasoning limitations'}]->(dataset);

-- Academic relationships
MATCH (paper1:ResearchPaper {identifier: 'The Illusion of Thinking'})
MATCH (paper2:ResearchPaper {identifier: 'Kambhampati Paper'})
CREATE (paper1)-[:BUILDS_ON {relationship: 'Responds to critique about anthropomorphization'}]->(paper2);
```

## Example Queries

### Find all papers by year
```sql
MATCH (p:ResearchPaper)
RETURN p.identifier, p.title, p.year
ORDER BY p.year DESC;
```

### Get all concepts from a specific domain
```sql
MATCH (c:Concept {domain: 'AI Assessment'})
RETURN c.name, c.description;
```

### Find papers that use specific datasets
```sql
MATCH (p:ResearchPaper)-[:USES_DATASET]->(d:Dataset)
RETURN p.title, d.name, d.purpose;
```

### Trace concept evolution through papers
```sql
MATCH (p1:ResearchPaper)-[:PROPOSES]->(c:Concept)<-[:PROPOSES]-(p2:ResearchPaper)
WHERE p1.year < p2.year
RETURN p1.title as earlier, c.name as concept, p2.title as later
ORDER BY p1.year, p2.year;
```

### Get research lineage for a paper
```sql
MATCH (p:ResearchPaper {identifier: 'The Illusion of Thinking'})-[:BUILDS_ON*1..3]->(related:ResearchPaper)
RETURN p.identifier as main_paper, related.identifier as builds_on, related.year
ORDER BY related.year DESC;
```

## Performance Considerations

### Indexes
```sql
-- Create indexes for frequently queried fields
CREATE INDEX paper_year_idx FOR ResearchPaper ON (year);
CREATE INDEX paper_arxiv_idx FOR ResearchPaper ON (arxiv_id);
CREATE INDEX concept_domain_idx FOR Concept ON (domain);
CREATE INDEX dataset_purpose_idx FOR Dataset ON (purpose);
```

### Array Handling
- Use `STRING[]` for observations arrays
- CSV import: Use JSON format: `"['observation1','observation2','observation3']"`
- Query array contents: `observations[0]` for first element

### Batch Operations
- Use `COPY FROM` for bulk data loading
- Process papers in batches of 100-1000 records
- Create relationships after all nodes are loaded

## Migration Notes

### Data Type Mapping
- Neo4j `String[]` → Kuzu `STRING[]`
- Neo4j `Integer` → Kuzu `INT16` (for years)
- Neo4j `Array<String>` → Kuzu `STRING[]` (for authors)

### Relationship Cardinality
- Paper → Concept: Many-to-Many (papers can propose multiple concepts)
- Paper → Dataset: Many-to-Many (papers can use multiple datasets)
- Paper → Framework: Many-to-Many (papers can implement multiple frameworks)
- Paper → Paper: Many-to-Many (papers can build on multiple papers)

### Vector Extension (Optional)
If semantic search is needed for paper content:
```sql
INSTALL vector;
LOAD EXTENSION vector;

-- Add embedding column to ResearchPaper
ALTER TABLE ResearchPaper ADD embedding FLOAT[768];
```

## Usage Patterns

### Academic Research Tracking
1. **Literature Review**: Trace concept development through papers
2. **Citation Analysis**: Find papers that build on specific research
3. **Dataset Usage**: Track which papers use specific datasets
4. **Framework Evolution**: Monitor how frameworks evolve across papers

### Knowledge Discovery
1. **Concept Clustering**: Group papers by proposed concepts
2. **Temporal Analysis**: Analyze research trends over time
3. **Collaboration Networks**: Identify co-authorship patterns
4. **Methodology Transfer**: Find cross-domain methodology applications

## Schema Extensibility

The schema can be extended with:
- **Author nodes**: Separate authors as entities for collaboration analysis
- **Venue nodes**: Conferences and journals for publication tracking
- **Keyword nodes**: Enhanced categorization and search
- **Citation counts**: Quantitative impact metrics
- **Review status**: Peer review and acceptance tracking