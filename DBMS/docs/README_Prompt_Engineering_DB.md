# Prompt Engineering KuzuDB Database

## Overview

This database contains the first extracted domain from the Neo4j knowledge graph: **Prompt Engineering Research**. It's a comprehensive knowledge base for prompt engineering techniques, based on The Prompt Report systematic survey, implemented in KuzuDB with vector similarity search capabilities.

## 🎯 Domain Summary

**Prompt Engineering Research** is the root community containing:
- 6 major subcommunities covering different technique categories
- Query templates and documentation resources
- Support for 58 distinct text-based prompting methods
- Hierarchical organization with semantic search capabilities

## 📁 Files Generated

### Core Database Files
- `prompt_engineering.kuzu` - Main KuzuDB database file
- `prompt_engineering_schema.md` - Comprehensive schema documentation
- `prompt_engineering_schema.cypher` - DDL statements for schema creation

### Data Ingestion Scripts
- `create_prompt_engineering_db.py` - Database creation and data ingestion
- `demo_queries.py` - Working demonstration queries
- `sample_queries.py` - Advanced query examples (some KuzuDB syntax limitations)

### Documentation
- `README_Prompt_Engineering_DB.md` - This file
- `test_schema.py` - Schema validation script

## 🏗️ Database Structure

### Node Tables

#### Community
```cypher
identifier STRING PRIMARY KEY,
type STRING,
observations STRING[],
created_date DATE DEFAULT current_date(),
embedding FLOAT[384]  // For semantic search
```

**Communities (8 total):**
- **Prompt Engineering Research** (Root)
- **Text-Based Techniques** (58 methods, 6 categories)
- **Multilingual Techniques** (Cross-language methods)
- **Multimodal Techniques** (Vision/audio/text)
- **Agent Techniques** (Multi-agent systems)
- **Evaluation Methods** (Assessment techniques)
- **Security and Alignment** (Safety methods)
- **Query Templates** (Documentation)

#### Technique
```cypher
name STRING PRIMARY KEY,
description STRING,
category STRING,
observations STRING[],
embedding FLOAT[384]
```

#### UseCase
```cypher
name STRING PRIMARY KEY,
description STRING,
domain STRING,
embedding FLOAT[384]
```

### Relationship Tables

#### SUBCOMMUNITY
- **Community → Community** (MANY_ONE)
- Hierarchical relationships between communities

#### CONTAINS
- **Community → Technique**
- Links communities to their techniques

#### BEST_FOR
- **Technique → UseCase**
- Connects techniques to optimal use cases

## 🚀 Quick Start

### 1. Create Database
```bash
# Activate virtual environment
source .venv/bin/activate

# Create database with sample data
python create_prompt_engineering_db.py
```

### 2. Run Demo
```bash
# See database in action
python demo_queries.py
```

### 3. Interactive Queries
```python
import kuzu

# Connect to database
db = kuzu.Database("./prompt_engineering.kuzu")
conn = kuzu.Connection(db)

# Explore communities
result = conn.execute("MATCH (c:Community) RETURN c.identifier, c.type")
while result.has_next():
    row = result.get_next()
    print(f"{row[0]} ({row[1]})")
```

## 🧠 Vector Search Capabilities

### MLX Integration (Apple Silicon)
The database is optimized for Apple Silicon with MLX embeddings:

```python
# Install MLX for embeddings
pip install mlx-embeddings

# Load 4-bit quantized model
from mlx_embeddings.utils import load
model_name = "mlx-community/all-MiniLM-L6-v2-4bit"
model, tokenizer = load(model_name)

# Generate 384-dimensional embeddings
def get_embedding(text):
    inputs = tokenizer.encode(text, return_tensors="mlx")
    outputs = model(inputs)
    return outputs.text_embeds.tolist()
```

### Semantic Search Example
```cypher
-- Find similar communities
MATCH (c:Community)
WHERE c.embedding IS NOT NULL
WITH c, array_cosine_similarity(c.embedding, $query_embedding) as similarity
RETURN c.identifier, similarity
ORDER BY similarity DESC
LIMIT 5;
```

## 📊 Sample Query Results

The demo shows:
- **8 communities** with hierarchical structure
- **7 subcommunity relationships**
- **Query templates** with 15 ready-to-use examples
- **Text-based techniques** covering 58 methods
- **Vector extension** ready for semantic search

## 🔍 Key Features

### Hierarchical Organization
- Root community with 6 specialized subcommunities
- Clear parent-child relationships with MANY_ONE constraints
- Documentation nested under relevant communities

### Vector Similarity Search
- 384-dimensional embeddings (MLX compatible)
- Native KuzuDB vector functions
- Apple Silicon optimized with MLX

### Query Templates
- Standardized Cypher queries for common operations
- 15 ready-to-use examples for practitioners
- Optimized patterns for technique discovery

### Extensible Design
- Ready for Technique and UseCase population
- Supports CONTAINS and BEST_FOR relationships
- Prepared for semantic search implementation

## 📈 Usage Statistics

Based on the Neo4j extraction:
- **2831 nodes scanned** in original database
- **70 valuable nodes extracted** for this domain
- **600 total observations** captured
- **8 communities** with full hierarchy
- **4 domain types** identified in full extraction

## 🔄 Migration from Neo4j

### Key Differences Handled
- **Array syntax**: `STRING[]` vs Neo4j lists
- **Vector extension**: Native `FLOAT[384]` support
- **Multiplicity constraints**: `MANY_ONE` relationships
- **Index creation**: Different syntax from Neo4j
- **Query limitations**: Some Cypher features not supported

### Safe DDL Patterns
- `IF NOT EXISTS` prevents re-run errors
- Parameterized queries for data safety
- Batch processing for large datasets

## 🛠️ Development Notes

### Tested Features
✅ Database creation and schema
✅ Community data ingestion
✅ Hierarchy relationships
✅ Basic Cypher queries
✅ Vector extension loading
✅ Demo functionality

### Known Limitations
⚠️ Some advanced Cypher features not supported
⚠️ ORDER BY on nodes requires client-side sorting
⚠️ Path variable syntax limitations
⚠️ Vector embeddings need manual addition

## 🎉 Next Steps

1. **Add MLX Embeddings**: Generate semantic vectors for all communities
2. **Populate Techniques**: Add the 58 text-based prompting methods
3. **Create Use Cases**: Define application scenarios
4. **Build Relationships**: Link techniques to communities and use cases
5. **Implement Search**: Create semantic discovery functionality

## 📚 Related Files

- `neo4j_extract.json` - Source data from Neo4j extraction
- `CLAUDE.md` - Project setup and KuzuDB guidance
- `NEO4J_TO_KUZU_MIGRATION.md` - Migration best practices

---

**Generated**: 2025-10-09
**Source**: Neo4j extracted knowledge graph
**Domain**: Prompt Engineering Research (Domain 1 of 4)
**Database**: KuzuDB with vector extension