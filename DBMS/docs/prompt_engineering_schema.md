# KuzuDB Schema: Prompt Engineering Knowledge Base

## Overview
This schema defines a comprehensive knowledge base for prompt engineering techniques, based on The Prompt Report systematic survey. It supports hierarchical community organization, technique categorization, and vector similarity search for semantic discovery of prompting methods.

## Database Setup
```python
import kuzu

# Initialize database
db = kuzu.Database("./prompt_engineering.kuzu")
conn = kuzu.Connection(db)

# Install vector extension for semantic search
conn.execute("INSTALL vector;")
conn.execute("LOAD EXTENSION vector;")
```

## Schema Definition

### Node Tables

#### Community
Represents hierarchical communities and subcommunities for organizing prompt engineering knowledge.

```cypher
CREATE NODE TABLE IF NOT EXISTS Community(
    identifier STRING PRIMARY KEY,
    type STRING,
    observations STRING[],
    created_date DATE DEFAULT current_date(),
    embedding FLOAT[384]  -- For semantic similarity search
);
```

**Fields:**
- `identifier`: Unique community name (Primary Key)
- `type`: Community type (e.g., "Community", "Documentation")
- `observations`: Array of knowledge observations and descriptions
- `created_date`: Auto-populated creation timestamp
- `embedding`: 384-dimensional vector for semantic search (MLX compatible)

#### Technique
Stores individual prompt engineering techniques with their classifications.

```cypher
CREATE NODE TABLE IF NOT EXISTS Technique(
    name STRING PRIMARY KEY,
    description STRING,
    category STRING,
    observations STRING[],
    embedding FLOAT[384]
);
```

**Fields:**
- `name`: Unique technique name (Primary Key)
- `description`: Detailed description of the technique
- `category`: Taxonomic category (e.g., "few-shot", "chain-of-thought")
- `observations`: Array of technique-specific insights
- `embedding`: Vector representation for similarity matching

#### UseCase
Defines specific use cases and application scenarios for prompt techniques.

```cypher
CREATE NODE TABLE IF NOT EXISTS UseCase(
    name STRING PRIMARY KEY,
    description STRING,
    domain STRING,
    embedding FLOAT[384]
);
```

**Fields:**
- `name`: Unique use case identifier (Primary Key)
- `description`: Detailed use case description
- `domain`: Application domain (e.g., "reasoning", "generation")
- `embedding`: Vector for use case similarity matching

### Relationship Tables

#### SUBCOMMUNITY
Establishes hierarchical relationships between communities with multiplicity constraints.

```cypher
CREATE REL TABLE IF NOT EXISTS SUBCOMMUNITY(
    FROM Community TO Community,
    MANY_ONE  -- Many subcommunities can belong to one parent
);
```

**Constraints:** `MANY_ONE` ensures clear hierarchical structure where each subcommunity belongs to exactly one parent.

#### CONTAINS
Links communities to the techniques they contain.

```cypher
CREATE REL TABLE IF NOT EXISTS CONTAINS(
    FROM Community TO Technique,
    context STRING
);
```

**Fields:**
- `context`: Additional context about the relationship

#### BEST_FOR
Connects techniques to their optimal use cases with evidence.

```cypher
CREATE REL TABLE IF NOT EXISTS BEST_FOR(
    FROM Technique TO UseCase,
    evidence STRING[]
);
```

**Fields:**
- `evidence`: Array of supporting evidence or examples

## Indexes for Performance

```cypher
-- Create indexes for frequently queried fields
CREATE INDEX community_identifier_idx FOR Community ON (identifier);
CREATE INDEX community_type_idx FOR Community ON (type);
CREATE INDEX technique_name_idx FOR Technique ON (name);
CREATE INDEX technique_category_idx FOR Technique ON (category);
CREATE INDEX usecase_domain_idx FOR UseCase ON (domain);
```

## Sample Data Ingestion

### Root Community
```cypher
CREATE (:Community {
    identifier: 'Prompt Engineering Research',
    type: 'Community',
    observations: [
        'Root community for all prompt engineering knowledge',
        'Contains comprehensive research on prompting techniques',
        'Based on systematic survey of prompt engineering methods',
        'Encompasses text-based, multimodal, multilingual, and agent techniques'
    ],
    created_date: date('2025-01-01')
});
```

### Subcommunities
```cypher
CREATE (:Community {
    identifier: 'Text-Based Techniques',
    type: 'Community',
    observations: [
        'Primary community for all text-based prompting techniques',
        'Contains 58 distinct prompting methods',
        'Organized into 6 major taxonomic categories',
        'Core focus of The Prompt Report systematic survey'
    ],
    created_date: date('2025-01-01')
});

CREATE (:Community {
    identifier: 'Multilingual Techniques',
    type: 'Community',
    observations: [
        'Community for cross-language prompting methods',
        'Techniques for handling multiple languages in prompts',
        'Includes translation and multilingual reasoning approaches'
    ],
    created_date: date('2025-01-01')
});
```

### Hierarchy Relationships
```cypher
MATCH (parent:Community {identifier: 'Prompt Engineering Research'})
MATCH (child:Community {identifier: 'Text-Based Techniques'})
CREATE (child)-[:SUBCOMMUNITY]->(parent);

MATCH (parent:Community {identifier: 'Prompt Engineering Research'})
MATCH (child:Community {identifier: 'Multilingual Techniques'})
CREATE (child)-[:SUBCOMMUNITY]->(parent);
```

## Vector Similarity Search with MLX

### Setup MLX Embeddings (Apple Silicon Optimized)
```python
from mlx_embeddings.utils import load
import mlx.core as mx

# Load 4-bit quantized model for efficiency
model_name = "mlx-community/all-MiniLM-L6-v2-4bit"
model, tokenizer = load(model_name)

def get_embedding(text):
    """Generate 384-dimensional embedding using MLX"""
    inputs = tokenizer.encode(text, return_tensors="mlx")
    outputs = model(inputs)
    return outputs.text_embeds.tolist()
```

### Semantic Search Queries
```cypher
-- Find similar communities using vector similarity
MATCH (c:Community)
WHERE c.embedding IS NOT NULL
WITH c, array_cosine_similarity(c.embedding, $query_embedding) as similarity
RETURN c.identifier, c.type, similarity
ORDER BY similarity DESC
LIMIT 5;

-- Find techniques for specific use cases with semantic matching
MATCH (t:Technique)-[:BEST_FOR]->(u:UseCase)
WHERE t.embedding IS NOT NULL AND u.embedding IS NOT NULL
WITH t, u,
     array_cosine_similarity(t.embedding, $query_embedding) as tech_sim,
     array_cosine_similarity(u.embedding, $context_embedding) as use_sim
RETURN t.name, u.description, (tech_sim + use_sim) / 2 as combined_score
ORDER BY combined_score DESC
LIMIT 10;
```

## Common Query Patterns

### 1. Community Hierarchy Traversal
```cypher
MATCH (root:Community {identifier: 'Prompt Engineering Research'})
<-[:SUBCOMMUNITY]-(subcommunities:Community)
OPTIONAL MATCH (subcommunities)<-[:SUBCOMMUNITY]-(subsub:Community)
RETURN root.identifier as root,
       subcommunities.identifier as level1,
       collect(DISTINCT subsub.identifier) as level2;
```

### 2. Technique Discovery by Category
```cypher
MATCH (c:Community {identifier: 'Text-Based Techniques'})
-[:CONTAINS]->(t:Technique {category: 'few-shot'})
RETURN t.name, t.description
ORDER BY t.name;
```

### 3. Use Case Recommendations
```cypher
MATCH (t:Technique)-[:BEST_FOR]->(u:UseCase)
WHERE u.domain = 'reasoning'
RETURN t.name as technique, u.description as use_case, t.evidence
ORDER BY t.name;
```

### 4. Semantic Technique Search
```cypher
-- Find techniques semantically similar to a query
WITH $query_text as query
MATCH (t:Technique)
WHERE t.embedding IS NOT NULL
WITH t, array_cosine_similarity(t.embedding, $query_embedding) as similarity
RETURN t.name, t.category, similarity, t.description
WHERE similarity > 0.7
ORDER BY similarity DESC;
```

## Batch Processing with MLX

### Efficient Embedding Generation
```python
def get_batch_embeddings(texts):
    """Generate embeddings for multiple texts efficiently"""
    inputs = tokenizer.batch_encode_plus(
        texts,
        return_tensors="mlx",
        padding=True,
        truncation=True,
        max_length=512
    )
    outputs = model(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    return outputs.text_embeds.tolist()

# Process communities in batches
batch_size = 8
communities = []
result = conn.execute("MATCH (c:Community) WHERE c.embedding IS NULL RETURN c.identifier, c.observations")

while result.has_next():
    row = result.get_next()
    communities.append(row)

    if len(communities) >= batch_size:
        texts = [' '.join(obs if obs else [identifier]) for identifier, obs in communities]
        embeddings = get_batch_embeddings(texts)

        for (identifier, _), embedding in zip(communities, embeddings):
            conn.execute("""
                MATCH (c:Community {identifier: $id})
                SET c.embedding = $embedding
            """, {"id": identifier, "embedding": embedding})

        communities.clear()
```

## Performance Optimizations

### Memory Management
- Use batch processing for embedding generation (8 items per batch)
- Clear intermediate results to prevent memory buildup
- Leverage MLX's Apple Silicon optimization for 4-bit quantized models

### Query Optimization
- Create indexes on frequently accessed fields
- Use parameterized queries to prevent injection
- Filter by `embedding IS NOT NULL` before similarity calculations

### Bulk Loading
```python
# For large datasets, use COPY FROM with CSV format
# CSV columns: identifier,type,observations,created_date
# observations format: "['obs1','obs2','obs3']"
conn.execute("COPY Community FROM 'communities.csv' (HEADER=true);")
```

## Schema Extensions

### Future Enhancements
```cypher
-- Add evaluation metrics
CREATE NODE TABLE IF NOT EXISTS Evaluation(
    name STRING PRIMARY KEY,
    metric_type STRING,
    score FLOAT,
    methodology STRING
);

-- Add technique relationships
CREATE REL TABLE IF NOT EXISTS COMPOSED_WITH(
    FROM Technique TO Technique,
    combination_type STRING
);

-- Add temporal aspects
CREATE REL TABLE IF NOT EXISTS EVOLVED_FROM(
    FROM Technique TO Technique,
    evolution_reason STRING
);
```

## Migration Notes

### From Neo4j to KuzuDB
- **Array Handling**: Use JSON format in CSV: `"['item1','item2','item3']"`
- **Vector Storage**: KuzuDB supports fixed-size arrays `FLOAT[384]` natively
- **Multiplicity Constraints**: Add `MANY_ONE`, `ONE_MANY`, or `ONE_ONE` to relationships
- **Extension Loading**: Must `INSTALL vector; LOAD EXTENSION vector;` before using vector functions

### Safe DDL Patterns
- Always use `IF NOT EXISTS` to prevent errors on re-runs
- Use parameterized queries for data insertion
- Validate schema with test queries before bulk loading

## Usage Example

```python
import kuzu
from mlx_embeddings.utils import load

# Initialize database with vector support
db = kuzu.Database("./prompt_engineering.kuzu")
conn = kuzu.Connection(db)
conn.execute("INSTALL vector;")
conn.execute("LOAD EXTENSION vector;")

# Create schema
with open('prompt_engineering_schema.cypher', 'r') as f:
    schema = f.read()
for statement in schema.split(';'):
    if statement.strip():
        conn.execute(statement)

# Load MLX model for embeddings
model_name = "mlx-community/all-MiniLM-L6-v2-4bit"
model, tokenizer = load(model_name)

# Perform semantic search
query_text = "improve reasoning capabilities"
query_embedding = get_embedding(query_text)

result = conn.execute("""
    MATCH (t:Technique)
    WHERE t.embedding IS NOT NULL
    WITH t, array_cosine_similarity(t.embedding, $query) as similarity
    RETURN t.name, t.category, similarity
    ORDER BY similarity DESC
    LIMIT 5;
""", {"query": query_embedding})

while result.has_next():
    row = result.get_next()
    print(f"{row[0]} ({row[1]}): {row[3]:.3f}")
```

This schema provides a robust foundation for building a comprehensive prompt engineering knowledge base with advanced semantic search capabilities, optimized for Apple Silicon performance with MLX embeddings.