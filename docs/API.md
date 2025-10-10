# Kuzu Memory Graph MCP Server API Documentation

This document provides detailed information about all MCP tools, resources, their parameters, and example usage for the Kuzu Memory Graph MCP Server with multi-database support.

## Project Structure

The server has been simplified to use only two main files:

- `src/kuzu_memory_server.py` - Main MCP server with all tools, database management, and multi-database support
- `src/semantic_search.py` - Semantic search utilities (used as fallback for embeddings)

## Table of Contents

- [Kuzu Memory Graph MCP Server API Documentation](#kuzu-memory-graph-mcp-server-api-documentation)
  - [Project Structure](#project-structure)
  - [Table of Contents](#table-of-contents)
  - [Multi-Database Architecture](#multi-database-architecture)
    - [Database Discovery](#database-discovery)
    - [Database Management](#database-management)
    - [Example Database Structure](#example-database-structure)
  - [Graph Schema](#graph-schema)
    - [Entity Node Table](#entity-node-table)
    - [Relationship Table](#relationship-table)
  - [MCP Resources](#mcp-resources)
    - [kuzu://databases/list](#kuzudatabaseslist)
  - [MCP Tools](#mcp-tools)
    - [create\_entity](#create_entity)
    - [create\_relationship](#create_relationship)
    - [add\_observations](#add_observations)
    - [search\_entities](#search_entities)
    - [semantic\_search](#semantic_search)
    - [get\_related\_entities](#get_related_entities)
    - [get\_graph\_summary](#get_graph_summary)
  - [Cypher Query Examples](#cypher-query-examples)
    - [Basic Entity Queries](#basic-entity-queries)
    - [Relationship Queries](#relationship-queries)
    - [Advanced Queries](#advanced-queries)
  - [Error Handling](#error-handling)
    - [Common Error Formats](#common-error-formats)
    - [Specific Error Cases](#specific-error-cases)
    - [Best Practices for Error Handling](#best-practices-for-error-handling)

The knowledge graph uses the following schema:

### Entity Node Table

```cypher
CREATE NODE TABLE Entity (
    name STRING PRIMARY KEY,
    type STRING,
    observations STRING[],
    embedding FLOAT[384],
    created_date DATE DEFAULT current_date(),
    updated_date DATE DEFAULT current_date()
)
```

**Fields:**

- `name`: Unique identifier for the entity
- `type`: Category or type of the entity (e.g., "person", "concept", "document")
- `observations`: List of facts or observations about the entity
- `embedding`: 384-dimensional vector for semantic search
- `created_date`: When the entity was created
- `updated_date`: When the entity was last modified

### Relationship Table

```cypher
CREATE REL TABLE RELATED_TO (
    FROM Entity TO Entity,
    relationship_type STRING,
    confidence FLOAT DEFAULT 1.0,
    created_date DATE DEFAULT current_date()
)
```

**Fields:**

- `relationship_type`: Type of relationship (e.g., "KNOWS", "WORKS_WITH", "EXPERT_IN")
- `confidence`: Confidence score for the relationship (0.0-1.0)
- `created_date`: When the relationship was created

## MCP Resources

### kuzu://databases/list

Lists all available Kuzu databases in the workspace with metadata.

**Returns:**

```json
{
  "databases": [
    {
      "name": "memory",
      "description": "Database: memory",
      "is_primary": true,
      "is_attached": true
    },
    {
      "name": "prompt_engineer",
      "description": "Database: prompt_engineer",
      "is_primary": false,
      "is_attached": false
    },
    {
      "name": "research_papers",
      "description": "Database: research_papers",
      "is_primary": false,
      "is_attached": false
    }
  ],
  "count": 3,
  "primary_database": "memory"
}
```

**Example Usage:**

```python
# Query the resource to discover available databases
databases = await access_resource("kuzu://databases/list")
```

## MCP Tools

### create_entity

Creates a new entity in the specified knowledge graph with automatic embedding generation.

**Parameters:**

- `database` (string, required): Database name (query kuzu://databases/list to see available)
- `name` (string, required): Unique name for the entity
- `entity_type` (string, required): Type/category of the entity
- `observations` (list[string], optional): List of observations/facts about the entity

**Returns:**

```json
{
  "status": "created|exists",
  "database": "database_name",
  "name": "entity_name",
  "type": "entity_type",
  "observations_count": 3,
  "embedded": true
}
```

**Example Request:**

```json
{
  "database": "memory",
  "name": "Alice Johnson",
  "entity_type": "person",
  "observations": [
    "Software engineer at TechCorp",
    "Lives in San Francisco",
    "Expert in Python and machine learning"
  ]
}
```

**Example Response:**

```json
{
  "status": "created",
  "database": "memory",
  "name": "Alice Johnson",
  "type": "person",
  "observations_count": 3,
  "embedded": true
}
```

### create_relationship

Creates a relationship between two existing entities in the specified database.

**Parameters:**

- `database` (string, required): Database name (query kuzu://databases/list to see available)
- `from_entity` (string, required): Name of the source entity
- `to_entity` (string, required): Name of the target entity
- `relationship_type` (string, required): Type of relationship
- `confidence` (float, optional): Confidence score (0.0-1.0, default: 1.0)

**Returns:**

```json
{
  "status": "created|error",
  "database": "database_name",
  "from": "source_entity",
  "to": "target_entity",
  "relationship_type": "relationship_type",
  "confidence": 0.9
}
```

**Example Request:**

```json
{
  "database": "memory",
  "from_entity": "Alice Johnson",
  "to_entity": "Machine Learning",
  "relationship_type": "EXPERT_IN",
  "confidence": 0.9
}
```

**Example Response:**

```json
{
  "status": "created",
  "database": "memory",
  "from": "Alice Johnson",
  "to": "Machine Learning",
  "relationship_type": "EXPERT_IN",
  "confidence": 0.9
}
```

### add_observations

Adds new observations to an existing entity in the specified database and updates its embedding.

**Parameters:**

- `database` (string, required): Database name (query kuzu://databases/list to see available)
- `entity_name` (string, required): Name of the entity to update
- `observations` (list[string], required): List of new observations to add

**Returns:**

```json
{
  "status": "updated|no_change|error",
  "database": "database_name",
  "entity_name": "entity_name",
  "added_observations": ["new_observation"],
  "total_observations": 5,
  "reembedded": true
}
```

**Example Request:**

```json
{
  "database": "memory",
  "entity_name": "Alice Johnson",
  "observations": [
    "Recently completed TensorFlow certification",
    "Contributing to open source projects"
  ]
}
```

**Example Response:**

```json
{
  "status": "updated",
  "database": "memory",
  "entity_name": "Alice Johnson",
  "added_observations": [
    "Recently completed TensorFlow certification",
    "Contributing to open source projects"
  ],
  "total_observations": 5,
  "reembedded": true
}
```

### search_entities

Searches entities using text-based queries across names, types, and observations in the specified database.

**Parameters:**

- `database` (string, required): Database name (query kuzu://databases/list to see available)
- `query` (string, required): Search query string
- `limit` (integer, optional): Maximum number of results (default: 10)

**Returns:**

```json
{
  "query": "search_query",
  "database": "database_name",
  "entities": [
    {
      "name": "entity_name",
      "type": "entity_type",
      "observations": ["observation1", "observation2"],
      "created_date": "2024-01-01"
    }
  ],
  "count": 1
}
```

**Example Request:**

```json
{
  "database": "memory",
  "query": "software engineer",
  "limit": 5
}
```

**Example Response:**

```json
{
  "query": "software engineer",
  "database": "memory",
  "entities": [
    {
      "name": "Alice Johnson",
      "type": "person",
      "observations": [
        "Software engineer at TechCorp",
        "Lives in San Francisco",
        "Expert in Python and machine learning"
      ],
      "created_date": "2024-01-15"
    }
  ],
  "count": 1
}
```

### semantic_search

Performs semantic similarity search using vector embeddings in the specified database.

**Parameters:**

- `database` (string, required): Database name (query kuzu://databases/list to see available)
- `query` (string, required): Search query for semantic matching
- `limit` (integer, optional): Maximum number of results (default: 10)
- `threshold` (float, optional): Minimum similarity threshold (0.0-1.0, default: 0.3)

**Returns:**

```json
{
  "query": "search_query",
  "database": "database_name",
  "entities": [
    {
      "name": "entity_name",
      "type": "entity_type",
      "observations": ["observation1", "observation2"],
      "similarity": 0.85
    }
  ],
  "count": 1,
  "threshold": 0.3,
  "method": "vector_similarity|manual_calculation"
}
```

**Example Request:**

```json
{
  "database": "research_papers",
  "query": "AI programming expertise",
  "limit": 5,
  "threshold": 0.4
}
```

**Example Response:**

```json
{
  "query": "AI programming expertise",
  "database": "research_papers",
  "entities": [
    {
      "name": "Alice Johnson",
      "type": "person",
      "observations": [
        "Software engineer at TechCorp",
        "Lives in San Francisco",
        "Expert in Python and machine learning"
      ],
      "similarity": 0.87
    }
  ],
  "count": 1,
  "threshold": 0.4,
  "method": "vector_similarity"
}
```

### get_related_entities

Finds entities related to a specified entity through relationship traversal in the specified database.

**Parameters:**

- `database` (string, required): Database name (query kuzu://databases/list to see available)
- `entity_name` (string, required): Name of the entity to find relations for
- `max_depth` (integer, optional): Maximum relationship depth (default: 2)
- `limit` (integer, optional): Maximum number of results (default: 20)

**Returns:**

```json
{
  "entity_name": "source_entity",
  "database": "database_name",
  "entities": [
    {
      "name": "related_entity",
      "type": "entity_type",
      "observations": ["observation1"],
      "distance": 1,
      "relationship_path": ["EXPERT_IN"],
      "confidence_path": [0.9]
    }
  ],
  "count": 1,
  "max_depth": 2
}
```

**Example Request:**

```json
{
  "database": "prompt_engineer",
  "entity_name": "Chain of Thought",
  "max_depth": 2,
  "limit": 10
}
```

**Example Response:**

```json
{
  "entity_name": "Chain of Thought",
  "database": "prompt_engineer",
  "entities": [
    {
      "name": "Tree of Thoughts",
      "type": "prompt_pattern",
      "observations": [
        "Extends chain of thought",
        "Multiple reasoning paths"
      ],
      "distance": 1,
      "relationship_path": ["EXTENDS"],
      "confidence_path": [0.9]
    }
  ],
  "count": 1,
  "max_depth": 2
}
```

### get_graph_summary

Provides statistics about a specific database or all databases.

**Parameters:**

- `database` (string, optional): Specific database to summarize, or None for all databases

**Returns (Single Database):**

```json
{
  "scope": "single_database",
  "database": "database_name",
  "stats": {
    "entities": 100,
    "relationships": 150,
    "entity_types": 5,
    "relationship_types": 8
  },
  "entity_types": [
    {"type": "person", "count": 40},
    {"type": "concept", "count": 30}
  ],
  "relationship_types": [
    {"type": "KNOWS", "count": 50},
    {"type": "EXPERT_IN", "count": 30}
  ]
}
```

**Returns (All Databases):**

```json
{
  "scope": "all_databases",
  "databases": {
    "memory": {
      "entities": 50,
      "relationships": 75
    },
    "prompt_engineer": {
      "entities": 25,
      "relationships": 30
    },
    "research_papers": {
      "entities": 75,
      "relationships": 120
    }
  },
  "total_databases": 3
}
```

**Example Request (Single Database):**

```json
{
  "database": "memory"
}
```

**Example Request (All Databases):**

```json
{}
```

**Example Response (Single Database):**

```json
{
  "scope": "single_database",
  "database": "memory",
  "stats": {
    "entities": 50,
    "relationships": 75,
    "entity_types": 3,
    "relationship_types": 4
  },
  "entity_types": [
    {"type": "person", "count": 20},
    {"type": "concept", "count": 15},
    {"type": "organization", "count": 15}
  ],
  "relationship_types": [
    {"type": "KNOWS", "count": 25},
    {"type": "EXPERT_IN", "count": 20},
    {"type": "WORKS_FOR", "count": 15},
    {"type": "RELATED_TO", "count": 15}
  ]
}
```

## Cypher Query Examples

### Basic Entity Queries

```cypher
-- Find all entities of a specific type
MATCH (e:Entity {type: "person"})
RETURN e.name, e.observations;

-- Get entities created after a certain date
MATCH (e:Entity)
WHERE e.created_date > "2024-01-01"
RETURN e.name, e.type, e.created_date;

-- Find entities with specific observation
MATCH (e:Entity)
WHERE "machine learning" IN e.observations
RETURN e.name, e.type;
```

### Relationship Queries

```cypher
-- Find all relationships for an entity
MATCH (e:Entity {name: "Alice Johnson"})-[r:RELATED_TO]->(related)
RETURN related.name, r.relationship_type, r.confidence;

-- Find bidirectional relationships
MATCH (e1:Entity)-[r:RELATED_TO]-(e2:Entity)
WHERE e1.name = "Alice Johnson"
RETURN e2.name, r.relationship_type;

-- Find entities connected through multiple hops
MATCH path = (start:Entity {name: "Alice Johnson"})-[:RELATED_TO*1..3]-(end:Entity)
RETURN end.name, length(path) as distance;
```

### Advanced Queries

```cypher
-- Find similar entities based on observation overlap
MATCH (e1:Entity), (e2:Entity)
WHERE e1.name = "Alice Johnson" 
  AND e1 <> e2
  AND size([obs IN e1.observations WHERE obs IN e2.observations]) > 0
RETURN e2.name, size([obs IN e1.observations WHERE obs IN e2.observations]) as common_observations
ORDER BY common_observations DESC;

-- Find entities with high-confidence relationships
MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity)
WHERE r.confidence > 0.8
RETURN e1.name, e2.name, r.relationship_type, r.confidence;

-- Get relationship network statistics
MATCH (e:Entity)
OPTIONAL MATCH (e)-[r:RELATED_TO]-()
RETURN e.type, count(DISTINCT r) as relationship_count
ORDER BY relationship_count DESC;
```

## Error Handling

All tools return structured error responses when issues occur:

### Common Error Formats

```json
{
  "status": "error",
  "message": "Descriptive error message"
}
```

### Specific Error Cases

1. **Entity Not Found**:

   ```json
   {
     "status": "error",
     "message": "Entity 'EntityName' not found"
   }
   ```

2. **Duplicate Entity**:

   ```json
   {
     "status": "exists",
     "name": "EntityName",
     "type": "person",
     "message": "Entity already exists"
   }
   ```

3. **Relationship Creation Error**:

   ```json
   {
     "status": "error",
     "message": "One or both entities not found: Entity1, Entity2"
   }
   ```

4. **Invalid Parameters**:

   ```json
   {
     "status": "error",
     "message": "Invalid parameter: threshold must be between 0.0 and 1.0"
   }
   ```

### Best Practices for Error Handling

1. Always check the `status` field in responses
2. Handle duplicate entity creation gracefully
3. Validate parameters before making requests
4. Implement retry logic for temporary failures
5. Log errors for debugging purposes
