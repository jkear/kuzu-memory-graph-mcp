# Prompt Engineering KuzuDB Database

## Overview

This database contains prompt engineering techniques and methods for working with Large Language Models (LLMs). It's accessible through the **Kuzu Memory Graph MCP Server** and uses a unified Entity model for consistency across all databases.

## 🎯 Database Purpose

Store and explore prompt engineering knowledge including:

- Prompting techniques (Chain-of-Thought, Few-Shot, Zero-Shot, etc.)
- Use cases and applications
- Best practices and principles
- Relationships between techniques

## 🏗️ Schema (Unified Entity Model)

### Entity Node

```cypher
CREATE NODE TABLE IF NOT EXISTS Entity (
    name STRING PRIMARY KEY,
    type STRING,
    observations STRING[],
    embedding FLOAT[384],
    created_date DATE DEFAULT current_date(),
    updated_date DATE DEFAULT current_date()
)
```

**Entity Types in this database**:

- `technique`: Specific prompting methods
- `use_case`: Application scenarios
- `principle`: General guidelines
- `category`: Technique categories

### Relationship

```cypher
CREATE REL TABLE IF NOT EXISTS RELATED_TO (
    FROM Entity TO Entity,
    relationship_type STRING,
    confidence FLOAT DEFAULT 1.0,
    created_date DATE DEFAULT current_date()
)
```

**Relationship Types**:

- `BEST_FOR`: Technique → Use Case
- `COMBINES_WITH`: Technique → Technique
- `CATEGORY_OF`: Category → Technique
- `RELATED_TO`: General relationships

## 🚀 Quick Start with MCP Server

### 1. Access via Claude Desktop

The database is available through the MCP server. See [MCP Integration Guide](./MCP_INTEGRATION_GUIDE.md) for setup.

### 2. List Available Databases

```
Query Resource: kuzu://databases/list
```

This shows `prompt_engineering` along with other available databases.

### 3. Create Entities

```json
{
  "tool": "create_entity",
  "parameters": {
    "database": "prompt_engineering",
    "name": "Chain-of-Thought",
    "entity_type": "technique",
    "observations": [
      "Prompting technique that encourages LLMs to show reasoning steps",
      "Improves performance on complex reasoning tasks",
      "Introduced in 'Chain-of-Thought Prompting' paper (2022)"
    ]
  }
}
```

### 4. Search Techniques

**Text Search**:

```json
{
  "tool": "search_entities",
  "parameters": {
    "database": "prompt_engineering",
    "query": "reasoning",
    "limit": 10
  }
}
```

**Semantic Search**:

```json
{
  "tool": "semantic_search",
  "parameters": {
    "database": "prompt_engineering",
    "query": "techniques for improving mathematical problem solving",
    "limit": 5,
    "threshold": 0.6
  }
}
```

## 📚 Common Use Cases

### 1. Exploring Techniques by Category

```json
{
  "tool": "search_entities",
  "parameters": {
    "database": "prompt_engineering",
    "query": "chain",
    "limit": 10
  }
}
```

Returns entities with "chain" in name/observations (Chain-of-Thought, etc.)

### 2. Finding Related Techniques

```json
{
  "tool": "get_related_entities",
  "parameters": {
    "database": "prompt_engineering",
    "entity_name": "Few-Shot Learning",
    "max_depth": 2,
    "limit": 20
  }
}
```

Explores 2-hop relationships from Few-Shot Learning.

### 3. Discovering Similar Methods

```json
{
  "tool": "semantic_search",
  "parameters": {
    "database": "prompt_engineering",
    "query": "techniques that help with reasoning and problem decomposition",
    "limit": 5,
    "threshold": 0.5
  }
}
```

Uses vector similarity to find conceptually related techniques.

### 4. Getting Database Overview

```json
{
  "tool": "get_graph_summary",
  "parameters": {
    "database": "prompt_engineering"
  }
}
```

Returns statistics on entities and relationships.

## 🎯 Example Entities to Create

### Core Techniques

```json
{
  "name": "Zero-Shot",
  "entity_type": "technique",
  "observations": [
    "Prompting without providing examples",
    "Relies on model's pre-trained knowledge",
    "Good baseline for many tasks"
  ]
}

{
  "name": "Few-Shot Learning",
  "entity_type": "technique",
  "observations": [
    "Provides 1-5 examples in the prompt",
    "Helps model understand task format",
    "Effective for structured outputs"
  ]
}

{
  "name": "Chain-of-Thought",
  "entity_type": "technique",
  "observations": [
    "Shows reasoning steps in examples",
    "Improves complex reasoning tasks",
    "Works best with larger models (>100B params)"
  ]
}

{
  "name": "Self-Consistency",
  "entity_type": "technique",
  "observations": [
    "Generate multiple reasoning paths",
    "Select most consistent answer via voting",
    "Improves reliability on reasoning tasks"
  ]
}

{
  "name": "ReAct",
  "entity_type": "technique",
  "observations": [
    "Combines reasoning and acting",
    "Alternates between thought and action",
    "Useful for tool-using agents"
  ]
}
```

### Use Cases

```json
{
  "name": "Mathematical Problem Solving",
  "entity_type": "use_case",
  "observations": [
    "Solving math word problems",
    "Requires step-by-step reasoning",
    "Benefits from Chain-of-Thought"
  ]
}

{
  "name": "Code Generation",
  "entity_type": "use_case",
  "observations": [
    "Generating code from descriptions",
    "Requires clear specification",
    "Benefits from few-shot examples"
  ]
}

{
  "name": "Question Answering",
  "entity_type": "use_case",
  "observations": [
    "Answering factual questions",
    "May require retrieval augmentation",
    "Zero-shot often sufficient"
  ]
}
```

### Relationships

```json
// Chain-of-Thought best for Math
{
  "from_entity": "Chain-of-Thought",
  "to_entity": "Mathematical Problem Solving",
  "relationship_type": "BEST_FOR",
  "confidence": 0.95
}

// Few-Shot best for Code
{
  "from_entity": "Few-Shot Learning",
  "to_entity": "Code Generation",
  "relationship_type": "BEST_FOR",
  "confidence": 0.90
}

// Self-Consistency builds on Chain-of-Thought
{
  "from_entity": "Self-Consistency",
  "to_entity": "Chain-of-Thought",
  "relationship_type": "BUILDS_ON",
  "confidence": 1.0
}
```

## 📊 Current Database State

The database is ready for population via MCP tools. Initial structure includes:

- **Schema**: Unified Entity model with vector embeddings
- **Vector Extension**: Loaded for semantic search
- **Vector Index**: `entity_embedding_idx` on embeddings
- **MLX Model**: 384-dimensional embeddings (Apple Silicon optimized)

## 🔄 Migration from Neo4j

For migrating existing Neo4j data, see:

- [Neo4j to MCP Migration Guide](./neo4j_to_mcp_migration.md)
- [MCP Integration Guide](./MCP_INTEGRATION_GUIDE.md)

The key transformation is mapping Neo4j schema to the unified Entity model:

```
Neo4j → MCP Entity Model
├── Community → Entity (type: "category")
├── Technique → Entity (type: "technique")
├── UseCase → Entity (type: "use_case")
├── SUBCOMMUNITY → RELATED_TO (relationship_type: "SUBCOMMUNITY")
├── CONTAINS → RELATED_TO (relationship_type: "CONTAINS")
└── BEST_FOR → RELATED_TO (relationship_type: "BEST_FOR")
```

## 📚 Additional Resources

- **[MCP Integration Guide](./MCP_INTEGRATION_GUIDE.md)**: Complete MCP server documentation
- **[Schema Documentation](./prompt_engineering_schema.md)**: Detailed schema reference
- **[Neo4j Migration Guide](./neo4j_to_mcp_migration.md)**: Step-by-step migration instructions

## 🚀 Next Steps

1. **Start MCP Server**: Configure Claude Desktop with server
2. **Query Database List**: Use `kuzu://databases/list` resource
3. **Create Core Techniques**: Use `create_entity` for fundamental methods
4. **Build Relationships**: Link techniques to use cases
5. **Semantic Search**: Test vector similarity on populated data

---

**Database**: `prompt_engineering.kuzu`  
**Location**: `/DBMS/prompt_engineering.kuzu`  
**Schema Version**: 1.0 (Unified Entity Model)  
**Last Updated**: 2025-10-10

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
