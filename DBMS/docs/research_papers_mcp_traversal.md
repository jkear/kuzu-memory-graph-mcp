# Research Papers Database - MCP Tool Traversal Guide

## Overview

This guide explains how to traverse the Research Papers knowledge graph using Kuzu MCP tools. The database contains academic research papers, concepts, datasets, and frameworks with rich relationships for semantic navigation.

## 📊 Database Schema (MCP Compatible)

### Node Tables
- **ResearchPaper**: Academic papers with metadata
- **Concept**: Theoretical frameworks and research concepts
- **Dataset**: Research datasets and resources
- **Framework**: Methodological frameworks

### Relationship Tables
- **PROPOSES**: Papers → Concepts (with contribution descriptions)
- **USES_DATASET**: Papers → Datasets (with methodology)
- **IMPLEMENTS**: Papers → Frameworks
- **BUILDS_ON**: Papers → Papers (citations and academic relationships)

## 🛠️ MCP Tool Usage

### Getting Schema Information

**Use MCP Tool: `mcp__kuzu__getSchema`**
```python
# Returns complete schema information
# No parameters needed - works with connected database
```

**Expected Output:**
- All node tables with their properties
- All relationship tables with their properties
- Data types and constraints

### Running Traversal Queries

**Use MCP Tool: `mcp__kuzu__query`**

#### Basic Node Queries
```sql
-- Get all research papers
MATCH (p:ResearchPaper)
RETURN p.identifier, p.title, p.year, p.authors

-- Get concepts by domain
MATCH (c:Concept {domain: 'AI Assessment'})
RETURN c.name, c.description

-- Find datasets by purpose
MATCH (d:Dataset)
WHERE d.purpose CONTAINS 'reasoning'
RETURN d.name, d.purpose
```

#### Relationship Traversals
```sql
-- Papers and their proposed concepts
MATCH (p:ResearchPaper)-[:PROPOSES]->(c:Concept)
RETURN p.title, c.name, p.year
ORDER BY p.year DESC

-- Datasets used by papers
MATCH (p:ResearchPaper)-[r:USES_DATASET]->(d:Dataset)
RETURN p.title, d.name, r.methodology

-- Academic citation networks
MATCH (p1:ResearchPaper)-[:BUILDS_ON]->(p2:ResearchPaper)
RETURN p1.title as citing_paper, p2.title as cited_paper
```

#### Pattern Matching Queries
```sql
-- Find research communities (papers that build on each other)
MATCH path = (p1:ResearchPaper)-[:BUILDS_ON*1..3]->(p2:ResearchPaper)
RETURN p1.title, p2.title, length(path) as distance

-- Multi-hop traversals: Paper → Concept → Related Papers
MATCH (p1:ResearchPaper)-[:PROPOSES]->(c:Concept)<-[:PROPOSES]-(p2:ResearchPaper)
WHERE p1.year < p2.year
RETURN p1.title as earlier, c.name as concept, p2.title as later
ORDER BY p1.year

-- Complex pattern: Papers using datasets that propose concepts
MATCH (p:ResearchPaper)-[:USES_DATASET]->(d:Dataset)
MATCH (p)-[:PROPOSES]->(c:Concept)
RETURN p.title, d.name, c.name
```

## 🚀 Key Traversal Patterns

### 1. Literature Review Analysis
```sql
-- Trace concept evolution through time
MATCH (p1:ResearchPaper)-[:PROPOSES]->(c:Concept)<-[:PROPOSES]-(p2:ResearchPaper)
WHERE p1.year < p2.year
WITH c, collect({earlier: p1.title, later: p2.title, year_diff: p2.year - p1.year}) as evolution
RETURN c.name, evolution
ORDER BY size(evolution) DESC
```

### 2. Citation Network Analysis
```sql
-- Find influential papers (most cited)
MATCH (p:ResearchPaper)<-[:BUILDS_ON]-(citing:ResearchPaper)
WITH p, count(citing) as citation_count
RETURN p.title, citation_count
ORDER BY citation_count DESC
LIMIT 10

-- Find research lineages
MATCH path = (base:ResearchPaper)<-[:BUILDS_ON*]-(descendant:ResearchPaper)
WHERE size(nodes(path)) > 2
RETURN base.title as foundation, descendant.title as latest, size(nodes(path)) as depth
ORDER BY depth DESC
LIMIT 5
```

### 3. Methodology Analysis
```sql
-- Find papers using similar methodologies
MATCH (p1:ResearchPaper)-[:USES_DATASET]->(d:Dataset)<-[:USES_DATASET]-(p2:ResearchPaper)
WHERE p1.identifier <> p2.identifier
RETURN p1.title, p2.title, d.name
LIMIT 10

-- Framework adoption patterns
MATCH (f:Framework)<-[:IMPLEMENTS]-(p:ResearchPaper)
WITH f, count(p) as adoption_count, collect(p.title) as papers
RETURN f.name, adoption_count, papers
ORDER BY adoption_count DESC
```

### 4. Cross-Domain Research
```sql
-- Find papers spanning multiple domains
MATCH (p:ResearchPaper)-[:PROPOSES]->(c1:Concept)
MATCH (p)-[:PROPOSES]->(c2:Concept)
WHERE c1.domain <> c2.domain
RETURN p.title, c1.domain, c2.domain, collect([c1.name, c2.name]) as concepts
```

## 📈 Analytical Queries

### Temporal Analysis
```sql
-- Research trends by year
MATCH (p:ResearchPaper)
WHERE p.year IS NOT NULL
RETURN p.year, count(p) as paper_count
ORDER BY p.year

-- Concept emergence over time
MATCH (p:ResearchPaper)-[:PROPOSES]->(c:Concept)
WHERE p.year IS NOT NULL
WITH c, min(p.year) as first_appearance
RETURN c.name, first_appearance
ORDER BY first_appearance
```

### Network Analysis
```sql
-- Research collaboration networks
MATCH (p1:ResearchPaper)-[:BUILDS_ON]->(p2:ResearchPaper)
RETURN p1.authors[0] as author1, p2.authors[0] as author2, count(*) as collaborations

-- Concept similarity (based on shared papers)
MATCH (c1:Concept)<-[:PROPOSES]-(p:ResearchPaper)-[:PROPOSES]->(c2:Concept)
WHERE c1.name < c2.name  // Avoid duplicates
WITH c1, c2, count(p) as shared_papers
RETURN c1.name, c2.name, shared_papers
ORDER BY shared_papers DESC
LIMIT 10
```

## 🎯 Sample Use Cases

### 1. Finding Related Research
```sql
-- Find papers similar to a target paper
MATCH (target:ResearchPaper {identifier: 'The Illusion of Thinking'})
MATCH (target)-[:PROPOSES]->(shared:Concept)<-[:PROPOSES]-(similar:ResearchPaper)
WHERE similar.identifier <> target.identifier
RETURN similar.title, shared.name as common_concept
```

### 2. Dataset Discovery
```sql
-- Find datasets used in specific research areas
MATCH (p:ResearchPaper)-[:USES_DATASET]->(d:Dataset)
MATCH (p)-[:PROPOSES]->(c:Concept {domain: 'AI Assessment'})
RETURN d.name, d.purpose, count(p) as usage_count
ORDER BY usage_count DESC
```

### 3. Methodology Transfer
```sql
-- Find methodologies transferred between domains
MATCH (p1:ResearchPaper)-[:USES_DATASET]->(d:Dataset)<-[:USES_DATASET]-(p2:ResearchPaper)
MATCH (p1)-[:PROPOSES]->(c1:Concept)
MATCH (p2)-[:PROPOSES]->(c2:Concept)
WHERE c1.domain <> c2.domain
RETURN p1.title, c1.domain, p2.title, c2.domain, d.name as shared_methodology
```

## 🔧 Performance Optimization

### Index Usage
The schema includes these indexes for efficient traversal:
- Primary keys on all node tables
- Foreign key relationships through connection tables

### Query Tips
1. **Use specific node labels** instead of generic `(n)` patterns
2. **Limit result sets** with `LIMIT` for exploration queries
3. **Use parameters** for repeated queries with different values
4. **Filter early** in the query to reduce working set size

## 📋 Quick Reference Commands

### Schema Exploration
```sql
-- List all node tables
CALL table_info('ResearchPaper')
CALL table_info('Concept')
CALL table_info('Dataset')
CALL table_info('Framework')

-- Sample data from each table
MATCH (p:ResearchPaper) RETURN p LIMIT 3
MATCH (c:Concept) RETURN c LIMIT 3
MATCH (d:Dataset) RETURN d LIMIT 3
MATCH (f:Framework) RETURN f LIMIT 3
```

### Common Traversals
```sql
-- Paper relationships
MATCH (p:ResearchPaper)-[r]->(related) RETURN type(r), labels(related)[0]

-- Concept networks
MATCH (c1:Concept)<-[:PROPOSES]-(p:ResearchPaper)-[:PROPOSES]->(c2:Concept)
RETURN c1.name, c2.name

-- Dataset usage
MATCH (d:Dataset)<-[:USES_DATASET]-(p:ResearchPaper)
RETURN d.name, count(p) as usage_count
```

## 🎉 Getting Started

1. **Connect to database**: Use MCP tools with `research_papers.kuzu`
2. **Explore schema**: Use `getSchema` to understand the structure
3. **Run basic queries**: Start with node and relationship traversals
4. **Try patterns**: Use the example queries above for common use cases
5. **Build complex queries**: Combine patterns for specific research questions

The database is ready for knowledge graph traversal and supports both simple lookups and complex multi-hop analytical queries!