-- KuzuDB Schema for Prompt Engineering Knowledge Base
-- Based on The Prompt Report systematic survey

-- Install vector extension for semantic search
INSTALL vector;
LOAD EXTENSION vector;

-- ========================================
-- NODE TABLES
-- ========================================

-- Community table for hierarchical organization
CREATE NODE TABLE IF NOT EXISTS Community(
    identifier STRING PRIMARY KEY,
    type STRING,
    observations STRING[],
    created_date DATE DEFAULT current_date(),
    embedding FLOAT[384]  -- For semantic similarity search
);

-- Technique table for individual prompting methods
CREATE NODE TABLE IF NOT EXISTS Technique(
    name STRING PRIMARY KEY,
    description STRING,
    category STRING,
    observations STRING[],
    embedding FLOAT[384]
);

-- UseCase table for application scenarios
CREATE NODE TABLE IF NOT EXISTS UseCase(
    name STRING PRIMARY KEY,
    description STRING,
    domain STRING,
    embedding FLOAT[384]
);

-- ========================================
-- RELATIONSHIP TABLES
-- ========================================

-- Hierarchical community relationships
CREATE REL TABLE IF NOT EXISTS SUBCOMMUNITY(
    FROM Community TO Community,
    MANY_ONE  -- Many subcommunities can belong to one parent
);

-- Community to technique relationships
CREATE REL TABLE IF NOT EXISTS CONTAINS(
    FROM Community TO Technique,
    context STRING
);

-- Technique to use case relationships
CREATE REL TABLE IF NOT EXISTS BEST_FOR(
    FROM Technique TO UseCase,
    evidence STRING[]
);

-- ========================================
-- INDEXES
-- ========================================

-- Note: KuzuDB syntax for indexes is different from Neo4j
-- These are created as part of node table definitions above

-- ========================================
-- SAMPLE DATA INSERTION
-- ========================================

-- Root community
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

-- Subcommunities
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

CREATE (:Community {
    identifier: 'Multimodal Techniques',
    type: 'Community',
    observations: [
        'Community for techniques involving multiple modalities',
        'Handles text, image, audio, and video inputs',
        'Includes vision-language and audio-language prompting'
    ],
    created_date: date('2025-01-01')
});

CREATE (:Community {
    identifier: 'Agent Techniques',
    type: 'Community',
    observations: [
        'Community for agent-based prompting approaches',
        'Includes tool use and multi-agent systems',
        'Covers planning and execution frameworks'
    ],
    created_date: date('2025-01-01')
});

CREATE (:Community {
    identifier: 'Evaluation Methods',
    type: 'Community',
    observations: [
        'Community for prompt evaluation techniques',
        'Includes benchmarking and assessment methods',
        'Covers performance measurement approaches'
    ],
    created_date: date('2025-01-01')
});

CREATE (:Community {
    identifier: 'Security and Alignment',
    type: 'Community',
    observations: [
        'Community for safety and alignment techniques',
        'Includes jailbreak prevention and alignment methods',
        'Covers robustness and safety considerations'
    ],
    created_date: date('2025-01-01')
});

-- Key memories documentation
CREATE (:Community {
    identifier: 'Query Templates',
    type: 'Documentation',
    observations: [
        'Collection of standardized Cypher queries for prompt engineering guidance',
        'Optimized patterns for technique discovery and implementation',
        'Templates for common prompt engineering use cases',
        '15 ready-to-use query examples for practitioners'
    ],
    created_date: date('2025-01-01')
});

-- ========================================
-- HIERARCHY RELATIONSHIPS
-- ========================================

-- Connect subcommunities to root
MATCH (parent:Community {identifier: 'Prompt Engineering Research'})
MATCH (child:Community {identifier: 'Text-Based Techniques'})
CREATE (child)-[:SUBCOMMUNITY]->(parent);

MATCH (parent:Community {identifier: 'Prompt Engineering Research'})
MATCH (child:Community {identifier: 'Multilingual Techniques'})
CREATE (child)-[:SUBCOMMUNITY]->(parent);

MATCH (parent:Community {identifier: 'Prompt Engineering Research'})
MATCH (child:Community {identifier: 'Multimodal Techniques'})
CREATE (child)-[:SUBCOMMUNITY]->(parent);

MATCH (parent:Community {identifier: 'Prompt Engineering Research'})
MATCH (child:Community {identifier: 'Agent Techniques'})
CREATE (child)-[:SUBCOMMUNITY]->(parent);

MATCH (parent:Community {identifier: 'Prompt Engineering Research'})
MATCH (child:Community {identifier: 'Evaluation Methods'})
CREATE (child)-[:SUBCOMMUNITY]->(parent);

MATCH (parent:Community {identifier: 'Prompt Engineering Research'})
MATCH (child:Community {identifier: 'Security and Alignment'})
CREATE (child)-[:SUBCOMMUNITY]->(parent);

-- Connect documentation to relevant community
MATCH (parent:Community {identifier: 'Text-Based Techniques'})
MATCH (child:Community {identifier: 'Query Templates'})
CREATE (child)-[:SUBCOMMUNITY]->(parent);