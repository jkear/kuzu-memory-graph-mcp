# Kuzu Memory Graph MCP Server - Development Guide

This guide provides comprehensive information to understand, contribute to, or extend the Kuzu Memory Graph MCP Server with multi-database support.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Multi-Database Architecture](#multi-database-architecture)
- [Development Setup](#development-setup)
- [Code Organization](#code-organization)
- [Key Components](#key-components)
- [Testing](#testing)
- [Debugging](#debugging)
- [Contributing](#contributing)
- [Extension Points](#extension-points)

## Architecture Overview

The Kuzu Memory Graph MCP Server follows a simplified architecture with only two main files:

```graph
┌─────────────────────────────────────────┐
│              MCP Protocol Layer         │
│         (FastMCP Server Implementation) │
├─────────────────────────────────────────┤
│        Main Server Implementation       │
│     (src/kuzu_memory_server.py)         │
│   - Tool Definitions & Handlers         │
│   - Multi-Database Management           │
│   - Database Discovery & Attachment     │
│   - Embedding Generation (MLX/ST)       │
├─────────────────────────────────────────┤
│          Semantic Search Module         │
│      (src/semantic_search.py)           │
│   - Fallback Embedding Provider         │
│   - Vector Similarity Computation       │
│   - Embedding Caching                   │
├─────────────────────────────────────────┤
│              Data Layer                 │
│         (Multiple KuzuDB Databases)     │
│         (Vector Index per Database)     │
└─────────────────────────────────────────┘
```

## Multi-Database Architecture

The server supports multiple Kuzu databases with dynamic primary (writable) database switching:

### Database Management Components

1. **Writable Databases Configuration**
   - `KUZU_WRITABLE_DATABASES`: Comma-separated list of databases that can be made primary
   - Example: `"memory,prompt_engineering,research_papers"`
   - Only configured databases can be made writable

2. **Database Manager**
   - `WritableDatabaseManager`: Global singleton managing database connections
   - `initialize()`: Setup with list of writable databases
   - `switch_to(db_name)`: Close current, open new primary database
   - `get_connection()`: Get current writable connection
   - `is_writable(db_name)`: Check if database is in writable list

3. **Database Discovery**
   - `discover_databases()`: Scans directory for `.kuzu` files
   - Auto-detects available databases at startup

4. **Dynamic Switching**
   - `switch_primary_database()`: MCP tool for AI to switch databases
   - Schema and vector index automatically created on switch
   - No server restart required

5. **Resource-Based Discovery**
   - `kuzu://databases/list` MCP Resource exposes available databases
   - Returns JSON with database metadata including writable status

### Database Lifecycle

```graph
Server Startup
    ↓
Parse KUZU_WRITABLE_DATABASES Configuration
    ↓
Initialize Database Manager
    ↓
Switch to Initial Primary Database
    ↓
Ensure Schema & Vector Index
    ↓
Load MLX Embedding Model
    ↓
Ready for Requests
    ↓
Tool Call: switch_primary_database(database="prompt_engineering")
    ↓
Close Current Primary Connection
    ↓
Open New Primary Database
    ↓
Ensure Schema & Vector Index
    ↓
Ready for Write Operations to New Database
    ↓
Tool Call: create_entity(database="prompt_engineering", ...)
    ↓
Execute Write Query on Current Primary
    ↓
Return Result
    ↓
Server Shutdown
    ↓
Cleanup All Connections
```

### Environment Variables

- `KUZU_MEMORY_DB_PATH`: Initial primary database file path (default: `./DBMS/memory.kuzu`)
- `KUZU_DATABASES_DIR`: Directory containing all .kuzu databases (default: `./DBMS`)
- `KUZU_WRITABLE_DATABASES`: Comma-separated list of writable databases (optional, enables multi-primary mode)

### Core Components

1. **Main Server (`src/kuzu_memory_server.py`)**
   - FastMCP-based server implementation with all tools
   - Dynamic database switching via `WritableDatabaseManager`
   - Database discovery and metadata management
   - Embedding generation with MLX (Apple Silicon) and Sentence Transformers fallback
   - Application lifecycle management

2. **Semantic Search Module (`src/semantic_search.py`)**
   - Fallback embedding provider using Sentence Transformers
   - Vector similarity computation utilities
   - Embedding caching functionality

3. **Database Layer**
   - `WritableDatabaseManager`: Global singleton for database lifecycle
   - Dynamic connection management with proper cleanup
   - Schema initialization and vector index creation per database
   - Entity and relationship storage across databases

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- Git
- Optional: Apple Silicon for MLX acceleration

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/jkear/kuzu-memory-graph-mcp.git
cd kuzu-memory-graph-mcp

# Install dependencies including development tools
uv sync --dev

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Development Environment

The project uses the following development tools:

- **Code Formatting**: Black (configured in `pyproject.toml`)
- **Linting**: Ruff (configured in `pyproject.toml`)
- **Testing**: pytest with async support
- **Type Checking**: mypy (optional, add with `uv add --dev mypy`)

### Running Tests

```bash
# Run basic functionality test
python test_server.py

# Run pytest suite (if available)
pytest tests/ -v

# Run with coverage (if installed)
pytest --cov=src tests/
```

### Development Server

```bash
# Run in development mode
uv run python src/kuzu_memory_server.py

# Or using the script entry point
uv run kuzu-memory-server
```

## Code Organization

The project uses a minimal file structure:

```bash
kuzu-memory-graph-mcp/
├── src/
│   ├── kuzu_memory_server.py    # Main MCP server with all tools
│   └── semantic_search.py       # Semantic search utilities (fallback)
├── docs/                        # Documentation
│   ├── API.md                   # API documentation
│   └── DEVELOPMENT.md           # Development guide
├── pyproject.toml               # Project configuration
├── README.md                    # Project overview
├── test_server.py               # Basic test script
├── .gitignore                   # Git ignore file
└── .python-version              # Python version specification
```

### Architecture Philosophy

The project follows a streamlined two-file implementation:

- **Single server file**: All MCP tools and database management in `src/kuzu_memory_server.py`
- **Utility module**: Semantic search utilities in `src/semantic_search.py` (used as fallback)
- **Minimal complexity**: Direct implementation without separate config, logging, validation, or connection management layers
- **Direct dependencies**: All dependencies managed through pyproject.toml

## Key Components

### MCP Server Implementation

The server uses FastMCP for MCP protocol handling:

```python
from mcp.server.fastmcp import FastMCP, Context

# Create server with lifespan management
mcp = FastMCP("kuzu-memory-graph", lifespan=app_lifespan)

@mcp.tool()
async def switch_primary_database(ctx: Context, database: str):
    """Switch the active primary (writable) database."""
    app_ctx = ctx.request_context.lifespan_context
    success, msg = app_ctx.db_manager.switch_to(database)
    return {
        "success": success,
        "message": msg,
        "current_primary": app_ctx.db_manager.get_current_name()
    }

@mcp.tool()
async def create_entity(ctx: Context, database: str, name: str, entity_type: str, observations: list[str] = None):
    """Create a new entity in the knowledge graph."""
    app_ctx = ctx.request_context.lifespan_context
    
    # Check if target database is current primary
    if database != app_ctx.db_manager.get_current_name():
        if app_ctx.db_manager.is_writable(database):
            return {"status": "error", "message": f"Use switch_primary_database(database='{database}') first"}
        else:
            return {"status": "error", "message": f"Database '{database}' is not writable"}
    
    conn = app_ctx.db_manager.get_connection()
    # Implementation...
```

### Application Lifecycle Management

The server uses a context manager for resource management with dynamic database switching:

```python
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with database and model initialization."""
    # Parse writable databases configuration
    writable_dbs_str = os.getenv('KUZU_WRITABLE_DATABASES', '')
    writable_databases = [db.strip() for db in writable_dbs_str.split(',') if db.strip()]
    
    # Get databases directory
    databases_dir = os.getenv('KUZU_DATABASES_DIR', './DBMS')
    
    # Load embedding models
    embedding_model = load_embeddings()
    tokenizer = load_tokenizer()
    
    # Initialize database manager
    db_manager.initialize(writable_databases, databases_dir, embedding_model, tokenizer)
    
    # Switch to initial primary database
    initial_db = writable_databases[0] if writable_databases else get_primary_db_name(db_path)
    success, msg = db_manager.switch_to(initial_db)
    
    if not success:
        raise RuntimeError(f"Failed to initialize primary database: {msg}")
    
    try:
        yield AppContext(
            db_manager=db_manager,
            embedding_model=embedding_model,
            tokenizer=tokenizer,
            databases_dir=databases_dir
        )
    finally:
        # Cleanup all connections
        db_manager.cleanup()
```

### Embedding Generation

The server supports both MLX (Apple Silicon) and Sentence Transformers:

```python
def generate_embedding(model: Any, tokenizer: Any, text: str) -> list[float]:
    """Generate 384-dimensional embedding using MLX or fallback."""
    if not text or not text.strip():
        return [0.0] * 384
    
    # Try MLX first
    try:
        import mlx.core as mx
        inputs = tokenizer.encode(text.strip(), return_tensors="mlx")
        outputs = model(inputs)
        return outputs.text_embeds.tolist()
    except:
        # Fallback to sentence transformers
        embedding = embedding_model.encode(text.strip(), convert_to_numpy=True)
        return embedding.tolist()
```

### Database Schema

The graph schema is created programmatically:

```python
# Entity node table
conn.execute("""
    CREATE NODE TABLE IF NOT EXISTS Entity (
        name STRING PRIMARY KEY,
        type STRING,
        observations STRING[],
        embedding FLOAT[384],
        created_date DATE DEFAULT current_date(),
        updated_date DATE DEFAULT current_date()
    )
""")

# Relationship table
conn.execute("""
    CREATE REL TABLE IF NOT EXISTS RELATED_TO (
        FROM Entity TO Entity,
        relationship_type STRING,
        confidence FLOAT DEFAULT 1.0,
        created_date DATE DEFAULT current_date()
    )
""")
```

## Testing

### Test Structure

The project includes multiple testing approaches:

1. **Basic Functionality Test** (`test_server.py`)
   - Tests database initialization
   - Verifies embedding generation
   - Validates basic query execution

2. **Unit Tests** (in `tests/` directory)
   - Individual component testing
   - Mock dependencies for isolation
   - Edge case validation

3. **Integration Tests**
   - End-to-end workflow testing
   - MCP protocol validation
   - Performance benchmarks

### Writing Tests

Example test for entity creation:

```python
import pytest
from unittest.mock import Mock, AsyncMock
from src.kuzu_memory_server import create_entity

@pytest.mark.asyncio
async def test_create_entity():
    # Setup mock context
    mock_ctx = Mock()
    mock_ctx.request_context.lifespan_context = Mock()
    mock_ctx.request_context.lifespan_context.conn = AsyncMock()
    
    # Mock database response
    mock_ctx.request_context.lifespan_context.conn.execute.return_value = Mock()
    mock_ctx.request_context.lifespan_context.conn.execute.return_value.has_next.return_value = False
    
    # Test entity creation
    result = await create_entity(
        mock_ctx,
        name="Test Entity",
        entity_type="test",
        observations=["Test observation"]
    )
    
    # Assertions
    assert result["status"] == "created"
    assert result["name"] == "Test Entity"
    assert result["type"] == "test"
```

### Test Data Management

For testing, use a separate database:

```python
# In test setup
test_db_path = "./test_memory.kuzu"
os.environ["KUZU_MEMORY_DB_PATH"] = test_db_path

# Cleanup after tests
if os.path.exists(test_db_path):
    shutil.rmtree(test_db_path)
```

## Debugging

### Logging

Add logging to debug issues:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In your code
logger.info(f"Creating entity: {name}")
logger.debug(f"Generated embedding dimension: {len(embedding)}")
```

### Common Debugging Scenarios

1. **Database Connection Issues**:

   ```python
   # Check database path and permissions
   print(f"text {var}", file=sys.stderr)f"Database path: {db_path}")
   print(f"text {var}", file=sys.stderr)f"Database exists: {os.path.exists(db_path)}")
   ```

2. **Embedding Generation Problems**:

   ```python
   # Test embedding generation
   test_embedding = generate_embedding(model, tokenizer, "test")
   print(f"text {var}", file=sys.stderr)f"Embedding dimension: {len(test_embedding)}")
   print(f"text {var}", file=sys.stderr)f"Sample values: {test_embedding[:5]}")
   ```

3. **MCP Tool Registration**:

   ```python
   # List registered tools
   print(f"text {var}", file=sys.stderr)"Registered tools:", list(mcp.tools.keys()))
   ```

### Performance Profiling

Use Python's built-in profiler:

```python
import cProfile
import pstats

# Profile embedding generation
profiler = cProfile.Profile()
profiler.enable()

# Your code here
embedding = generate_embedding(model, tokenizer, text)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

## Contributing

### Code Style

Follow the project's code style:

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Fix linting issues
ruff check src/ tests/ --fix
```

### Adding New MCP Tools

1. Define the tool function with proper type hints
2. Add `database` parameter (first parameter after `ctx`) if the tool needs database access
3. Add comprehensive docstring with parameter descriptions
4. Check if database is writable and is the current primary for write operations
5. Implement error handling with helpful messages
6. Add tests for the new tool
7. Update API documentation

Example (Read Operation):

```python
@mcp.tool()
async def new_read_tool(
    ctx: Context[ServerSession, AppContext],
    database: str,
    param1: str,
    param2: int = 10
) -> dict[str, Any]:
    """New read tool description.
    
    Args:
        database: Database name (query kuzu://databases/list to see available)
        param1: Description of parameter 1
        param2: Description of parameter 2 (default: 10)
        
    Returns:
        Dictionary with tool results
    """
    app_ctx = ctx.request_context.lifespan_context
    
    # Read operations can work with any discovered database
    try:
        # Implementation for reading data
        return {"status": "success", "database": database, "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e), "database": database}
```

Example (Write Operation):

```python
@mcp.tool()
async def new_write_tool(
    ctx: Context[ServerSession, AppContext],
    database: str,
    param1: str
) -> dict[str, Any]:
    """New write tool description.
    
    Args:
        database: Database name (must be current primary)
        param1: Description of parameter 1
        
    Returns:
        Dictionary with tool results
    """
    app_ctx = ctx.request_context.lifespan_context
    current_primary = app_ctx.db_manager.get_current_name()
    
    # Check if target database is the current primary
    if database != current_primary:
        if app_ctx.db_manager.is_writable(database):
            return {
                "status": "error",
                "message": f"Use switch_primary_database(database='{database}') first",
                "database": database
            }
        else:
            return {
                "status": "error",
                "message": f"Database '{database}' is not writable",
                "database": database
            }
    
    conn = app_ctx.db_manager.get_connection()
    
    try:
        # Implementation for writing data
        return {"status": "success", "database": database, "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e), "database": database}
```

### Schema Changes

When modifying the database schema:

1. Create migration scripts
2. Update schema documentation
3. Test with existing databases
4. Handle backward compatibility

Example migration:

```python
def migrate_schema(conn: kuzu.Connection):
    """Migrate database to new schema version."""
    try:
        # Add new column
        conn.execute("ALTER TABLE Entity ADD COLUMN new_field STRING;")
        print(f"text {var}", file=sys.stderr)"Schema migration completed")
    except Exception as e:
        print(f"text {var}", file=sys.stderr)f"Migration failed: {e}")
```

## Extension Points

### Custom Embedding Models

Add support for new embedding models:

```python
class CustomEmbeddingProvider:
    def __init__(self, model_name: str):
        self.model = load_custom_model(model_name)
    
    def encode(self, text: str) -> list[float]:
        # Custom encoding logic
        return self.model.encode(text)
    
    def batch_encode(self, texts: list[str]) -> list[list[float]]:
        # Batch encoding logic
        return [self.encode(text) for text in texts]
```

### Multi-Database Extensions

Add custom database management features:

```python
def custom_database_filter(
    databases: dict[str, dict[str, str]],
    filter_criteria: dict[str, Any]
) -> dict[str, dict[str, str]]:
    """Filter databases based on custom criteria."""
    filtered = {}
    for name, info in databases.items():
        # Apply custom filtering logic
        if matches_criteria(info, filter_criteria):
            filtered[name] = info
    return filtered

def cross_database_query(
    ctx: Context,
    query: str,
    databases: list[str]
) -> dict[str, Any]:
    """Execute query across multiple databases."""
    results = {}
    for db_name in databases:
        # Switch to each database and execute query
        success, _ = ensure_database_attached(...)
        if success:
            switch_database_context(...)
            result = execute_query(query)
            results[db_name] = result
    return {"results": results}
```

### Additional Entity Types

Extend the entity system:

```python
# Custom entity validation
def validate_person_entity(observations: list[str]) -> bool:
    """Validate person-specific observations."""
    required_patterns = ["name", "age", "occupation"]
    return any(pattern in " ".join(observations).lower() for pattern in required_patterns)

# Custom relationship types
VALID_RELATIONSHIPS = {
    "person": ["KNOWS", "WORKS_WITH", "FRIEND_OF"],
    "concept": ["RELATED_TO", "EXAMPLE_OF", "APPLICATION_OF"],
    "document": ["REFERENCES", "DESCRIBES", "CONTAINS"]
}
```

### Custom Search Algorithms

Implement specialized search:

```python
async def hybrid_search(
    ctx: Context,
    query: str,
    text_weight: float = 0.5,
    semantic_weight: float = 0.5,
    limit: int = 10
) -> dict[str, Any]:
    """Combine text and semantic search with weighted scoring."""
    # Get text search results
    text_results = await search_entities(ctx, query, limit * 2)
    
    # Get semantic search results
    semantic_results = await semantic_search(ctx, query, limit * 2)
    
    # Combine and re-rank
    combined_results = combine_results(
        text_results, semantic_results, 
        text_weight, semantic_weight
    )
    
    return {"results": combined_results[:limit]}
```

### Performance Monitoring

Add metrics collection:

```python
import time
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def record_operation(self, operation: str, duration: float):
        self.metrics[operation].append(duration)
    
    def get_stats(self) -> dict[str, dict]:
        stats = {}
        for op, times in self.metrics.items():
            stats[op] = {
                "count": len(times),
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times)
            }
        return stats

# Usage
monitor = PerformanceMonitor()

@mcp.tool()
async def search_entities(ctx: Context, query: str, limit: int = 10):
    start_time = time.time()
    # Implementation...
    duration = time.time() - start_time
    monitor.record_operation("search_entities", duration)
    return result
```

This developer guide provides a comprehensive foundation for understanding and extending the Kuzu Memory Graph MCP Server. For specific implementation details, refer to the source code and API documentation.
