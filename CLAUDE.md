# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kuzu Memory Graph MCP Server - a high-performance LLM memory server using Kuzu graph database with semantic search capabilities. The server provides AI assistants with persistent graph-based memory storage through the Model Context Protocol (MCP).

## Common Development Commands

### Running the Server

```bash
# Development mode (recommended)
uv run kuzu-memory-server

# Direct Python execution
python -m kuzu_memory_server

# Test basic functionality
python test_server.py
```

### Testing

```bash
# Run basic functionality test
python test_server.py

# Run architecture verification test
python test_architecture.py

# Run pytest suite (if available)
pytest tests/ -v

# Run with coverage (if installed)
pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Fix linting issues
ruff check src/ tests/ --fix
```

## Architecture Overview

The server has been simplified to a two-file architecture:

### Core Components

**Main Server** (`src/kuzu_memory_server.py`)

- FastMCP-based server implementation with all tools
- Multi-database support with dynamic primary switching via `WritableDatabaseManager`
- Database discovery and metadata management
- Embedding generation with MLX (Apple Silicon) and Sentence Transformers fallback
- Application lifecycle management with proper resource cleanup

### Multi-Database Architecture

The server supports multiple Kuzu databases with dynamic primary (writable) database switching:

- **WritableDatabaseManager**: Global singleton managing database connections
- **switch_primary_database()**: MCP tool for AI to switch between writable databases
- **Database Discovery**: Automatic scanning of `KUZU_DATABASES_DIR` for `.kuzu` files
- **Write Isolation**: Only the current primary database accepts writes
- **Environment Configuration**:
  - `KUZU_WRITABLE_DATABASES`: Comma-separated list of databases that can be made primary
  - `KUZU_DATABASES_DIR`: Directory containing database files
  - `KUZU_MEMORY_DB_PATH`: Initial primary database path

### ✅ Architecture Status: RESOLVED

**Status**: ✅ **FIXED** - Multi-primary database architecture now works correctly

**Solution Implemented**: Complete architectural refactor to ensure single source of truth:

**What Was Fixed**:

1. ✅ **Removed deprecated fields**: `AppContext` no longer has `db`, `conn`, `attached_databases`, or other direct database references
2. ✅ **Single source of truth**: `db_manager` is now the only way to access database connections
3. ✅ **Updated all tools**: All MCP tools now use `app_ctx.db_manager.get_connection()` exclusively
4. ✅ **Connection lifecycle**: Database switching properly closes old connections and creates new ones
5. ✅ **Write isolation**: Each database maintains separate data when switching

**Architecture Pattern**:

- All database access goes through `app_ctx.db_manager.get_connection()`
- Current primary database name via `app_ctx.db_manager.get_current_name()`
- Write operations check if target database is current primary before proceeding
- Read operations can work with any database (creates temporary connections for non-primary databases)

**Verification**:

- ✅ Created comprehensive test suite (`test_architecture.py`) that verifies:
  - Database initialization and switching
  - Write isolation between databases
  - Connection lifecycle management
  - Single source of truth pattern
- ✅ All tests pass, confirming the architecture works correctly

**Status**: 🟢 **WORKING** - Multi-primary database feature fully functional

## Key Technical Details

### Database Schema

```cypher
CREATE NODE TABLE Entity (
    name STRING PRIMARY KEY,
    type STRING,
    observations STRING[],
    embedding FLOAT[384],
    created_date DATE DEFAULT current_date(),
    updated_date DATE DEFAULT current_date()
)

CREATE REL TABLE RELATED_TO (
    FROM Entity TO Entity,
    relationship_type STRING,
    confidence FLOAT DEFAULT 1.0,
    created_date DATE DEFAULT current_date()
)
```

### MCP Tools Available

- `switch_primary_database()` - Switch active writable database
- `create_entity()` - Create entities in current primary database
- `create_relationship()` - Create relationships between entities
- `add_observations()` - Add observations to existing entities
- `search_entities()` - Text-based entity search
- `semantic_search()` - Vector-based semantic search
- `get_related_entities()` - Find entities through relationships
- `get_graph_summary()` - Get database statistics

### MCP Resources

- `kuzu://databases/list` - Discover available databases with metadata

## Development Patterns

### Database Access Pattern

For write operations, always use `db_manager.get_connection()` to get the current writable database connection:

```python
app_ctx = ctx.request_context.lifespan_context
conn = app_ctx.db_manager.get_connection()
current_primary = app_ctx.db_manager.get_current_name()

# Check if target database is current primary for writes
if database != current_primary:
    return {"status": "error", "message": f"Use switch_primary_database(database='{database}') first"}
```

### Embedding Generation

The server supports both MLX (Apple Silicon) and Sentence Transformers:

```python
# MLX path (preferred on Apple Silicon)
embedding = generate_embedding(app_ctx.embedding_model, app_ctx.tokenizer, text)

# Automatic fallback to Sentence Transformers
# Handled automatically in generate_embedding()
```

### Error Handling Pattern

All tools return structured error responses:

```python
return {
    "status": "error",
    "message": "Descriptive error message",
    "database": database,
    "current_primary": current_primary
}
```

## Testing Strategy

### Test Database Setup

Use separate test databases to avoid conflicts:

```python
test_db_path = "./test_memory.kuzu"
os.environ["KUZU_MEMORY_DB_PATH"] = test_db_path

# Cleanup after tests
if os.path.exists(test_db_path):
    shutil.rmtree(test_db_path)
```

### Test Coverage Areas

1. Database switching functionality
2. Write isolation between databases
3. Entity and relationship operations
4. Semantic search accuracy
5. MCP tool integration
6. Connection lifecycle management

## Configuration

### Development Environment

- Python 3.11+ required
- Use `uv` package manager for dependency management
- KuzuDB 0.11.2+ for graph database functionality
- MLX embeddings for Apple Silicon acceleration

### MCP Client Configuration Example

```json
{
  "mcpServers": {
    "kuzu-memory": {
      "command": "uv",
      "args": ["--directory", "/path/to/kuzu-memory-graph-mcp", "run", "kuzu-memory-server"],
      "env": {
        "KUZU_MEMORY_DB_PATH": "/path/to/DBMS/memory.kuzu",
        "KUZU_DATABASES_DIR": "/path/to/DBMS",
        "KUZU_WRITABLE_DATABASES": "memory,prompt_engineering,research_papers"
      }
    }
  }
}
```

## Important Constraints

- **Connection State Management**: Database state must exist in exactly ONE place to avoid inconsistency
- **Write Isolation**: Only the current primary database accepts write operations
- **Kuzu Connection Lifecycle**: Database objects must outlive Connection objects
- **Resource Cleanup**: Proper cleanup of connections and databases on server shutdown
- **Error Handling**: Partial failures during database switching must be handled gracefully
