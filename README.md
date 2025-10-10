# Kuzu Memory Graph MCP Server

A high-performance LLM memory server using Kuzu graph database with semantic search capabilities, built with the Model Context Protocol (MCP) for seamless integration with AI assistants and agents.

## 🌟 Features

- **Graph-based Memory Storage**: Uses KuzuDB for efficient relationship traversal and complex queries
- **Multi-Database Support**: Work with multiple Kuzu databases simultaneously with dynamic switching
- **Database Discovery**: Automatic discovery of available databases via MCP Resources
- **Dynamic Primary Switching**: Switch between writable databases without server restart
- **Semantic Search**: Vector-based similarity search using sentence transformers and MLX embeddings
- **Hybrid Search**: Combines text-based and semantic search for comprehensive results
- **Fast & Reliable**: Optimized for AI/agent memory use cases with Apple Silicon acceleration
- **Flexible Entity Model**: Support for custom entity types and relationships
- **Automatic Embedding Generation**: MLX-optimized embeddings for Apple Silicon with fallback support

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- KuzuDB 0.11.2+

### Installation

#### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/jkear/kuzu-memory-graph-mcp.git
cd kuzu-memory-graph-mcp

# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/jkear/kuzu-memory-graph-mcp.git
cd kuzu-memory-graph-mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Running the Server

```bash
# Run using uv in development mode (recommended)
uv run kuzu-memory-server

# Or run directly with Python after activating the venv
python -m kuzu_memory_server

# Or use uvx for testing (installs from PyPI - for published package only)
# uvx kuzu-memory-server
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KUZU_MEMORY_DB_PATH` | Initial primary database file path | `./DBMS/memory.kuzu` |
| `KUZU_DATABASES_DIR` | Directory containing all .kuzu databases | `./DBMS` |
| `KUZU_WRITABLE_DATABASES` | Comma-separated list of databases that can be made primary (writable) | Single-primary mode |
| `SEMANTIC_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `EMBEDDING_CACHE_DIR` | Embedding cache directory | `./.embeddings_cache` |

### MCP Client Configuration

Add to your MCP client configuration (e.g., Claude Desktop `~/Library/Application Support/Claude/claude_desktop_config.json`):

#### For Development (Local Project)

```json
{
  "mcpServers": {
    "kuzu-memory": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/kuzu-memory-graph-mcp",
        "run",
        "kuzu-memory-server"
      ],
      "env": {
        "KUZU_MEMORY_DB_PATH": "/path/to/DBMS/memory.kuzu",
        "KUZU_DATABASES_DIR": "/path/to/DBMS",
        "KUZU_WRITABLE_DATABASES": "memory,prompt_engineering,research_papers"
      }
    }
  }
}
```

#### For Production (Published Package)

```json
{
  "mcpServers": {
    "kuzu-memory": {
      "command": "uvx",
      "args": ["kuzu-memory-server"],
      "env": {
        "KUZU_MEMORY_DB_PATH": "/path/to/DBMS/memory.kuzu",
        "KUZU_DATABASES_DIR": "/path/to/DBMS",
        "KUZU_WRITABLE_DATABASES": "memory,prompt_engineering,research_papers"
      }
    }
  }
}
```

## 📚 Usage Examples

### Multi-Database Setup

Organize your databases in a dedicated directory and configure which databases can be written to:

```bash
/path/to/databases/
├── memory.kuzu           # Writable database
├── prompt_engineer.kuzu  # Writable database
└── research_papers.kuzu  # Writable database
```

Configure writable databases in your environment:

```bash
export KUZU_WRITABLE_DATABASES="memory,prompt_engineer,research_papers"
export KUZU_DATABASES_DIR="/path/to/databases"
```

### Switching Between Writable Databases

Use the `switch_primary_database` tool to change which database receives write operations:

```python
# Switch to prompt_engineer database
switch_primary_database(database="prompt_engineer")

# Now create entities in prompt_engineer
create_entity(
    database="prompt_engineer",
    name="Chain of Thought",
    entity_type="prompt_pattern",
    observations=["Improves reasoning", "Step-by-step thinking"]
)
```

### Database Discovery

Query available databases via MCP Resource:

```python
# List all available databases
resource_result = await access_resource("kuzu://databases/list")
# Returns JSON with database metadata including writable status
```

### Creating Entities

```python
# Create a person entity in the 'memory' database
create_entity(
    database="memory",
    name="Jordan Kearfott",
    entity_type="person",
    observations=["Vibe engineer for Sales & Marketing", "Studies LLM Memory techniques", "Lives in Gainesville", "Mid at Python"]
)

# Switch to prompt_engineer and create an entity
switch_primary_database(database="prompt_engineer")
create_entity(
    database="prompt_engineer",
    name="Chain of Thought",
    entity_type="prompt_pattern",
    observations=["Improves reasoning", "Step-by-step thinking"]
)

# Switch to research_papers and create an entity
switch_primary_database(database="research_papers")
create_entity(
    database="research_papers",
    name="Attention Is All You Need",
    entity_type="paper",
    observations=["Transformer architecture", "Self-attention mechanism"]
)
```

### Creating Relationships

```python
# Create relationship between entities in specific database
create_relationship(
    database="memory",
    from_entity="Sam Altman",
    to_entity="OpenAI",
    relationship_type="IS_CEO",
    confidence=0.9
)
```

### Searching Entities

```python
# Search in specific database
search_entities(database="memory", query="software engineer", limit=10)

# Semantic search across research papers
semantic_search(database="research_papers", query="transformer architecture", limit=10, threshold=0.3)

# Get related entities in prompt database
get_related_entities(database="prompt_engineer", entity_name="Chain of Thought", max_depth=2)
```

### Cross-Database Operations

```python
# Get overview of all databases
summary = await get_graph_summary()  # No database param = all databases

# Get detailed summary of specific database
summary = await get_graph_summary(database="memory")
```

## 🛠️ MCP Tools & Resources

The server provides the following MCP tools and resources:

### Resources

| Resource | Description |
|----------|-------------|
| `kuzu://databases/list` | List all available Kuzu databases with metadata |

### Tools

| Tool | Description |
|------|-------------|
| `switch_primary_database` | Switch the active writable database |
| `create_entity` | Create new entities in the specified knowledge graph |
| `create_relationship` | Create relationships between entities in specified database |
| `add_observations` | Add observations to existing entities in specified database |
| `search_entities` | Search entities using text-based queries in specified database |
| `semantic_search` | Search entities using semantic similarity in specified database |
| `get_related_entities` | Find entities related through relationships in specified database |
| `get_graph_summary` | Get statistics about specific database or all databases |

**Note**: Write operations (`create_entity`, `create_relationship`, `add_observations`) only work on databases listed in `KUZU_WRITABLE_DATABASES`. Use `switch_primary_database` to change the active writable database.

## 🏗️ Architecture

The Kuzu Memory Graph MCP Server has been completely refactored to eliminate dual-state inconsistency:

### Core Components

1. **MCP Server Layer**: FastMCP-based protocol implementation
2. **Database Manager**: Single-source-of-truth database state management with connection pool
3. **Graph Database Layer**: KuzuDB for entity and relationship storage
4. **Semantic Search Layer**: MLX/Sentence Transformers for embedding generation
5. **Query Layer**: Cypher query execution with vector similarity search

### Database Connection Architecture

The server implements a **hybrid connection pool pattern**:

- **WritableDatabaseManager**: Centralized database state manager
- **Connection Pool**: Databases stay open, connections created on-demand
- **Single Source of Truth**: All database state exists only in WritableDatabaseManager
- **Auto-Recovery**: Built-in connection health checks and recovery
- **Fast Switching**: Database switching in <100ms without reopening databases

### Refactored Architecture Benefits

✅ **Eliminated Dual-State Issue**: No more stale connection references
✅ **Centralized State Management**: Single source of truth for all database operations
✅ **Connection Pool Pattern**: Databases remain open for fast switching
✅ **Auto-Recovery**: Automatic detection and recovery from connection failures
✅ **Safety Features**: Path validation, health checks, comprehensive error handling
✅ **Clean Separation**: Write operations vs read operations with proper isolation

## 🧪 Testing

Run the test suite:

```zsh
# Run basic functionality test
uv python test_server.py

# Run architecture verification test
uv python test_architecture.py

# Run multi-primary database tests
uv python test_multi_primary.py

# Run with pytest (if installed)
pytest tests/
```

### Architecture Testing

The refactored architecture includes comprehensive tests to verify:

- ✅ **Single Source of Truth**: All database state centralized in WritableDatabaseManager
- ✅ **Connection Pool Pattern**: Databases remain open, connections created on-demand
- ✅ **Write Isolation**: Entities written to one database don't appear in others
- ✅ **Auto-Recovery**: Connection health checks and automatic recovery
- ✅ **Safety Features**: Path validation and error handling
- ✅ **Legacy Code Removal**: Old dual-state functions eliminated

## 📦 Dependencies

- **kuzu>=0.11.2**: High-performance graph database
- **modelcontextprotocol>=0.1.0**: MCP protocol implementation
- **sentence-transformers>=5.1.1**: Text embedding models
- **mlx-embeddings>=0.0.4**: Apple Silicon optimized embeddings
- **numpy>=2.3.3**: Numerical computations
- **polars>=1.34.0**: Data manipulation
- **pyarrow>=21.0.0**: Columnar data format
- **networkx>=3.5**: Graph algorithms
- **scipy>=1.16.2**: Scientific computing
- **mcp>=1.16.0**: MCP client/server library

## 🔧 Development Setup

1. Clone the repository
2. Install with `uv sync --dev`
3. Run tests with `python test_server.py`
4. Start development server with `uvx run .` or `uvx run . kuzu-memory-server`

## 🚀 Deployment

For production deployment considerations, see [DEPLOYMENT.md](DEPLOYMENT.md).

## 📚 Documentation

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [KuzuDB](https://kuzudb.com/) for the high-performance graph database
- [FastMCP](https://github.com/jlowin/fastmcp) for the MCP framework
- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [MLX](https://ml-explore.github.io/mlx/) for Apple Silicon optimization
