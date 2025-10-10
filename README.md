# Kuzu Memory Graph MCP Server

A high-performance LLM memory server using Kuzu graph database with semantic search capabilities, built with the Model Context Protocol (MCP) for seamless integration with AI assistants and agents.

## 🌟 Features

- **Graph-based Memory Storage**: Uses KuzuDB for efficient relationship traversal and complex queries
- **Semantic Search**: Vector-based similarity search using sentence transformers and MLX embeddings
- **MCP Protocol**: Full Model Context Protocol implementation for AI assistant integration
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
git clone https://github.com/your-org/kuzu-memory-graph-mcp.git
cd kuzu-memory-graph-mcp

# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Option 2: Using pip (But why would you?)

```bash
# Clone the repository
git clone https://github.com/your-org/kuzu-memory-graph-mcp.git
cd kuzu-memory-graph-mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Running the Server

```bash
# Run as MCP server
uv run kuzu-memory-server

# Or directly with Python. Again, you could but should you?
python -m src.kuzu_memory_server
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KUZU_MEMORY_DB_PATH` | Database file path | `./memory.kuzu` |
| `SEMANTIC_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `EMBEDDING_CACHE_DIR` | Embedding cache directory | `./.embeddings_cache` |

### MCP Client Configuration

Add to your MCP client configuration (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "kuzu-memory": {
      "command": "uv",
      "args": ["run", "kuzu-memory-server"],
      "env": {
        "KUZU_MEMORY_DB_PATH": "/path/to/your/memory.kuzu"
      }
    }
  }
}
```

## 📚 Usage Examples

### Creating Entities

```python
# Create a person entity
create_entity(
    name="Jordan Kearfott",
    entity_type="person",
    observations=["Software vibe-ineer", "Studies LLM Memory techniques", "Lives in Gainesville", "Mid at Python but improving"]
)

# Create a concept entity
create_entity(
    name="Machine Learning",
    entity_type="concept",
    observations=["Subset of AI", "Uses statistical techniques", "Enables computers to learn"]
)
```

### Creating Relationships

```python
# Create relationship between entities
create_relationship(
    from_entity="Sam Altman",
    to_entity="OpenAI",
    relationship_type="IS_CEO",
    confidence=0.9
)
```

### Searching Entities

```python
# Text-based search
search_entities(query="software engineer", limit=10)

# Semantic search
semantic_search(query="AI programming", limit=10, threshold=0.3)

# Get related entities
get_related_entities(entity_name="Alice Johnson", max_depth=2)
```

## 🛠️ MCP Tools

The server provides the following MCP tools:

| Tool | Description |
|------|-------------|
| `create_entity` | Create new entities in the knowledge graph |
| `create_relationship` | Create relationships between entities |
| `add_observations` | Add observations to existing entities |
| `search_entities` | Search entities using text-based queries |
| `semantic_search` | Search entities using semantic similarity |
| `get_related_entities` | Find entities related through relationships |
| `get_graph_summary` | Get statistics about the knowledge graph |

## 🏗️ Architecture

The Kuzu Memory Graph MCP Server consists of:

1. **MCP Server Layer**: FastMCP-based protocol implementation
2. **Graph Database Layer**: KuzuDB for entity and relationship storage
3. **Semantic Search Layer**: MLX/Sentence Transformers for embedding generation
4. **Query Layer**: Cypher query execution with vector similarity search

## 🧪 Testing

Run the test suite:

```zsh
# Run basic functionality test
uv python test_server.py

# Run with pytest (if installed)
pytest tests/
```

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
4. Start development server with `uv run kuzu-memory-server`

See [DEVELOPER.md](DEVELOPER.md) for detailed development instructions.

## 🚀 Deployment

For production deployment considerations, see [DEPLOYMENT.md](DEPLOYMENT.md).

## 🐛 Troubleshooting

For common issues and solutions, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## 📚 Documentation

### Application Lifecycle Management

The server uses a context manager for resource management:

```python
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with database and model initialization."""
    # Database setup
    # Model loading
    # Schema creation
    try:
        yield AppContext(db=db, conn=conn, embedding_model=model, tokenizer=tokenizer)
    finally:
        # Cleanup
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
   print(f"Database path: {db_path}")
   print(f"Database exists: {os.path.exists(db_path)}")
   ```

2. **Embedding Generation Problems**:

   ```python
   # Test embedding generation
   test_embedding = generate_embedding(model, tokenizer, "test")
   print(f"Embedding dimension: {len(test_embedding)}")
   print(f"Sample values: {test_embedding[:5]}")
   ```

3. **MCP Tool Registration**:

   ```python
   # List registered tools
   print("Registered tools:", list(mcp.tools.keys()))
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
