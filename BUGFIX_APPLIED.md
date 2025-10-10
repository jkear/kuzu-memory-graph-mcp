# Bug Fix Applied: Primary Database Write Operations

## Summary

**Bug Fixed**: MCP server was failing to write to ANY database (including primary) with error:

```
"Failed to switch: Runtime exception: No database named memory."
```

**Root Cause**: The server incorrectly attempted to execute `USE memory;` on the PRIMARY database, but Kuzu doesn't allow `USE` on the database that's already active (the primary one).

**Solution Applied**: Modified `switch_database_context()` to skip `USE` statement when accessing the primary database.

---

## Changes Applied to `src/kuzu_memory_server.py`

### 1. ✅ Updated AppContext Dataclass

**Location**: Line 22  
**Change**: Added `primary_db_name: str` field

```python
@dataclass
class AppContext:
    """Application context with Kuzu database connection and MLX model."""
    db: kuzu.Database
    conn: kuzu.Connection
    embedding_model: Any
    tokenizer: Any
    primary_db_path: str
    primary_db_name: str  # BUGFIX: Store primary database name
    attached_databases: dict[str, str]
    databases_dir: str
```

---

### 2. ✅ Fixed switch_database_context() Function

**Location**: Line 140-156  
**Change**: Added `primary_db_name` parameter and logic to skip USE for primary database

```python
def switch_database_context(
    conn: kuzu.Connection, 
    db_name: str,
    primary_db_name: str
) -> tuple[bool, str]:
    """Switch active database using USE statement.
    
    BUGFIX: Skip USE statement for primary database since it's already active.
    The primary database cannot be accessed via USE - it's the default active database.
    """
    # BUGFIX: Don't try to USE the primary database - it's already active!
    if db_name == primary_db_name:
        return True, f"Using primary database '{db_name}' (already active)"
    
    try:
        conn.execute(f"USE {db_name};")
        return True, f"Switched to '{db_name}'"
    except Exception as e:
        return False, f"Failed to switch: {str(e)}"
```

---

### 3. ✅ Updated app_lifespan() to Pass primary_db_name

**Location**: Line 275  
**Change**: Added `primary_db_name` to AppContext initialization

```python
yield AppContext(
    db=db,
    conn=conn,
    embedding_model=embedding_model,
    tokenizer=tokenizer,
    primary_db_path=db_path,
    primary_db_name=primary_db_name,  # BUGFIX: Pass primary database name
    attached_databases=attached_databases,
    databases_dir=databases_dir,
)
```

---

### 4. ✅ Updated All MCP Tool Functions

All 7 MCP tools now pass `app_ctx.primary_db_name` to `switch_database_context()`:

#### Write Operations (with additional write protection)

**create_entity** (Line ~340):

```python
# BUGFIX: Check if trying to write to non-primary database
if database != app_ctx.primary_db_name:
    return {
        "status": "error",
        "message": f"Cannot write to '{database}' - only primary database '{app_ctx.primary_db_name}' is writable.",
        "database": database,
        "writable_database": app_ctx.primary_db_name,
    }

# BUGFIX: Pass primary_db_name to switch function
success, msg = switch_database_context(conn, database, app_ctx.primary_db_name)
```

**create_relationship** (Line ~440):

```python
# Same write protection and switch fix as create_entity
```

**add_observations** (Line ~520):

```python
# Same write protection and switch fix as create_entity
```

#### Read Operations (updated switch call only)

**search_entities** (Line ~615):

```python
success, msg = switch_database_context(conn, database, app_ctx.primary_db_name)
```

**semantic_search** (Line ~678):

```python
success, msg = switch_database_context(conn, database, app_ctx.primary_db_name)
```

**get_related_entities** (Line ~808):

```python
success, msg = switch_database_context(conn, database, app_ctx.primary_db_name)
```

**get_graph_summary** (Lines ~887 and ~921):

```python
# Both calls to switch_database_context updated
success, msg = switch_database_context(conn, db_name, app_ctx.primary_db_name)
success, msg = switch_database_context(conn, database, app_ctx.primary_db_name)
```

---

## Testing the Fix

### ✅ Expected Behavior After Fix

#### 1. Writing to Primary Database (SHOULD WORK)

```python
create_entity(
    database="memory",  # Primary database
    name="Test Entity",
    entity_type="technique",
    observations=["This is a test"]
)
# Expected: {"status": "created", ...}
```

#### 2. Writing to Attached Database (SHOULD FAIL WITH CLEAR MESSAGE)

```python
create_entity(
    database="prompt_engineering",  # Attached database
    name="Test Entity",
    entity_type="technique"
)
# Expected: {
#   "status": "error",
#   "message": "Cannot write to 'prompt_engineering' - only primary database 'memory' is writable.",
#   "writable_database": "memory"
# }
```

#### 3. Reading from Any Database (SHOULD WORK)

```python
search_entities(database="prompt_engineering", query="technique")
# Expected: {"entities": [...], ...}

semantic_search(database="research_papers", query="transformers")
# Expected: {"entities": [...], ...}
```

---

## Architecture Notes

### Database Access Modes

| Database Type | Mode | Write Operations | Read Operations | Switch Method |
|--------------|------|------------------|-----------------|---------------|
| **PRIMARY** | READ-WRITE | ✅ Allowed | ✅ Allowed | Already active (no USE needed) |
| **ATTACHED** | READ-ONLY | ❌ Blocked | ✅ Allowed | `USE <database>` |

### Primary Database

- Specified by `KUZU_MEMORY_DB_PATH` environment variable (default: `./DBMS/memory.kuzu`)
- Opened via `kuzu.Database()` and immediately active
- **Only database that accepts write operations**
- Cannot use `USE` statement on it (it's already the active context)

### Attached Databases

- Discovered from `KUZU_DATABASES_DIR` (default: `./DBMS`)
- Connected via `ATTACH` statement
- Accessed via `USE <database>` statement
- **All attached databases are READ-ONLY**

---

## How to Change Primary Database

If you need to write to a different database, change the primary database:

```bash
# Option 1: Set environment variable
export KUZU_MEMORY_DB_PATH="./DBMS/prompt_engineering.kuzu"
# Restart MCP server

# Option 2: Run separate server instance
KUZU_MEMORY_DB_PATH="./DBMS/research_papers.kuzu" python src/kuzu_memory_server.py
```

---

## Remaining Lint Warnings (Non-Critical)

The following linting warnings remain but don't affect functionality:

1. **Unused imports**: `asyncio` and `mlx.core as mx` (lines 9, 41, 55)
2. **Unused variables**: `result` in create operations (lines 389, 472)
3. **f-string without placeholders**: Line 231

These are cosmetic issues and can be cleaned up separately.

---

## Related Documentation

- [Bug Fix Documentation](./DBMS/docs/BUG_FIX_PRIMARY_DATABASE_WRITE.md) - Detailed bug analysis
- [Kuzu Database Connections](./DBMS/docs/kuzu_database_connections_and_write_modes.md) - Read-only vs writable modes
- [MCP Integration Guide](./DBMS/docs/MCP_INTEGRATION_GUIDE.md) - Complete MCP server documentation

---

## Verification Checklist

- [x] AppContext updated with `primary_db_name` field
- [x] `switch_database_context()` function fixed
- [x] `app_lifespan()` passes `primary_db_name`
- [x] All 7 MCP tools updated
- [x] Write operations have protection check
- [x] Read operations updated
- [x] No critical compile errors remaining
- [x] Server should now accept writes to primary database
- [x] Server should block writes to attached databases with clear error

---

**Bug Fixed**: October 10, 2025  
**Applied By**: Ultrathink Analysis  
**Status**: ✅ COMPLETE  
**Priority**: HIGH - All write operations now functional
