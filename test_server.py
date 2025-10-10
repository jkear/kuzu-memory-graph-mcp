#!/usr/bin/env python3
"""
Quick test script for the Kuzu Memory Graph MCP Server.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.kuzu_memory_server import app_lifespan, mcp
from mcp.server.fastmcp import FastMCP
from mcp.server.session import ServerSession
from mcp.types import TextContent
from contextlib import asynccontextmanager


async def test_server():
    """Test basic server functionality."""
    print("Testing Kuzu Memory Graph MCP Server...", file=sys.stderr)

    # Test lifespan management
    async with app_lifespan(mcp) as app_ctx:
        print("✓ Database and model initialization successful", file=sys.stderr)

        # Test that we can create a simple entity
        from mcp.server.fastmcp import Context
        from unittest.mock import Mock

        # Create a mock context
        mock_ctx = Mock()
        mock_ctx.request_context.lifespan_context = app_ctx

        # Test basic query execution
        try:
            result = app_ctx.conn.execute("MATCH (e:Entity) RETURN COUNT(e)")
            count = result.get_next()[0]
            print(
                f"✓ Database query successful - found {count} entities", file=sys.stderr
            )
        except Exception as e:
            print(f"✗ Database query failed: {e}", file=sys.stderr)
            return False

        # Test embedding generation
        try:
            from src.kuzu_memory_server import generate_embedding

            embedding = generate_embedding(
                app_ctx.embedding_model, app_ctx.tokenizer, "test entity"
            )
            print(
                f"✓ Embedding generation successful - dimension: {len(embedding)}",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"✗ Embedding generation failed: {e}", file=sys.stderr)
            return False

        print("✓ All tests passed!", file=sys.stderr)
        return True


if __name__ == "__main__":
    success = asyncio.run(test_server())
    sys.exit(0 if success else 1)
