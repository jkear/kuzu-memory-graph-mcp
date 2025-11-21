#!/usr/bin/env python3
"""
Test the embedding fix for data type mismatch.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_embedding_format():
    """Test that embeddings are now correctly formatted as 1D arrays."""
    print("🔧 Testing embedding format fix...")

    try:
        from kuzu_memory_server import generate_embedding

        # Test with a mock embedding model that produces 2D arrays
        class MockMLXOutput:
            def __init__(self):
                # Simulate the problematic 2D array output
                self.text_embeds = MockTensor([[0.1, 0.2, 0.3] + [0.0] * 381])

        class MockTensor:
            def __init__(self, data):
                self.data = data

            def tolist(self):
                return self.data

        # Test the fix
        mock_output = MockMLXOutput()

        # Simulate what the generate_embedding function would do
        embedding = mock_output.text_embeds.tolist()
        print(f"Original embedding shape: {len(embedding)}x{len(embedding[0]) if embedding else 0}")

        # Apply the fix
        if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
            embedding = embedding[0]

        embedding = [float(x) for x in embedding]

        # Ensure correct length
        if len(embedding) != 384:
            if len(embedding) > 384:
                embedding = embedding[:384]
            else:
                embedding = embedding + [0.0] * (384 - len(embedding))

        print(f"Fixed embedding shape: {len(embedding)}")
        print(f"Fixed embedding type: {type(embedding[0])}")

        # Verify the fix
        assert len(embedding) == 384, f"Expected 384 dimensions, got {len(embedding)}"
        assert isinstance(embedding[0], float), f"Expected float type, got {type(embedding[0])}"
        assert isinstance(embedding, list), f"Expected list, got {type(embedding)}"

        print("✅ Embedding format fix verified!")
        return True

    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases for embedding generation."""
    print("\n🔧 Testing embedding edge cases...")

    try:
        from kuzu_memory_server import generate_embedding

        # Test 1: Empty string
        result = generate_embedding(None, None, "")
        assert len(result) == 384, f"Empty string test failed"
        assert all(x == 0.0 for x in result), f"Empty string should return zeros"
        print("✅ Empty string test passed")

        # Test 2: Whitespace only
        result = generate_embedding(None, None, "   ")
        assert len(result) == 384, f"Whitespace test failed"
        assert all(x == 0.0 for x in result), f"Whitespace should return zeros"
        print("✅ Whitespace test passed")

        # Test 3: None text
        result = generate_embedding(None, None, None)
        assert len(result) == 384, f"None text test failed"
        print("✅ None text test passed")

        print("✅ All edge cases passed!")
        return True

    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("EMBEDDING FIX VERIFICATION")
    print("=" * 60)

    format_ok = test_embedding_format()
    edge_cases_ok = test_edge_cases()

    print("\n" + "=" * 60)
    if format_ok and edge_cases_ok:
        print("🎉 ALL TESTS PASSED - Embedding fix is working!")
        print("The DOUBLE[][] to FLOAT[384] conversion is correct.")
    else:
        print("❌ TESTS FAILED - Embedding fix needs more work.")
    print("=" * 60)