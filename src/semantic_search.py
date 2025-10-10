"""
Semantic Search Module for Kuzu Memory Graph

Provides vector-based semantic search capabilities using sentence transformers.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """High-performance semantic search using sentence transformers."""

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None
    ):
        """Initialize semantic search engine.

        Args:
            model_name: Name of the sentence transformer model
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir or "./.embeddings_cache")
        self.cache_dir.mkdir(exist_ok=True)

        print(f"Loading sentence transformer model: {model_name}", file=sys.stderr)
        self.model = SentenceTransformer(model_name)
        print(
            f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}",
            file=sys.stderr,
        )

    def encode_text(self, text: str) -> List[float]:
        """Encode text to embedding vector.

        Args:
            text: Text to encode

        Returns:
            List of float values representing the embedding
        """
        if not text or not text.strip():
            embedding_dim = self.model.get_sentence_embedding_dimension()
            return [0.0] * (embedding_dim if embedding_dim is not None else 384)

        embedding = self.model.encode(text.strip(), convert_to_numpy=True)
        return embedding.tolist()

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts to embedding vectors.

        Args:
            texts: List of texts to encode

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [text.strip() for text in texts if text and text.strip()]

        if not valid_texts:
            embedding_dim = self.model.get_sentence_embedding_dimension()
            embedding_dim = embedding_dim if embedding_dim is not None else 384
            return [[0.0] * embedding_dim] * len(texts)

        embeddings = self.model.encode(valid_texts, convert_to_numpy=True)

        # Map back to original order with empty vectors for invalid texts
        result = []
        valid_idx = 0
        for text in texts:
            if text and text.strip():
                result.append(embeddings[valid_idx].tolist())
                valid_idx += 1
            else:
                embedding_dim = self.model.get_sentence_embedding_dimension()
                embedding_dim = embedding_dim if embedding_dim is not None else 384
                result.append([0.0] * embedding_dim)

        return result

    def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        if not embedding1 or not embedding2:
            return 0.0

        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Find most similar embeddings to query.

        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top results to return

        Returns:
            List of (index, similarity_score) tuples
        """
        if not query_embedding or not candidate_embeddings:
            return []

        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append((i, similarity))

        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _sanitize_cache_key(self, key: str) -> str:
        """Sanitize cache key to prevent path traversal attacks.

        Args:
            key: Original cache key

        Returns:
            Sanitized cache key safe for filesystem use
        """
        # Remove path separators and limit length
        safe_key = key.replace("/", "_").replace("\\", "_").replace("..", "_")
        # Limit to reasonable length to prevent filesystem issues
        return safe_key[:100] if len(safe_key) > 100 else safe_key

    def cache_embedding(self, key: str, embedding: List[float]) -> bool:
        """Cache an embedding to disk using JSON serialization.

        Args:
            key: Cache key
            embedding: Embedding vector

        Returns:
            True if caching succeeded, False otherwise
        """
        try:
            safe_key = self._sanitize_cache_key(key)
            cache_file = self.cache_dir / f"{safe_key}.json"

            with open(cache_file, "w") as f:
                json.dump(
                    {
                        "embedding": embedding,
                        "model": self.model_name,
                        "version": "1.0",
                    },
                    f,
                )

            logger.debug(f"Cached embedding for key: {safe_key}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache embedding for key {key}: {e}")
            return False

    def load_cached_embedding(self, key: str) -> Optional[List[float]]:
        """Load cached embedding from disk using JSON deserialization.

        Args:
            key: Cache key

        Returns:
            Cached embedding if exists, None otherwise
        """
        try:
            safe_key = self._sanitize_cache_key(key)
            cache_file = self.cache_dir / f"{safe_key}.json"

            if cache_file.exists():
                with open(cache_file, "r") as f:
                    data = json.load(f)

                # Validate cache data structure
                if (
                    isinstance(data, dict)
                    and "embedding" in data
                    and isinstance(data["embedding"], list)
                ):
                    # Verify embedding dimension matches expected
                    if len(data["embedding"]) == self.get_embedding_dimension():
                        logger.debug(f"Loaded cached embedding for key: {safe_key}")
                        return data["embedding"]
                    else:
                        logger.warning(
                            f"Cached embedding dimension mismatch for key: {safe_key}"
                        )

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in cache file for key {key}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load cached embedding for key {key}: {e}")

        return None

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        dim = self.model.get_sentence_embedding_dimension()
        return dim if dim is not None else 384

    def encode_with_cache(self, text: str, cache_key: str) -> List[float]:
        """Encode text with caching support.

        Args:
            text: Text to encode
            cache_key: Key for caching

        Returns:
            Embedding vector
        """
        # Try to load from cache first
        cached = self.load_cached_embedding(cache_key)
        if cached is not None:
            return cached

        # Generate new embedding
        embedding = self.encode_text(text)

        # Cache for future use
        self.cache_embedding(cache_key, embedding)

        return embedding
