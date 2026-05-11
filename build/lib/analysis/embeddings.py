"""Embedding helpers for post text using real multilingual models.

Improvement #1: Replaces the hash-based embedding with sentence-transformers
using BAAI/bge-m3 for proper semantic understanding of Japanese text.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from typing import Sequence

LOGGER = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

DEFAULT_MODEL = "BAAI/bge-m3"


class TextEmbedder:
    """Embeds text using a sentence-transformers model.

    The model is loaded once on first use and reused for all subsequent calls.
    Uses BAAI/bge-m3 by default for strong multilingual + Japanese support.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
        self._model = None

    def _load_model(self):
        from sentence_transformers import SentenceTransformer

        LOGGER.info("Loading embedding model '%s' (this may download ~2.3GB on first run)...", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        LOGGER.info("Embedding model loaded successfully.")
        return self._model

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Encode a batch of texts into dense embedding vectors.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to process at once.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        if not texts:
            return []

        LOGGER.info("Embedding %d texts with model '%s'...", len(texts), self.model_name)
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return [emb.tolist() for emb in embeddings]

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        result = self.embed_texts([text])
        return result[0] if result else []


# --- Legacy hash-based embedding (kept for comparison) ---


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def embed_text_hash(text: str, dimensions: int = 256) -> list[float]:
    """Create a deterministic hashed embedding vector for text.

    This is the legacy hash-based approach kept for comparison/fallback.
    It has NO semantic understanding — use TextEmbedder for real embeddings.
    """
    vector = [0.0] * dimensions
    tokens = tokenize(text)
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % dimensions
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[idx] += sign

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


# Backward compatibility alias
def embed_text(text: str, dimensions: int = 256) -> list[float]:
    """Legacy hash-based embedding. Prefer TextEmbedder for real semantic embeddings."""
    return embed_text_hash(text, dimensions)
