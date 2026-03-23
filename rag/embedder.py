"""
rag/embedder.py — OpenAI embedding wrapper.

Model : text-embedding-3-small  (1536-dim)
Cost  : ~$0.00002 / 1K tokens — negligible for a small knowledge base
Docs are embedded once at startup; queries are embedded per request.

Vectors are L2-normalised so cosine similarity == dot product (fast NumPy).
"""
from __future__ import annotations

from typing import List

import numpy as np
from openai import OpenAI

from config import OPENAI_API_KEY

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536


class Embedder:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model  = EMBEDDING_MODEL
        self.dim    = EMBEDDING_DIM

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.

        Returns
        -------
        np.ndarray  shape (N, dim), dtype float32, L2-normalised
        """
        # OpenAI recommends replacing newlines with spaces
        cleaned = [t.replace("\n", " ") for t in texts]

        response = self.client.embeddings.create(
            model=self.model,
            input=cleaned,
        )

        vecs  = np.array([d.embedding for d in response.data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-9)   # L2 normalise

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (dim,)."""
        return self.embed([text])[0]
