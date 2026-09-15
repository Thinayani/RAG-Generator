"""In-memory vector store: one collection at a time, cosine-similarity search.

No persistence, no external vector DB -- a numpy matrix and a parallel
list of chunk metadata is sufficient at this scale. clear() is what makes
swapping document sets a data operation instead of a restart.
"""

import numpy as np


class VectorStore:
    def __init__(self):
        self.vectors: np.ndarray | None = None  # (N, D), assumed L2-normalized
        self.chunks: list[dict] = []  # metadata: {"text": ..., "source": ...}

    def clear(self) -> None:
        """Reset the store -- used when a new document set is ingested."""
        self.vectors = None
        self.chunks = []

    def add(self, vectors: np.ndarray, chunks: list[dict]) -> None:
        """Append embedded chunks to the store."""
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must be the same length")
        if len(vectors) == 0:
            return
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[dict]:
        """Return the top_k most similar chunks, each with a similarity score.

        Since vectors are L2-normalized, cosine similarity is a dot product.
        """
        if self.is_empty():
            return []

        scores = self.vectors @ query_vector
        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            entry = dict(self.chunks[idx])
            entry["score"] = float(scores[idx])
            results.append(entry)
        return results

    def is_empty(self) -> bool:
        return self.vectors is None or len(self.chunks) == 0
