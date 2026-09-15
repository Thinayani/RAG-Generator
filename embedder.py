"""Embed text chunks into vectors using a local sentence-transformer model.

Loaded once as a module-level singleton -- avoids reloading the model
weights on every call, which matters once this sits behind a UI.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Return an (N, D) array of L2-normalized embeddings for N input texts.

    Normalizing at encode time means cosine similarity reduces to a plain
    dot product at search time.
    """
    if not texts:
        return np.zeros((0, _EMBEDDING_DIM), dtype=np.float32)

    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)
