from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np


class TripleEmbedder:
    """Encode realized triples for cosine retrieval.

    Paper-faithful runs should use sentence-transformers/all-MiniLM-L6-v2 with
    require_sentence_transformer=True. The deterministic fallback exists only
    for offline smoke tests and unit tests.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        require_sentence_transformer: bool = False,
        fallback_dim: int = 384,
        batch_size: int = 256,
    ) -> None:
        self.model_name = model_name
        self.fallback_dim = fallback_dim
        self.batch_size = batch_size
        self.using_fallback = False
        self.model = None

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            if require_sentence_transformer:
                raise RuntimeError(
                    "Paper-faithful runs require sentence-transformers and "
                    f"the embedding model {model_name}."
                ) from exc
            self.using_fallback = True

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.model is not None:
            vectors = self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vectors, dtype=np.float32)
        return np.asarray([self._hash_embed(text) for text in texts], dtype=np.float32)

    def _hash_embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.fallback_dim, dtype=np.float32)
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.fallback_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm
