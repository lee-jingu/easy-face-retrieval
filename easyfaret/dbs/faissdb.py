from __future__ import annotations
import faiss
import numpy as np


class FaissDB:
    def __init__(self, name: str | None = None):
        name = name if name is not None else "test"
        self._collection = faiss.IndexFlatL2(512)
        self._metadatas = []
        self._name = name

    def insert(self, embeddings: list[np.ndarray], metadatas: list[dict]) -> None:
        assert len(embeddings) == len(
            metadatas
        ), "embeddings, metadatas must have the same length"

        # for embedding, metadata in zip(embeddings, metadatas):
        self._collection.add(np.array(embeddings))
        self._metadatas = metadatas

    def search(self, embedding: np.ndarray, n_results: int = 3):
        embedding = np.array([embedding])
        _, idxs = self._collection.search(embedding, n_results)
        results = [self._metadatas[i] for i in idxs[0]]

        return results
