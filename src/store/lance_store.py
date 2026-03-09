"""LanceDB vector store implementation."""

from src.store.base import BaseVectorStore


class LanceStore(BaseVectorStore):
    def add_documents(self, documents: list) -> None:
        raise NotImplementedError

    def query(self, query: str, top_k: int = 5) -> list:
        raise NotImplementedError
