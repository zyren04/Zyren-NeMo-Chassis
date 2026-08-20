"""
FAISS-Based Local Vector Store Implementation

Uses FAISS.load_local() with langchain_community.vectorstores.FAISS for
code/documentation retrieval. Lightweight, pure-Python, no server required —
perfect for budget & resource-constrained environments (e.g., 16GB RAM, 
entry-level GPUs like GTX 1650, with no Milvus/Redis heavy-server footprint).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import BaseModel, Field


class FAISSVectorStoreConfig(BaseModel):
    """Configuration for FAISS vector store."""

    model_config = {"arbitrary_types_allowed": True}

    path: str = Field(..., description="Path to the FAISS index directory")
    embedder: Embeddings = Field(..., description="Embedding model for vectorization")
    allow_dangerous_deserialization: bool = Field(
        default=True, description="Allow deserialization of FAISS index (required for local files)"
    )


class FAISSVectorStore:
    """
    FAISS-based local vector store for document retrieval.

    Features:
    - Load existing FAISS index from disk
    - Create new index from documents
    - Retrieve documents via similarity search
    - No external server required (local-first)
    """

    def __init__(self, config: FAISSVectorStoreConfig) -> None:
        self.config = config
        self._db: FAISS = FAISS.load_local(
            config.path,
            config.embedder,
            allow_dangerous_deserialization=config.allow_dangerous_deserialization,
        )

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        """
        Get a retriever for this vector store.

        Args:
            **kwargs: Arguments passed to FAISS.as_retriever()
                     (e.g., search_type="similarity", search_kwargs={"k": 4})

        Returns:
            VectorStoreRetriever for querying the index.
        """
        return self._db.as_retriever(**kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """
        Perform similarity search directly.

        Args:
            query: Search query string.
            k: Number of results to return.
            **kwargs: Additional arguments passed to FAISS.similarity_search().

        Returns:
            List of matching Documents.
        """
        return self._db.similarity_search(query, k=k, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.

        Args:
            query: Search query string.
            k: Number of results to return.
            **kwargs: Additional arguments.

        Returns:
            List of (Document, score) tuples.
        """
        return self._db.similarity_search_with_score(query, k=k, **kwargs)

    @staticmethod
    def create_from_documents(
        documents: list[Document],
        embedder: Embeddings,
        path: str,
        allow_dangerous_deserialization: bool = True,
    ) -> FAISSVectorStore:
        """
        Create a new FAISS index from documents and save to disk.

        Args:
            documents: List of Documents to index.
            embedder: Embedding model for vectorization.
            path: Directory path to save the index.
            allow_dangerous_deserialization: Allow deserialization when loading.

        Returns:
            FAISSVectorStore instance with the new index.
        """
        # Ensure directory exists
        Path(path).mkdir(parents=True, exist_ok=True)

        db = FAISS.from_documents(documents, embedder)
        db.save_local(path)

        config = FAISSVectorStoreConfig(
            path=path,
            embedder=embedder,
            allow_dangerous_deserialization=allow_dangerous_deserialization,
        )
        return FAISSVectorStore(config)

    @staticmethod
    def create_from_texts(
        texts: list[str],
        embedder: Embeddings,
        path: str,
        metadatas: list[dict[str, Any]] | None = None,
        allow_dangerous_deserialization: bool = True,
    ) -> FAISSVectorStore:
        """
        Create a new FAISS index from raw texts and save to disk.

        Args:
            texts: List of text strings to index.
            embedder: Embedding model for vectorization.
            path: Directory path to save the index.
            metadatas: Optional metadata for each text.
            allow_dangerous_deserialization: Allow deserialization when loading.

        Returns:
            FAISSVectorStore instance with the new index.
        """
        documents = [
            Document(page_content=text, metadata=metadatas[i] if metadatas else {})
            for i, text in enumerate(texts)
        ]
        return FAISSVectorStore.create_from_documents(
            documents, embedder, path, allow_dangerous_deserialization
        )

    def add_documents(self, documents: list[Document]) -> None:
        """
        Add documents to the existing index and save.

        Args:
            documents: List of Documents to add.
        """
        self._db.add_documents(documents)
        self._db.save_local(self.config.path)

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Add texts to the existing index and save.

        Args:
            texts: List of text strings to add.
            metadatas: Optional metadata for each text.
        """
        documents = [
            Document(page_content=text, metadata=metadatas[i] if metadatas else {})
            for i, text in enumerate(texts)
        ]
        self.add_documents(documents)

    def __repr__(self) -> str:
        return f"FAISSVectorStore(path={self.config.path})"


__all__ = [
    "FAISSVectorStore",
    "FAISSVectorStoreConfig",
]
