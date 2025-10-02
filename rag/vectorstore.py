"""Vector store helper utilities."""

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS

from .config import AppConfig


def build_vectorstore(
    documents: Sequence[Document],
    embeddings: Embeddings,
) -> FAISS:
    """Create an in-memory FAISS index from documents."""

    if not documents:
        raise ValueError("No documents provided to build the vector store.")

    return FAISS.from_documents(documents, embeddings)


def as_retriever(vector_store: FAISS, config: AppConfig):
    """Return a similarity search retriever with sensible defaults."""

    return vector_store.as_retriever(search_kwargs={"k": config.retriever_k})
