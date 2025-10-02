"""Conversational retrieval chain construction."""

from __future__ import annotations

from langchain.chains import ConversationalRetrievalChain

from .config import AppConfig
from .llm import build_chat_model
from .vectorstore import as_retriever


def build_conversational_chain(vector_store, config: AppConfig):
    """Create a conversational retrieval chain anchored to the vector store."""

    llm = build_chat_model(config)

    retriever = as_retriever(vector_store, config)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
