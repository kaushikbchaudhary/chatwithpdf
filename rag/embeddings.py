"""Embedding model factory."""

from __future__ import annotations

from typing import Any

from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from .config import AppConfig


def build_embeddings(config: AppConfig) -> Embeddings:
    """Return an embedding model based on configuration."""

    if config.embedding_provider == "openai":
        return OpenAIEmbeddings(
            api_key=config.openai_api_key,
            model=config.openai_embedding_model,
        )

    if config.embedding_provider == "huggingface":
        model_kwargs: dict[str, Any] = {}
        if config.huggingface_token:
            model_kwargs["token"] = config.huggingface_token
        return HuggingFaceEmbeddings(
            model_name=config.huggingface_embedding_model,
            model_kwargs=model_kwargs,
        )

    raise RuntimeError(f"Unsupported embedding provider: {config.embedding_provider}")
