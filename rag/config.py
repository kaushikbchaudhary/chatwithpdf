"""Application configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Configuration values sourced from environment variables."""

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()

    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model_name: str = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    huggingface_token: Optional[str] = os.getenv("HUGGINGFACE_API_TOKEN")
    huggingface_embedding_model: str = os.getenv(
        "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    huggingface_chat_model: Optional[str] = os.getenv("HUGGINGFACE_CHAT_MODEL")

    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")

    retriever_k: int = int(os.getenv("RETRIEVER_K", "4"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    def ensure_valid(self) -> None:
        """Raise helpful errors when critical configuration is missing."""

        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai."
            )
        if self.embedding_provider not in {"openai", "huggingface"}:
            raise RuntimeError(
                "EMBEDDING_PROVIDER must be either 'openai' or 'huggingface'."
            )
        if self.chunk_overlap >= self.chunk_size:
            raise RuntimeError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        if self.llm_provider not in {"openai", "ollama", "huggingface"}:
            raise RuntimeError(
                "LLM_PROVIDER must be one of 'openai', 'ollama', or 'huggingface'."
            )
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai."
            )
        if self.llm_provider == "huggingface" and not self.huggingface_chat_model:
            raise RuntimeError(
                "HUGGINGFACE_CHAT_MODEL must be set when LLM_PROVIDER=huggingface."
            )
        if self.llm_provider == "huggingface" and not self.huggingface_token:
            raise RuntimeError(
                "HUGGINGFACE_API_TOKEN must be set when LLM_PROVIDER=huggingface."
            )

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Factory method that ensures values are validated."""

        config = cls()
        config.ensure_valid()
        return config
