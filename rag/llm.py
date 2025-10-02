"""Chat model factory."""

from __future__ import annotations

from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models.huggingface import ChatHuggingFace

from .config import AppConfig


def build_chat_model(config: AppConfig) -> BaseChatModel:
    """Return a chat model configured via environment variables."""

    if config.llm_provider == "openai":
        return ChatOpenAI(
            api_key=config.openai_api_key,
            model=config.openai_model_name,
            temperature=0,
        )

    if config.llm_provider == "ollama":
        return ChatOllama(model=config.ollama_model)

    if config.llm_provider == "huggingface":
        return ChatHuggingFace(
            repo_id=config.huggingface_chat_model,
            huggingfacehub_api_token=config.huggingface_token,
        )

    raise RuntimeError(f"Unsupported LLM provider: {config.llm_provider}")
