"""Factories for the LLM and embedding model (Google GenAI or local Ollama)."""

from __future__ import annotations

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM

from docqa.config import Settings


def build_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "google":
        from llama_index.llms.google_genai import GoogleGenAI

        return GoogleGenAI(model=settings.llm_model, temperature=settings.llm_temperature)

    if settings.llm_provider == "ollama":
        try:
            from llama_index.llms.ollama import Ollama
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install the 'ollama' extra: pip install -e .[ollama]") from exc

        return Ollama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            request_timeout=180.0,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def build_embed_model(settings: Settings) -> BaseEmbedding:
    if settings.embed_provider == "google":
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

        return GoogleGenAIEmbedding(model_name=settings.embed_model)

    if settings.embed_provider == "ollama":
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install the 'ollama' extra: pip install -e .[ollama]") from exc

        return OllamaEmbedding(model_name=settings.embed_model, base_url=settings.ollama_base_url)

    raise ValueError(f"Unsupported embedding provider: {settings.embed_provider}")
