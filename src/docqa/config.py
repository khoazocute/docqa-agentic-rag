"""Runtime configuration loaded from environment variables (.env supported)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from docqa.images import CAPTION_VERSION

NOT_FOUND_MESSAGE = "Tôi không tìm thấy thông tin này trong tài liệu đã cung cấp."
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".txt", ".md", ".pdf", ".docx")

LLM_PROVIDERS = ("google", "ollama")
VECTOR_STORES = ("simple", "qdrant")
RETRIEVAL_MODES = ("vector", "bm25", "hybrid")
# sentence: fixed-size sentence splitting of each file/page (original behaviour).
# structure: split by headings/tables first (see docqa.structure), with fallback.
# semantic: cut where the meaning of consecutive sentences shifts (one embedding per sentence).
CHUNK_STRATEGIES = ("sentence", "semantic", "structure")
# How structure-aware chunking splits files that have no headings/tables.
STRUCTURE_FALLBACKS = ("semantic", "sentence")
# Prompt-injection defence: off | spotlight (delimit excerpts, guard output, gate tools)
# | full (spotlight + remove sentences that address the AI). See docqa.injection.
INJECTION_DEFENSES = ("off", "spotlight", "full")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    value = _env_str(name, default).lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return value


def _env_path(name: str, default: str) -> Path:
    path = Path(_env_str(name, default)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    # Models
    llm_provider: str = "google"
    llm_model: str = "gemini-3.5-flash-lite"
    llm_temperature: float = 0.1
    embed_provider: str = "google"
    embed_model: str = "gemini-embedding-2"
    agent_model: str = "gemini-3.5-flash-lite"
    agent_max_llm_calls: int = 15  # guardrail against runaway agent loops (per user turn)
    ollama_base_url: str = "http://localhost:11434"

    # Storage
    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage")
    vector_store: str = "simple"
    # Qdrant: server mode when qdrant_url is set (e.g. http://localhost:6333),
    # otherwise embedded mode stored under storage/qdrant (single process only).
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    # Chunking
    chunk_strategy: str = "sentence"
    structure_fallback: str = "semantic"
    semantic_breakpoint_percentile: float = 95.0
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_part_chars: int = 200
    # Describe images with the (multimodal) LLM at ingestion so their content is searchable.
    image_captions: bool = False
    max_images_per_file: int = 40

    # Retrieval
    retrieval_mode: str = "hybrid"
    candidate_top_k: int = 8
    top_k: int = 4
    rrf_k: int = 60
    min_similarity: float = 0.55

    # Security
    injection_defense: str = "off"

    # Behaviour
    auto_sync: bool = True
    max_upload_mb: int = 20
    api_key: str | None = None

    @classmethod
    def from_env(cls, **overrides: Any) -> Settings:
        load_dotenv()
        settings = cls(
            llm_provider=_env_choice("LLM_PROVIDER", "google", LLM_PROVIDERS),
            llm_model=_env_str("LLM_MODEL", "gemini-3.5-flash-lite"),
            llm_temperature=_env_float("LLM_TEMPERATURE", 0.1),
            embed_provider=_env_choice("EMBED_PROVIDER", "google", LLM_PROVIDERS),
            embed_model=_env_str("EMBED_MODEL", "gemini-embedding-2"),
            agent_model=_env_str("ADK_MODEL", "gemini-3.5-flash-lite"),
            agent_max_llm_calls=_env_int("AGENT_MAX_LLM_CALLS", 15, minimum=2),
            ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
            data_dir=_env_path("DATA_DIR", "data"),
            storage_dir=_env_path("STORAGE_DIR", "storage"),
            vector_store=_env_choice("VECTOR_STORE", "simple", VECTOR_STORES),
            qdrant_url=os.getenv("QDRANT_URL") or None,
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            chunk_strategy=_env_choice("CHUNK_STRATEGY", "sentence", CHUNK_STRATEGIES),
            structure_fallback=_env_choice("STRUCTURE_FALLBACK", "semantic", STRUCTURE_FALLBACKS),
            semantic_breakpoint_percentile=_env_float("SEMANTIC_BREAKPOINT_PERCENTILE", 95.0),
            chunk_size=_env_int("CHUNK_SIZE", 512, minimum=64),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 64, minimum=0),
            min_part_chars=_env_int("MIN_PART_CHARS", 200, minimum=0),
            image_captions=_env_bool("IMAGE_CAPTIONS", False),
            max_images_per_file=_env_int("MAX_IMAGES_PER_FILE", 40, minimum=0),
            retrieval_mode=_env_choice("RETRIEVAL_MODE", "hybrid", RETRIEVAL_MODES),
            candidate_top_k=_env_int("CANDIDATE_TOP_K", 8, minimum=1),
            top_k=_env_int("TOP_K", 4, minimum=1),
            rrf_k=_env_int("RRF_K", 60, minimum=1),
            min_similarity=_env_float("MIN_SIMILARITY", 0.55),
            injection_defense=_env_choice("INJECTION_DEFENSE", "off", INJECTION_DEFENSES),
            auto_sync=_env_bool("AUTO_SYNC", True),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 20, minimum=1),
            api_key=os.getenv("DOCQA_API_KEY") or None,
        )
        return settings.with_overrides(**overrides) if overrides else settings

    def with_overrides(self, **overrides: Any) -> Settings:
        return replace(self, **overrides)

    def index_fingerprint(self) -> dict[str, Any]:
        """Settings that invalidate a persisted index when they change."""
        return {
            "embed_provider": self.embed_provider,
            "embed_model": self.embed_model,
            "chunk_strategy": self.chunk_strategy,
            "structure_fallback": self.structure_fallback,
            "semantic_breakpoint_percentile": self.semantic_breakpoint_percentile,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_part_chars": self.min_part_chars,
            # Captions change document text; the model and prompt version are part of it.
            "image_captions": (
                f"{self.llm_provider}/{self.llm_model}/{CAPTION_VERSION}"
                if self.image_captions
                else False
            ),
            "vector_store": self.vector_store,
        }
