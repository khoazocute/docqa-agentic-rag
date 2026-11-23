"""Offline test doubles: deterministic embeddings and a rule-based LLM (no API key needed)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import Field

from docqa.config import NOT_FOUND_MESSAGE, Settings
from docqa.engine import DocQAEngine

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOP = {
    "và",
    "của",
    "là",
    "có",
    "các",
    "được",
    "cho",
    "trong",
    "gì",
    "nào",
    "bao",
    "nhiêu",
    "không",
}


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


class FakeEmbedding(BaseEmbedding):
    """Hashed bag-of-words vectors: texts sharing words get high cosine similarity."""

    dim: int = 256
    embedded_texts: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    fail: bool = False

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokens(text):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def _get_text_embedding(self, text: str) -> list[float]:
        if self.fail:
            raise RuntimeError("embedding service unavailable")
        self.embedded_texts.append(text)
        return self._vector(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        self.queries.append(query)
        return self._vector(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


class FakeLLM(CustomLLM):
    """Answers with the context block that shares the most words with the question."""

    prompts: list[str] = Field(default_factory=list)
    forced_answer: str | None = None

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="fake-llm")

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        self.prompts.append(prompt)
        if "KIỂM TRA câu trả lời" in prompt:
            return CompletionResponse(text=self._verify(prompt))
        if self.forced_answer is not None:
            return CompletionResponse(text=self.forced_answer)

        context = prompt.split("Đoạn trích:", 1)[1].rsplit("Câu hỏi:", 1)[0]
        question = prompt.rsplit("Câu hỏi:", 1)[1].split("Trả lời:", 1)[0]
        q_tokens = set(tokens(question))
        best, best_overlap = None, 0
        for block in re.split(r"\n\n(?=\[\d+\])", context.strip()):
            match = re.match(r"\[(\d+)\][^\n]*\n(.*)", block, re.S)
            if not match:
                continue
            overlap = len(q_tokens & set(tokens(match.group(2))))
            if overlap > best_overlap:
                best, best_overlap = match, overlap
        if best is None or best_overlap < 2:
            return CompletionResponse(text=NOT_FOUND_MESSAGE)
        lines = [line for line in best.group(2).splitlines() if line.strip()]
        best_line = max(lines, key=lambda line: len(q_tokens & set(tokens(line))))
        return CompletionResponse(text=f"{best_line.strip()} [{best.group(1)}]")

    @staticmethod
    def _verify(prompt: str) -> str:
        """Judge stand-in: a claim is unsupported if it states a number absent from the context."""
        context = prompt.split("Đoạn trích:", 1)[1].split("Câu trả lời nháp:", 1)[0]
        # Only passage text counts as evidence: random passage ids ("[#a16f3c2e]") and source
        # lines must not, or "16" would be "found" in about 3% of runs (a flaky test).
        context = re.sub(r"\[#[0-9a-fA-F]+\]\s*\(nguồn:[^)]*\)", "", context)
        answer = prompt.split("Câu trả lời nháp:", 1)[1].split("Chỉ trả về JSON", 1)[0]
        answer = re.sub(r"\[#[0-9a-fA-F]+\]", "", answer)
        missing = [n for n in re.findall(r"\d+(?:[.,]\d+)*", answer) if n not in context]
        claims = [f"Số liệu {n} không có trong tài liệu" for n in missing]
        return json.dumps(
            {"supported": not claims, "unsupported_claims": claims}, ensure_ascii=False
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        yield self.complete(prompt)


DEMO_DOCS = {
    "quy_dinh.txt": (
        "Thời gian thực tập\n"
        "Sinh viên thực tập từ thứ Hai đến thứ Sáu, bắt đầu lúc 8:30 và kết thúc lúc 17:30.\n\n"
        "Báo cáo công việc\n"
        "Mỗi sinh viên cập nhật tiến độ công việc hằng ngày vào cuối buổi."
    ),
    "bao_mat.md": (
        "# Chính sách mật khẩu\n\n"
        "Mật khẩu phải có ít nhất 12 ký tự và được đổi định kỳ 90 ngày một lần.\n\n"
        "# Sự cố bảo mật\n\n"
        "Nhân viên báo cáo sự cố bảo mật cho bộ phận an ninh trong vòng 24 giờ."
    ),
}


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "data"
    directory.mkdir()
    for name, text in DEMO_DOCS.items():
        (directory / name).write_text(text, encoding="utf-8")
    return directory


@pytest.fixture
def settings(tmp_path: Path, data_dir: Path) -> Settings:
    return Settings(
        data_dir=data_dir,
        storage_dir=tmp_path / "storage",
        chunk_size=128,
        chunk_overlap=0,
        candidate_top_k=4,
        top_k=3,
        min_similarity=0.2,
    )


@pytest.fixture
def fake_embed() -> FakeEmbedding:
    return FakeEmbedding()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def engine(settings: Settings, fake_llm: FakeLLM, fake_embed: FakeEmbedding) -> DocQAEngine:
    return DocQAEngine(settings, llm=fake_llm, embed_model=fake_embed)
