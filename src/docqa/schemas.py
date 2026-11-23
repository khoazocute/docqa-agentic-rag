"""Public data contracts shared by the engine, ADK tools, API and evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

QueryStatus = Literal["ok", "not_found", "error"]


class Source(BaseModel):
    id: int = Field(description="Citation number used in the answer, e.g. [1].")
    file: str
    page: str | None = None
    section: str | None = Field(
        default=None, description="Heading path of the chunk, e.g. '2. Methods > 2.1 Goals'."
    )
    score: float = Field(description="Final ranking score (RRF in hybrid mode).")
    vector_score: float | None = None
    bm25_score: float | None = None
    cited: bool = False
    snippet: str


class QueryResult(BaseModel):
    status: QueryStatus
    answer: str
    sources: list[Source] = Field(default_factory=list)
    reason: str | None = Field(default=None, description="Why the status is not 'ok'.")
    latency_ms: dict[str, float] = Field(default_factory=dict)


class SyncReport(BaseModel):
    mode: Literal["sync", "rebuild"]
    added: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list, description="Files that could not be parsed.")
    unchanged: int = 0
    documents: int = 0
    chunks: int = 0
    index_version: str | None = None
    duration_ms: float = 0.0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.deleted)


class DocumentInfo(BaseModel):
    source: str
    parts: int
    chunks: int


class IndexStats(BaseModel):
    index_version: str | None
    created_at: str | None
    documents: int
    chunks: int
    retrieval_mode: str
    vector_store: str
    embed_model: str
    llm_model: str


class Passage(BaseModel):
    """A retrieved excerpt handed to the agent as evidence. Cite it in answers as [#id]."""

    id: str = Field(description="Short stable id of the chunk (prefix of the node id).")
    file: str
    page: str | None = None
    section: str | None = Field(
        default=None, description="Heading path of the chunk, e.g. '2. Methods > 2.1 Goals'."
    )
    score: float | None = None
    text: str


class PassageSearch(BaseModel):
    status: Literal["ok", "not_found", "error"]
    passages: list[Passage] = Field(default_factory=list)
    reason: str | None = None


class AnswerCheck(BaseModel):
    """Result of verifying a draft answer against the passages it cites."""

    status: Literal["ok", "unsupported", "unknown"]
    cited_ids: list[str] = Field(default_factory=list)
    unknown_ids: list[str] = Field(default_factory=list, description="Cited ids that do not exist.")
    uncited_sentences: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)


class AgentStep(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str = ""


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    llm_calls: int = 0
