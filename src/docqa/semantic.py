"""Semantic chunking for documents without usable structure (transcripts, notes, plain text).

Where a document has headings, structure-aware chunking (docqa.structure) already knows where
topics change. Where it has none, cutting every N tokens splits ideas at arbitrary points, so the
split points are found from the text itself, following Kamradt's percentile method:

1. split into sentences (PDF hard line breaks are re-joined first; list items stay separate);
2. embed a window around each sentence (the sentence and its neighbours, to smooth noise);
3. measure the cosine distance between consecutive windows;
4. cut where the distance is above the given percentile (the sharpest topic shifts);
5. re-split groups that exceed the chunk size with the sentence splitter, and merge tiny ones.

Cost: one embedding per sentence, on top of the chunk embeddings. Embeddings go through the
engine's batched, quota-aware embedding function, and very long documents fall back to sentence
splitting instead of spending thousands of calls on split points.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

import numpy as np
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser.interface import MetadataAwareTextSplitter
from llama_index.core.utils import get_tokenizer

logger = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], list[list[float]]]

_SENTENCE_END_RE = re.compile(r"(?<=[.!?…;])\s+(?=\S)")
_LIST_ITEM_RE = re.compile(r"^([-*•●▪–]|\d+[.)]|[a-zA-Z][.)])\s+")
_ENDS_SENTENCE = (".", "!", "?", "…", ":", ";")


def split_sentences(text: str) -> list[tuple[str, int]]:
    """Sentences with the index of the paragraph they belong to.

    Lines inside a paragraph are re-joined unless the next line starts a list item or the
    previous one ended a sentence, because PDF extraction breaks lines mid-sentence.
    """
    sentences: list[tuple[str, int]] = []
    for paragraph_index, paragraph in enumerate(re.split(r"\n\s*\n", text)):
        units: list[str] = []
        for line in (raw.strip() for raw in paragraph.splitlines()):
            if not line:
                continue
            if units and not _LIST_ITEM_RE.match(line) and not units[-1].endswith(_ENDS_SENTENCE):
                units[-1] = f"{units[-1]} {line}"
            else:
                units.append(line)
        for unit in units:
            sentences.extend(
                (part.strip(), paragraph_index)
                for part in _SENTENCE_END_RE.split(unit)
                if part.strip()
            )
    return sentences


def join_sentences(sentences: list[tuple[str, int]]) -> str:
    """Re-assemble sentences, keeping paragraph breaks."""
    out = ""
    previous: int | None = None
    for text, paragraph in sentences:
        if previous is None:
            out = text
        else:
            out += (" " if paragraph == previous else "\n\n") + text
        previous = paragraph
    return out


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def cosine_distances(vectors: list[list[float]]) -> list[float]:
    matrix = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1.0, norms)
    return [float(1.0 - matrix[i] @ matrix[i + 1]) for i in range(len(matrix) - 1)]


class SemanticSplitter(MetadataAwareTextSplitter):
    """Cut where consecutive sentence windows are semantically far apart (see module doc)."""

    chunk_size: int = Field(default=512, description="Maximum tokens per chunk.")
    breakpoint_percentile: float = Field(default=95.0, ge=0.0, le=100.0)
    buffer_size: int = Field(default=1, ge=0, description="Neighbour sentences per window.")
    min_chunk_tokens: int = Field(default=48, ge=0)
    max_sentences: int = Field(
        default=1500, description="Longer texts fall back to sentence splitting (embedding cost)."
    )

    _embed: EmbedFn = PrivateAttr()
    _fallback: SentenceSplitter = PrivateAttr()
    _tokenizer: Callable = PrivateAttr()
    _embedded: int = PrivateAttr(default=0)

    def __init__(self, embed: EmbedFn, **kwargs) -> None:
        super().__init__(**kwargs)
        self._embed = embed
        self._fallback = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=0)
        self._tokenizer = get_tokenizer()
        self._embedded = 0

    @classmethod
    def class_name(cls) -> str:
        return "DocQASemanticSplitter"

    @property
    def embedded_windows(self) -> int:
        """Embeddings spent on finding split points (for cost reporting)."""
        return self._embedded

    def _tokens(self, text: str) -> int:
        return len(self._tokenizer(text))

    def split_text(self, text: str) -> list[str]:
        return self.split_text_metadata_aware(text, "")

    def split_text_metadata_aware(self, text: str, metadata_str: str) -> list[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        budget = max(self.chunk_size - self._tokens(metadata_str) - 8, self.chunk_size // 2)
        if len(sentences) > self.max_sentences:
            logger.warning(
                "%d sentences: too many for semantic splitting, using sentence splitting",
                len(sentences),
            )
            return self._fallback.split_text_metadata_aware(text, metadata_str)

        pieces: list[str] = []
        for group in self._semantic_groups(sentences):
            group_text = join_sentences(group)
            if self._tokens(group_text) > budget:
                pieces.extend(self._fallback.split_text_metadata_aware(group_text, metadata_str))
            else:
                pieces.append(group_text)
        return self._merge_tiny(pieces, budget)

    def _semantic_groups(self, sentences: list[tuple[str, int]]) -> list[list[tuple[str, int]]]:
        if len(sentences) < 3:
            return [sentences]
        texts = [s for s, _ in sentences]
        b = self.buffer_size
        windows = [" ".join(texts[max(0, i - b) : i + b + 1]) for i in range(len(texts))]
        vectors = self._embed(windows)
        self._embedded += len(windows)
        distances = cosine_distances(vectors)
        threshold = percentile(distances, self.breakpoint_percentile)

        groups: list[list[tuple[str, int]]] = [[sentences[0]]]
        for i, distance in enumerate(distances):
            if distance > threshold:
                groups.append([])
            groups[-1].append(sentences[i + 1])
        return groups

    def _merge_tiny(self, pieces: list[str], budget: int) -> list[str]:
        """Join a tiny piece with its neighbour, never producing a chunk above ``budget``."""
        merged: list[str] = []
        for piece in pieces:
            if merged:
                previous, current = self._tokens(merged[-1]), self._tokens(piece)
                tiny = min(previous, current) < self.min_chunk_tokens
                if tiny and previous + current <= budget:
                    merged[-1] = f"{merged[-1]}\n\n{piece}"
                    continue
            merged.append(piece)
        return merged
