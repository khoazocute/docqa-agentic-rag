"""Token-level evaluation of chunking strategies (Chroma's method) on human-labelled spans.

Hit@k says whether *a* chunk containing the answer was retrieved; it does not penalise the
irrelevant text that comes with it. Following Chroma's "Evaluating Chunking Strategies for
Retrieval" (2024), every question has a ground-truth excerpt (character span in a document), and
chunks are compared with it token by token:

- recall    = relevant tokens retrieved / relevant tokens
- precision = relevant tokens retrieved / tokens retrieved
- IoU       = relevant tokens retrieved / (relevant tokens ∪ tokens retrieved)
- precision Ω = precision when exactly the chunks that overlap the excerpt are retrieved: the
  best precision a chunking allows, independent of the retriever (its recall is 1 by definition).

Ground truth comes from UIT-ViQuAD 2.0 (VLSP 2021): human-annotated answer spans. Each Wikipedia
article is rebuilt from its paragraphs, and the excerpt is the sentence(s) containing the answer.
Unanswerable questions have no excerpt and are skipped here. IoU rewards small chunks, so it must
be read together with recall and with answer quality, not optimised alone.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core.schema import Document, TextNode
from llama_index.core.utils import get_tokenizer

Span = tuple[int, int]  # [start, end) character offsets in one document

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?…])\s+")
PARAGRAPH_SEPARATOR = "\n\n"


@dataclass
class ChunkBench:
    name: str
    documents: dict[str, str] = field(default_factory=dict)  # doc id -> full text
    questions: dict[str, str] = field(default_factory=dict)  # question id -> text
    excerpts: dict[str, tuple[str, list[Span]]] = field(default_factory=dict)  # qid -> (doc, spans)


# ----------------------------------------------------------------------------- loading
def sentence_spans(text: str) -> list[Span]:
    spans, start = [], 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def excerpt_for_answer(paragraph: str, answer: str, answer_start: int) -> Span | None:
    """The sentence(s) of ``paragraph`` that contain the answer, or None if it cannot be found."""
    if paragraph[answer_start : answer_start + len(answer)] != answer:
        answer_start = paragraph.find(answer)  # some mirrors have shifted offsets
        if answer_start < 0:
            return None
    end = answer_start + len(answer)
    covering = [s for s in sentence_spans(paragraph) if s[0] < end and s[1] > answer_start]
    return (covering[0][0], covering[-1][1]) if covering else None


def load_viquad(parquet: Path, name: str = "UIT-ViQuAD 2.0") -> ChunkBench:
    """Rebuild articles from paragraphs (in order of first appearance) and map answer spans."""
    import pyarrow.parquet as pq

    rows = pq.read_table(str(parquet)).to_pylist()
    paragraphs: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["context"] not in paragraphs[row["title"]]:
            paragraphs[row["title"]].append(row["context"])

    bench = ChunkBench(name=name)
    offsets: dict[tuple[str, str], int] = {}
    for title, paras in paragraphs.items():
        position = 0
        for para in paras:
            offsets[(title, para)] = position
            position += len(para) + len(PARAGRAPH_SEPARATOR)
        bench.documents[title] = PARAGRAPH_SEPARATOR.join(paras)

    for row in rows:
        answers = row.get("answers") or {}
        if row.get("is_impossible") or not answers.get("text"):
            continue
        span = excerpt_for_answer(row["context"], answers["text"][0], answers["answer_start"][0])
        if span is None:
            continue
        base = offsets[(row["title"], row["context"])]
        bench.questions[row["id"]] = row["question"]
        bench.excerpts[row["id"]] = (row["title"], [(base + span[0], base + span[1])])
    return bench


# ----------------------------------------------------------------------------- spans
def merge(spans: list[Span]) -> list[Span]:
    merged: list[Span] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def intersect(a: list[Span], b: list[Span]) -> list[Span]:
    out = []
    for s1, e1 in merge(a):
        for s2, e2 in merge(b):
            start, end = max(s1, s2), min(e1, e2)
            if start < end:
                out.append((start, end))
    return merge(out)


class TokenCounter:
    def __init__(self) -> None:
        self._tokenize = get_tokenizer()

    def __call__(self, text: str, spans: list[Span]) -> int:
        return sum(len(self._tokenize(text[s:e])) for s, e in merge(spans))


# ----------------------------------------------------------------------------- chunking
@dataclass
class Chunk:
    doc_id: str
    span: Span
    text: str


def chunk_with(splitter, bench: ChunkBench) -> list[Chunk]:
    """Chunk every document and recover each chunk's character span in the original text."""
    chunks: list[Chunk] = []
    for doc_id, text in bench.documents.items():
        nodes = splitter.get_nodes_from_documents([Document(text=text, id_=doc_id)])
        cursor = 0
        for node in nodes:
            body = node.get_content()
            start = node.start_char_idx
            if start is None or text[start : start + len(body)] != body:
                found = text.find(body, max(0, cursor - len(body)))
                start = found if found >= 0 else text.find(body)
            if start is None or start < 0:
                continue  # text rewritten by the splitter: cannot be scored at token level
            chunks.append(Chunk(doc_id, (start, start + len(body)), body))
            cursor = start + len(body)
    return chunks


def paragraph_chunks(bench: ChunkBench) -> list[Chunk]:
    """Natural-boundary baseline: one chunk per paragraph."""
    chunks = []
    for doc_id, text in bench.documents.items():
        start = 0
        for para in text.split(PARAGRAPH_SEPARATOR):
            chunks.append(Chunk(doc_id, (start, start + len(para)), para))
            start += len(para) + len(PARAGRAPH_SEPARATOR)
    return chunks


# ----------------------------------------------------------------------------- metrics
def evaluate_chunks(
    bench: ChunkBench,
    chunks: list[Chunk],
    retrieve: Callable[[str, int], list[int]] | None = None,
    k: int = 5,
    count: TokenCounter | None = None,
) -> dict[str, float]:
    """Precision Ω (no retriever) and, with ``retrieve``, recall/precision/IoU of the top-k."""
    count = count or TokenCounter()
    by_doc: dict[str, list[int]] = defaultdict(list)
    for i, chunk in enumerate(chunks):
        by_doc[chunk.doc_id].append(i)

    omega, recall, precision, iou = [], [], [], []
    for qid, (doc_id, excerpt) in bench.excerpts.items():
        text = bench.documents[doc_id]
        relevant = count(text, excerpt)
        if not relevant:
            continue
        touching = [chunks[i].span for i in by_doc[doc_id] if intersect([chunks[i].span], excerpt)]
        omega.append(count(text, intersect(touching, excerpt)) / max(count(text, touching), 1))
        if retrieve is None:
            continue
        retrieved = [chunks[i] for i in retrieve(bench.questions[qid], k)]
        same_doc = [c.span for c in retrieved if c.doc_id == doc_id]
        hit = count(text, intersect(same_doc, excerpt))
        total = sum(count(bench.documents[c.doc_id], [c.span]) for c in retrieved)
        recall.append(hit / relevant)
        precision.append(hit / max(total, 1))
        iou.append(hit / max(total + relevant - hit, 1))

    tokens = [count(bench.documents[c.doc_id], [c.span]) for c in chunks]
    report = {
        "chunks": len(chunks),
        "mean_chunk_tokens": round(sum(tokens) / max(len(tokens), 1), 1),
        "precision_omega": round(sum(omega) / max(len(omega), 1), 4),
        "questions": len(omega),
    }
    if retrieve is not None:
        report.update(
            {
                f"recall@{k}": round(sum(recall) / len(recall), 4),
                f"precision@{k}": round(sum(precision) / len(precision), 4),
                f"iou@{k}": round(sum(iou) / len(iou), 4),
            }
        )
    return report


def bm25_retriever(chunks: list[Chunk]) -> Callable[[str, int], list[int]]:
    """Rank chunks with the project's BM25 (no API cost), returning chunk indexes."""
    from docqa.retrieval import build_bm25

    nodes = [TextNode(id_=str(i), text=c.text) for i, c in enumerate(chunks)]
    cache: dict[int, object] = {}

    def retrieve(question: str, k: int) -> list[int]:
        if k not in cache:
            cache[k] = build_bm25(nodes, k)
        return [int(hit.node.node_id) for hit in cache[k].retrieve(question)]

    return retrieve
