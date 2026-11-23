"""Public-benchmark evaluation in the BEIR format (corpus.jsonl, queries.jsonl, qrels/*.jsonl).

Benchmarks such as the Zalo AI 2021 Legal Text Retrieval set ship passages that are already
segmented (one legal article each), so, following BEIR practice, every passage becomes one node
and chunking is bypassed: these numbers measure retrieval (BM25, dense, hybrid), not chunking.

The full corpus is used for BM25 (no API cost). Dense and hybrid retrieval need one embedding per
passage, so they run on a reproducible subset: every relevant passage of the sampled queries plus
random distractors drawn with a fixed seed. Subset scores are comparable between our own
configurations, not with published full-corpus results, and reports say which one was used.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core.schema import TextNode

from docqa.ir_metrics import evaluate_rankings


@dataclass
class BeirDataset:
    name: str
    corpus: dict[str, dict] = field(default_factory=dict)  # id -> {"title", "text"}
    queries: dict[str, str] = field(default_factory=dict)  # id -> question
    qrels: dict[str, set[str]] = field(default_factory=dict)  # query id -> relevant ids

    @property
    def description(self) -> str:
        return f"{self.name}: {len(self.corpus)} passages, {len(self.qrels)} labelled queries"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_beir(directory: Path, split: str = "test", name: str | None = None) -> BeirDataset:
    dataset = BeirDataset(name=name or directory.name)
    for row in _read_jsonl(directory / "corpus.jsonl"):
        dataset.corpus[str(row["_id"])] = {"title": row.get("title", ""), "text": row["text"]}
    for row in _read_jsonl(directory / "qrels" / f"{split}.jsonl"):
        if float(row.get("score", 1)) > 0:
            dataset.qrels.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))
    wanted = set(dataset.qrels)
    for row in _read_jsonl(directory / "queries.jsonl"):
        if str(row["_id"]) in wanted:
            dataset.queries[str(row["_id"])] = row["text"]
    # Drop labels that point outside the corpus or to queries without text.
    dataset.qrels = {
        q: {d for d in docs if d in dataset.corpus}
        for q, docs in dataset.qrels.items()
        if q in dataset.queries
    }
    dataset.qrels = {q: docs for q, docs in dataset.qrels.items() if docs}
    return dataset


def subset(
    dataset: BeirDataset, n_queries: int | None, n_distractors: int, seed: int = 42
) -> BeirDataset:
    """Sampled queries, all their relevant passages, plus ``n_distractors`` random passages."""
    rng = random.Random(seed)
    query_ids = sorted(dataset.qrels)
    if n_queries is not None and n_queries < len(query_ids):
        query_ids = sorted(rng.sample(query_ids, n_queries))
    relevant = {d for q in query_ids for d in dataset.qrels[q]}
    others = sorted(set(dataset.corpus) - relevant)
    distractors = set(rng.sample(others, min(n_distractors, len(others))))
    keep = relevant | distractors
    return BeirDataset(
        name=f"{dataset.name}[q={len(query_ids)},docs={len(keep)},seed={seed}]",
        corpus={d: dataset.corpus[d] for d in sorted(keep)},
        queries={q: dataset.queries[q] for q in query_ids},
        qrels={q: dataset.qrels[q] for q in query_ids},
    )


def to_nodes(dataset: BeirDataset) -> list[TextNode]:
    """One node per passage; the title (e.g. "Điều 5. ...") is part of the indexed text."""
    nodes = []
    for doc_id, doc in dataset.corpus.items():
        text = f"{doc['title']}\n{doc['text']}" if doc.get("title") else doc["text"]
        nodes.append(
            TextNode(
                id_=doc_id,
                text=text,
                metadata={"source": doc_id},
                # The id is bookkeeping: keep it out of the embedded and BM25-indexed text.
                excluded_embed_metadata_keys=["source"],
                excluded_llm_metadata_keys=["source"],
            )
        )
    return nodes


def run_bm25(dataset: BeirDataset, k: int = 100) -> dict:
    """BM25 over the full corpus with the project's Vietnamese tokenisation settings."""
    from docqa.retrieval import build_bm25

    retriever = build_bm25(to_nodes(dataset), top_k=k)
    rankings = {
        q: [hit.node.node_id for hit in retriever.retrieve(text)]
        for q, text in dataset.queries.items()
    }
    return {
        "dataset": dataset.description,
        "mode": "bm25",
        "metrics": evaluate_rankings(rankings, dataset.qrels),
    }


class EmbeddingCache:
    """Append-only on-disk cache of passage embeddings, keyed by model and text.

    A dense run over a few thousand passages exceeds a free tier's daily embedding quota; with
    the cache, a run interrupted by the quota resumes the next day without re-embedding.
    """

    def __init__(self, path: Path, model_id: str) -> None:
        self.path = path
        self.model_id = model_id
        self._vectors: dict[str, list[float]] = {}
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        self._vectors[row["k"]] = row["v"]

    def key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_id}\n{text}".encode()).hexdigest()

    def embed(
        self, texts: list[str], embed_fn: Callable[[list[str]], list[list[float]]], batch: int = 32
    ) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if self.key(t) not in self._vectors))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(missing), batch):
            chunk = missing[start : start + batch]
            vectors = embed_fn(chunk)
            with self.path.open("a", encoding="utf-8") as handle:  # saved batch by batch
                for text, vector in zip(chunk, vectors, strict=True):
                    key = self.key(text)
                    self._vectors[key] = [round(float(x), 6) for x in vector]
                    handle.write(json.dumps({"k": key, "v": self._vectors[key]}) + "\n")
        return [self._vectors[self.key(t)] for t in texts]


def run_retrieval(
    dataset: BeirDataset,
    modes: list[str],
    *,
    embed_texts: Callable[[list[str]], list[list[float]]] | None = None,
    embed_query: Callable[[str], list[float]] | None = None,
    embed_model=None,
    cache: EmbeddingCache | None = None,
    k: int = 100,
    rrf_k: int = 60,
) -> list[dict]:
    """Evaluate the project's HybridRetriever in each mode on a BEIR dataset."""
    from llama_index.core import VectorStoreIndex
    from llama_index.core.schema import MetadataMode

    from docqa.retrieval import HybridRetriever, build_bm25

    nodes = to_nodes(dataset)
    vector_retriever = None
    if any(m in ("vector", "hybrid") for m in modes):
        if embed_texts is None or embed_query is None or embed_model is None:
            raise ValueError("vector and hybrid modes need an embedding model")
        texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes]
        vectors = cache.embed(texts, embed_texts) if cache else embed_texts(texts)
        for node, vector in zip(nodes, vectors, strict=True):
            node.embedding = vector
        index = VectorStoreIndex(nodes=nodes, embed_model=embed_model, show_progress=False)
        vector_retriever = index.as_retriever(similarity_top_k=k)
    retriever = HybridRetriever(
        vector_retriever,
        build_bm25(nodes, k),
        top_k=k,
        rrf_k=rrf_k,
        embed_query=embed_query,
    )
    results = []
    for mode in modes:
        rankings = {
            q: [c.node.node_id for c in retriever.search(text, mode=mode, top_k=k).chunks]
            for q, text in dataset.queries.items()
        }
        results.append(
            {
                "dataset": dataset.description,
                "mode": mode,
                "metrics": evaluate_rankings(rankings, dataset.qrels),
            }
        )
    return results


def format_report(results: list[dict]) -> str:
    metric_names = [m for m in results[0]["metrics"] if m != "n_queries"]
    lines = [
        "| Dataset | Mode | " + " | ".join(metric_names) + " |",
        "|---" * (len(metric_names) + 2) + "|",
    ]
    for result in results:
        cells = [
            f"{v['mean']:.3f} [{v['ci_low']:.3f}–{v['ci_high']:.3f}]"
            for m, v in result["metrics"].items()
            if m != "n_queries"
        ]
        lines.append(f"| {result['dataset']} | {result['mode']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
