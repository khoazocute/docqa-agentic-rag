"""Offline evaluation of retrieval quality and answer behaviour on a labelled QA set.

Metrics
- Retrieval (single-hop answerable questions): Hit@k and MRR@k. A retrieved chunk counts as
  relevant when it comes from the expected file and contains every ``evidence`` phrase.
- Answers, for the one-shot RAG pipeline (``/query``) and optionally for the agentic RAG agent:
  keyword accuracy (overall and per question kind), correct-rejection rate for unanswerable
  questions, false-rejection rate, citation rate, latency, and for the agent the cost in LLM calls
  and tool calls per question.
- Optional LLM-as-judge faithfulness (LlamaIndex ``FaithfulnessEvaluator``) for the pipeline.
- Similarity calibration: suggests a ``MIN_SIMILARITY`` threshold from score distributions.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from docqa.config import NOT_FOUND_MESSAGE
from docqa.engine import DocQAEngine
from docqa.prompts import REFUSAL_RE
from docqa.retrieval import RetrievedChunk

KINDS = ("single", "multi_hop")


@dataclass
class EvalCase:
    id: str
    question: str
    answerable: bool = True
    kind: str = "single"  # "single" (one passage answers it) or "multi_hop" (needs several)
    source: str | None = None
    sources: list[str] = field(default_factory=list)  # informational, for multi-hop cases
    evidence: list[str] = field(default_factory=list)
    answer_keywords: list[str] = field(default_factory=list)
    # Free-form labels (e.g. "table", "section") used to break retrieval metrics down by group.
    tags: list[str] = field(default_factory=list)


def load_cases(path: Path) -> list[EvalCase]:
    cases = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            case = EvalCase(**json.loads(line))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no}: invalid eval case: {exc}") from exc
        if case.kind not in KINDS:
            raise ValueError(f"{path}:{line_no}: kind must be one of {KINDS}")
        cases.append(case)
    return cases


def category(case: EvalCase) -> str:
    """ "single", "multi_hop" or "unanswerable": used to select questions on the CLI."""
    return case.kind if case.answerable else "unanswerable"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join(re.sub(r"[*_`#>]", " ", text).split())


def is_relevant(chunk: RetrievedChunk, case: EvalCase) -> bool:
    if case.source and not chunk.source.endswith(case.source):
        return False
    text = normalize(chunk.text)
    return all(normalize(phrase) in text for phrase in case.evidence)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return round(ordered[index], 1)


def _rate(items: list[dict], predicate: Callable[[dict], bool]) -> float | None:
    return round(sum(1 for r in items if predicate(r)) / len(items), 3) if items else None


# --------------------------------------------------------------------------- retrieval
def evaluate_retrieval(
    engine: DocQAEngine, cases: list[EvalCase], modes: list[str], k: int
) -> dict[str, Any]:
    # Multi-hop questions have no single relevant chunk, so they are scored on answers only.
    answerable = [c for c in cases if c.answerable and c.kind == "single" and c.evidence]
    results: dict[str, Any] = {}
    for mode in modes:
        ranks: dict[str, int | None] = {}
        for case in answerable:
            chunks = engine.retrieve(case.question, mode=mode, top_k=k).chunks
            ranks[case.id] = next(
                (i for i, c in enumerate(chunks, start=1) if is_relevant(c, case)), None
            )
        results[mode] = {
            **_rank_metrics(list(ranks.values()), k),
            "misses": [case_id for case_id, rank in ranks.items() if rank is None],
        }
        tags = sorted({t for c in answerable for t in c.tags})
        if tags:
            results[mode]["by_tag"] = {
                tag: _rank_metrics([ranks[c.id] for c in answerable if tag in c.tags], k)
                for tag in tags
            }
    return results


def _rank_metrics(ranks: list[int | None], k: int) -> dict[str, Any]:
    """Hit@1, Hit@k and MRR@k from the 1-based rank of the first relevant chunk (None = miss)."""
    total = len(ranks) or 1
    return {
        "hit@1": round(sum(r == 1 for r in ranks) / total, 3),
        f"hit@{k}": round(sum(r is not None for r in ranks) / total, 3),
        f"mrr@{k}": round(sum(1.0 / r for r in ranks if r) / total, 3),
        "n": len(ranks),
    }


# ----------------------------------------------------------------------------- answers
def _score(case: EvalCase, answer: str, rejected: bool) -> bool:
    if not case.answerable:
        return rejected
    normalized = normalize(answer)
    return not rejected and all(normalize(k) in normalized for k in case.answer_keywords)


def _error_row(case: EvalCase, exc: Exception) -> dict[str, Any]:
    return {
        "id": case.id,
        "kind": case.kind,
        "answerable": case.answerable,
        "status": "error",
        "reason": f"{type(exc).__name__}: {str(exc)[:120]}",
        "cited": False,
        "correct": False,
        "answer": "",
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Shared metrics for any answering system. Errors are excluded and counted separately."""
    done = [r for r in rows if r["status"] != "error"]
    answerable = [r for r in done if r["answerable"]]
    unanswerable = [r for r in done if not r["answerable"]]
    answered = [r for r in answerable if r["status"] == "ok"]
    latencies = [r["latency_ms"] for r in done if r.get("latency_ms") is not None]
    metrics: dict[str, Any] = {
        "answer_accuracy": _rate(answerable, lambda r: r["correct"]),
        "accuracy_by_kind": {
            kind: _rate([r for r in answerable if r["kind"] == kind], lambda r: r["correct"])
            for kind in KINDS
            if any(r["kind"] == kind for r in answerable)
        },
        "correct_rejection_rate": _rate(unanswerable, lambda r: r["correct"]),
        "false_rejection_rate": _rate(answerable, lambda r: r["status"] == "not_found"),
        "citation_rate": _rate(answered, lambda r: r["cited"]),
        "latency_ms_p50": percentile(latencies, 0.5),
        "latency_ms_p95": percentile(latencies, 0.95),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "errors": len(rows) - len(done),
    }
    if any("llm_calls" in r for r in done):
        metrics["llm_calls_mean"] = round(statistics.mean(r["llm_calls"] for r in done), 2)
        metrics["llm_calls_p95"] = percentile([r["llm_calls"] for r in done], 0.95)
        metrics["searches_mean"] = round(statistics.mean(r["searches"] for r in done), 2)
        metrics["verified_rate"] = _rate(answered, lambda r: r["verified"])
    if any("faithful" in r for r in done):
        metrics["faithfulness"] = _rate(
            [r for r in done if "faithful" in r], lambda r: r["faithful"]
        )
    return metrics


def evaluate_answers(
    engine: DocQAEngine, cases: list[EvalCase], judge: bool = False, delay: float = 0.0
) -> dict[str, Any]:
    """One-shot pipeline: retrieve once, generate once (the `/query` endpoint)."""
    faithfulness_evaluator = None
    if judge:
        from llama_index.core.evaluation import FaithfulnessEvaluator

        faithfulness_evaluator = FaithfulnessEvaluator(llm=engine.llm)

    rows = []
    for index, case in enumerate(cases):
        if delay and index:
            time.sleep(delay)  # stay under free-tier requests-per-minute limits
        try:
            result = engine.query(case.question)
        except Exception as exc:  # e.g. quota exhausted: record and keep going
            rows.append(_error_row(case, exc))
            continue
        row: dict[str, Any] = {
            "id": case.id,
            "kind": case.kind,
            "answerable": case.answerable,
            "status": result.status,
            "reason": result.reason,
            "cited": any(s.cited for s in result.sources),
            "correct": _score(case, result.answer, result.status == "not_found"),
            "latency_ms": result.latency_ms.get("total"),
            "llm_calls": 1 if "generation" in result.latency_ms else 0,
            "searches": 1,
            "verified": False,
        }
        if faithfulness_evaluator and result.status == "ok":
            contexts = [c.text for c in engine.retrieve(case.question).chunks]
            verdict = faithfulness_evaluator.evaluate(
                query=case.question, response=result.answer, contexts=contexts
            )
            row["faithful"] = bool(verdict.passing)
        row["answer"] = result.answer
        rows.append(row)
    return {**aggregate(rows), "rows": rows}


def evaluate_agent(
    cases: list[EvalCase], chat_factory: Callable[[], Any], delay: float = 0.0
) -> dict[str, Any]:
    """Agentic RAG through the real ADK runtime; every question starts a fresh session."""
    service = chat_factory()
    not_found = normalize(NOT_FOUND_MESSAGE)

    async def run() -> list[dict[str, Any]]:
        rows = []
        for index, case in enumerate(cases):
            if delay and index:
                await asyncio.sleep(delay)
            start = time.perf_counter()
            try:
                reply = await service.chat(case.question, user_id="eval")
            except Exception as exc:
                rows.append(_error_row(case, exc))
                continue
            text = normalize(reply.answer)
            refused = not_found in text or bool(REFUSAL_RE.search(text))
            # A partial answer may say "tài liệu không đề cập phần X" and still be correct.
            answered_ok = case.answerable and _score(case, reply.answer, False)
            rejected = refused and not answered_ok
            rows.append(
                {
                    "id": case.id,
                    "kind": case.kind,
                    "answerable": case.answerable,
                    "status": "not_found" if rejected else "ok",
                    "cited": bool(reply.sources),
                    "correct": _score(case, reply.answer, rejected),
                    "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                    "llm_calls": reply.llm_calls,
                    "searches": sum(s.tool == "search_passages" for s in reply.steps),
                    "verified": any(s.tool == "check_answer" for s in reply.steps),
                    "steps": [
                        f"{s.tool}({json.dumps(s.args, ensure_ascii=False)})" for s in reply.steps
                    ],
                    "answer": reply.answer,
                }
            )
        return rows

    rows = asyncio.run(run())
    return {**aggregate(rows), "rows": rows}


# ------------------------------------------------------------------------- calibration
def calibrate_similarity(engine: DocQAEngine, cases: list[EvalCase]) -> dict[str, Any]:
    """Best vector score per question, split by answerable / unanswerable."""
    positives, negatives = [], []
    for case in cases:
        score = engine.retrieve(case.question, mode="vector", top_k=1).best_vector_score
        if score is not None:
            (positives if case.answerable else negatives).append(score)
    if not positives or not negatives:
        return {"suggested_min_similarity": None}

    candidates = sorted(set(positives + negatives))
    best_threshold, best_accuracy = None, -1.0
    for threshold in candidates:
        correct = sum(s >= threshold for s in positives) + sum(s < threshold for s in negatives)
        accuracy = correct / (len(positives) + len(negatives))
        if accuracy > best_accuracy:
            best_threshold, best_accuracy = threshold, accuracy
    safe_threshold = round(min(positives) - 0.02, 3)
    return {
        "answerable_scores": {
            "min": round(min(positives), 3),
            "mean": round(statistics.mean(positives), 3),
        },
        "unanswerable_scores": {
            "max": round(max(negatives), 3),
            "mean": round(statistics.mean(negatives), 3),
        },
        "best_separating_threshold": round(best_threshold, 3) if best_threshold else None,
        "separation_accuracy": round(best_accuracy, 3),
        "suggested_min_similarity": min(safe_threshold, round(best_threshold or safe_threshold, 3)),
    }


# ----------------------------------------------------------------------------- runner
def run_evaluation(
    engine: DocQAEngine,
    dataset: Path,
    *,
    modes: list[str],
    k: int,
    retrieval_only: bool = False,
    judge: bool = False,
    agent: bool = False,
    chat_factory: Callable[[], Any] | None = None,
    kinds: list[str] | None = None,
    limit: int | None = None,
    delay: float = 0.0,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    all_cases = load_cases(dataset)
    engine.ensure_ready()
    # Kinds/limit select which questions are *answered* (LLM cost); retrieval uses all of them.
    answer_cases = [c for c in all_cases if not kinds or category(c) in kinds][: limit or None]
    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "cases": len(all_cases),
        "answered_cases": len(answer_cases),
        "config": {
            "embed_model": f"{engine.settings.embed_provider}/{engine.settings.embed_model}",
            "llm_model": f"{engine.settings.llm_provider}/{engine.settings.llm_model}",
            "agent_model": engine.settings.agent_model,
            "chunk_size": engine.settings.chunk_size,
            "chunk_overlap": engine.settings.chunk_overlap,
            "retrieval_mode": engine.settings.retrieval_mode,
            "top_k": k,
            "min_similarity": engine.settings.min_similarity,
            "vector_store": engine.settings.vector_store,
        },
        "retrieval": evaluate_retrieval(engine, all_cases, modes, k),
        "calibration": calibrate_similarity(engine, all_cases),
    }
    if not retrieval_only:
        report["answers"] = evaluate_answers(engine, answer_cases, judge=judge, delay=delay)
    if agent:
        if chat_factory is None:
            from docqa.chat import ChatService

            def chat_factory() -> ChatService:
                return ChatService(settings=engine.settings)

        report["agent"] = evaluate_agent(answer_cases, chat_factory, delay=delay)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        (out_dir / f"eval-{stamp}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / f"eval-{stamp}.md").write_text(to_markdown(report), encoding="utf-8")
        report["written_to"] = str(out_dir / f"eval-{stamp}.md")
    return report


_PERCENT_METRICS = ("rate", "accuracy", "faithfulness", "hit@", "mrr@")


def _fmt(value: Any, metric: str = "accuracy") -> str:
    """Rates and accuracies as percentages; counts, call averages and latencies as plain numbers."""
    if value is None:
        return "-"
    if isinstance(value, float) and any(marker in metric for marker in _PERCENT_METRICS):
        return f"{value * 100:.1f}%"
    return str(value)


COMPARE_METRICS = (
    "answer_accuracy",
    "correct_rejection_rate",
    "false_rejection_rate",
    "citation_rate",
    "faithfulness",
    "llm_calls_mean",
    "llm_calls_p95",
    "searches_mean",
    "verified_rate",
    "latency_ms_p50",
    "latency_ms_p95",
    "errors",
)


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Evaluation report ({report['timestamp']})",
        "",
        f"Dataset: `{report['dataset']}` ({report['cases']} questions,"
        f" {report.get('answered_cases', report['cases'])} answered)",
        "",
        "Config: " + ", ".join(f"{k}=`{v}`" for k, v in report["config"].items()),
        "",
        "## Retrieval (single-hop answerable questions)",
        "",
    ]
    retrieval = report["retrieval"]
    skip = ("n", "misses", "by_tag")
    metric_names = [m for m in next(iter(retrieval.values())) if m not in skip]
    lines.append("| Mode | " + " | ".join(metric_names) + " | Misses |")
    lines.append("|---" * (len(metric_names) + 2) + "|")
    for mode, metrics in retrieval.items():
        values = " | ".join(_fmt(metrics[m], m) for m in metric_names)
        lines.append(f"| {mode} | {values} | {', '.join(metrics['misses']) or '-'} |")
    if any("by_tag" in m for m in retrieval.values()):
        lines += ["", "By question tag:", ""]
        lines.append("| Mode | Tag | n | " + " | ".join(metric_names) + " |")
        lines.append("|---" * (len(metric_names) + 3) + "|")
        for mode, metrics in retrieval.items():
            for tag, tag_metrics in metrics.get("by_tag", {}).items():
                values = " | ".join(_fmt(tag_metrics[m], m) for m in metric_names)
                lines.append(f"| {mode} | {tag} | {tag_metrics['n']} | {values} |")

    lines += [
        "",
        "## Similarity calibration",
        "",
        "```json",
        json.dumps(report["calibration"], indent=2),
        "```",
    ]

    systems = {
        name: report[key]
        for name, key in (("one-shot RAG", "answers"), ("agentic RAG", "agent"))
        if key in report
    }
    if systems:
        names = list(systems)
        lines += [
            "",
            "## Answers",
            "",
            "| Metric | " + " | ".join(names) + " |",
            "|---" * (len(names) + 1) + "|",
        ]
        kinds = sorted({k for s in systems.values() for k in s.get("accuracy_by_kind", {})})
        for kind in kinds:
            values = " | ".join(
                _fmt(s.get("accuracy_by_kind", {}).get(kind)) for s in systems.values()
            )
            lines.append(f"| accuracy ({kind}) | {values} |")
        for key in COMPARE_METRICS:
            if any(key in s for s in systems.values()):
                values = " | ".join(_fmt(s.get(key), key) for s in systems.values())
                lines.append(f"| {key} | {values} |")

        for name, system in systems.items():
            failures = [r for r in system["rows"] if not r["correct"]]
            if not failures:
                continue
            lines += [
                "",
                f"### Failed cases: {name}",
                "",
                "| id | kind | status | answer |",
                "|---|---|---|---|",
            ]
            for row in failures:
                answer = (
                    (row.get("answer") or row.get("reason") or "")
                    .replace("\n", " ")
                    .replace("|", "/")[:160]
                )
                lines.append(f"| {row['id']} | {row['kind']} | {row['status']} | {answer} |")
    return "\n".join(lines) + "\n"


def summary(report: dict[str, Any]) -> dict[str, Any]:
    """Compact view for console output."""
    out: dict[str, Any] = {
        "retrieval": {
            m: {k: v for k, v in r.items() if k != "misses"} for m, r in report["retrieval"].items()
        },
        "calibration": report["calibration"].get("suggested_min_similarity"),
    }
    for key in ("answers", "agent"):
        if key in report:
            out[key] = {k: v for k, v in report[key].items() if k != "rows"}
    return out


__all__ = ["EvalCase", "evaluate_agent", "load_cases", "run_evaluation", "summary", "to_markdown"]
