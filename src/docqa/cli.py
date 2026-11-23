"""Command line interface: ``docqa sync | rebuild | stats | docs | ask | eval | serve``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from docqa.config import RETRIEVAL_MODES


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    for noisy in ("llama_index", "httpx", "httpcore", "google_genai", "qdrant_client", "bm25s"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _print(data) -> None:
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    elif isinstance(data, list):
        data = [d.model_dump() if hasattr(d, "model_dump") else d for d in data]
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="docqa", description="Vietnamese document QA (ADK + LlamaIndex)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sync", help="Incrementally index new/changed/deleted files in DATA_DIR")
    sub.add_parser("rebuild", help="Re-embed every document into a new index version")
    sub.add_parser("stats", help="Show index statistics")
    sub.add_parser("docs", help="List indexed documents")

    ask = sub.add_parser("ask", help="Ask one question (direct RAG, no agent)")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="Print the raw JSON result")

    ev = sub.add_parser("eval", help="Evaluate retrieval and answers on a labelled dataset")
    ev.add_argument("--dataset", type=Path, default=Path("eval/dataset.jsonl"))
    ev.add_argument("--modes", default="vector,bm25,hybrid", help="Retrieval modes to compare")
    ev.add_argument("--k", type=int, default=None, help="Top-k (default: TOP_K)")
    ev.add_argument("--retrieval-only", action="store_true", help="Skip LLM answer generation")
    ev.add_argument("--judge", action="store_true", help="Add LLM-as-judge faithfulness")
    ev.add_argument(
        "--delay", type=float, default=0.0, help="Seconds between LLM calls (free-tier rate limits)"
    )
    ev.add_argument("--agent", action="store_true", help="Also evaluate the agentic RAG agent")
    ev.add_argument(
        "--kinds",
        default="",
        help="Only answer these question kinds: single,multi_hop,unanswerable (default: all)",
    )
    ev.add_argument("--limit", type=int, default=None, help="Answer at most N questions")
    ev.add_argument("--out", type=Path, default=Path("eval/results"))

    bench = sub.add_parser(
        "bench", help="Evaluate retrieval on a public benchmark in BEIR format (e.g. Zalo Legal)"
    )
    bench.add_argument(
        "directory", type=Path, help="Folder with corpus.jsonl, queries.jsonl, qrels/"
    )
    bench.add_argument("--name", default=None, help="Dataset name shown in the report")
    bench.add_argument("--split", default="test")
    bench.add_argument("--modes", default="bm25", help="bm25, vector, hybrid (comma-separated)")
    bench.add_argument("--queries", type=int, default=None, help="Sample N queries (subset)")
    bench.add_argument(
        "--distractors",
        type=int,
        default=None,
        help="Subset corpus: relevant passages + N random distractors (needed for vector/hybrid)",
    )
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--k", type=int, default=100, help="Retrieval depth")
    bench.add_argument("--out", type=Path, default=Path("eval/results"))

    chunk_bench = sub.add_parser(
        "chunk-bench", help="Token-level chunking evaluation (Chroma metrics) on UIT-ViQuAD 2.0"
    )
    chunk_bench.add_argument("parquet", type=Path, help="UIT-ViQuAD 2.0 validation parquet file")
    chunk_bench.add_argument(
        "--configs",
        default="paragraph,128/0,256/0,256/64,512/0,512/64,512/128,1024/0",
        help="'paragraph' and/or chunk_size/overlap pairs for the sentence splitter",
    )
    chunk_bench.add_argument(
        "--k", type=int, default=5, help="Chunks retrieved (BM25) per question"
    )

    inj = sub.add_parser(
        "injection-eval", help="Indirect prompt-injection suite: attack success rate per defence"
    )
    inj.add_argument("--modes", default="oneshot", help="oneshot and/or agent (comma-separated)")
    inj.add_argument("--defenses", default="off,spotlight,full", help="Defence levels to compare")
    inj.add_argument("--payloads", default="", help="Only these payload types (default: all)")
    inj.add_argument("--delay", type=float, default=0.0, help="Seconds between calls (rate limits)")
    inj.add_argument("--out", type=Path, default=Path("eval/results"))

    detector = sub.add_parser(
        "injection-detector", help="Pattern detector recall/FPR on deepset/prompt-injections"
    )
    detector.add_argument(
        "directory", type=Path, nargs="?", default=Path("data/benchmarks/deepset_injections")
    )

    serve = sub.add_parser("serve", help="Run the REST API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    args = parser.parse_args(argv)
    _configure_logging()

    if args.command == "serve":
        import uvicorn

        uvicorn.run("docqa.api.app:app", host=args.host, port=args.port, reload=args.reload)
        return 0

    if args.command == "bench":
        return _bench(args, parser)
    if args.command == "chunk-bench":
        return _chunk_bench(args)
    if args.command == "injection-detector":
        from docqa.injection_eval import evaluate_detector

        _print(evaluate_detector(args.directory))
        return 0
    if args.command == "injection-eval":
        return _injection_eval(args, parser)

    from docqa.engine import get_engine

    engine = get_engine()  # shared with the agent tools (one index, one Qdrant client)

    if args.command == "sync":
        _print(engine.sync())
    elif args.command == "rebuild":
        _print(engine.rebuild())
    elif args.command == "stats":
        _print(engine.stats())
    elif args.command == "docs":
        _print(engine.list_documents())
    elif args.command == "ask":
        result = engine.query(args.question)
        if args.json:
            _print(result)
        else:
            print(result.answer)
            for source in result.sources:
                mark = "*" if source.cited else " "
                page = f" (trang {source.page})" if source.page else ""
                print(f" {mark}[{source.id}] {source.file}{page}  score={source.score}")
            print(f"\nstatus={result.status} latency={result.latency_ms}")
    elif args.command == "eval":
        from docqa.evaluation import run_evaluation, summary

        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        invalid = set(modes) - set(RETRIEVAL_MODES)
        if invalid:
            parser.error(f"unknown modes: {invalid}")
        report = run_evaluation(
            engine,
            args.dataset,
            modes=modes,
            k=args.k or engine.settings.top_k,
            retrieval_only=args.retrieval_only,
            judge=args.judge,
            delay=args.delay,
            agent=args.agent,
            kinds=[k.strip() for k in args.kinds.split(",") if k.strip()] or None,
            limit=args.limit,
            out_dir=args.out,
        )
        _print(summary(report))
        print(f"\nReport: {report.get('written_to')}")
    return 0


def _bench(args, parser) -> int:
    """Public-benchmark retrieval evaluation; writes eval/results/bench-<timestamp>.md|json."""
    import json
    from datetime import datetime

    from docqa.benchmarks import EmbeddingCache, format_report, load_beir, run_retrieval, subset

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    invalid = set(modes) - set(RETRIEVAL_MODES)
    if invalid:
        parser.error(f"unknown modes: {invalid}")
    dataset = load_beir(args.directory, split=args.split, name=args.name)
    if args.queries is not None or args.distractors is not None:
        distractors = args.distractors if args.distractors is not None else len(dataset.corpus)
        dataset = subset(dataset, args.queries, distractors, seed=args.seed)

    kwargs = {}
    if set(modes) & {"vector", "hybrid"}:
        if len(dataset.corpus) > 20_000:
            parser.error("vector/hybrid need one embedding per passage: use --distractors")
        from docqa.engine import get_engine

        engine = get_engine()
        settings = engine.settings
        kwargs = {
            "embed_texts": engine._embed_texts,
            "embed_query": engine._embed_query,
            "embed_model": engine.embed_model,
            "cache": EmbeddingCache(
                settings.storage_dir / "bench_cache" / f"{args.directory.name}.jsonl",
                f"{settings.embed_provider}/{settings.embed_model}",
            ),
        }
    results = run_retrieval(dataset, modes, k=args.k, **kwargs)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.out.mkdir(parents=True, exist_ok=True)
    report = format_report(results)
    header = (
        f"# Benchmark report ({stamp})\n\n{dataset.description}. Mean with 95% bootstrap CI.\n\n"
    )
    (args.out / f"bench-{stamp}.md").write_text(header + report + "\n", encoding="utf-8")
    (args.out / f"bench-{stamp}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(report)
    print(f"\nReport: {args.out / f'bench-{stamp}.md'}")
    return 0


def _injection_eval(args, parser) -> int:
    """Run the 30-case indirect-injection suite against one-shot RAG and/or the agent."""
    from docqa.config import INJECTION_DEFENSES, Settings
    from docqa.engine import DocQAEngine
    from docqa.injection_eval import (
        PAYLOADS,
        agent_runner,
        build_cases,
        oneshot_runner,
        run_cases,
        summarize,
        to_markdown,
        write_report,
    )

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    defenses = [d.strip() for d in args.defenses.split(",") if d.strip()]
    payloads = [p.strip() for p in args.payloads.split(",") if p.strip()]
    if set(modes) - {"oneshot", "agent"}:
        parser.error("modes: oneshot, agent")
    if set(defenses) - set(INJECTION_DEFENSES) or set(payloads) - set(PAYLOADS):
        parser.error(f"defenses: {INJECTION_DEFENSES}; payloads: {sorted(PAYLOADS)}")

    base = Settings.from_env().with_overrides(auto_sync=False)
    shared = DocQAEngine(base)  # one LLM/embedding client for every case corpus
    engines: dict[tuple, DocQAEngine] = {}

    def make_engine(settings: Settings) -> DocQAEngine:
        key = (str(settings.storage_dir), settings.injection_defense)
        if key not in engines:
            engines[key] = DocQAEngine(settings, llm=shared.llm, embed_model=shared.embed_model)
        return engines[key]

    runners = {"oneshot": oneshot_runner(make_engine), "agent": agent_runner(make_engine)}
    cases = [c for c in build_cases() if not payloads or c.payload in payloads]
    rows = run_cases(
        base,
        {m: runners[m] for m in modes},
        defenses,
        workdir=base.storage_dir / "injection_eval",
        delay=args.delay,
        cases=cases,
    )
    print(to_markdown(summarize(rows)))
    print(f"\nReport: {write_report(rows, args.out)}")
    return 0


def _chunk_bench(args) -> int:
    """Precision Ω (no retriever) and BM25 recall/precision/IoU@k per chunking configuration."""
    from llama_index.core.node_parser import SentenceSplitter

    from docqa.chunk_eval import (
        TokenCounter,
        bm25_retriever,
        chunk_with,
        evaluate_chunks,
        load_viquad,
        paragraph_chunks,
    )

    bench = load_viquad(args.parquet)
    count = TokenCounter()
    k = args.k
    print(
        f"{bench.name}: {len(bench.documents)} articles, {len(bench.excerpts)} labelled excerpts\n"
    )
    print(
        f"| Chunking | Chunks | Tokens/chunk | Precision Ω | Recall@{k} | Precision@{k} | IoU@{k} |"
    )
    print("|---|---|---|---|---|---|---|")
    for config in [c.strip() for c in args.configs.split(",") if c.strip()]:
        if config == "paragraph":
            chunks = paragraph_chunks(bench)
        else:
            size, overlap = (int(x) for x in config.split("/"))
            chunks = chunk_with(SentenceSplitter(chunk_size=size, chunk_overlap=overlap), bench)
            config = f"sentence {config}"
        r = evaluate_chunks(bench, chunks, bm25_retriever(chunks), k=k, count=count)
        print(
            f"| {config} | {r['chunks']} | {r['mean_chunk_tokens']} | {r['precision_omega']:.3f} "
            f"| {r[f'recall@{k}']:.3f} | {r[f'precision@{k}']:.3f} | {r[f'iou@{k}']:.3f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
