import json

from docqa.evaluation import load_cases, run_evaluation


def write_dataset(path):
    rows = [
        {
            "id": "pwd",
            "question": "Mật khẩu phải có ít nhất bao nhiêu ký tự?",
            "source": "bao_mat.md",
            "evidence": ["12 ký tự"],
            "answer_keywords": ["12 ký tự"],
        },
        {
            "id": "hours",
            "question": "Sinh viên thực tập bắt đầu lúc mấy giờ?",
            "source": "quy_dinh.txt",
            "evidence": ["8:30"],
            "answer_keywords": ["8:30"],
        },
        {"id": "paris", "question": "Thủ đô nước Pháp tên gì?", "answerable": False},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


def test_load_cases(tmp_path):
    cases = load_cases(write_dataset(tmp_path / "ds.jsonl"))
    assert [c.id for c in cases] == ["pwd", "hours", "paris"]
    assert cases[2].answerable is False


def test_run_evaluation_end_to_end(engine, tmp_path):
    dataset = write_dataset(tmp_path / "ds.jsonl")

    report = run_evaluation(
        engine, dataset, modes=["vector", "bm25", "hybrid"], k=3, out_dir=tmp_path / "out"
    )

    for mode in ("vector", "bm25", "hybrid"):
        assert report["retrieval"][mode]["hit@3"] == 1.0
    answers = report["answers"]
    assert answers["answer_accuracy"] == 1.0
    assert answers["correct_rejection_rate"] == 1.0
    assert answers["citation_rate"] == 1.0
    assert report["calibration"]["suggested_min_similarity"] is not None
    markdown = (tmp_path / "out").glob("eval-*.md")
    assert "## Retrieval" in next(markdown).read_text(encoding="utf-8")


def test_agent_evaluation_reports_cost_and_compares_systems(
    engine, settings, tmp_path, monkeypatch
):
    import docqa.agent as agent_module
    from docqa.chat import ChatService
    from tests.test_agent import ScriptedLlm, search_then_verify

    monkeypatch.setattr(agent_module, "get_engine", lambda: engine)
    dataset = write_dataset(tmp_path / "ds.jsonl")

    def chat_factory():
        model = ScriptedLlm(policy=search_then_verify, requests=[])
        return ChatService(agent=agent_module.build_agent(settings, model=model), settings=settings)

    report = run_evaluation(
        engine,
        dataset,
        modes=["hybrid"],
        k=3,
        agent=True,
        chat_factory=chat_factory,
        kinds=["single"],
        out_dir=tmp_path / "out",
    )

    agent = report["agent"]
    assert report["answered_cases"] == 2  # the unanswerable case is filtered out by kinds
    assert agent["llm_calls_mean"] == 3 and agent["searches_mean"] == 1
    assert agent["verified_rate"] == 1.0 and agent["citation_rate"] == 1.0
    assert set(agent["accuracy_by_kind"]) == {"single"}
    markdown = next((tmp_path / "out").glob("eval-*.md")).read_text(encoding="utf-8")
    assert "| Metric | one-shot RAG | agentic RAG |" in markdown


def test_report_formats_rates_as_percent_and_counts_as_numbers():
    from docqa.evaluation import _fmt

    assert _fmt(0.86, "llm_calls_mean") == "0.86"
    assert _fmt(0.5, "correct_rejection_rate") == "50.0%"
    assert _fmt(0.941, "hit@1") == "94.1%"
    assert _fmt(None, "errors") == "-"


def test_retrieval_metrics_are_broken_down_by_tag(engine, tmp_path):
    rows = [
        {
            "id": "pwd",
            "question": "Mật khẩu phải có ít nhất bao nhiêu ký tự?",
            "source": "bao_mat.md",
            "evidence": ["12 ký tự"],
            "tags": ["text"],
        },
        {
            "id": "missing",
            "question": "Sinh viên thực tập bắt đầu lúc mấy giờ?",
            "source": "quy_dinh.txt",
            "evidence": ["cụm từ không tồn tại"],
            "tags": ["table"],
        },
    ]
    dataset = tmp_path / "tagged.jsonl"
    dataset.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), "utf-8")

    report = run_evaluation(
        engine, dataset, modes=["hybrid"], k=3, retrieval_only=True, out_dir=tmp_path / "out"
    )

    hybrid = report["retrieval"]["hybrid"]
    assert hybrid["hit@1"] == 0.5 and hybrid["hit@3"] == 0.5 and hybrid["misses"] == ["missing"]
    assert hybrid["by_tag"]["text"] == {"hit@1": 1.0, "hit@3": 1.0, "mrr@3": 1.0, "n": 1}
    assert hybrid["by_tag"]["table"]["hit@3"] == 0.0
    markdown = next((tmp_path / "out").glob("eval-*.md")).read_text(encoding="utf-8")
    assert "By question tag" in markdown
