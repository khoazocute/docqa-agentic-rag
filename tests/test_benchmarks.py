import json
import math
from pathlib import Path

import pytest

from docqa.benchmarks import EmbeddingCache, load_beir, run_retrieval, subset, to_nodes
from docqa.ir_metrics import (
    bootstrap_ci,
    evaluate_rankings,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from tests.conftest import FakeEmbedding


def test_metrics_match_hand_computed_values():
    ranked = ["a", "x", "b", "y"]
    relevant = {"a", "b"}
    assert recall_at_k(ranked, relevant, 1) == 0.5
    assert recall_at_k(ranked, relevant, 3) == 1.0
    assert hit_at_k(["x", "a"], relevant, 1) == 0.0 and hit_at_k(["x", "a"], relevant, 2) == 1.0
    assert mrr_at_k(["x", "y", "b"], relevant, 10) == pytest.approx(1 / 3)
    ideal = 1 + 1 / math.log2(3)
    assert ndcg_at_k(ranked, relevant, 10) == pytest.approx((1 + 1 / math.log2(4)) / ideal)
    assert ndcg_at_k(["a", "b"], relevant, 10) == pytest.approx(1.0)


def test_bootstrap_interval_brackets_the_mean():
    values = [0.0] * 30 + [1.0] * 70
    low, high = bootstrap_ci(values)
    assert low < 0.7 < high and high - low < 0.25
    report = evaluate_rankings({"q1": ["a"], "q2": ["x"]}, {"q1": {"a"}, "q2": {"b"}})
    assert report["recall@1"]["mean"] == 0.5 and report["n_queries"]["mean"] == 2


LAWS = {
    "luat_lao_dong+14": (
        "Điều 14. Hình thức hợp đồng",
        "Hợp đồng lao động phải được giao kết bằng văn bản.",
    ),
    "luat_lao_dong+113": (
        "Điều 113. Nghỉ hằng năm",
        "Người lao động được nghỉ hằng năm 12 ngày làm việc.",
    ),
    "luat_giao_thong+5": ("Điều 5. Tốc độ", "Ô tô trong khu dân cư chạy tối đa 50 km/h."),
    "luat_thue+3": ("Điều 3. Thuế suất", "Thuế suất thuế giá trị gia tăng là 10 phần trăm."),
}


def write_beir(directory: Path) -> Path:
    (directory / "qrels").mkdir(parents=True)
    with (directory / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for doc_id, (title, text) in LAWS.items():
            f.write(
                json.dumps({"_id": doc_id, "title": title, "text": text}, ensure_ascii=False) + "\n"
            )
    queries = {
        "q1": "Hợp đồng lao động giao kết bằng hình thức gì?",
        "q2": "Người lao động được nghỉ hằng năm bao nhiêu ngày?",
        "q3": "Câu hỏi không có nhãn",
    }
    with (directory / "queries.jsonl").open("w", encoding="utf-8") as f:
        for qid, text in queries.items():
            f.write(json.dumps({"_id": qid, "text": text}, ensure_ascii=False) + "\n")
    with (directory / "qrels" / "test.jsonl").open("w", encoding="utf-8") as f:
        for qid, doc in [
            ("q1", "luat_lao_dong+14"),
            ("q2", "luat_lao_dong+113"),
            ("q2", "missing"),
        ]:
            f.write(json.dumps({"query-id": qid, "corpus-id": doc, "score": 1}) + "\n")
    return directory


def test_load_beir_keeps_only_valid_labels(tmp_path):
    dataset = load_beir(write_beir(tmp_path / "zalo"))
    assert len(dataset.corpus) == 4
    assert dataset.qrels == {"q1": {"luat_lao_dong+14"}, "q2": {"luat_lao_dong+113"}}
    assert set(dataset.queries) == {"q1", "q2"}


def test_subset_is_reproducible_and_keeps_relevant_passages(tmp_path):
    dataset = load_beir(write_beir(tmp_path / "zalo"))
    first = subset(dataset, n_queries=1, n_distractors=1, seed=7)
    second = subset(dataset, n_queries=1, n_distractors=1, seed=7)
    assert first.corpus.keys() == second.corpus.keys() and len(first.corpus) == 2
    (query,) = first.qrels
    assert first.qrels[query] <= set(first.corpus)


def test_passage_ids_stay_out_of_indexed_text(tmp_path):
    from llama_index.core.schema import MetadataMode

    node = to_nodes(load_beir(write_beir(tmp_path / "zalo")))[0]
    assert node.node_id not in node.get_content(metadata_mode=MetadataMode.EMBED)
    assert node.get_content().startswith("Điều 14.")


def test_all_modes_rank_the_labelled_article_and_reuse_the_cache(tmp_path):
    dataset = load_beir(write_beir(tmp_path / "zalo"))
    embed = FakeEmbedding()
    calls = []

    def embed_texts(texts):
        calls.append(len(texts))
        return [embed.get_text_embedding(t) for t in texts]

    cache = EmbeddingCache(tmp_path / "cache.jsonl", "fake")
    kwargs = {
        "embed_texts": embed_texts,
        "embed_query": embed.get_query_embedding,
        "embed_model": embed,
        "cache": cache,
        "k": 4,
    }

    results = run_retrieval(dataset, ["bm25", "vector", "hybrid"], **kwargs)

    for result in results:
        assert result["metrics"]["recall@1"]["mean"] == 1.0, result["mode"]
    assert sum(calls) == 4
    again = EmbeddingCache(tmp_path / "cache.jsonl", "fake")  # e.g. the next day
    run_retrieval(dataset, ["vector"], **{**kwargs, "cache": again})
    assert sum(calls) == 4  # nothing re-embedded
