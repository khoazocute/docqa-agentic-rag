from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode

from docqa.config import NOT_FOUND_MESSAGE
from docqa.prompts import (
    build_qa_prompt,
    cited_passage_ids,
    extract_citations,
    is_not_found_answer,
    parse_verdict,
    uncited_sentences,
)
from docqa.retrieval import (
    HybridRetriever,
    RetrievedChunk,
    protect_leaders,
    reciprocal_rank_fusion,
)


def hits(*pairs):
    return [
        NodeWithScore(node=TextNode(id_=node_id, text=node_id), score=score)
        for node_id, score in pairs
    ]


class StaticRetriever(BaseRetriever):
    def __init__(self, results):
        super().__init__()
        self.results = results

    def _retrieve(self, query_bundle):
        return self.results


def test_rrf_rewards_documents_ranked_well_by_both_retrievers():
    vector = hits(("a", 0.9), ("b", 0.8), ("c", 0.7))
    bm25 = hits(("d", 5.0), ("b", 3.0), ("e", 1.0))

    ranking = reciprocal_rank_fusion([vector, bm25], k=60)

    ids = [node_id for node_id, _ in ranking]
    assert ids[0] == "b"  # ranked 2nd by both lists beats ranked 1st by only one
    assert set(ids) == {"a", "b", "c", "d", "e"}
    assert abs(dict(ranking)["b"] - (1 / 62 + 1 / 62)) < 1e-12


def test_hybrid_search_merges_scores_and_drops_zero_bm25_hits():
    retriever = HybridRetriever(
        StaticRetriever(hits(("a", 0.9), ("b", 0.5))),
        StaticRetriever(hits(("b", 2.0), ("z", 0.0))),
        mode="hybrid",
        top_k=5,
    )

    result = retriever.search("q")

    ids = [c.node.node_id for c in result.chunks]
    assert "z" not in ids
    assert result.best_vector_score == 0.9
    by_id = {c.node.node_id: c for c in result.chunks}
    assert by_id["b"].vector_score == 0.5 and by_id["b"].bm25_score == 2.0
    assert by_id["a"].bm25_score is None


def test_single_modes_use_raw_scores():
    retriever = HybridRetriever(
        StaticRetriever(hits(("a", 0.9))), StaticRetriever(hits(("b", 3.0))), top_k=3
    )
    assert [c.score for c in retriever.search("q", mode="vector").chunks] == [0.9]
    bm25 = retriever.search("q", mode="bm25")
    assert [c.node.node_id for c in bm25.chunks] == ["b"]
    assert bm25.best_vector_score is None


def test_prompt_numbers_context_and_includes_rules():
    node = TextNode(
        text="Mật khẩu tối thiểu 12 ký tự.", metadata={"source": "bao_mat.md", "page_label": "2"}
    )
    prompt = build_qa_prompt("Mật khẩu dài bao nhiêu?", [RetrievedChunk(node=node, score=1.0)])

    assert "[1] (nguồn: bao_mat.md, trang 2)" in prompt
    assert NOT_FOUND_MESSAGE in prompt
    assert prompt.rstrip().endswith("Trả lời:")


def test_not_found_detection_and_citations():
    assert is_not_found_answer(f'"{NOT_FOUND_MESSAGE}"')
    assert is_not_found_answer(NOT_FOUND_MESSAGE.upper())
    assert not is_not_found_answer("Mật khẩu tối thiểu 12 ký tự [1].")
    assert extract_citations("A [1], B [3][2], C [9]", max_id=3) == {1, 2, 3}


def test_each_retrievers_top_hit_survives_fusion():
    # "definition" is dense #1 but absent from BM25; plain RRF would rank it below b, c, d.
    vector = hits(("definition", 0.78), ("b", 0.74), ("c", 0.71), ("d", 0.70))
    bm25 = hits(("b", 0.27), ("c", 0.26), ("d", 0.25), ("e", 0.24))
    ranking = reciprocal_rank_fusion([vector, bm25])
    assert "definition" not in [node_id for node_id, _ in ranking[:3]]

    protected = protect_leaders(ranking, [vector, bm25], top_k=3)

    ids = [node_id for node_id, _ in protected]
    assert len(ids) == 3
    assert "definition" in ids and "b" in ids  # both leaders kept
    assert ids == sorted(ids, key=lambda i: -dict(ranking)[i])  # still ordered by RRF


def test_hybrid_search_keeps_dense_leader():
    retriever = HybridRetriever(
        StaticRetriever(hits(("definition", 0.78), ("b", 0.74), ("c", 0.71))),
        StaticRetriever(hits(("b", 0.27), ("c", 0.26), ("e", 0.24))),
        top_k=2,
    )
    assert "definition" in [c.node.node_id for c in retriever.search("RAG là gì").chunks]


def test_passage_citation_helpers():
    answer = "Nghỉ 12 ngày [#82CBF963]. Cộng thêm 1 ngày mỗi 5 năm [#f02fcf1d][#82cbf963]."
    assert cited_passage_ids(answer) == ["82cbf963", "f02fcf1d"]

    draft = "\n".join(
        [
            "Nhân viên được nghỉ phép năm mười hai ngày có lương.",
            "Nguồn: so_tay.md",
            "Có [#82cbf963].",
        ]
    )
    assert uncited_sentences(draft) == ["Nhân viên được nghỉ phép năm mười hai ngày có lương."]


def test_parse_verdict_tolerates_fences_and_rejects_garbage():
    fenced = "\n".join(
        ["```json", '{"supported": false, "unsupported_claims": ["16 ký tự"]}', "```"]
    )
    assert parse_verdict(fenced) == {"supported": False, "unsupported_claims": ["16 ký tự"]}
    assert parse_verdict("không phải JSON") is None
    assert parse_verdict('{"other": 1}') is None


def test_grouped_citations_are_recognised_by_helpers():
    answer = "Ý một [#82cbf963, #f02fcf1d]. Ý hai [#97661d2f]."
    assert cited_passage_ids(answer) == ["82cbf963", "f02fcf1d", "97661d2f"]
