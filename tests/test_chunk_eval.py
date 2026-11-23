import pytest
from llama_index.core.node_parser import SentenceSplitter

from docqa.chunk_eval import (
    Chunk,
    ChunkBench,
    TokenCounter,
    bm25_retriever,
    chunk_with,
    evaluate_chunks,
    excerpt_for_answer,
    intersect,
    load_viquad,
    merge,
    paragraph_chunks,
)

PARA_1 = "Paris là thủ đô của Pháp. Thành phố nằm bên sông Seine."
PARA_2 = "Hà Nội là thủ đô của Việt Nam. Thành phố có hồ Hoàn Kiếm."


def bench() -> ChunkBench:
    text = PARA_1 + "\n\n" + PARA_2
    start = text.index("Hà Nội")
    return ChunkBench(
        name="tiny",
        documents={"doc": text},
        questions={"q": "Thủ đô của Việt Nam là gì?"},
        excerpts={"q": ("doc", [(start, start + len("Hà Nội là thủ đô của Việt Nam."))])},
    )


def test_excerpt_is_the_sentence_containing_the_answer():
    span = excerpt_for_answer(PARA_2, "hồ Hoàn Kiếm", PARA_2.index("hồ Hoàn Kiếm"))
    assert PARA_2[span[0] : span[1]] == "Thành phố có hồ Hoàn Kiếm."
    # Shifted offsets (seen in some mirrors) are recovered by searching the answer text.
    assert excerpt_for_answer(PARA_2, "Hà Nội", 40) == (0, len("Hà Nội là thủ đô của Việt Nam."))
    assert excerpt_for_answer(PARA_2, "không có", 0) is None


def test_span_helpers():
    assert merge([(5, 8), (0, 3), (2, 4)]) == [(0, 4), (5, 8)]
    assert intersect([(0, 10)], [(5, 20), (30, 40)]) == [(5, 10)]


def test_precision_omega_rewards_chunks_that_match_the_excerpt():
    b = bench()
    count = TokenCounter()
    exact = [Chunk("doc", span, "") for span in b.excerpts["q"][1]]
    whole = [Chunk("doc", (0, len(b.documents["doc"])), "")]

    assert evaluate_chunks(b, exact, count=count)["precision_omega"] == 1.0
    assert 0.2 < evaluate_chunks(b, whole, count=count)["precision_omega"] < 0.5


def test_chunk_spans_point_back_to_the_original_text():
    b = bench()
    for chunk in chunk_with(SentenceSplitter(chunk_size=24, chunk_overlap=0), b):
        assert b.documents["doc"][chunk.span[0] : chunk.span[1]] == chunk.text
    assert [c.text for c in paragraph_chunks(b)] == [PARA_1, PARA_2]


def test_retrieval_metrics_with_bm25():
    b = bench()
    chunks = paragraph_chunks(b)
    report = evaluate_chunks(b, chunks, bm25_retriever(chunks), k=1)
    assert report["recall@1"] == 1.0
    # With full recall, IoU reduces to precision: relevant / retrieved.
    assert report["iou@1"] == pytest.approx(report["precision@1"])
    assert 0 < report["iou@1"] < 1


def test_load_viquad_rebuilds_articles_and_skips_unanswerable(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    rows = [
        {
            "id": "1",
            "title": "Việt Nam",
            "context": PARA_2,
            "question": "Thủ đô của Việt Nam là gì?",
            "answers": {"text": ["Hà Nội"], "answer_start": [0]},
            "is_impossible": False,
        },
        {
            "id": "2",
            "title": "Việt Nam",
            "context": PARA_1,
            "question": "Dân số Paris năm 3000?",
            "answers": {"text": [], "answer_start": []},
            "is_impossible": True,
        },
    ]
    path = tmp_path / "v.parquet"
    pq.write_table(pa.Table.from_pylist(rows), str(path))

    b = load_viquad(path)

    assert b.documents == {"Việt Nam": PARA_2 + "\n\n" + PARA_1}
    assert list(b.questions) == ["1"]
    doc, spans = b.excerpts["1"]
    assert b.documents[doc][spans[0][0] : spans[0][1]] == "Hà Nội là thủ đô của Việt Nam."
