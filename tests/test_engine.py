import json

import pytest

from docqa.engine import DocQAEngine
from tests.conftest import FakeEmbedding, FakeLLM

# ----------------------------------------------------------------------- ingestion


def test_first_use_builds_and_persists_index(engine, settings):
    stats = engine.stats()

    assert stats.documents == 2
    assert stats.chunks >= 2
    current = (settings.storage_dir / "CURRENT").read_text(encoding="utf-8")
    manifest = json.loads(
        (settings.storage_dir / "versions" / current / "manifest.json").read_text("utf-8")
    )
    assert manifest["fingerprint"] == settings.index_fingerprint()


def test_sync_only_embeds_changed_files(engine, settings, fake_embed):
    engine.ensure_ready()
    embedded_before = len(fake_embed.embedded_texts)

    (settings.data_dir / "moi.txt").write_text(
        "Nhân viên được nghỉ phép năm 12 ngày.", encoding="utf-8"
    )
    (settings.data_dir / "bao_mat.md").write_text("Mật khẩu tối thiểu 16 ký tự.", encoding="utf-8")
    (settings.data_dir / "quy_dinh.txt").unlink()

    report = engine.sync()

    assert report.added == ["moi.txt"]
    assert report.updated == ["bao_mat.md"]
    assert report.deleted == ["quy_dinh.txt"]
    assert len(fake_embed.embedded_texts) - embedded_before == 2  # one chunk per changed file
    assert {d.source for d in engine.list_documents()} == {"bao_mat.md", "moi.txt"}

    again = engine.sync()
    assert not again.changed and again.unchanged == 2


def test_index_survives_restart(engine, settings):
    engine.ensure_ready()
    restarted = DocQAEngine(settings, llm=FakeLLM(), embed_model=FakeEmbedding())

    result = restarted.query("Mật khẩu phải có ít nhất bao nhiêu ký tự?")

    assert result.status == "ok"
    assert restarted.stats().index_version == engine.stats().index_version


def test_sync_on_fresh_engine_reports_the_initial_build(engine):
    report = engine.sync()
    assert report.mode == "rebuild"
    assert report.added == ["bao_mat.md", "quy_dinh.txt"]


def test_failed_rebuild_keeps_previous_index(engine, settings, fake_embed):
    version = engine.stats().index_version
    fake_embed.fail = True

    with pytest.raises(RuntimeError):
        engine.rebuild()

    fake_embed.fail = False
    assert engine._storage.current_version() == version
    assert engine.query("Mật khẩu phải có ít nhất bao nhiêu ký tự?").status == "ok"


def test_failed_sync_does_not_corrupt_index(engine, settings, fake_embed):
    engine.ensure_ready()
    (settings.data_dir / "quy_dinh.txt").write_text("Nội dung mới hoàn toàn.", encoding="utf-8")
    fake_embed.fail = True

    with pytest.raises(RuntimeError):
        engine.sync()

    fake_embed.fail = False
    # The persisted (old) version is reloaded, then the pending change is applied.
    assert engine.sync().updated == ["quy_dinh.txt"]


def test_changed_chunking_triggers_rebuild(engine, settings):
    first_version = engine.stats().index_version
    other = DocQAEngine(
        settings.with_overrides(chunk_size=256), llm=FakeLLM(), embed_model=FakeEmbedding()
    )
    assert other.stats().index_version != first_version


def test_engine_reloads_index_written_by_another_process(engine, settings):
    engine.ensure_ready()
    writer = DocQAEngine(settings, llm=FakeLLM(), embed_model=FakeEmbedding())
    (settings.data_dir / "moi.txt").write_text(
        "Phụ cấp ăn trưa 50.000 đồng mỗi ngày.", encoding="utf-8"
    )
    writer.sync()

    assert "moi.txt" in {d.source for d in engine.list_documents()}


def test_only_previous_version_is_kept(engine, settings):
    for i in range(3):
        (settings.data_dir / f"f{i}.txt").write_text(f"tài liệu số {i} nội dung", encoding="utf-8")
        engine.sync()
    assert len(list((settings.storage_dir / "versions").iterdir())) == 2


def test_add_and_remove_file(engine, settings):
    engine.ensure_ready()
    report = engine.add_file(
        "../../evil name?.md", "Quy định gửi xe: miễn phí cho nhân viên.".encode()
    )
    assert report.added == ["uploads/evil name_.md"]
    assert (settings.data_dir / "uploads" / "evil name_.md").exists()

    assert engine.remove_file("uploads/evil name_.md").deleted == ["uploads/evil name_.md"]
    with pytest.raises(ValueError):
        engine.remove_file("../outside.txt")
    with pytest.raises(ValueError):
        engine.add_file("virus.exe", b"x")


# ------------------------------------------------------------------------ querying


def test_answer_contains_citations_and_sources(engine):
    result = engine.query("Mật khẩu phải có ít nhất bao nhiêu ký tự?")

    assert result.status == "ok"
    assert "12 ký tự" in result.answer and "[1]" in result.answer
    assert result.sources[0].file == "bao_mat.md"
    assert result.sources[0].cited is True
    assert {"retrieval", "generation", "total"} <= result.latency_ms.keys()


def test_low_similarity_question_skips_llm(engine, fake_llm):
    result = engine.query("Thủ đô nước Pháp tên gì?")

    assert result.status == "not_found"
    assert result.reason == "low_similarity"
    assert fake_llm.prompts == []


def test_llm_refusal_becomes_not_found(engine, fake_llm):
    fake_llm.forced_answer = "Tôi không tìm thấy thông tin này trong tài liệu đã cung cấp."
    result = engine.query("Mật khẩu phải có ít nhất bao nhiêu ký tự?")

    assert result.status == "not_found"
    assert result.reason == "answer_not_in_context"
    assert result.sources == []


def test_empty_corpus_returns_not_found(settings, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    engine = DocQAEngine(
        settings.with_overrides(data_dir=empty), llm=FakeLLM(), embed_model=FakeEmbedding()
    )

    assert engine.query("Bất kỳ câu hỏi nào?").status == "not_found"


def test_blank_question_is_rejected(engine):
    with pytest.raises(ValueError):
        engine.query("   ")


def test_query_embeddings_are_cached(engine, fake_embed):
    engine.ensure_ready()

    for mode in ("vector", "hybrid", "vector"):
        engine.retrieve("Mật khẩu bao nhiêu ký tự?", mode=mode)
    engine.retrieve("  mật khẩu   bao nhiêu ký tự? ", mode="hybrid")
    engine.retrieve("Câu khác", mode="bm25")  # BM25 needs no embedding

    assert fake_embed.queries == ["Mật khẩu bao nhiêu ký tự?"]


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_all_retrieval_modes_find_the_right_document(engine, mode):
    result = engine.retrieve("báo cáo sự cố bảo mật trong bao lâu", mode=mode)
    assert result.chunks[0].source == "bao_mat.md"


@pytest.fixture
def qdrant_settings(settings, tmp_path):
    pytest.importorskip("qdrant_client")
    return settings.with_overrides(vector_store="qdrant", storage_dir=tmp_path / "qdrant_storage")


def qdrant_collections(engine):
    return sorted(c.name for c in engine._storage._qdrant().get_collections().collections)


def test_qdrant_backend_query_sync_and_reload(qdrant_settings, settings):
    engine = DocQAEngine(qdrant_settings, llm=FakeLLM(), embed_model=FakeEmbedding())
    result = engine.query("Mật khẩu phải có ít nhất bao nhiêu ký tự?")
    assert result.status == "ok"
    assert result.sources[0].vector_score is not None  # cosine similarity from Qdrant

    (settings.data_dir / "quy_dinh.txt").unlink()
    assert engine.sync().deleted == ["quy_dinh.txt"]
    assert [d.source for d in engine.list_documents()] == ["bao_mat.md"]
    hits = engine.retrieve("Sinh viên thực tập mấy giờ", mode="vector").chunks
    assert all(c.source == "bao_mat.md" for c in hits)  # deleted vectors are gone from Qdrant

    restarted = DocQAEngine(qdrant_settings, llm=FakeLLM(), embed_model=FakeEmbedding())
    assert restarted.query("Mật khẩu phải có ít nhất bao nhiêu ký tự?").status == "ok"


def test_qdrant_rebuild_replaces_collection(qdrant_settings):
    engine = DocQAEngine(qdrant_settings, llm=FakeLLM(), embed_model=FakeEmbedding())
    engine.ensure_ready()
    first = qdrant_collections(engine)

    engine.rebuild()
    second = qdrant_collections(engine)
    engine.rebuild()
    third = qdrant_collections(engine)

    assert len(first) == 1 and len(second) == 2  # current + previous kept for rollback
    assert len(third) == 2 and first[0] not in third  # older collections are dropped


def test_missing_qdrant_collection_triggers_rebuild(qdrant_settings):
    engine = DocQAEngine(qdrant_settings, llm=FakeLLM(), embed_model=FakeEmbedding())
    version = engine.stats().index_version
    client = engine._storage._qdrant()
    for name in qdrant_collections(engine):
        client.delete_collection(name)

    restarted = DocQAEngine(qdrant_settings, llm=FakeLLM(), embed_model=FakeEmbedding())
    assert restarted.stats().index_version != version
    assert restarted.query("Mật khẩu phải có ít nhất bao nhiêu ký tự?").status == "ok"


# ------------------------------------------------------------- agentic primitives


def test_search_passages_returns_evidence_with_short_ids(engine):
    result = engine.search_passages("Mật khẩu phải có ít nhất bao nhiêu ký tự?")

    assert result.status == "ok"
    top = result.passages[0]
    assert top.file == "bao_mat.md" and "12 ký tự" in top.text
    assert len(top.id) == 8
    found, unknown = engine.get_passages([top.id, "deadbeef"])
    assert [p.id for p in found] == [top.id] and unknown == ["deadbeef"]


def test_search_passages_scoped_to_one_document(engine):
    scoped = engine.search_passages("thời gian", source="quy_dinh.txt")
    assert scoped.status == "ok" and {p.file for p in scoped.passages} == {"quy_dinh.txt"}
    assert engine.search_passages("thời gian", source="nope.md").status == "error"


def test_search_passages_applies_similarity_gate(engine):
    result = engine.search_passages("Thủ đô nước Pháp tên gì?")
    assert result.status == "not_found" and result.reason == "low_similarity"


def test_read_context_returns_neighbours_in_reading_order(settings, fake_llm, fake_embed):
    long_doc = "\n\n".join(f"Điều {i}: quy định số {i} về chủ đề {i}. " * 6 for i in range(1, 7))
    (settings.data_dir / "dai.md").write_text(long_doc, encoding="utf-8")
    engine = DocQAEngine(settings, llm=fake_llm, embed_model=fake_embed)
    engine.ensure_ready()
    chunks = sorted(
        (n for n in engine._index.docstore.docs.values() if n.metadata["source"] == "dai.md"),
        key=lambda n: n.start_char_idx,
    )
    assert len(chunks) >= 3
    middle = engine.passage_id(chunks[1])

    context = engine.read_context(middle)

    assert [p.id for p in context] == [engine.passage_id(c) for c in chunks[:3]]
    assert engine.read_context(engine.passage_id(chunks[0]))[0].id == engine.passage_id(chunks[0])


def test_check_answer_verifies_citations_and_claims(engine):
    pid = engine.search_passages("mật khẩu 12 ký tự").passages[0].id

    assert engine.check_answer(f"Mật khẩu tối thiểu 12 ký tự [#{pid}].").status == "ok"

    wrong = engine.check_answer(f"Mật khẩu tối thiểu 16 ký tự [#{pid}].")
    assert wrong.status == "unsupported" and "16" in wrong.unsupported_claims[0]

    fake_id = engine.check_answer(f"Mật khẩu tối thiểu 12 ký tự [#{pid}] [#deadbeef].")
    assert fake_id.status == "unsupported" and fake_id.unknown_ids == ["deadbeef"]

    uncited = engine.check_answer("Mật khẩu phải có tối thiểu mười hai ký tự theo quy định.")
    assert uncited.status == "unsupported" and uncited.uncited_sentences


def test_qdrant_supports_document_filter(qdrant_settings):
    engine = DocQAEngine(qdrant_settings, llm=FakeLLM(), embed_model=FakeEmbedding())
    scoped = engine.search_passages("mật khẩu sự cố thực tập", source="quy_dinh.txt")
    assert scoped.status == "ok" and {p.file for p in scoped.passages} == {"quy_dinh.txt"}
