from pathlib import Path

import pytest
from llama_index.core.schema import Document

from docqa.config import Settings
from docqa.documents import (
    PART_SEPARATOR,
    discover_files,
    load_directory,
    merge_short_parts,
)


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("CHUNK_SIZE", "not-a-number")
    monkeypatch.setenv("RETRIEVAL_MODE", "BM25")
    monkeypatch.setenv("DATA_DIR", "my_docs")
    monkeypatch.setenv("AUTO_SYNC", "false")

    settings = Settings.from_env()

    assert settings.top_k == 7
    assert settings.chunk_size == 512  # invalid value falls back to default
    assert settings.retrieval_mode == "bm25"
    assert settings.data_dir == (tmp_path / "my_docs").resolve()
    assert settings.auto_sync is False


def test_invalid_choice_is_rejected(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "pinecone")
    with pytest.raises(ValueError, match="VECTOR_STORE"):
        Settings.from_env()


def test_fingerprint_tracks_index_relevant_settings():
    base = Settings()
    assert base.index_fingerprint() == base.with_overrides(top_k=10).index_fingerprint()
    assert base.index_fingerprint() != base.with_overrides(chunk_size=256).index_fingerprint()
    assert base.index_fingerprint() != base.with_overrides(embed_model="other").index_fingerprint()


def test_discover_skips_hidden_unsupported_and_temp_files(tmp_path: Path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "b.exe").write_text("x", encoding="utf-8")
    (tmp_path / "~$draft.docx").write_text("x", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "c.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.TXT").write_text("x", encoding="utf-8")

    names = [p.relative_to(tmp_path).as_posix() for p in discover_files(tmp_path)]
    assert names == ["a.md", "sub/d.TXT"]


def test_documents_have_stable_ids_and_content_hashes(data_dir: Path):
    first = load_directory(data_dir).documents
    (data_dir / "quy_dinh.txt").touch()  # mtime change only
    second = load_directory(data_dir).documents

    assert [d.id_ for d in first] == ["bao_mat.md::0", "quy_dinh.txt::0"]
    assert [d.hash for d in first] == [d.hash for d in second]
    assert first[0].metadata["source"] == "bao_mat.md"


def test_unreadable_file_is_reported_not_fatal(data_dir: Path):
    (data_dir / "broken.pdf").write_bytes(b"this is not a pdf")
    (data_dir / "empty.txt").write_text("   ", encoding="utf-8")

    result = load_directory(data_dir)

    sources = {d.metadata["source"] for d in result.documents}
    assert sources == {"bao_mat.md", "quy_dinh.txt"}
    assert "empty.txt" not in sources


def test_short_pages_are_merged_into_the_next_page():
    pages = [
        Document(text="Quy trình của RAG", metadata={"file_name": "s.pdf", "page_label": "7"}),
        Document(text="   ", metadata={"page_label": "8"}),
        Document(text="Bước 1: chia tài liệu thành đoạn. " * 5, metadata={"page_label": "9"}),
        Document(text="Nội dung đầy đủ của trang mười. " * 5, metadata={"page_label": "10"}),
        Document(text="Cảm ơn", metadata={"page_label": "11"}),
    ]

    parts = merge_short_parts(pages, "s.pdf", min_chars=100)

    assert [p.id_ for p in parts] == ["s.pdf::0", "s.pdf::1"]
    assert parts[0].text.startswith("Quy trình của RAG" + PART_SEPARATOR + "Bước 1")
    assert parts[0].metadata["page_label"] == "7-9"
    assert parts[1].metadata["page_label"] == "10-11"  # short tail joins the previous group
    assert parts[1].text.endswith("Cảm ơn")
