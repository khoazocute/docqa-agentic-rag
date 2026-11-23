from pathlib import Path

import pytest
from llama_index.core.schema import MetadataMode

from docqa.config import Settings
from docqa.documents import load_directory
from docqa.engine import DocQAEngine, QuotaExhaustedError, retry_delay
from docqa.structure import (
    Block,
    build_sections,
    infer_heading_level,
    merge_short_sections,
    read_docx,
    read_markdown,
    read_plain_text,
    render_table,
    structured_documents,
)


def make_docx(path: Path) -> Path:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("Quy chế làm việc", level=0)  # Word "Title" style: not a section
    document.add_paragraph("Quy chế áp dụng cho toàn bộ nhân viên chính thức của công ty.")
    document.add_heading("1. Làm việc từ xa", level=1)
    document.add_paragraph("Nhân viên đã qua thử việc được làm việc từ xa tối đa 2 ngày mỗi tuần.")
    document.add_heading("1.1 Số ngày theo cấp bậc", level=2)
    # Some templates style captions as headings: it must not open a new section.
    document.add_heading("Bảng 1. Số ngày làm từ xa theo cấp bậc", level=6)
    table = document.add_table(rows=3, cols=3)
    for row, values in zip(
        table.rows,
        [
            ["Cấp bậc", "Được phép", "Ngày/tuần"],
            ["Nhân viên", "Có", "2"],
            ["Trưởng nhóm", "Có", "3"],
        ],
        strict=True,
    ):
        for cell, value in zip(row.cells, values, strict=True):
            cell.text = value
    document.add_heading("2. Trách nhiệm", level=1)
    document.add_paragraph("Nhân viên phải bật camera khi họp trực tuyến và phản hồi tin nhắn.")
    document.save(str(path))
    return path


def test_docx_sections_keep_heading_path_and_table_columns(tmp_path):
    path = make_docx(tmp_path / "quy_che.docx")

    sections = build_sections(read_docx(path))

    paths = [s.path for s in sections]
    assert ["1. Làm việc từ xa"] in paths
    table = next(s for s in sections if s.is_table)
    assert table.path == [
        "1. Làm việc từ xa",
        "1.1 Số ngày theo cấp bậc",
        "Bảng 1. Số ngày làm từ xa theo cấp bậc",
    ]
    # Two adjacent cells with the same text ("Có") must both stay, so columns do not shift.
    assert "Cấp bậc: Trưởng nhóm; Được phép: Có; Ngày/tuần: 3" in table.text()
    # The document title is not part of any breadcrumb.
    assert all("Quy chế làm việc" not in p for p in paths)


def test_merged_cells_are_kept_once(tmp_path):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    table = document.add_table(rows=2, cols=3)
    table.cell(0, 0).merge(table.cell(0, 1)).text = "Thông tin"
    table.cell(0, 2).text = "Giá trị"
    for cell, value in zip(table.rows[1].cells, ["A", "B", "C"], strict=True):
        cell.text = value
    document.save(str(tmp_path / "t.docx"))

    rows = next(b for b in read_docx(tmp_path / "t.docx") if b.kind == "table").rows

    assert rows[0] == ["Thông tin", "Giá trị"]


def test_markdown_headings_and_tables():
    blocks = read_markdown(
        "# Sổ tay\n\nGiới thiệu chung.\n\n## 1. Nghỉ phép\n\nNghỉ 12 ngày mỗi năm.\n\n"
        "## 2. Phụ cấp\n\n| Loại | Mức |\n|---|---|\n| Ăn trưa | 30.000 |\n"
    )
    kinds = [b.kind for b in blocks]
    assert kinds.count("heading") == 3 and "table" in kinds
    table = next(b for b in blocks if b.kind == "table")
    assert render_table(table.rows) == "Loại: Ăn trưa; Mức: 30.000"


@pytest.mark.parametrize(
    ("line", "level"),
    [
        ("2.1 Mục tiêu đề tài", 2),
        ("3. Trách nhiệm", 1),
        ("Chương II Quyền và nghĩa vụ", 2),
        ("Điều 14. Hình thức hợp đồng lao động", 4),
        ("1. Hợp đồng lao động phải được giao kết bằng văn bản.", None),  # list item, not title
        ("Sinh viên thực tập từ thứ Hai đến thứ Sáu", None),
    ],
)
def test_numbered_heading_inference(line, level):
    assert infer_heading_level(line) == level


def test_implausible_structure_returns_none():
    only_text = [Block("paragraph", "a"), Block("paragraph", "b")]
    headings_everywhere = [Block("heading", f"h{i}", level=1) for i in range(5)] + [
        Block("paragraph", "x"),
        Block("paragraph", "y"),
    ]
    assert build_sections(only_text) is None
    assert build_sections(headings_everywhere) is None


def test_short_intro_merges_forward_but_captioned_table_stays_alone():
    sections = build_sections(
        [
            Block("heading", "1. A", level=1),
            Block("paragraph", "Ngắn."),
            Block("heading", "1.1 B", level=2),
            Block("paragraph", "x" * 300),
            Block("caption", "Bảng 1. Số liệu"),
            Block("table", rows=[["k", "v"], ["a", "1"]]),
            Block("heading", "2. C", level=1),
            Block("paragraph", "y" * 300),
        ]
    )

    groups = merge_short_sections(sections, min_chars=200)

    assert [[s.path[-1] for s in g] for g in groups] == [
        ["1. A", "1.1 B"],
        ["Bảng 1. Số liệu"],
        ["2. C"],
    ]


def test_structured_documents_embed_the_section_path(tmp_path):
    path = make_docx(tmp_path / "quy_che.docx")

    docs = structured_documents(path, "quy_che.docx", 50, ["source"], ["source", "file_name"])

    assert [d.id_ for d in docs] == [f"quy_che.docx::{i}" for i in range(len(docs))]
    table_doc = next(d for d in docs if "Bảng 1" in d.metadata.get("section", ""))
    embed_text = table_doc.get_content(metadata_mode=MetadataMode.EMBED)
    assert "section: 1. Làm việc từ xa > 1.1 Số ngày theo cấp bậc > Bảng 1." in embed_text


def test_unstructured_files_fall_back_to_the_plain_loader(data_dir):
    plain = load_directory(data_dir, strategy="sentence").documents
    structured = load_directory(data_dir, strategy="structure").documents

    by_source = lambda docs: {d.id_: d.text for d in docs if d.id_.startswith("quy_dinh")}  # noqa: E731
    assert by_source(structured) == by_source(plain)  # .txt without headings: unchanged
    sections = [d.metadata.get("section") for d in structured if d.id_.startswith("bao_mat")]
    # Both sections are shorter than min_part_chars, so they are merged into one part.
    assert sections == ["Chính sách mật khẩu; Sự cố bảo mật"]


def test_chunk_strategy_is_part_of_the_index_fingerprint():
    base = Settings()
    assert (
        base.index_fingerprint()
        != base.with_overrides(chunk_strategy="structure").index_fingerprint()
    )


# ----------------------------------------------------------------------------- quota handling
class RateLimited(Exception):
    code = 429


def test_embedding_waits_out_per_minute_limits(settings, fake_embed, fake_llm, monkeypatch):
    calls = {"n": 0}
    original = fake_embed.get_text_embedding_batch

    def flaky(texts, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimited("429 RESOURCE_EXHAUSTED. Please retry in 7.5s.")
        return original(texts, **kwargs)

    monkeypatch.setattr(type(fake_embed), "get_text_embedding_batch", lambda self, t, **k: flaky(t))
    slept = []
    monkeypatch.setattr(DocQAEngine, "_sleep", staticmethod(slept.append))
    engine = DocQAEngine(settings, llm=fake_llm, embed_model=fake_embed)

    report = engine.rebuild()

    assert report.chunks > 0 and slept == [8.5]


def test_daily_quota_fails_fast(settings, fake_embed, fake_llm, monkeypatch):
    def exhausted(self, texts, **kwargs):
        raise RateLimited("429 RESOURCE_EXHAUSTED per day. Please retry in 14h51m28.5s.")

    monkeypatch.setattr(type(fake_embed), "get_text_embedding_batch", exhausted)
    monkeypatch.setattr(DocQAEngine, "_sleep", staticmethod(lambda s: pytest.fail("slept")))
    engine = DocQAEngine(settings, llm=fake_llm, embed_model=fake_embed)

    with pytest.raises(QuotaExhaustedError):
        engine.rebuild()
    assert retry_delay(RateLimited("retry in 2m5s"), 0) == 126.0


def test_answers_and_passages_cite_the_section(settings, fake_llm, fake_embed):
    engine = DocQAEngine(
        settings.with_overrides(chunk_strategy="structure"), llm=fake_llm, embed_model=fake_embed
    )

    result = engine.query("Mật khẩu phải có ít nhất bao nhiêu ký tự?")
    passages = engine.search_passages("Mật khẩu phải có ít nhất bao nhiêu ký tự?").passages

    cited = [s for s in result.sources if s.cited]
    assert cited and "Chính sách mật khẩu" in (cited[0].section or "")
    assert any("Chính sách mật khẩu" in (p.section or "") for p in passages)
    assert ", mục Chính sách mật khẩu" in fake_llm.prompts[-1]


def test_plain_text_without_blank_lines_keeps_every_heading():
    # Text copied from Word: one paragraph per line, no blank lines between sections.
    text = (
        "1.1.1. Sơ đồ tổ chức\n"
        "Công ty được chia thành các khối chức năng, bao gồm:\n"
        "-\tKhối Kỹ thuật: thực hiện dự án phần mềm.\n"
        "-\tKhối Kinh doanh: phát triển khách hàng.\n"
        "Hình 2 Sơ đồ tổ chức công ty\n"
        "1.1.2. Tầm nhìn và sứ mệnh\n"
        "Tầm nhìn: trở thành công ty gia công phần mềm hàng đầu,\n"
        "góp phần đưa Việt Nam lên bản đồ thế giới.\n"
    )
    blocks = read_plain_text(text)

    assert [b.text for b in blocks if b.kind == "heading"] == [
        "1.1.1. Sơ đồ tổ chức",
        "1.1.2. Tầm nhìn và sứ mệnh",
    ]
    assert [b.kind for b in blocks].count("paragraph") == 4  # intro, 2 list items, vision
    assert any(b.kind == "caption" and b.text.startswith("Hình 2") for b in blocks)
    # A sentence wrapped across two lines (PDF-style) is re-joined.
    assert any(
        b.text.endswith("hàng đầu, góp phần đưa Việt Nam lên bản đồ thế giới.") for b in blocks
    )


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Hình 2 Sơ đồ tổ chức công ty và vị trí thực tập", "caption"),
        ("Bảng 3. Tiêu chí đánh giá", "caption"),
        ("Hình 2 cho thấy sáu module nghiệp vụ liên kết quanh trục Gateway.", "paragraph"),
    ],
)
def test_captions_with_or_without_punctuation(text, kind):
    assert read_plain_text(text)[0].kind == kind
