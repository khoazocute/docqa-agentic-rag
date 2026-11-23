import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from docqa.config import Settings
from docqa.documents import load_directory
from docqa.engine import DocQAEngine
from docqa.images import (
    NO_INFO_MARKER,
    CaptionService,
    pdf_page_images,
    prepare_image,
)
from docqa.structure import structured_documents


def picture(color: str, size=(400, 300), label: str = "") -> Image.Image:
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    for x in range(0, size[0], 7):  # noise so the encoded image is not trivially small
        draw.line([(x, 0), (size[0] - x, size[1])], fill=(x % 255, 80, 160))
    if label:
        draw.text((10, 10), label, fill="black")
    return image


def png_bytes(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


class FakeCaptioner:
    """Describes an image by its dominant colour; red images are 'decorative'."""

    def __init__(self, fail_with: Exception | None = None):
        self.calls = 0
        self.fail_with = fail_with

    def __call__(self, data: bytes, mime: str, prompt: str) -> str:
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        assert mime == "image/jpeg" and "Ngữ cảnh" in prompt
        r, g, b = Image.open(io.BytesIO(data)).convert("RGB").resize((1, 1)).getpixel((0, 0))
        if r > 150 and g < 100:
            return NO_INFO_MARKER
        return "Sơ đồ quy trình phê duyệt gồm ba bước: tạo đề nghị, quản lý duyệt, kế toán chi."


def service(tmp_path: Path, captioner, **kwargs) -> CaptionService:
    return CaptionService(captioner, tmp_path / "captions", "fake/model", **kwargs)


def test_prepare_image_skips_icons_and_reencodes():
    assert prepare_image(png_bytes(picture("white", size=(64, 64)))) is None
    rgba = Image.new("RGBA", (2400, 1200), (0, 0, 255, 128))
    data, mime = prepare_image(png_bytes(rgba))
    assert mime == "image/jpeg" and max(Image.open(io.BytesIO(data)).size) == 1600


def test_descriptions_are_cached_and_decorations_remembered(tmp_path):
    captioner = FakeCaptioner()
    captions = service(tmp_path, captioner)
    diagram, logo = png_bytes(picture("white")), png_bytes(picture("red"))

    first = captions.describe(diagram, "ctx")
    again = captions.describe(diagram, "ctx")
    assert first == again and "ba bước" in first
    assert captions.describe(logo, "ctx") is None
    assert captions.describe(logo, "ctx") is None
    assert captioner.calls == 2  # each image described once, decoration included
    # A new service (e.g. after a restart or an index rebuild) reuses the disk cache.
    assert (
        service(tmp_path, FakeCaptioner(fail_with=RuntimeError("no call"))).describe(diagram, "ctx")
        == first
    )


def test_budget_caps_calls_per_file(tmp_path):
    captioner = FakeCaptioner()
    captions = service(tmp_path, captioner, max_images_per_file=1)
    budget = captions.new_budget()
    images = [png_bytes(picture(c)) for c in ("white", "blue", "green")]

    assert len(captions.describe_all(images, "ctx", budget)) == 1
    assert captioner.calls == 1 and captions.stats["over_budget"] == 2


def test_daily_quota_skips_images_without_failing(tmp_path):
    class DailyQuota(Exception):
        code = 429

    captioner = FakeCaptioner(fail_with=DailyQuota("RESOURCE_EXHAUSTED retry in 14h2m3s"))
    captions = service(tmp_path, captioner, sleep=lambda s: pytest.fail("must not wait hours"))
    images = [png_bytes(picture(c)) for c in ("white", "blue", "green")]

    assert captions.describe_all(images, "ctx", captions.new_budget()) == []
    assert captioner.calls == 1  # later images are skipped, not retried one by one
    assert not (tmp_path / "captions").exists()  # nothing cached: retried on a later sync


def test_per_minute_quota_is_waited_out(tmp_path):
    class PerMinute(Exception):
        code = 429

    calls = {"n": 0}

    def flaky(data, mime, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PerMinute("429 RESOURCE_EXHAUSTED. Please retry in 4s.")
        return "Bảng so sánh ba gói dịch vụ theo giá và dung lượng lưu trữ."

    slept = []
    captions = service(tmp_path, flaky, sleep=slept.append)

    assert "Bảng so sánh" in captions.describe(png_bytes(picture("white")), "ctx")
    assert slept == [5.0]


def make_pdf(path: Path) -> Path:
    background = picture("lightgray", label="logo")
    pages = [
        picture("white", label="Trang 1"),  # page 1: unique diagram
        background,  # pages 2-4: same image = decoration
        background,
        background,
    ]
    pages[0].save(path, save_all=True, append_images=pages[1:])
    return path


def test_pdf_images_skip_decorations_repeated_across_pages(tmp_path):
    images = pdf_page_images(make_pdf(tmp_path / "slides.pdf"))
    assert len(images[0]) == 1
    assert all(not images.get(i) for i in (1, 2, 3))


def test_image_only_pdf_page_becomes_searchable(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    make_pdf(data / "slides.pdf")

    without = load_directory(data).documents
    with_captions = load_directory(data, captions=service(tmp_path, FakeCaptioner())).documents

    assert without == []  # image-only PDF: no text at all, nothing indexed
    assert len(with_captions) == 1
    doc = with_captions[0]
    assert "[Nội dung hình (trang 1)]" in doc.text and "ba bước" in doc.text
    assert doc.metadata["page_label"].startswith("1")


def test_docx_figure_description_lands_in_its_section_with_its_caption(tmp_path):
    docx = pytest.importorskip("docx")
    image_path = tmp_path / "diagram.png"
    picture("white").save(image_path)
    document = docx.Document()
    document.add_heading("1. Tổng quan", level=1)
    document.add_paragraph("Phần này giới thiệu quy trình chung của công ty. " * 6)
    document.add_heading("2. Quy trình phê duyệt", level=1)
    document.add_paragraph("Các bước được mô tả trong hình dưới đây. " * 6)
    document.add_picture(str(image_path))
    document.add_paragraph("Hình 1. Sơ đồ quy trình phê duyệt")
    document.save(str(tmp_path / "quy_trinh.docx"))

    docs = structured_documents(
        tmp_path / "quy_trinh.docx",
        "quy_trinh.docx",
        100,
        ["source"],
        ["source"],
        captions=service(tmp_path, FakeCaptioner()),
    )

    section = next(d for d in docs if "ba bước" in d.text)
    assert section.metadata["section"] == "2. Quy trình phê duyệt"
    assert "[Nội dung hình (Hình 1. Sơ đồ quy trình phê duyệt)]" in section.text


def test_captions_are_part_of_the_index_fingerprint():
    base = Settings()
    assert base.index_fingerprint() != base.with_overrides(image_captions=True).index_fingerprint()


def test_engine_answers_from_an_image_description(settings, fake_llm, fake_embed):
    make_pdf(settings.data_dir / "slides.pdf")
    engine = DocQAEngine(
        settings.with_overrides(image_captions=True),
        llm=fake_llm,
        embed_model=fake_embed,
        captioner=FakeCaptioner(),
    )

    result = engine.query("Quy trình phê duyệt gồm mấy bước?")

    cited = [s for s in result.sources if s.cited]
    assert cited and cited[0].file == "slides.pdf" and cited[0].page.startswith("1")
    assert "ba bước" in result.answer
