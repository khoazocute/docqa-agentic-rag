import pytest

from docqa.config import Settings
from docqa.engine import DocQAEngine
from docqa.semantic import SemanticSplitter, cosine_distances, split_sentences
from tests.conftest import FakeEmbedding

PASSWORD = [
    "Mật khẩu tài khoản phải có ít nhất mười hai ký tự.",
    "Mật khẩu tài khoản phải gồm chữ hoa, chữ thường và ký tự đặc biệt.",
    "Mật khẩu tài khoản được đổi định kỳ chín mươi ngày.",
]
LEAVE = [
    "Nhân viên chính thức được nghỉ phép năm mười hai ngày có lương.",
    "Nhân viên chính thức nghỉ phép phải đăng ký trước ba ngày làm việc.",
    "Nhân viên chính thức nghỉ phép quá năm ngày cần giám đốc phê duyệt.",
]


class CountingEmbed:
    def __init__(self):
        self.model = FakeEmbedding()
        self.calls: list[int] = []

    def __call__(self, texts):
        self.calls.append(len(texts))
        return [self.model.get_text_embedding(t) for t in texts]


def test_split_sentences_rejoins_pdf_lines_and_keeps_list_items():
    text = "Hệ thống sẽ tìm kiếm các \nthông tin liên quan. Sau đó trả lời.\n● Mục một\n● Mục hai\n\nĐoạn mới."
    assert split_sentences(text) == [
        ("Hệ thống sẽ tìm kiếm các thông tin liên quan.", 0),
        ("Sau đó trả lời.", 0),
        ("● Mục một", 0),
        ("● Mục hai", 0),
        ("Đoạn mới.", 1),
    ]


def test_cuts_where_the_topic_changes():
    embed = CountingEmbed()
    splitter = SemanticSplitter(embed, chunk_size=512, min_chunk_tokens=0)

    chunks = splitter.split_text(" ".join(PASSWORD + LEAVE))

    assert len(chunks) == 2
    assert all("Mật khẩu" in s for s in chunks[0].split(". "))
    assert chunks[1].startswith("Nhân viên chính thức được nghỉ phép")
    assert splitter.embedded_windows == 6 and sum(embed.calls) == 6  # one window per sentence


def test_oversized_groups_are_cut_to_size_and_tiny_ones_merged():
    long_text = " ".join(PASSWORD * 12)  # one topic, far above the chunk size
    splitter = SemanticSplitter(CountingEmbed(), chunk_size=80)

    chunks = splitter.split_text(long_text)

    assert len(chunks) > 1
    assert all(len(splitter._tokenizer(c)) <= 80 for c in chunks)
    tiny = SemanticSplitter(CountingEmbed(), chunk_size=512, min_chunk_tokens=200)
    assert len(tiny.split_text(" ".join(PASSWORD + LEAVE))) == 1


def test_very_long_texts_skip_semantic_split_to_save_embeddings():
    embed = CountingEmbed()
    splitter = SemanticSplitter(embed, chunk_size=128, max_sentences=5)

    assert splitter.split_text(" ".join(PASSWORD + LEAVE))
    assert embed.calls == []


def test_cosine_distances():
    assert cosine_distances([[1, 0], [1, 0], [0, 1]]) == pytest.approx([0.0, 1.0])


def test_structure_strategy_uses_semantic_only_for_unstructured_files(
    settings, fake_llm, fake_embed, data_dir
):
    # A plain-text file without headings, long enough to contain a topic shift.
    (data_dir / "ghi_chu.txt").write_text(" ".join(PASSWORD + LEAVE), encoding="utf-8")
    calls = []
    original = fake_embed.get_text_embedding_batch

    def spy(texts, **kwargs):
        calls.append(list(texts))
        return original(texts, **kwargs)

    fake_embed.__dict__["get_text_embedding_batch"] = spy  # instance-level spy (pydantic model)
    engine = DocQAEngine(
        settings.with_overrides(chunk_strategy="structure", structure_fallback="semantic"),
        llm=fake_llm,
        embed_model=fake_embed,
    )

    engine.rebuild()

    window_texts = [t for batch in calls for t in batch if "file_name:" not in t]
    assert any("Mật khẩu tài khoản" in t for t in window_texts)  # semantic windows were embedded
    assert not any("Chính sách mật khẩu" in t for t in window_texts)  # .md had structure
    nodes = list(engine._index.docstore.docs.values())
    notes = [n for n in nodes if n.metadata.get("file_name") == "ghi_chu.txt"]
    assert len(notes) == 2 and "section" not in notes[0].metadata


def test_semantic_strategy_and_fallback_are_in_the_fingerprint():
    base = Settings()
    assert (
        base.index_fingerprint()
        != base.with_overrides(chunk_strategy="semantic").index_fingerprint()
    )
    assert (
        base.index_fingerprint()
        != base.with_overrides(structure_fallback="sentence").index_fingerprint()
    )
