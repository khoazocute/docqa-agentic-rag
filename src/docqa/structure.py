"""Structure-aware parsing: turn a file into sections (heading path + blocks) before chunking.

The fixed-size splitter sees a document as flat text, so a chunk can start in the middle of one
section and end in the next, and a table becomes a list of loose cells. Here every supported
format is first read into the same small block model (heading, paragraph, table, caption), the
blocks are grouped into sections by heading level, and each section becomes its own Document
whose ``section`` metadata (the heading path, e.g. "2. Methods > 2.1 Goals") is embedded and
indexed together with every chunk cut from it.

Structure is detected in three tiers: declared (Word heading styles, Markdown ``#``), inferred
(numbered headings such as "1.2", "Chương II", "Điều 5" in plain text) and none. When no
plausible structure is found the caller falls back to the plain page-based loader, so a document
without structure is chunked exactly as before.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from llama_index.core.schema import Document

from docqa.images import CaptionService, docx_paragraph_images, format_description

logger = logging.getLogger(__name__)

STRUCTURED_EXTENSIONS = (".docx", ".md", ".txt")
SECTION_SEPARATOR = " > "
PARAGRAPH_SEPARATOR = "\n\n"

# Captions are recognised by their text, whatever style the author used (some templates style
# captions as headings, which would otherwise open bogus sections).
CAPTION_RE = re.compile(
    r"^(hình|bảng|biểu đồ|sơ đồ|figure|fig\.|table|chart)\s*\d+(\.\d+)*\s*[.:\-–]", re.IGNORECASE
)
# "Hình 2 Sơ đồ tổ chức" (no punctuation) is common too; accepted only for short lines that do
# not end like a sentence, so body text such as "Hình 2 cho thấy ... ." is not taken as a caption.
_LOOSE_CAPTION_RE = re.compile(
    r"^(hình|bảng|biểu đồ|sơ đồ|figure|fig\.|table|chart)\s*\d+(\.\d+)*\s+\S", re.IGNORECASE
)
MAX_CAPTION_CHARS = 150
_LIST_ITEM_RE = re.compile(r"^([-*•●▪–]|\d+[.)]|[a-zA-Z][.)])\s+")
_SENTENCE_END = (".", "!", "?", ":", ";", "…")
_WORD_HEADING_RE = re.compile(r"^heading\s*(\d)$", re.IGNORECASE)
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_MD_TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")
# Numbered headings in plain text: "2.1 Goals", "Chương II ...", "Điều 5. ..."
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,4})\.?\s+(\S.*)$")
_LEGAL_LEVELS = {"phần": 1, "chương": 2, "mục": 3, "điều": 4}
_LEGAL_HEADING_RE = re.compile(
    r"^(phần|chương|mục|điều)\s+([0-9]+|[ivxlcdm]+)\b[.:]?\s*(.*)$", re.IGNORECASE
)
MAX_HEADING_CHARS = 150


@dataclass
class Block:
    kind: Literal["heading", "paragraph", "table", "caption", "image"]
    text: str = ""  # for images: the generated description
    level: int = 0
    rows: list[list[str]] = field(default_factory=list)
    label: str = ""  # for images: the figure caption, when there is one


@dataclass
class Section:
    path: list[str]
    blocks: list[Block] = field(default_factory=list)
    is_table: bool = False

    @property
    def label(self) -> str:
        return SECTION_SEPARATOR.join(self.path)

    def text(self) -> str:
        return PARAGRAPH_SEPARATOR.join(render_block(b) for b in self.blocks if render_block(b))


# ----------------------------------------------------------------------------- readers
def read_docx(path: Path, captions: CaptionService | None = None) -> list[Block]:
    import docx
    from docx.table import Table

    blocks: list[Block] = []
    budget = captions.new_budget() if captions else None
    for item in docx.Document(str(path)).iter_inner_content():
        if isinstance(item, Table):
            rows = [_row_cells(row) for row in item.rows]
            rows = [r for r in rows if any(r)]
            if rows:
                blocks.append(Block("table", rows=rows))
            continue
        text = item.text.strip()
        if text:
            style = (item.style.name if item.style is not None else "") or ""
            blocks.append(_classify(text, _word_heading_level(item, style), style))
        if captions is not None:
            images = docx_paragraph_images(item)
            if images:
                context = path.name + "\n" + "\n".join(b.text for b in blocks[-3:] if b.text)
                for description in captions.describe_all(images, context, budget):
                    blocks.append(Block("image", description))
    _label_images(blocks)
    return blocks


def _label_images(blocks: list[Block]) -> None:
    """Name each figure description after its caption ("Hình 5. ..."), which in Word
    usually follows the picture and sometimes precedes it."""
    for i, block in enumerate(blocks):
        if block.kind != "image":
            continue
        for j in (i + 1, i - 1):
            if 0 <= j < len(blocks) and blocks[j].kind == "caption":
                block.label = blocks[j].text
                break


def _word_heading_level(paragraph, style: str) -> int | None:
    match = _WORD_HEADING_RE.match(style.strip())
    if match:
        return int(match.group(1))
    # Direct outline level (w:outlineLvl is 0-based) also marks a heading.
    levels = paragraph._p.xpath("./w:pPr/w:outlineLvl/@w:val")
    if levels and levels[0].isdigit() and int(levels[0]) < 9:
        return int(levels[0]) + 1
    return None


def _row_cells(row) -> list[str]:
    """Cell texts of a row, keeping a horizontally merged cell once.

    python-docx returns a merged cell once per grid column it spans. Duplicates are recognised by
    the underlying XML element, never by text: two adjacent cells may legitimately both say "Có".
    """
    seen: set[int] = set()
    cells = []
    for cell in row.cells:
        if id(cell._tc) in seen:
            continue
        seen.add(id(cell._tc))
        cells.append(cell.text.strip())
    return cells


def read_markdown(text: str) -> list[Block]:
    blocks: list[Block] = []
    paragraph: list[str] = []
    table: list[list[str]] = []

    def flush() -> None:
        if paragraph:
            blocks.append(_classify(" ".join(paragraph), None, ""))
            paragraph.clear()
        if table:
            blocks.append(Block("table", rows=[row[:] for row in table]))
            table.clear()

    for raw in text.splitlines():
        line = raw.strip()
        heading = _MD_HEADING_RE.match(line)
        if heading:
            flush()
            blocks.append(_classify(heading.group(2), len(heading.group(1)), ""))
        elif line.startswith("|"):
            if paragraph:
                flush()
            if not _MD_TABLE_SEPARATOR_RE.match(line):
                table.append([c.strip() for c in line.strip("|").split("|")])
        elif not line:
            flush()
        else:
            if table:
                flush()
            paragraph.append(line)
    flush()
    return blocks


def read_plain_text(text: str) -> list[Block]:
    """Numbered or legal-style lines become headings, wherever they are.

    Text copied from Word has one paragraph per line and no blank lines, while text extracted
    from PDFs wraps sentences across lines. So every line is checked: headings, list items and
    lines after a finished sentence start a new block; other lines continue the previous one.
    """
    blocks: list[Block] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(_classify(" ".join(paragraph), None, ""))
            paragraph.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        level = infer_heading_level(line)
        if level is not None:
            flush()
            blocks.append(Block("heading", line, level=level))
            continue
        if paragraph and (_LIST_ITEM_RE.match(line) or paragraph[-1].endswith(_SENTENCE_END)):
            flush()
        paragraph.append(line)
    flush()
    return blocks


def infer_heading_level(line: str) -> int | None:
    """Level of a numbered heading ("2.1 Goals" -> 2, "Điều 5. ..." -> 4), else None."""
    if len(line) > MAX_HEADING_CHARS or line.endswith((".", ",", ";", ":")):
        return None
    legal = _LEGAL_HEADING_RE.match(line)
    if legal:
        return _LEGAL_LEVELS[legal.group(1).lower()]
    numbered = _NUMBERED_HEADING_RE.match(line)
    if numbered and len(numbered.group(2).split()) <= 15:
        return numbered.group(1).count(".") + 1
    return None


def _classify(text: str, heading_level: int | None, style: str) -> Block:
    loose_caption = (
        _LOOSE_CAPTION_RE.match(text)
        and len(text) <= MAX_CAPTION_CHARS
        and not text.endswith((".", "!", "?"))
    )
    if CAPTION_RE.match(text) or loose_caption or style.strip().lower() == "caption":
        return Block("caption", text)
    if heading_level is not None and len(text) <= MAX_HEADING_CHARS:
        return Block("heading", text, level=heading_level)
    return Block("paragraph", text)


# ----------------------------------------------------------------------------- sections
def render_block(block: Block) -> str:
    if block.kind == "table":
        return render_table(block.rows)
    if block.kind == "image":
        return format_description(block.text, block.label or "hình")
    return block.text


def render_table(rows: list[list[str]]) -> str:
    """One line per data row as "header: value" pairs, so every cell keeps its row and column.

    A flat dump ("Chatbase", "Không", "Hạn chế") loses which value belongs to which product;
    "Tiêu chí: Đa mô hình LLM; Chatbase: Hạn chế (1 model/bot)" does not, and rows can be
    split between chunks without losing the header.
    """
    width = max(len(r) for r in rows)
    if len(rows) < 2 or width < 2:
        return PARAGRAPH_SEPARATOR.join(" | ".join(c for c in r if c) for r in rows)
    header = rows[0]
    lines = []
    for row in rows[1:]:
        pairs = []
        for i, value in enumerate(row):
            if not value:
                continue
            name = header[i] if i < len(header) and header[i] and header[i] != value else ""
            pairs.append(f"{name}: {value}" if name else value)
        if pairs:
            lines.append("; ".join(pairs))
    return PARAGRAPH_SEPARATOR.join(lines)


def build_sections(blocks: list[Block]) -> list[Section] | None:
    """Group blocks under their heading path. Returns None when the structure is not plausible."""
    blocks = _demote_document_title(blocks)
    headings = [b for b in blocks if b.kind == "heading"]
    content = [b for b in blocks if b.kind in ("paragraph", "table", "image")]
    if len(headings) < 2 or len(headings) > 2 * max(len(content), 1):
        # No structure, or "headings" everywhere (e.g. bold lines misread as titles). One heading
        # per paragraph (FAQ-style documents) is still a plausible structure.
        return None

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    current = Section(path=[])
    pending_caption: Block | None = None

    def close() -> None:
        nonlocal current
        if current.blocks:
            sections.append(current)
        current = Section(path=[title for _, title in stack])

    for block in blocks:
        if block.kind == "heading":
            close()
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.text))
            current = Section(path=[title for _, title in stack])
        elif block.kind == "caption":
            pending_caption = block
            current.blocks.append(block)
        elif block.kind == "table":
            # A table is its own section, titled by its caption when it has one, so its chunks
            # never mix with prose and always carry the table name.
            caption = pending_caption.text if pending_caption else None
            if caption and current.blocks and current.blocks[-1] is pending_caption:
                current.blocks.pop()
            close()
            path = current.path + ([caption] if caption else [])
            sections.append(Section(path=path, blocks=[block], is_table=bool(caption)))
            pending_caption = None
        else:
            pending_caption = None
            current.blocks.append(block)
    close()
    return sections


def _demote_document_title(blocks: list[Block]) -> list[Block]:
    """A lone top-level heading that opens the document is its title, not a section.

    Kept in the path it would be repeated in every chunk's breadcrumb, where its generic words
    ("company", "policy") add BM25 noise; the file name already identifies the document.
    """
    headings = [b for b in blocks if b.kind == "heading"]
    if len(headings) < 2:
        return blocks
    top = min(h.level for h in headings)
    first = headings[0]
    if first.level != top or sum(h.level == top for h in headings) != 1:
        return blocks
    return [Block("paragraph", b.text) if b is first else b for b in blocks]


def merge_short_sections(sections: list[Section], min_chars: int) -> list[list[Section]]:
    """Group a short prose section (often a parent's intro line) with the following sections.

    Captioned tables always stay on their own; short uncaptioned tables (cover pages, signature
    boxes) are merged like prose so they do not become tiny chunks that BM25 over-scores.
    """
    groups: list[list[Section]] = []
    current: list[Section] = []
    for section in sections:
        if section.is_table:
            if current:
                groups.append(current)
                current = []
            groups.append([section])
            continue
        current.append(section)
        if sum(len(s.text()) for s in current) >= min_chars:
            groups.append(current)
            current = []
    if current:
        if groups and not groups[-1][0].is_table:
            groups[-1].extend(current)
        else:
            groups.append(current)
    return groups


def _group_label(group: list[Section]) -> str:
    labels = list(dict.fromkeys(s.label for s in group if s.label))
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    # Shared prefix once, then the distinct leaf titles: "1. Overview > 1.1 A; 1.1.1 B".
    paths = [s.path for s in group if s.path]
    prefix: list[str] = []
    for parts in zip(*paths, strict=False):
        if len(set(parts)) != 1:
            break
        prefix.append(parts[0])
    leaves = list(dict.fromkeys(SECTION_SEPARATOR.join(p[len(prefix) :]) for p in paths))
    leaves = [leaf for leaf in leaves if leaf]
    label = SECTION_SEPARATOR.join(prefix)
    if leaves:
        label = (label + SECTION_SEPARATOR if label else "") + "; ".join(leaves)
    return label[:300]


# ----------------------------------------------------------------------------- entry point
def read_blocks(path: Path, captions: CaptionService | None = None) -> list[Block]:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return read_docx(path, captions)
    text = path.read_text(encoding="utf-8", errors="replace")
    return read_markdown(text) if suffix == ".md" else read_plain_text(text)


def structured_documents(
    path: Path,
    source: str,
    min_chars: int,
    excluded_embed: list[str],
    excluded_llm: list[str],
    captions: CaptionService | None = None,
) -> list[Document] | None:
    """One Document per (merged) section, or None when the file has no usable structure."""
    if path.suffix.lower() not in STRUCTURED_EXTENSIONS:
        return None
    sections = build_sections(read_blocks(path, captions))
    if not sections:
        return None
    documents = []
    for part, group in enumerate(merge_short_sections(sections, min_chars)):
        text = PARAGRAPH_SEPARATOR.join(s.text() for s in group if s.text())
        if not text.strip():
            continue
        metadata = {"file_name": path.name, "source": source}
        label = _group_label(group)
        if label:
            metadata["section"] = label
        documents.append(
            Document(
                id_=f"{source}::{part}",
                text=text,
                metadata=metadata,
                excluded_embed_metadata_keys=list(excluded_embed),
                excluded_llm_metadata_keys=list(excluded_llm),
            )
        )
    logger.info("%s: %d sections -> %d parts", source, len(sections), len(documents))
    return documents or None
