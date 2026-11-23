"""Load local files into LlamaIndex Documents with stable, content-based identity."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

from docqa.config import SUPPORTED_EXTENSIONS
from docqa.images import CaptionService, format_description, pdf_page_images
from docqa.structure import structured_documents

logger = logging.getLogger(__name__)

# Only content-relevant metadata is kept, so a document's hash changes when its
# text changes, not when the file is merely touched or copied.
_KEEP_METADATA = ("file_name", "page_label")
# file_name and section (heading path) help retrieval; the rest is bookkeeping.
EXCLUDED_EMBED_KEYS = ["source", "page_label"]
EXCLUDED_LLM_KEYS = ["source", "file_name", "page_label"]


@dataclass
class LoadResult:
    documents: list[Document] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)  # relative paths that could not be parsed
    # Ids of section-based Documents (structure strategy); the rest have no usable structure.
    structured_ids: set[str] = field(default_factory=set)


def discover_files(data_dir: Path) -> list[Path]:
    """Return supported files under ``data_dir``, skipping hidden and temporary files."""
    if not data_dir.exists():
        return []
    files = []
    for path in data_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        relative = path.relative_to(data_dir)
        if any(part.startswith((".", "~$")) for part in relative.parts):
            continue
        files.append(path)
    return sorted(files)


DEFAULT_MIN_PART_CHARS = 200
PART_SEPARATOR = chr(10) * 2  # blank line between merged pages


def load_directory(
    data_dir: Path,
    min_part_chars: int = DEFAULT_MIN_PART_CHARS,
    strategy: str = "sentence",
    captions: CaptionService | None = None,
) -> LoadResult:
    """Load every supported file. One file may produce several Documents (e.g. PDF pages).

    Each Document gets the id ``<relative/path>::<part index>`` so re-ingesting the same
    folder yields the same ids, which is what incremental sync relies on. A file that cannot
    be parsed is reported in ``failed`` instead of aborting the whole ingestion.

    Pages shorter than ``min_part_chars`` (slide titles, image-only pages) are merged into the
    following page: on their own they become tiny chunks that BM25 over-scores and that carry
    no answer, pushing real content out of the top-k.

    With ``strategy="structure"``, files with a detectable structure (headings, tables) produce
    one Document per section instead; files without one fall back to the loader above.

    With ``captions``, images are described by a multimodal model and the descriptions are
    inserted where the image was: on its PDF page, or in its Word section (structure strategy).
    """
    data_dir = data_dir.resolve()
    result = LoadResult()

    for file_path in discover_files(data_dir):
        source = file_path.relative_to(data_dir).as_posix()
        if strategy == "structure":
            structured = _load_structured(file_path, source, min_part_chars, captions)
            if structured:
                result.documents.extend(structured)
                result.structured_ids.update(d.id_ for d in structured)
                continue
        try:
            raw_documents = SimpleDirectoryReader(input_files=[str(file_path)]).load_data(
                show_progress=False
            )
        except Exception as exc:
            logger.warning("Skipping %s: %s", source, exc)
            result.failed.append(source)
            continue
        if captions is not None and file_path.suffix.lower() == ".pdf":
            _describe_pdf_images(raw_documents, file_path, source, captions)
        result.documents.extend(merge_short_parts(raw_documents, source, min_part_chars))

    logger.info(
        "Loaded %d document parts (%d files failed)", len(result.documents), len(result.failed)
    )
    return result


def load_documents(data_dir: Path, min_part_chars: int = DEFAULT_MIN_PART_CHARS) -> list[Document]:
    return load_directory(data_dir, min_part_chars).documents


def _load_structured(
    path: Path, source: str, min_chars: int, captions: CaptionService | None = None
) -> list[Document] | None:
    """Section-based Documents, or None to fall back to the plain loader."""
    try:
        return structured_documents(
            path, source, min_chars, EXCLUDED_EMBED_KEYS, EXCLUDED_LLM_KEYS, captions
        )
    except Exception as exc:  # a parsing problem must not lose the file: use the plain loader
        logger.warning("Structure parsing failed for %s, using plain text: %s", source, exc)
        return None


def merge_short_parts(raw_parts: list[Document], source: str, min_chars: int) -> list[Document]:
    """Group consecutive parts of one file until each group has at least ``min_chars``."""
    groups: list[list[Document]] = []
    current: list[Document] = []
    for raw in raw_parts:
        if not (raw.text or "").strip():
            continue
        current.append(raw)
        if sum(len(r.text.strip()) for r in current) >= min_chars:
            groups.append(current)
            current = []
    if current:
        if groups:  # a short tail (e.g. "Thank you" slide) joins the previous group
            groups[-1].extend(current)
        else:
            groups.append(current)
    return [_to_document(group, source, part) for part, group in enumerate(groups)]


def _to_document(group: list[Document], source: str, part: int) -> Document:
    first = group[0]
    metadata = {key: first.metadata[key] for key in _KEEP_METADATA if first.metadata.get(key)}
    pages = [str(r.metadata["page_label"]) for r in group if r.metadata.get("page_label")]
    if pages:
        metadata["page_label"] = pages[0] if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"
    metadata["source"] = source
    return Document(
        id_=f"{source}::{part}",
        text=PART_SEPARATOR.join(r.text.strip() for r in group),
        metadata=metadata,
        excluded_embed_metadata_keys=list(EXCLUDED_EMBED_KEYS),
        excluded_llm_metadata_keys=list(EXCLUDED_LLM_KEYS),
    )


def _describe_pdf_images(
    pages: list[Document], path: Path, source: str, captions: CaptionService
) -> None:
    """Append image descriptions to the text of the page each image is on (in place).

    Image-only slides had no text at all and were dropped; with a description they become
    searchable and keep their page number for citations.
    """
    try:
        images = pdf_page_images(path)
    except Exception as exc:
        logger.warning("Could not extract images from %s: %s", source, exc)
        return
    budget = captions.new_budget()
    for index, page in enumerate(pages):
        page_images = images.get(index)
        if not page_images:
            continue
        label = str(page.metadata.get("page_label") or index + 1)
        text = (page.text or "").strip()
        context = f"{source}, trang {label}\n{text[:600]}"
        descriptions = captions.describe_all(page_images, context, budget)
        if descriptions:
            parts = [text] + [format_description(d, f"trang {label}") for d in descriptions]
            page.set_content(PART_SEPARATOR.join(p for p in parts if p))
