"""Describe images at ingestion time so their content becomes searchable text.

Slides and reports often carry their key content in figures: diagrams, comparison tables pasted
as pictures, image-only slides. Text extraction drops all of it, so no chunking or reranking can
ever retrieve it. Each meaningful image is sent once to a multimodal LLM that transcribes the
visible text and explains the figure; the description is inserted where the image was (the PDF
page, or the Word section) and is chunked, embedded and cited like any other text.

Cost control:
- decorations are skipped before any call: images repeated on many pages (backgrounds, logos)
  and tiny images (icons, bullets);
- the model answers ``NO_INFO_MARKER`` for decorative images, which is cached too;
- descriptions are cached on disk by image hash, so rebuilding an index never re-describes an
  image, and the cache survives index versions;
- a per-file budget caps the calls for very image-heavy files;
- quota errors never fail ingestion: the image is skipped and described on a later sync.
"""

from __future__ import annotations

import hashlib
import io
import logging
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from docqa.ratelimit import is_rate_limited, retry_delay

logger = logging.getLogger(__name__)

# Bump when the prompt changes so cached descriptions are regenerated.
CAPTION_VERSION = "v1"
NO_INFO_MARKER = "KHONG_CO_THONG_TIN"
MIN_IMAGE_SIDE = 120  # px; smaller images are icons or bullets
MIN_IMAGE_BYTES = 5_000
MAX_IMAGE_SIDE = 1600  # px; larger images are downscaled before upload
MAX_DECORATION_PAGES = 2  # an image on more pages than this is a background or logo
MAX_WAIT_S = 90.0

CAPTION_PROMPT = (
    "Bạn mô tả một hình trong tài liệu để hệ thống tìm kiếm có thể tìm thấy nội dung của nó.\n"
    "1. Chép lại NGUYÊN VĂN mọi chữ nhìn thấy trong hình: tiêu đề, nhãn, chú thích, nội dung ô "
    "bảng (giữ hàng và cột, mỗi hàng một dòng), số liệu.\n"
    "2. Sau đó giải thích ngắn gọn bằng tiếng Việt hình thể hiện điều gì: sơ đồ, quy trình các "
    "bước, so sánh, kết luận chính.\n"
    "Chỉ mô tả những gì có trong hình, không suy đoán thêm. Không nêu danh tính người trong ảnh.\n"
    f"Nếu hình chỉ để trang trí (logo, icon, ảnh minh họa không có thông tin), trả lời đúng một "
    f"từ: {NO_INFO_MARKER}\n\n"
    "Ngữ cảnh quanh hình (tên file và chữ gần hình):\n{context}"
)

# (image bytes, MIME type, prompt) -> description
Captioner = Callable[[bytes, str, str], str]


def prepare_image(data: bytes) -> tuple[bytes, str] | None:
    """Decode any supported format (incl. JPEG 2000), drop tiny images, downscale, re-encode JPEG.

    Returns None for images that are too small to carry information or cannot be decoded.
    """
    if len(data) < MIN_IMAGE_BYTES:
        return None
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        return None
    if min(image.size) < MIN_IMAGE_SIDE:
        return None
    if image.mode not in ("RGB", "L"):
        background = Image.new("RGB", image.size, "white")
        rgba = image.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        image = background
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    out = io.BytesIO()
    image.convert("RGB").save(out, format="JPEG", quality=85)
    return out.getvalue(), "image/jpeg"


def pdf_page_images(path: Path) -> dict[int, list[bytes]]:
    """Images per 0-based page index, without decorations repeated across pages."""
    from pypdf import PdfReader

    per_page: dict[int, list[bytes]] = defaultdict(list)
    pages_of: Counter[str] = Counter()
    for index, page in enumerate(PdfReader(str(path)).pages):
        seen: set[str] = set()
        for image in page.images:
            digest = hashlib.sha256(image.data).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            pages_of[digest] += 1
            per_page[index].append(image.data)
    return {
        index: [
            d for d in images if pages_of[hashlib.sha256(d).hexdigest()] <= MAX_DECORATION_PAGES
        ]
        for index, images in per_page.items()
    }


def docx_paragraph_images(paragraph) -> list[bytes]:
    """Images embedded in a python-docx paragraph, in reading order."""
    blobs = []
    for rel_id in paragraph._p.xpath(".//a:blip/@r:embed"):
        part = paragraph.part.related_parts.get(rel_id)
        if part is not None and getattr(part, "blob", None):
            blobs.append(part.blob)
    return blobs


def llm_captioner(llm) -> Captioner:
    """A captioner backed by a LlamaIndex multimodal LLM (Gemini, or a vision model on Ollama)."""
    from llama_index.core.base.llms.types import ChatMessage, ImageBlock, MessageRole, TextBlock

    def caption(data: bytes, mime: str, prompt: str) -> str:
        message = ChatMessage(
            role=MessageRole.USER,
            blocks=[ImageBlock(image=data, image_mimetype=mime), TextBlock(text=prompt)],
        )
        return llm.chat([message]).message.content or ""

    return caption


class CaptionService:
    """Describes images with caching, a per-file budget and quota-aware retries."""

    def __init__(
        self,
        captioner: Captioner,
        cache_dir: Path,
        model_id: str,
        max_images_per_file: int = 40,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._captioner = captioner
        self._cache_dir = cache_dir
        self._model_id = model_id
        self._max_per_file = max_images_per_file
        self._sleep = sleep
        self._lock = threading.Lock()
        self._exhausted_until = 0.0
        self.stats: Counter[str] = Counter()

    def describe_all(self, images: list[bytes], context: str, budget: list[int]) -> list[str]:
        """Descriptions of the informative images; ``budget`` is a shared per-file counter."""
        descriptions = []
        for data in images:
            text = self.describe(data, context, budget)
            if text:
                descriptions.append(text)
        return descriptions

    def new_budget(self) -> list[int]:
        return [self._max_per_file]

    def describe(self, data: bytes, context: str, budget: list[int] | None = None) -> str | None:
        prepared = prepare_image(data)
        if prepared is None:
            self.stats["skipped_small"] += 1
            return None
        image, mime = prepared
        key = hashlib.sha256(
            f"{CAPTION_VERSION}|{self._model_id}|".encode() + hashlib.sha256(data).digest()
        ).hexdigest()
        cache_file = self._cache_dir / f"{key}.txt"
        if cache_file.exists():
            self.stats["cached"] += 1
            return cache_file.read_text(encoding="utf-8") or None

        if budget is not None:
            if budget[0] <= 0:
                self.stats["over_budget"] += 1
                return None
            budget[0] -= 1
        if time.monotonic() < self._exhausted_until:
            self.stats["quota_skipped"] += 1
            return None

        text = self._call(image, mime, CAPTION_PROMPT.format(context=context[:800]))
        if text is None:
            return None
        informative = NO_INFO_MARKER not in text and len(text.strip()) > 20
        result = text.strip() if informative else ""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(result, encoding="utf-8")
        self.stats["described" if informative else "decorative"] += 1
        return result or None

    def _call(self, image: bytes, mime: str, prompt: str) -> str | None:
        for attempt in range(4):
            try:
                with self._lock:  # one call at a time keeps us under per-minute quotas
                    return self._captioner(image, mime, prompt)
            except Exception as exc:
                if is_rate_limited(exc):
                    delay = retry_delay(exc, attempt)
                    if delay <= MAX_WAIT_S and attempt < 3:
                        logger.warning("Image captioning rate-limited, retrying in %.0fs", delay)
                        self._sleep(delay)
                        continue
                    # Daily quota: stop trying for a while instead of failing every image.
                    self._exhausted_until = time.monotonic() + min(delay, 3600.0)
                    logger.warning("Image captioning quota exhausted; images left undescribed")
                    self.stats["quota_skipped"] += 1
                    return None
                logger.warning("Image captioning failed: %s", exc)
                self.stats["failed"] += 1
                return None
        return None


def format_description(description: str, where: str) -> str:
    """How a description appears in the document text (and therefore in chunks and citations)."""
    return f"[Nội dung hình ({where})]\n{description}"
