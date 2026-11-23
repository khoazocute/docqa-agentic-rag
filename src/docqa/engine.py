"""DocQA engine: ingestion (incremental sync / safe rebuild), hybrid retrieval, grounded answers."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, Document, MetadataMode

from docqa.config import NOT_FOUND_MESSAGE, SUPPORTED_EXTENSIONS, Settings
from docqa.documents import LoadResult, load_directory
from docqa.images import Captioner, CaptionService, llm_captioner
from docqa.injection import leaks_instructions, sanitize, strip_unknown_links
from docqa.prompts import (
    QA_PROMPT,
    VERIFY_PROMPT,
    build_qa_prompt,
    cited_passage_ids,
    excerpt_text,
    extract_citations,
    format_passages_for_check,
    is_not_found_answer,
    parse_verdict,
    snippet,
    uncited_sentences,
)
from docqa.providers import build_embed_model, build_llm
from docqa.ratelimit import QuotaExhaustedError, is_rate_limited, retry_delay
from docqa.retrieval import HybridRetriever, RetrievalResult
from docqa.schemas import (
    AnswerCheck,
    DocumentInfo,
    IndexStats,
    Passage,
    PassageSearch,
    QueryResult,
    Source,
    SyncReport,
)
from docqa.semantic import SemanticSplitter
from docqa.storage import IndexStorage

logger = logging.getLogger(__name__)

UPLOAD_SUBDIR = "uploads"
QUERY_CACHE_SIZE = 512
EMBED_BATCH_SIZE = 32
EMBED_MAX_RETRIES = 6
# Waits longer than this are daily quotas: failing fast beats blocking an upload for hours.
EMBED_MAX_WAIT_S = 120.0


PASSAGE_ID_LENGTH = 8
PASSAGE_MAX_CHARS = 1500
_UNSAFE_FILENAME_CHARS = re.compile(r"[\\/:*?\"<>|\x00-\x1f]")


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class DocQAEngine:
    """Thread-safe facade used by the ADK tools, the REST API, the CLI and the evaluator.

    The LLM and embedding model are created lazily and can be injected (tests use fakes).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLM | None = None,
        embed_model: BaseEmbedding | None = None,
        captioner: Captioner | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self._llm = llm
        self._captioner = captioner
        self._captions: CaptionService | None = None
        self._embed_model = embed_model
        self._storage = IndexStorage(self.settings)
        self._lock = threading.RLock()
        self._index: VectorStoreIndex | None = None
        self._retriever: HybridRetriever | None = None
        self._manifest: dict | None = None
        self._auto_synced = False
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------ models
    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    @property
    def embed_model(self) -> BaseEmbedding:
        if self._embed_model is None:
            self._embed_model = build_embed_model(self.settings)
        return self._embed_model

    def _embed_query(self, text: str) -> list[float]:
        """LRU-cached query embedding: repeated questions cost no API call."""
        key = " ".join(text.split()).lower()
        with self._cache_lock:
            if key in self._query_cache:
                self._query_cache.move_to_end(key)
                return self._query_cache[key]
        embedding = self.embed_model.get_query_embedding(text)
        with self._cache_lock:
            self._query_cache[key] = embedding
            if len(self._query_cache) > QUERY_CACHE_SIZE:
                self._query_cache.popitem(last=False)
        return embedding

    def _load(self) -> LoadResult:
        return load_directory(
            self.settings.data_dir,
            self.settings.min_part_chars,
            self.settings.chunk_strategy,
            captions=self.captions,
        )

    @property
    def captions(self) -> CaptionService | None:
        """Image describer, when IMAGE_CAPTIONS is on. Cached on disk across index versions."""
        if not self.settings.image_captions:
            return None
        if self._captions is None:
            self._captions = CaptionService(
                self._captioner or llm_captioner(self.llm),
                self.settings.storage_dir / "captions",
                model_id=f"{self.settings.llm_provider}/{self.settings.llm_model}",
                max_images_per_file=self.settings.max_images_per_file,
            )
        return self._captions

    def _split(self, documents: list[Document], structured_ids: set[str]) -> list[BaseNode]:
        """Chunk documents according to CHUNK_STRATEGY.

        sentence: fixed-size sentence splitting. semantic: split points from embedding distance.
        structure: section Documents are only cut to size (their boundaries are already the
        topic boundaries); files without structure use STRUCTURE_FALLBACK.
        """
        strategy = self.settings.chunk_strategy
        if strategy == "sentence":
            return self._splitter().get_nodes_from_documents(documents)
        if strategy == "semantic":
            return self._semantic_splitter().get_nodes_from_documents(documents)
        structured = [d for d in documents if d.id_ in structured_ids]
        plain = [d for d in documents if d.id_ not in structured_ids]
        fallback = (
            self._semantic_splitter()
            if self.settings.structure_fallback == "semantic"
            else self._splitter()
        )
        nodes = self._splitter().get_nodes_from_documents(structured) if structured else []
        if plain:
            nodes += fallback.get_nodes_from_documents(plain)
        return nodes

    def _semantic_splitter(self) -> SemanticSplitter:
        return SemanticSplitter(
            self._embed_texts,
            chunk_size=self.settings.chunk_size,
            breakpoint_percentile=self.settings.semantic_breakpoint_percentile,
        )

    def _splitter(self) -> SentenceSplitter:
        return SentenceSplitter(
            chunk_size=self.settings.chunk_size, chunk_overlap=self.settings.chunk_overlap
        )

    # ------------------------------------------------------------ index state
    def ensure_ready(self) -> SyncReport | None:
        """Load the current index (or build it), and reload when another process changed it.

        Returns the report of the build/sync it had to run, if any.
        """
        with self._lock:
            current = self._storage.current_version()
            if self._index is not None and self._manifest and current == self._manifest["version"]:
                return None

            loaded = self._storage.load(self.embed_model) if current else None
            if loaded is None:
                logger.info("No compatible index found, building a new one")
                self._auto_synced = True
                return self._rebuild_locked()

            self._activate(loaded.index, loaded.manifest)
            if self.settings.auto_sync and not self._auto_synced:
                self._auto_synced = True
                return self._sync_locked()
            return None

    def _activate(self, index: VectorStoreIndex, manifest: dict) -> None:
        self._index = index
        self._manifest = manifest
        self._retriever = HybridRetriever.from_index(index, self.settings, self._embed_query)

    def _invalidate(self) -> None:
        self._index = None
        self._retriever = None
        self._manifest = None

    def _embed_nodes(self, nodes: list[BaseNode]) -> None:
        """Embed before touching the index, so API failures cannot leave it half-updated."""
        if not nodes:
            return
        texts = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes]
        for node, embedding in zip(nodes, self._embed_texts(texts), strict=True):
            node.embedding = embedding

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed in batches through the quota-aware path (also used by semantic chunking)."""
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            embeddings.extend(self._embed_batch(texts[start : start + EMBED_BATCH_SIZE]))
        return embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch, waiting out per-minute quota errors instead of failing the ingest.

        Free tiers cap embeddings per minute (e.g. 100/min), so a large document would otherwise
        fail half-way; the wait the provider asks for is honoured, then the batch is retried.
        """
        for attempt in range(EMBED_MAX_RETRIES + 1):
            try:
                return self.embed_model.get_text_embedding_batch(texts)
            except Exception as exc:
                if not is_rate_limited(exc) or attempt == EMBED_MAX_RETRIES:
                    raise
                delay = retry_delay(exc, attempt)
                if delay > EMBED_MAX_WAIT_S:
                    raise QuotaExhaustedError(
                        f"Embedding quota exhausted; the provider asks to wait {delay / 3600:.1f} h"
                    ) from exc
                logger.warning("Embedding rate-limited, retrying in %.0fs", delay)
                self._sleep(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def _sleep(seconds: float) -> None:
        time.sleep(seconds)

    def _chunk_count(self) -> int:
        return len(self._index.docstore.docs) if self._index is not None else 0

    # ---------------------------------------------------------------- ingestion
    def rebuild(self) -> SyncReport:
        """Re-embed everything into a new index version (old version keeps serving until done)."""
        with self._lock:
            return self._rebuild_locked()

    def _rebuild_locked(self) -> SyncReport:
        start = time.perf_counter()
        loaded = self._load()
        documents = loaded.documents
        nodes = self._split(documents, loaded.structured_ids)
        self._embed_nodes(nodes)

        context, collection = self._storage.new_storage_context()
        try:
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=context,
                embed_model=self.embed_model,
                store_nodes_override=True,
                show_progress=False,
            )
            for document in documents:
                context.docstore.set_document_hash(document.id_, document.hash)
            manifest = self._storage.commit(
                index, collection=collection, documents=len(documents), chunks=len(nodes)
            )
        except Exception:
            self._storage.discard_collection(collection)
            raise

        self._activate(index, manifest)
        logger.info(
            "Rebuilt index %s: %d parts, %d chunks", manifest["version"], len(documents), len(nodes)
        )
        return SyncReport(
            mode="rebuild",
            added=sorted({d.metadata["source"] for d in documents}),
            failed=loaded.failed,
            documents=len(documents),
            chunks=len(nodes),
            index_version=manifest["version"],
            duration_ms=_elapsed_ms(start),
        )

    def sync(self) -> SyncReport:
        """Incrementally apply changes in the data folder: only new/changed files are embedded."""
        with self._lock:
            self._auto_synced = True  # this call is the sync; don't run another one on load
            report = self.ensure_ready()
            if report is not None and report.mode == "rebuild":
                return report  # a fresh build already contains every current file
            return self._sync_locked()

    def _sync_locked(self) -> SyncReport:
        start = time.perf_counter()
        index = self._index
        assert index is not None
        docstore = index.docstore

        loaded = self._load()
        documents = loaded.documents
        current: dict[str, Document] = {doc.id_: doc for doc in documents}
        # ref_doc_info only lists source documents (the hash table also contains node ids).
        known_ids = set(index.ref_doc_info.keys())
        failed = set(loaded.failed)

        added = [d for d in documents if d.id_ not in known_ids]
        updated = [
            d
            for d in documents
            if d.id_ in known_ids and docstore.get_document_hash(d.id_) != d.hash
        ]
        # A file that failed to parse keeps its previously indexed version.
        deleted_ids = sorted(
            doc_id
            for doc_id in known_ids - current.keys()
            if doc_id.rsplit("::", 1)[0] not in failed
        )
        unchanged = len(documents) - len(added) - len(updated)

        changed_docs = added + updated
        nodes = self._split(changed_docs, loaded.structured_ids)
        try:
            self._embed_nodes(nodes)  # may fail (network/quota): nothing mutated yet
            for doc_id in [d.id_ for d in updated] + deleted_ids:
                index.delete_ref_doc(doc_id, delete_from_docstore=True)
            if nodes:
                index.insert_nodes(nodes)
            for document in changed_docs:
                docstore.set_document_hash(document.id_, document.hash)

            if changed_docs or deleted_ids:
                self._manifest = self._storage.commit(
                    index,
                    collection=(self._manifest or {}).get("collection"),
                    documents=len(documents),
                    chunks=len(docstore.docs),
                )
                self._retriever = HybridRetriever.from_index(
                    index, self.settings, self._embed_query
                )
        except Exception:
            # In-memory index may be partially modified: drop it, the persisted version is intact.
            self._invalidate()
            raise

        def sources(docs: list[Document]) -> list[str]:
            return sorted({d.metadata["source"] for d in docs})

        report = SyncReport(
            mode="sync",
            added=sources(added),
            updated=sources(updated),
            deleted=sorted({doc_id.rsplit("::", 1)[0] for doc_id in deleted_ids}),
            failed=loaded.failed,
            unchanged=unchanged,
            documents=len(documents),
            chunks=len(docstore.docs),
            index_version=(self._manifest or {}).get("version"),
            duration_ms=_elapsed_ms(start),
        )
        if report.changed:
            logger.info("Sync: +%s ~%s -%s", report.added, report.updated, report.deleted)
        return report

    # ------------------------------------------------------------ file management
    def add_file(self, filename: str, content: bytes) -> SyncReport:
        """Save an uploaded file under data/uploads and sync it into the index."""
        name = _UNSAFE_FILENAME_CHARS.sub("_", Path(filename).name).strip(" .")
        if not name or name.startswith("."):
            raise ValueError("Tên file không hợp lệ.")
        if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Chỉ hỗ trợ các định dạng: {', '.join(SUPPORTED_EXTENSIONS)}")
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise ValueError(f"File vượt quá {self.settings.max_upload_mb} MB.")
        if not content:
            raise ValueError("File rỗng.")

        target_dir = self.settings.data_dir / UPLOAD_SUBDIR
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / name).write_bytes(content)
        return self.sync()

    def remove_file(self, source: str) -> SyncReport:
        """Delete a document file (path relative to the data folder) and sync."""
        data_dir = self.settings.data_dir.resolve()
        target = (data_dir / source).resolve()
        if not target.is_relative_to(data_dir) or target == data_dir:
            raise ValueError("Đường dẫn không hợp lệ.")
        if not target.is_file():
            raise FileNotFoundError(source)
        target.unlink()
        return self.sync()

    # ----------------------------------------------------------------- querying
    def retrieve(
        self,
        question: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        source: str | None = None,
    ) -> RetrievalResult:
        self.ensure_ready()
        retriever = self._retriever
        if retriever is None:
            return RetrievalResult(chunks=[])
        return retriever.search(question, mode=mode, top_k=top_k, source=source)

    # ---------------------------------------------------- agentic RAG primitives
    # The agent drives retrieval itself: it searches, reads the evidence, decides whether it is
    # enough, re-searches (rephrased, or restricted to one document), expands context, and
    # finally verifies its draft. These methods return evidence, never a generated answer.

    @staticmethod
    def passage_id(node: BaseNode) -> str:
        return node.node_id.replace("-", "")[:PASSAGE_ID_LENGTH]

    def _to_passage(self, node: BaseNode, score: float | None = None) -> Passage:
        meta = node.metadata or {}
        page = meta.get("page_label")
        text = node.get_content().strip()
        if self.settings.injection_defense == "full":
            text = sanitize(text).text  # the agent never sees sentences addressing the AI
        if len(text) > PASSAGE_MAX_CHARS:
            text = text[: PASSAGE_MAX_CHARS - 1].rstrip() + "…"
        return Passage(
            id=self.passage_id(node),
            file=meta.get("source") or meta.get("file_name") or "unknown",
            page=str(page) if page else None,
            section=meta.get("section") or None,
            score=round(score, 4) if score is not None else None,
            text=text,
        )

    def _node_by_passage_id(self, passage_id: str) -> BaseNode:
        self.ensure_ready()
        assert self._index is not None
        key = passage_id.strip().lstrip("#").replace("-", "").lower()
        if len(key) < 6:
            raise ValueError("Mã đoạn quá ngắn.")
        matches = [
            node
            for node_id, node in self._index.docstore.docs.items()
            if node_id.replace("-", "").lower().startswith(key)
        ]
        if not matches:
            raise KeyError(passage_id)
        if len(matches) > 1:
            raise ValueError(f"Mã đoạn '{passage_id}' không duy nhất.")
        return matches[0]

    def get_passages(self, passage_ids: list[str]) -> tuple[list[Passage], list[str]]:
        """Resolve ids to passages; returns (found, unknown_ids)."""
        found, unknown = [], []
        for passage_id in passage_ids:
            try:
                found.append(self._to_passage(self._node_by_passage_id(passage_id)))
            except (KeyError, ValueError):
                unknown.append(passage_id)
        return found, unknown

    def search_passages(
        self, query: str, *, source: str | None = None, top_k: int | None = None
    ) -> PassageSearch:
        """Retrieve evidence passages (no generation). ``source`` limits the search to one file."""
        query = (query or "").strip()
        if not query:
            raise ValueError("Câu truy vấn không được để trống.")
        source = (source or "").strip() or None
        if source:
            known = {d.source for d in self.list_documents()}
            if source not in known:
                return PassageSearch(
                    status="error",
                    reason=f"Không có tài liệu '{source}'. Tài liệu hiện có: "
                    + ", ".join(sorted(known)),
                )

        result = self.retrieve(query, top_k=top_k, source=source)
        if not result.chunks:
            return PassageSearch(status="not_found", reason="no_documents_matched")
        if (
            self.settings.retrieval_mode != "bm25"
            and result.best_vector_score is not None
            and result.best_vector_score < self.settings.min_similarity
        ):
            return PassageSearch(status="not_found", reason="low_similarity")
        return PassageSearch(
            status="ok", passages=[self._to_passage(c.node, c.score) for c in result.chunks]
        )

    def _neighbor(self, node: BaseNode, step: int) -> BaseNode | None:
        """Previous (-1) or next (+1) chunk, crossing into the adjacent page/part if needed."""
        assert self._index is not None
        docstore = self._index.docstore
        related = node.prev_node if step < 0 else node.next_node
        if related is not None and related.node_id in docstore.docs:
            return docstore.get_node(related.node_id)
        source, _, part = (node.ref_doc_id or "").rpartition("::")
        if not source or not part.isdigit():
            return None
        adjacent = self._index.ref_doc_info.get(f"{source}::{int(part) + step}")
        if adjacent is None or not adjacent.node_ids:
            return None
        node_id = adjacent.node_ids[-1] if step < 0 else adjacent.node_ids[0]
        return docstore.get_node(node_id) if node_id in docstore.docs else None

    def read_context(self, passage_id: str, window: int = 1) -> list[Passage]:
        """The passage plus up to ``window`` neighbouring chunks on each side, in reading order."""
        node = self._node_by_passage_id(passage_id)
        before: list[BaseNode] = []
        cursor: BaseNode | None = node
        for _ in range(window):
            cursor = self._neighbor(cursor, -1) if cursor else None
            if cursor is None:
                break
            before.insert(0, cursor)
        after: list[BaseNode] = []
        cursor = node
        for _ in range(window):
            cursor = self._neighbor(cursor, +1) if cursor else None
            if cursor is None:
                break
            after.append(cursor)
        return [self._to_passage(n) for n in [*before, node, *after]]

    def check_answer(self, answer: str) -> AnswerCheck:
        """Verify a draft answer: citations must exist and the cited passages must support it."""
        ids = cited_passage_ids(answer)
        passages, unknown = self.get_passages(ids)
        uncited = uncited_sentences(answer)
        if not passages:
            return AnswerCheck(
                status="unsupported",
                cited_ids=ids,
                unknown_ids=unknown,
                uncited_sentences=uncited,
                unsupported_claims=["Câu trả lời không trích dẫn đoạn tài liệu hợp lệ nào."],
            )

        prompt = VERIFY_PROMPT.format(
            context=format_passages_for_check(passages), answer=answer.strip()
        )
        verdict = parse_verdict(self.llm.complete(prompt).text)
        if verdict is None:
            return AnswerCheck(
                status="unknown", cited_ids=ids, unknown_ids=unknown, uncited_sentences=uncited
            )
        claims = [str(c) for c in verdict.get("unsupported_claims") or [] if str(c).strip()]
        supported = bool(verdict.get("supported")) and not claims and not unknown
        return AnswerCheck(
            status="ok" if supported else "unsupported",
            cited_ids=ids,
            unknown_ids=unknown,
            uncited_sentences=uncited,
            unsupported_claims=claims,
        )

    def _not_found(self, reason: str, latency: dict[str, float]) -> QueryResult:
        return QueryResult(
            status="not_found", answer=NOT_FOUND_MESSAGE, reason=reason, latency_ms=latency
        )

    def query(self, question: str) -> QueryResult:
        question = (question or "").strip()
        if not question:
            raise ValueError("Câu hỏi không được để trống.")

        self.ensure_ready()  # one-off load/sync cost is not part of the query latency
        total_start = time.perf_counter()
        retrieval_start = time.perf_counter()
        result = self.retrieve(question)
        latency = {"retrieval": _elapsed_ms(retrieval_start)}

        if not result.chunks:
            latency["total"] = _elapsed_ms(total_start)
            return self._not_found("no_documents_matched", latency)

        # Cheap pre-filter: if even the best semantic match is weak, skip the LLM call.
        mode = self.settings.retrieval_mode
        if (
            mode != "bm25"
            and result.best_vector_score is not None
            and result.best_vector_score < self.settings.min_similarity
        ):
            latency["total"] = _elapsed_ms(total_start)
            return self._not_found("low_similarity", latency)

        generation_start = time.perf_counter()
        defense = self.settings.injection_defense
        prompt = build_qa_prompt(question, result.chunks, defense=defense)
        answer = self.llm.complete(prompt).text.strip()
        latency["generation"] = _elapsed_ms(generation_start)
        latency["total"] = _elapsed_ms(total_start)

        if not answer or is_not_found_answer(answer):
            return self._not_found("answer_not_in_context", latency)
        if defense != "off":
            visible = [excerpt_text(c, defense) for c in result.chunks]
            answer, _ = strip_unknown_links(answer, visible)
            if leaks_instructions(answer, [QA_PROMPT.template.replace(NOT_FOUND_MESSAGE, "")]):
                logger.warning("Answer withheld: it reproduces the system instructions")
                return self._not_found("injection_guard", latency)

        cited = extract_citations(answer, max_id=len(result.chunks))
        sources = [
            Source(
                id=number,
                file=chunk.source,
                page=chunk.page,
                section=chunk.section,
                score=chunk.score,
                vector_score=round(chunk.vector_score, 4)
                if chunk.vector_score is not None
                else None,
                bm25_score=round(chunk.bm25_score, 4) if chunk.bm25_score is not None else None,
                cited=number in cited,
                snippet=snippet(chunk.text),
            )
            for number, chunk in enumerate(result.chunks, start=1)
        ]
        return QueryResult(status="ok", answer=answer, sources=sources, latency_ms=latency)

    # -------------------------------------------------------------------- info
    def list_documents(self) -> list[DocumentInfo]:
        self.ensure_ready()
        assert self._index is not None
        grouped: dict[str, DocumentInfo] = {}
        for ref_id, info in self._index.ref_doc_info.items():
            source = (info.metadata or {}).get("source") or ref_id.rsplit("::", 1)[0]
            entry = grouped.setdefault(source, DocumentInfo(source=source, parts=0, chunks=0))
            entry.parts += 1
            entry.chunks += len(info.node_ids)
        return sorted(grouped.values(), key=lambda d: d.source)

    def stats(self) -> IndexStats:
        self.ensure_ready()
        manifest = self._manifest or {}
        return IndexStats(
            index_version=manifest.get("version"),
            created_at=manifest.get("created_at"),
            documents=len(self.list_documents()),
            chunks=self._chunk_count(),
            retrieval_mode=self.settings.retrieval_mode,
            vector_store=self.settings.vector_store,
            embed_model=f"{self.settings.embed_provider}/{self.settings.embed_model}",
            llm_model=f"{self.settings.llm_provider}/{self.settings.llm_model}",
        )


@lru_cache(maxsize=1)
def get_engine() -> DocQAEngine:
    """Process-wide engine shared by the ADK tools and the API."""
    return DocQAEngine()
