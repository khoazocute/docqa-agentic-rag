"""Versioned, crash-safe persistence for the LlamaIndex index.

Layout::

    storage/
      CURRENT                 # name of the active version (switched atomically)
      versions/<version>/     # docstore.json, index_store.json, vector store, manifest.json
      qdrant/                 # only for VECTOR_STORE=qdrant in embedded mode (no QDRANT_URL)

Every write goes to a brand-new version folder and only then is ``CURRENT`` switched, so a
failure half-way through a rebuild never corrupts the index that is serving queries. Other
processes (API, ADK web, CLI) notice the new version and reload it.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.embeddings.base import BaseEmbedding

from docqa.config import Settings

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
KEEP_VERSIONS = 2  # current + one previous, for manual rollback
COLLECTION_PREFIX = "docqa_"

# Embedded Qdrant locks its folder, so one client per location must be shared per process.
_QDRANT_CLIENTS: dict[str, Any] = {}
_QDRANT_LOCK = threading.Lock()


@atexit.register
def _close_qdrant_clients() -> None:
    """Close clients before interpreter shutdown (embedded mode releases its folder lock)."""
    with _QDRANT_LOCK:
        for client in _QDRANT_CLIENTS.values():
            with contextlib.suppress(Exception):  # best effort at exit
                client.close()
        _QDRANT_CLIENTS.clear()


def qdrant_client_for(settings: Settings, local_path: Path):
    """Return a process-wide Qdrant client: server mode if QDRANT_URL is set, else embedded."""
    from qdrant_client import QdrantClient

    key = settings.qdrant_url or str(local_path.resolve())
    with _QDRANT_LOCK:
        if key not in _QDRANT_CLIENTS:
            if settings.qdrant_url:
                _QDRANT_CLIENTS[key] = QdrantClient(
                    url=settings.qdrant_url, api_key=settings.qdrant_api_key
                )
            else:
                local_path.mkdir(parents=True, exist_ok=True)
                _QDRANT_CLIENTS[key] = QdrantClient(path=str(local_path))
        return _QDRANT_CLIENTS[key]


@dataclass
class LoadedIndex:
    index: VectorStoreIndex
    manifest: dict[str, Any]


class IndexStorage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.storage_dir
        self.versions_dir = self.root / "versions"
        self.current_file = self.root / "CURRENT"
        self.qdrant_dir = self.root / "qdrant"

    # ------------------------------------------------------------------ versions
    def current_version(self) -> str | None:
        try:
            version = self.current_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        return version if version and (self.versions_dir / version).is_dir() else None

    def read_manifest(self, version: str | None = None) -> dict[str, Any] | None:
        version = version or self.current_version()
        if not version:
            return None
        try:
            return json.loads((self.versions_dir / version / MANIFEST_FILE).read_text("utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def is_compatible(self, manifest: dict[str, Any]) -> bool:
        return manifest.get("fingerprint") == self.settings.index_fingerprint()

    # -------------------------------------------------------------- vector store
    @property
    def uses_qdrant(self) -> bool:
        return self.settings.vector_store == "qdrant"

    def _qdrant(self):
        return qdrant_client_for(self.settings, self.qdrant_dir)

    def _vector_store(self, collection: str | None):
        if not self.uses_qdrant:
            return None
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        # The collection is created on first insert, with the embedding size and cosine distance.
        return QdrantVectorStore(collection_name=collection, client=self._qdrant())

    def new_storage_context(self) -> tuple[StorageContext, str | None]:
        """Empty storage context for a full rebuild (a new Qdrant collection if needed)."""
        collection = f"{COLLECTION_PREFIX}{uuid.uuid4().hex[:12]}" if self.uses_qdrant else None
        vector_store = self._vector_store(collection)
        context = (
            StorageContext.from_defaults(vector_store=vector_store)
            if vector_store is not None
            else StorageContext.from_defaults()
        )
        return context, collection

    def discard_collection(self, collection: str | None) -> None:
        if not collection or not self.uses_qdrant:
            return
        try:
            if self._qdrant().collection_exists(collection):
                self._qdrant().delete_collection(collection)
        except Exception:  # pragma: no cover - best effort cleanup
            logger.warning("Could not delete Qdrant collection %s", collection, exc_info=True)

    # ------------------------------------------------------------------ load/save
    def load(self, embed_model: BaseEmbedding) -> LoadedIndex | None:
        version = self.current_version()
        manifest = self.read_manifest(version)
        if not version or manifest is None:
            return None
        if not self.is_compatible(manifest):
            logger.warning(
                "Index %s was built with %s, current settings are %s: rebuild required",
                version,
                manifest.get("fingerprint"),
                self.settings.index_fingerprint(),
            )
            return None

        collection = manifest.get("collection")
        if (
            self.uses_qdrant
            and manifest.get("chunks")
            and not (collection and self._qdrant().collection_exists(collection))
        ):
            # e.g. QDRANT_URL now points to another server, or the collection was dropped.
            logger.warning("Qdrant collection %s is missing: rebuild required", collection)
            return None

        persist_dir = str(self.versions_dir / version)
        vector_store = self._vector_store(collection)
        context = (
            StorageContext.from_defaults(persist_dir=persist_dir, vector_store=vector_store)
            if vector_store is not None
            else StorageContext.from_defaults(persist_dir=persist_dir)
        )
        index = load_index_from_storage(context, embed_model=embed_model, store_nodes_override=True)
        return LoadedIndex(index=index, manifest=manifest)

    def commit(
        self, index: VectorStoreIndex, *, collection: str | None, documents: int, chunks: int
    ) -> dict[str, Any]:
        """Persist ``index`` as a new version and atomically make it current."""
        previous = self.current_version()
        version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        version_dir = self.versions_dir / version
        version_dir.mkdir(parents=True, exist_ok=False)

        index.storage_context.persist(persist_dir=str(version_dir))
        manifest = {
            "version": version,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "fingerprint": self.settings.index_fingerprint(),
            "collection": collection,
            "documents": documents,
            "chunks": chunks,
        }
        (version_dir / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        tmp = self.current_file.with_suffix(".tmp")
        tmp.write_text(version, encoding="utf-8")
        os.replace(tmp, self.current_file)

        self._cleanup(keep=[version, previous] if previous else [version])
        return manifest

    def _cleanup(self, keep: list[str]) -> None:
        if not self.versions_dir.exists():
            return
        versions = sorted(p.name for p in self.versions_dir.iterdir() if p.is_dir())
        keep_set = set(keep[:KEEP_VERSIONS])
        for name in versions:
            if name not in keep_set:
                shutil.rmtree(self.versions_dir / name, ignore_errors=True)

        if self.uses_qdrant:
            referenced = {m.get("collection") for v in keep_set if (m := self.read_manifest(v))}
            client = self._qdrant()
            for collection in client.get_collections().collections:
                if (
                    collection.name.startswith(COLLECTION_PREFIX)
                    and collection.name not in referenced
                ):
                    client.delete_collection(collection.name)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
