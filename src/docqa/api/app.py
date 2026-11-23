"""REST API (FastAPI) exposing direct RAG queries, agent chat and document management."""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from docqa import __version__
from docqa.engine import DocQAEngine, get_engine
from docqa.schemas import ChatResponse, DocumentInfo, IndexStats, QueryResult, SyncReport

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=128)


def create_app(
    engine_provider: Callable[[], DocQAEngine] = get_engine,
    chat_provider: Callable[[], object] | None = None,
) -> FastAPI:
    app = FastAPI(
        title="DocQA API",
        version=__version__,
        description="Hỏi đáp tài liệu tiếng Việt: Google ADK agent + LlamaIndex hybrid RAG.",
    )
    chat_state: dict[str, object] = {}

    def engine() -> DocQAEngine:
        return engine_provider()

    def chat_service():
        if "service" not in chat_state:
            if chat_provider is not None:
                chat_state["service"] = chat_provider()
            else:
                from docqa.chat import ChatService

                chat_state["service"] = ChatService()
        return chat_state["service"]

    def require_api_key(
        x_api_key: str | None = Header(default=None), eng: DocQAEngine = Depends(engine)
    ) -> None:
        expected = eng.settings.api_key
        if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    def run(fn, *args):
        try:
            return fn(*args)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Request failed")
            raise HTTPException(status_code=500, detail="Lỗi hệ thống, xem log server.") from exc

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/stats", response_model=IndexStats, tags=["system"])
    def stats(eng: DocQAEngine = Depends(engine)):
        return run(eng.stats)

    @app.post("/query", response_model=QueryResult, tags=["qa"])
    def query(body: QueryRequest, eng: DocQAEngine = Depends(engine)):
        """Stateless RAG: retrieve, then generate a cited answer."""
        return run(eng.query, body.question)

    @app.post("/chat", response_model=ChatResponse, tags=["qa"])
    async def chat(body: ChatRequest):
        """Multi-turn conversation with the ADK agent (keeps context per session_id)."""
        try:
            return await chat_service().chat(body.message, body.session_id)
        except Exception as exc:
            logger.exception("Chat failed")
            raise HTTPException(status_code=500, detail="Lỗi hệ thống, xem log server.") from exc

    @app.get("/documents", response_model=list[DocumentInfo], tags=["documents"])
    def documents(eng: DocQAEngine = Depends(engine)):
        return run(eng.list_documents)

    @app.post(
        "/documents",
        response_model=SyncReport,
        tags=["documents"],
        dependencies=[Depends(require_api_key)],
    )
    async def upload(file: UploadFile = File(...), eng: DocQAEngine = Depends(engine)):
        limit = eng.settings.max_upload_mb * 1024 * 1024
        content = await file.read(limit + 1)  # +1 byte so oversize files are detected
        return await run_in_threadpool(run, eng.add_file, file.filename or "", content)

    @app.delete(
        "/documents/{source:path}",
        response_model=SyncReport,
        tags=["documents"],
        dependencies=[Depends(require_api_key)],
    )
    def delete(source: str, eng: DocQAEngine = Depends(engine)):
        return run(eng.remove_file, source)

    @app.post(
        "/index/sync",
        response_model=SyncReport,
        tags=["index"],
        dependencies=[Depends(require_api_key)],
    )
    def sync(eng: DocQAEngine = Depends(engine)):
        return run(eng.sync)

    @app.post(
        "/index/rebuild",
        response_model=SyncReport,
        tags=["index"],
        dependencies=[Depends(require_api_key)],
    )
    def rebuild(eng: DocQAEngine = Depends(engine)):
        return run(eng.rebuild)

    return app


app = create_app()
