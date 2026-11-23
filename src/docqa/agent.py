"""Agentic RAG with Google ADK.

The agent drives retrieval itself (ReAct style): it plans sub-questions, searches for evidence,
judges whether the evidence is sufficient, re-searches with a rephrased query or within one
document, expands context around a passage, drafts a cited answer and verifies it before replying.
The tools are thin wrappers around the DocQA engine and return evidence, never final answers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from google.adk import Agent
from google.adk.tools import ToolContext

from docqa.config import NOT_FOUND_MESSAGE, Settings
from docqa.engine import get_engine

logger = logging.getLogger(__name__)

AGENT_NAME = "docqa_agent"
MAX_SEARCHES_PER_SUBQUESTION = 3
CHECKS_STATE_KEY = "temp:answer_checks"  # temp: state lives for one invocation only

INSTRUCTION = f"""Bạn là trợ lý hỏi đáp tài liệu nội bộ (agentic RAG), luôn trả lời bằng tiếng Việt.
Bạn chỉ được trả lời dựa trên các đoạn tài liệu do công cụ trả về, không dùng kiến thức riêng,
kể cả khi bạn biết câu trả lời.

QUY TRÌNH cho mỗi câu hỏi mang tính thông tin (quy định, chính sách, khái niệm, số liệu...):
1. LẬP KẾ HOẠCH: viết lại câu hỏi thành câu đầy đủ, độc lập theo ngữ cảnh hội thoại. Nếu câu hỏi
   gồm nhiều ý (ví dụ "A thế nào và B bao nhiêu?"), tách thành các câu hỏi con và xử lý từng câu.
2. TÌM BẰNG CHỨNG: gọi `search_passages(query)` cho từng câu hỏi con.
3. ĐÁNH GIÁ: đọc kỹ các đoạn trả về và tự hỏi "đoạn này đã đủ để trả lời câu hỏi con chưa?".
   Nếu CHƯA ĐỦ, chọn một cách rồi tìm lại:
   - diễn đạt lại bằng từ khóa có khả năng xuất hiện trong tài liệu (từ đồng nghĩa, thuật ngữ);
   - nếu biết tài liệu liên quan (xem `list_documents`), tìm trong đúng tài liệu đó bằng
     `search_passages(query, document="<tên file>")`;
   - nếu đoạn liên quan nhưng bị cắt ngang hoặc thiếu ngữ cảnh, gọi `read_context(passage_id)`.
   Tối đa {MAX_SEARCHES_PER_SUBQUESTION} lần tìm cho mỗi câu hỏi con. Hết số lần mà vẫn không có
   bằng chứng thì coi là tài liệu không đề cập ý đó.
4. SOẠN NHÁP: mỗi ý trong câu trả lời phải có trích dẫn dạng [#id], với id là trường "id" của
   đoạn đã dùng (ví dụ: "Nhân viên được nghỉ 12 ngày phép năm [#82cbf963]."). Không bịa id.
5. KIỂM TRA: gọi `check_answer(answer=<bản nháp>)` đúng MỘT lần. Nếu kết quả có
   "unsupported_claims" hoặc "unknown_ids", sửa hoặc bỏ các ý đó. Sau đó trả lời người dùng ngay,
   không kiểm tra lại.
6. TRẢ LỜI: ngắn gọn, dùng gạch đầu dòng khi có nhiều ý, giữ nguyên các trích dẫn [#id]. Ý nào tài
   liệu không đề cập thì nói rõ là tài liệu không đề cập. Nếu không có bằng chứng cho toàn bộ câu
   hỏi, trả lời đúng một câu: "{NOT_FOUND_MESSAGE}"

Trường hợp khác:
- "X là gì" hay câu trông giống kiến thức chung vẫn phải tìm trong tài liệu trước (bước 2), vì bạn
  không biết kho tài liệu chứa gì nếu không tìm.
- Người dùng hỏi có những tài liệu nào: gọi `list_documents`.
- Người dùng báo vừa thêm/sửa/xóa tài liệu hoặc yêu cầu cập nhật dữ liệu: gọi `sync_documents`.
- Chào hỏi, cảm ơn: trả lời ngắn gọn, không cần gọi công cụ.
- Công cụ trả status "error": thông báo ngắn gọn rằng hệ thống đang gặp lỗi kỹ thuật.
"""


def _error(exc: Exception) -> dict[str, Any]:
    logger.exception("Tool call failed")
    return {
        "status": "error",
        "reason": "Hệ thống tra cứu tài liệu đang gặp lỗi kỹ thuật.",
        "error": type(exc).__name__,
    }


# The tools are async and run the engine in a worker thread. This keeps the ADK event loop
# responsive, and is required because LlamaIndex's sync Gemini client calls asyncio.run(),
# which fails inside a running event loop.


def _search_passages(query: str, document: str) -> dict[str, Any]:
    try:
        result = get_engine().search_passages(query, source=document or None)
    except ValueError as exc:
        return {"status": "error", "reason": str(exc), "passages": []}
    except Exception as exc:
        return _error(exc)
    return result.model_dump(exclude_none=True)


def _read_context(passage_id: str) -> dict[str, Any]:
    try:
        passages = get_engine().read_context(passage_id)
    except KeyError:
        return {"status": "error", "reason": f"Không tìm thấy đoạn có id '{passage_id}'."}
    except ValueError as exc:
        return {"status": "error", "reason": str(exc)}
    except Exception as exc:
        return _error(exc)
    return {"status": "ok", "passages": [p.model_dump(exclude_none=True) for p in passages]}


def _check_answer(answer: str) -> dict[str, Any]:
    try:
        return get_engine().check_answer(answer).model_dump()
    except Exception as exc:
        return _error(exc)


def _list() -> dict[str, Any]:
    try:
        documents = get_engine().list_documents()
    except Exception as exc:
        return _error(exc)
    return {"status": "ok", "documents": [d.model_dump() for d in documents]}


def _sync() -> dict[str, Any]:
    try:
        report = get_engine().sync()
    except Exception as exc:
        return _error(exc)
    return {"status": "ok", **report.model_dump(exclude={"duration_ms"})}


async def search_passages(query: str, document: str = "") -> dict[str, Any]:
    """Tìm các đoạn tài liệu liên quan để làm bằng chứng (không tự trả lời).

    Args:
        query: Câu hỏi hoặc từ khóa đầy đủ, độc lập bằng tiếng Việt.
        document: Tùy chọn. Tên file (đúng như trong list_documents) để chỉ tìm trong tài liệu đó.

    Returns:
        dict gồm status ("ok" | "not_found" | "error") và passages: danh sách đoạn với id, file,
        page, score, text. Trích dẫn một đoạn trong câu trả lời bằng [#id].
    """
    return await asyncio.to_thread(_search_passages, query, document)


async def read_context(passage_id: str) -> dict[str, Any]:
    """Đọc thêm đoạn liền trước và liền sau một đoạn đã tìm được, khi đoạn đó thiếu ngữ cảnh.

    Args:
        passage_id: Trường id của đoạn (ví dụ "82cbf963").

    Returns:
        dict gồm status và passages theo thứ tự đọc (đoạn trước, đoạn gốc, đoạn sau).
    """
    return await asyncio.to_thread(_read_context, passage_id)


async def check_answer(answer: str, tool_context: ToolContext) -> dict[str, Any]:
    """Kiểm tra câu trả lời nháp: trích dẫn [#id] có tồn tại và đoạn được trích có ủng hộ nội dung không.

    Args:
        answer: Toàn bộ câu trả lời nháp, có trích dẫn dạng [#id].

    Returns:
        dict gồm status ("ok" | "unsupported" | "unknown"), unsupported_claims (nhận định không có
        căn cứ), unknown_ids (id không tồn tại) và uncited_sentences (câu chưa có trích dẫn).
    """
    # Enforced in code, not only in the prompt: small models tend to re-check in a loop.
    checks = tool_context.state.get(CHECKS_STATE_KEY, 0)
    tool_context.state[CHECKS_STATE_KEY] = checks + 1
    if checks >= 1:
        return {
            "status": "skipped",
            "reason": "Đã kiểm tra một lần. Hãy trả lời người dùng ngay: giữ các ý có trích dẫn, "
            "bỏ các ý bị đánh dấu không có căn cứ.",
        }
    return await asyncio.to_thread(_check_answer, answer)


async def list_documents() -> dict[str, Any]:
    """Liệt kê các tài liệu đang có trong kho (tên file, số phần, số đoạn đã lập chỉ mục)."""
    return await asyncio.to_thread(_list)


async def sync_documents() -> dict[str, Any]:
    """Cập nhật chỉ mục theo thư mục tài liệu: chỉ xử lý file mới, file đã sửa và file đã xóa."""
    return await asyncio.to_thread(_sync)


TOOLS = [search_passages, read_context, check_answer, list_documents, sync_documents]


def build_agent(settings: Settings | None = None, model: Any | None = None) -> Agent:
    settings = settings or Settings.from_env()
    return Agent(
        name=AGENT_NAME,
        model=model or settings.agent_model,
        description="Trợ lý hỏi đáp tài liệu nội bộ tiếng Việt (agentic RAG trên LlamaIndex).",
        instruction=INSTRUCTION,
        tools=TOOLS,
    )
