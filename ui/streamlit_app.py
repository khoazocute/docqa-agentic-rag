"""Streamlit chat UI for DocQA. Talks to the REST API (run `docqa serve` first).

streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("DOCQA_API_KEY", "")
TIMEOUT = httpx.Timeout(180.0, connect=5.0)

st.set_page_config(page_title="DocQA – Hỏi đáp tài liệu", page_icon="📄", layout="wide")


def api(method: str, path: str, **kwargs):
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    response = httpx.request(method, f"{API_URL}{path}", headers=headers, timeout=TIMEOUT, **kwargs)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text
        raise RuntimeError(f"{response.status_code}: {detail}")
    return response.json()


TOOL_LABELS = {
    "search_passages": "🔎 Tìm",
    "read_context": "📖 Đọc thêm ngữ cảnh",
    "check_answer": "✅ Kiểm tra câu trả lời",
    "list_documents": "📚 Liệt kê tài liệu",
    "sync_documents": "🔄 Đồng bộ",
}


def render_steps(steps: list[dict], llm_calls: int = 0) -> None:
    """Show how the agent reached its answer: searches, re-searches, context reads, checks."""
    if not steps:
        return
    with st.expander(f"Các bước agent đã thực hiện ({len(steps)} bước · {llm_calls} lần gọi LLM)"):
        for number, step in enumerate(steps, start=1):
            label = TOOL_LABELS.get(step["tool"], step["tool"])
            args = step.get("args") or {}
            detail = args.get("query") or args.get("passage_id") or ""
            if args.get("document"):
                detail += f"  (chỉ trong `{args['document']}`)"
            st.markdown(f"{number}. **{label}** {detail}")
            st.caption(f"↳ {step.get('result', '')}")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    cited = [s for s in sources if s.get("cited")] or sources
    with st.expander(f"Nguồn ({len(cited)})"):
        for source in cited:
            page = f" – trang {source['page']}" if source.get("page") else ""
            st.markdown(
                f"**[{source['id']}] {source['file']}{page}**  ·  score `{source['score']}`"
            )
            if source.get("section"):
                st.caption(f"📑 Mục: {source['section']}")
            st.caption(source["snippet"])


def render_sync_report(report: dict) -> None:
    parts = [
        f"thêm {len(report['added'])}",
        f"cập nhật {len(report['updated'])}",
        f"xóa {len(report['deleted'])}",
        f"không đổi {report['unchanged']}",
    ]
    st.success(f"Đã đồng bộ: {', '.join(parts)} · {report['chunks']} đoạn")
    if report.get("failed"):
        st.warning("Không đọc được: " + ", ".join(report["failed"]))


# ---------------------------------------------------------------- session state
st.session_state.setdefault("messages", [])
st.session_state.setdefault("session_id", None)

# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📄 DocQA")
    mode = st.radio(
        "Chế độ",
        ["Agent (ADK)", "RAG trực tiếp"],
        help="Agentic RAG: nhớ ngữ cảnh, tự tìm lại khi thiếu bằng chứng và tự kiểm tra câu trả lời; "
        "RAG trực tiếp trả lời từng câu độc lập.",
    )
    if st.button("🗨️ Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

    st.divider()
    try:
        stats = api("GET", "/stats")
        st.caption(
            f"Index `{stats['index_version']}` · {stats['documents']} tài liệu · "
            f"{stats['chunks']} đoạn · {stats['retrieval_mode']}"
        )
        documents = api("GET", "/documents")
        with st.expander(f"Tài liệu ({len(documents)})"):
            for doc in documents:
                st.markdown(f"- `{doc['source']}` ({doc['chunks']} đoạn)")
    except Exception as exc:  # API down or misconfigured
        st.error(f"Không kết nối được API ({API_URL}): {exc}")

    st.divider()
    uploaded = st.file_uploader("Thêm tài liệu", type=["pdf", "docx", "txt", "md"])
    if uploaded and st.button("⬆️ Tải lên & lập chỉ mục", use_container_width=True):
        with st.spinner("Đang xử lý tài liệu..."):
            try:
                report = api(
                    "POST", "/documents", files={"file": (uploaded.name, uploaded.getvalue())}
                )
                render_sync_report(report)
            except Exception as exc:
                st.error(str(exc))

    if st.button("🔄 Đồng bộ thư mục dữ liệu", use_container_width=True):
        with st.spinner("Đang đồng bộ..."):
            try:
                render_sync_report(api("POST", "/index/sync"))
            except Exception as exc:
                st.error(str(exc))

# ------------------------------------------------------------------------ chat
st.title("Hỏi đáp tài liệu nội bộ")
st.caption("Câu trả lời chỉ dựa trên tài liệu đã lập chỉ mục, kèm trích dẫn nguồn [n].")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_steps(message.get("steps", []), message.get("llm_calls", 0))
        render_sources(message.get("sources", []))

if question := st.chat_input("Đặt câu hỏi về tài liệu..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"), st.spinner("Đang tìm trong tài liệu..."):
        try:
            if mode == "Agent (ADK)":
                data = api(
                    "POST",
                    "/chat",
                    json={"message": question, "session_id": st.session_state.session_id},
                )
                st.session_state.session_id = data["session_id"]
            else:
                data = api("POST", "/query", json={"question": question})
            answer, sources = data["answer"], data.get("sources", [])
            steps, llm_calls = data.get("steps", []), data.get("llm_calls", 0)
        except Exception as exc:
            answer, sources, steps, llm_calls = f"⚠️ Lỗi: {exc}", [], [], 0
        st.markdown(answer)
        render_steps(steps, llm_calls)
        render_sources(sources)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "steps": steps,
            "llm_calls": llm_calls,
        }
    )
