"""Agentic RAG tests: scripted models drive the real ADK runtime, which executes the real tools."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable

import pytest
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types

import docqa.agent as agent_module
from docqa.chat import LIMIT_MESSAGE, ChatService, renumber_citations

Policy = Callable[[LlmRequest, list[dict]], types.Part]


def call(name: str, **args) -> types.Part:
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


def text(value: str) -> types.Part:
    return types.Part(text=value)


class ScriptedLlm(BaseLlm):
    """A fake model driven by a policy: (request, tool results so far this turn) -> next part."""

    model: str = "scripted"
    policy: Policy
    requests: list[LlmRequest] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        results = []
        for content in reversed(llm_request.contents):
            responses = [p.function_response for p in content.parts or [] if p.function_response]
            if not responses and content.role == "user":
                break  # start of the current user turn
            results[:0] = [{"name": r.name, **(r.response or {})} for r in responses]
        part = self.policy(llm_request, results)
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def user_message(request: LlmRequest) -> str:
    for content in reversed(request.contents):
        if content.role == "user" and content.parts and content.parts[0].text:
            return content.parts[0].text
    return ""


def search_then_verify(request: LlmRequest, results: list[dict]) -> types.Part:
    """search -> draft citing the first passage -> check_answer -> final answer."""
    if not results:
        return call("search_passages", query=user_message(request))
    last = results[-1]
    if last["name"] == "search_passages":
        passage = last["passages"][0]
        line = passage["text"].strip().splitlines()[-1]
        return call("check_answer", answer=f"{line} [#{passage['id']}]")
    if last["name"] == "check_answer":
        draft = next(r for r in results if r["name"] == "search_passages")["passages"][0]
        return text(f"{draft['text'].strip().splitlines()[-1]} [#{draft['id']}]")
    raise AssertionError(results)


class FakeToolContext:
    """Minimal stand-in for ADK's ToolContext when a tool is called directly."""

    def __init__(self) -> None:
        self.state: dict = {}


@pytest.fixture
def use_engine(monkeypatch, engine):
    monkeypatch.setattr(agent_module, "get_engine", lambda: engine)
    return engine


def service(settings, policy: Policy, **overrides) -> tuple[ChatService, ScriptedLlm]:
    llm = ScriptedLlm(policy=policy, requests=[])
    s = settings.with_overrides(**overrides)
    return ChatService(agent=agent_module.build_agent(s, model=llm), settings=s), llm


# ---------------------------------------------------------------------------- tools


def test_agent_exposes_agentic_tools():
    agent = agent_module.build_agent(model=ScriptedLlm(policy=search_then_verify))
    assert [t.__name__ for t in agent.tools] == [
        "search_passages",
        "read_context",
        "check_answer",
        "list_documents",
        "sync_documents",
    ]
    assert "check_answer" in agent.instruction and "[#id]" in agent.instruction


def test_search_tool_returns_evidence_not_answers(use_engine):
    result = asyncio.run(agent_module.search_passages("Mật khẩu phải có ít nhất bao nhiêu ký tự?"))
    assert result["status"] == "ok"
    top = result["passages"][0]
    assert top["file"] == "bao_mat.md" and "12 ký tự" in top["text"] and len(top["id"]) == 8
    assert "answer" not in result


def test_search_tool_can_be_restricted_to_one_document(use_engine):
    scoped = asyncio.run(agent_module.search_passages("thời gian", document="quy_dinh.txt"))
    assert {p["file"] for p in scoped["passages"]} == {"quy_dinh.txt"}

    unknown = asyncio.run(agent_module.search_passages("thời gian", document="khong_co.md"))
    assert unknown["status"] == "error" and "bao_mat.md" in unknown["reason"]


def test_read_context_and_check_answer_tools(use_engine):
    pid = asyncio.run(agent_module.search_passages("mật khẩu 12 ký tự"))["passages"][0]["id"]

    context = asyncio.run(agent_module.read_context(pid))
    assert context["status"] == "ok" and pid in [p["id"] for p in context["passages"]]
    assert asyncio.run(agent_module.read_context("deadbeef"))["status"] == "error"

    good = asyncio.run(
        agent_module.check_answer(f"Mật khẩu tối thiểu 12 ký tự [#{pid}].", FakeToolContext())
    )
    bad = asyncio.run(
        agent_module.check_answer(f"Mật khẩu tối thiểu 16 ký tự [#{pid}].", FakeToolContext())
    )
    assert good["status"] == "ok"
    assert bad["status"] == "unsupported" and bad["unsupported_claims"]


def test_tool_errors_are_reported_not_raised(monkeypatch):
    def broken():
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(agent_module, "get_engine", broken)
    result = asyncio.run(agent_module.search_passages("bất kỳ"))
    assert result["status"] == "error" and result["error"] == "RuntimeError"


# ------------------------------------------------------------------ agent loop (ADK)


def test_agent_searches_verifies_and_cites(use_engine, settings):
    chat, llm = service(settings, search_then_verify)

    reply = asyncio.run(chat.chat("Mật khẩu phải có ít nhất bao nhiêu ký tự?"))

    assert [s.tool for s in reply.steps] == ["search_passages", "check_answer"]
    assert reply.steps[0].args["query"] == "Mật khẩu phải có ít nhất bao nhiêu ký tự?"
    assert reply.steps[0].result.startswith("ok:") and reply.steps[1].result == "ok"
    assert reply.answer.endswith("[1]") and "[#" not in reply.answer
    assert reply.sources[0].file == "bao_mat.md" and reply.sources[0].cited
    assert reply.llm_calls == 3


def test_agent_researches_with_document_filter_and_context(use_engine, settings):
    def policy(request: LlmRequest, results: list[dict]) -> types.Part:
        if not results:
            return call("search_passages", query="lương thưởng cuối năm")  # poor first query
        last = results[-1]
        if last["name"] == "search_passages" and last["status"] != "ok":
            return call("search_passages", query="báo cáo sự cố bảo mật", document="bao_mat.md")
        if last["name"] == "search_passages":
            return call("read_context", passage_id=last["passages"][0]["id"])
        pid = last["passages"][0]["id"]
        return text(f"Báo cáo sự cố trong vòng 24 giờ [#{pid}].")

    chat, _ = service(settings, policy)
    reply = asyncio.run(chat.chat("Sự cố bảo mật phải báo cáo trong bao lâu?"))

    assert [s.tool for s in reply.steps] == ["search_passages", "search_passages", "read_context"]
    assert reply.steps[0].result.startswith("not_found")
    assert reply.steps[1].args["document"] == "bao_mat.md"
    assert reply.answer == "Báo cáo sự cố trong vòng 24 giờ [1]."
    assert reply.sources[0].file == "bao_mat.md"


def test_llm_call_limit_stops_runaway_loops(use_engine, settings):
    def endless(request: LlmRequest, results: list[dict]) -> types.Part:
        return call("search_passages", query=f"mật khẩu lần {len(results)}")

    chat, llm = service(settings, endless, agent_max_llm_calls=3)
    reply = asyncio.run(chat.chat("Mật khẩu?"))

    assert reply.answer == LIMIT_MESSAGE
    assert len(llm.requests) == 3


def test_conversation_history_is_kept_per_session(use_engine, settings):
    chat, llm = service(settings, search_then_verify)
    first = asyncio.run(chat.chat("Mật khẩu phải có ít nhất bao nhiêu ký tự?"))
    asyncio.run(chat.chat("Báo cáo sự cố bảo mật trong bao lâu?", session_id=first.session_id))

    # The second turn sees the whole first turn (question, tool calls, results, answer).
    assert len(llm.requests[-1].contents) > len(llm.requests[2].contents)


def test_tools_work_with_llms_that_call_asyncio_run(use_engine, settings, fake_llm, monkeypatch):
    """Regression: LlamaIndex's sync Gemini client calls asyncio.run() inside complete().

    That raises "asyncio.run() cannot be called from a running event loop" if a tool runs the
    engine directly on ADK's event loop. Tools must offload the work to a thread.
    """
    original = type(fake_llm).complete

    def complete_like_gemini(self, prompt, formatted=False, **kwargs):
        asyncio.run(asyncio.sleep(0))
        return original(self, prompt, formatted, **kwargs)

    monkeypatch.setattr(type(fake_llm), "complete", complete_like_gemini)
    chat, _ = service(settings, search_then_verify)

    reply = asyncio.run(chat.chat("Mật khẩu phải có ít nhất bao nhiêu ký tự?"))

    assert reply.steps[1].result == "ok"  # check_answer ran the (patched) LLM successfully
    assert reply.answer.endswith("[1]") and reply.sources[0].file == "bao_mat.md"


# --------------------------------------------------------------------- citations


def test_renumber_citations_maps_ids_and_drops_unknown_ones():
    passages = {
        "aaaa1111": {"id": "aaaa1111", "file": "a.md", "text": "A"},
        "bbbb2222": {"id": "bbbb2222", "file": "b.md", "page": "3", "text": "B"},
    }
    answer = "Ý một [#bbbb2222]. Ý hai [#aaaa1111] [#aaaa1111]. Bịa [#deadbeef]. Lại [#BBBB2222]."

    text_out, sources = renumber_citations(answer, passages)

    assert text_out == "Ý một [1]. Ý hai [2]. Bịa. Lại [1]."
    assert [(s.id, s.file, s.page) for s in sources] == [(1, "b.md", "3"), (2, "a.md", None)]


def test_grouped_citations_are_split_and_renumbered():
    passages = {
        "aaaa1111": {"id": "aaaa1111", "file": "a.md", "text": "A"},
        "bbbb2222": {"id": "bbbb2222", "file": "b.md", "text": "B"},
    }
    text_out, sources = renumber_citations(
        "Không đề cập [#aaaa1111, #bbbb2222; cccc3333].", passages
    )
    assert text_out == "Không đề cập [1][2]."
    assert [s.file for s in sources] == ["a.md", "b.md"]


def test_uncited_answer_after_search_becomes_standard_refusal(use_engine, settings):
    """Grounding guardrail: no valid citation after searching means no claim."""
    from docqa.config import NOT_FOUND_MESSAGE

    def paraphrased_refusal(request: LlmRequest, results: list[dict]) -> types.Part:
        if not results:
            return call("search_passages", query="thưởng Tết mấy tháng lương")
        return text("Tài liệu hiện tại không đề cập đến mức thưởng Tết.")

    chat, _ = service(settings, paraphrased_refusal)
    reply = asyncio.run(chat.chat("Thưởng Tết năm nay là mấy tháng lương?"))

    assert reply.answer == NOT_FOUND_MESSAGE and reply.sources == []


def test_answers_without_search_are_left_alone(use_engine, settings):
    def greet(request: LlmRequest, results: list[dict]) -> types.Part:
        return text("Xin chào! Tôi có thể giúp gì về tài liệu nội bộ?")

    chat, _ = service(settings, greet)
    assert asyncio.run(chat.chat("Xin chào")).answer.startswith("Xin chào!")


def test_check_answer_runs_at_most_once_per_turn(use_engine, settings):
    """Small models re-check in a loop; the limit is enforced in code, per user turn."""

    def recheck_twice(request: LlmRequest, results: list[dict]) -> types.Part:
        if not results:
            return call("search_passages", query="mật khẩu bao nhiêu ký tự")
        pid = results[0]["passages"][0]["id"]
        if len([r for r in results if r["name"] == "check_answer"]) < 2:
            return call("check_answer", answer=f"Mật khẩu tối thiểu 12 ký tự [#{pid}].")
        return text(f"Mật khẩu tối thiểu 12 ký tự [#{pid}].")

    chat, _ = service(settings, recheck_twice)
    first = asyncio.run(chat.chat("Mật khẩu?"))
    assert [s.result.split(":")[0] for s in first.steps] == ["ok", "ok", "skipped"]

    second = asyncio.run(chat.chat("Mật khẩu?", session_id=first.session_id))
    assert second.steps[1].result == "ok"  # a new turn may check again


def test_citations_without_hash_are_accepted():
    passages = {"ac7acd25": {"id": "ac7acd25", "file": "so_tay.md", "text": "A"}}
    text_out, sources = renumber_citations(
        "Đã qua thử việc mới được làm từ xa [ac7acd25].", passages
    )
    assert text_out == "Đã qua thử việc mới được làm từ xa [1]." and sources[0].file == "so_tay.md"


def test_fake_judge_ignores_passage_ids():
    """Regression: a random passage id containing "16" made the fake judge accept "16 ký tự"."""
    from tests.conftest import FakeLLM

    prompt = (
        "KIỂM TRA câu trả lời\nĐoạn trích:\n[#a16f3c2e] (nguồn: bao_mat.md)\n"
        "Mật khẩu tối thiểu 12 ký tự.\nCâu trả lời nháp:\n"
        "Mật khẩu tối thiểu 16 ký tự [#a16f3c2e].\nChỉ trả về JSON"
    )
    assert '"supported": false' in FakeLLM._verify(prompt)
