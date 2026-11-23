"""Multi-turn chat over the agentic RAG agent (used by the REST API, the web UI and evaluation).

Besides running the ADK agent, this layer:
- records every tool call as a step (what the agent searched for, and what came back),
- counts LLM calls and enforces a per-turn limit (guardrail against runaway loops),
- turns the agent's passage citations ``[#82cbf963]`` into numbered citations ``[1]`` and a
  matching source list, dropping citations to passages the agent never actually retrieved.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from google.adk import Agent
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from docqa.agent import INSTRUCTION, SECURITY_NOTE, build_agent
from docqa.config import NOT_FOUND_MESSAGE, Settings
from docqa.injection import GUARD_MESSAGE, leaks_instructions, strip_unknown_links
from docqa.prompts import PASSAGE_CITATION_RE, expand_citation_groups, snippet
from docqa.schemas import AgentStep, ChatResponse, Source

logger = logging.getLogger(__name__)

APP_NAME = "docqa"
LIMIT_MESSAGE = (
    "Câu hỏi cần quá nhiều bước tra cứu nên tôi chưa thể trả lời đầy đủ. "
    "Bạn hãy thử tách thành các câu hỏi nhỏ hơn."
)
EVIDENCE_TOOLS = ("search_passages", "read_context")


def _summarize_result(tool: str, payload: dict[str, Any]) -> str:
    status = payload.get("status", "?")
    if tool in EVIDENCE_TOOLS:
        passages = payload.get("passages") or []
        files = sorted({p.get("file", "?") for p in passages})
        detail = f"{len(passages)} đoạn" + (f" từ {', '.join(files)}" if files else "")
        reason = payload.get("reason")
        return f"{status}: {detail}" + (f" ({reason})" if reason and not passages else "")
    if tool == "check_answer":
        claims = payload.get("unsupported_claims") or []
        unknown = payload.get("unknown_ids") or []
        detail = "; ".join(str(c)[:80] for c in claims[:2])
        if unknown:
            detail += ("; " if detail else "") + f"id không tồn tại: {', '.join(unknown[:3])}"
        return f"{status}" + (f": {detail}" if detail else "")
    return str(status)


def renumber_citations(
    answer: str, passages: dict[str, dict[str, Any]]
) -> tuple[str, list[Source]]:
    """Replace [#id] with [n] (order of first use) and build the matching source list."""
    numbers: dict[str, int] = {}
    sources: list[Source] = []

    def resolve(passage_id: str) -> str | None:
        key = passage_id.lower()
        return next((pid for pid in passages if pid.startswith(key) or key.startswith(pid)), None)

    def replace(match: re.Match) -> str:
        pid = resolve(match.group(1))
        if pid is None:  # cites a passage the agent never retrieved: drop it
            return ""
        if pid not in numbers:
            numbers[pid] = len(numbers) + 1
            p = passages[pid]
            sources.append(
                Source(
                    id=numbers[pid],
                    file=p.get("file", "unknown"),
                    page=p.get("page"),
                    section=p.get("section"),
                    score=p.get("score") or 0.0,
                    cited=True,
                    snippet=snippet(p.get("text", "")),
                )
            )
        return f"[{numbers[pid]}]"

    text = PASSAGE_CITATION_RE.sub(replace, expand_citation_groups(answer))
    text = re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", text)  # "[1] [1]" -> "[1]"
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    return text.strip(), sources


class ChatService:
    def __init__(self, agent: Agent | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or Settings.from_env()
        self._agent = agent or build_agent(self._settings)
        self._sessions = InMemorySessionService()
        self._runner = Runner(
            app_name=APP_NAME,
            agent=self._agent,
            session_service=self._sessions,
            auto_create_session=True,
        )

    async def chat(
        self, message: str, session_id: str | None = None, user_id: str = "web"
    ) -> ChatResponse:
        session_id = session_id or uuid.uuid4().hex
        content = types.Content(role="user", parts=[types.Part(text=message)])
        run_config = RunConfig(max_llm_calls=self._settings.agent_max_llm_calls)

        answer_parts: list[str] = []
        steps: list[AgentStep] = []
        pending: dict[str, AgentStep] = {}
        passages: dict[str, dict[str, Any]] = {}
        llm_calls = 0

        try:
            async for event in self._runner.run_async(
                user_id=user_id, session_id=session_id, new_message=content, run_config=run_config
            ):
                if event.content and event.content.role == "model" and not event.partial:
                    llm_calls += 1
                for call in event.get_function_calls():
                    step = AgentStep(tool=call.name, args=dict(call.args or {}))
                    steps.append(step)
                    pending[call.id or call.name] = step
                for response in event.get_function_responses():
                    payload = response.response or {}
                    step = pending.pop(response.id or response.name, None)
                    if step is not None:
                        step.result = _summarize_result(response.name, payload)
                    if response.name in EVIDENCE_TOOLS:
                        for passage in payload.get("passages") or []:
                            passages[str(passage.get("id", "")).lower()] = passage
                if event.is_final_response() and event.content and event.content.parts:
                    answer_parts.extend(
                        p.text for p in event.content.parts if p.text and not p.thought
                    )
        except LlmCallsLimitExceededError:
            logger.warning("Agent hit the LLM call limit (%d)", self._settings.agent_max_llm_calls)
            answer_parts = [LIMIT_MESSAGE]

        answer, sources = renumber_citations("".join(answer_parts).strip(), passages)
        searched = any(s.tool == "search_passages" for s in steps)
        if searched and not sources and answer not in (LIMIT_MESSAGE, NOT_FOUND_MESSAGE):
            # Grounding guardrail: after searching, an answer without a single valid citation is
            # either a paraphrased refusal or an unsupported claim; both become the standard refusal.
            logger.info("Uncited answer replaced by the not-found message: %r", answer[:120])
            answer = NOT_FOUND_MESSAGE
        if self._settings.injection_defense != "off" and answer not in (
            LIMIT_MESSAGE,
            NOT_FOUND_MESSAGE,
        ):
            # Output guard: no links the excerpts did not contain, no copy of the instructions.
            answer, _ = strip_unknown_links(answer, [p.get("text", "") for p in passages.values()])
            instructions = (INSTRUCTION + SECURITY_NOTE).replace(NOT_FOUND_MESSAGE, "")
            if leaks_instructions(answer, [instructions]):
                logger.warning("Agent answer withheld: it reproduces the system instructions")
                answer, sources = GUARD_MESSAGE, []
        for step in steps:  # keep long arguments (draft answers) readable in the UI and logs
            step.args = {
                k: (v if len(json.dumps(v, ensure_ascii=False)) < 300 else "…")
                for k, v in step.args.items()
            }
        return ChatResponse(
            session_id=session_id,
            answer=answer,
            sources=sources,
            tool_calls=[s.tool for s in steps],
            steps=steps,
            llm_calls=llm_calls,
        )
