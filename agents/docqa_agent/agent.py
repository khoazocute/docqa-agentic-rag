"""ADK entrypoint. Run with `adk web agents` or `adk run agents/docqa_agent`."""

from docqa.agent import build_agent

root_agent = build_agent()
