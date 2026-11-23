# Guide for coding agents working on this repo

## Layout

- `src/docqa/`: installable package (`pip install -e ".[dev]"`). All logic lives here.
- `agents/docqa_agent/agent.py`: ADK entrypoint only (`root_agent = build_agent()`).
- `ui/streamlit_app.py`: UI; talks to the REST API only.
- `tests/`: offline tests; fakes live in `tests/conftest.py`.

## Rules

- Keep ADK tools thin: they call `get_engine()` and return plain dicts with a `status` key and never raise.
- Any change to ingestion must keep these invariants (covered by `tests/test_engine.py`):
  network calls happen before index mutation; writes go to a new version folder and then
  switch `CURRENT`; settings that affect vectors belong in `Settings.index_fingerprint()`.
- Do not add hard-coded keyword rules for specific questions. Improve retrieval and prompts and
  measure the effect with `docqa eval`.
- Never commit `.env`, `storage/`, personal files in `data/` or `.adk/` session databases.

## Checks before finishing

```bash
ruff check . && ruff format --check .
pytest
```
