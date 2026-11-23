---
name: llamaindex-document-agent
description: Maintain and extend the DocQA project (Google ADK agent + LlamaIndex hybrid RAG for Vietnamese documents). Use when changing ingestion, retrieval, prompts, the agent, the API or the evaluation set.
---

# DocQA maintenance skill

## Where things are

| Task | File |
|---|---|
| Change agent behaviour or tools | `src/docqa/agent.py` |
| Change the answer prompt or citation handling | `src/docqa/prompts.py` |
| Change retrieval (modes, fusion, BM25 setup) | `src/docqa/retrieval.py` |
| Change ingestion or sync | `src/docqa/documents.py`, `src/docqa/engine.py` |
| Change persistence | `src/docqa/storage.py` |
| Add a setting | `src/docqa/config.py` and `.env.example` and the README table |
| Add an endpoint | `src/docqa/api/app.py` and `tests/test_api.py` |

## Workflow for quality changes

1. Run a baseline: `docqa eval --out eval/results` (needs `GOOGLE_API_KEY`).
2. Make the change.
3. Re-run the evaluation and compare Hit@k, MRR, answer accuracy, correct and false rejection.
4. Add questions to `eval/dataset.jsonl` for any new failure mode. Each answerable case needs
   `source`, `evidence` (a phrase that appears verbatim in the source) and `answer_keywords`.

## Done criteria

- `ruff check .` and `pytest` pass offline.
- The README and `docs/` describe the actual behaviour.
