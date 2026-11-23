# Architecture

## Layers

| Layer | Module | Responsibility |
|---|---|---|
| Interfaces | `ui/streamlit_app.py`, `src/docqa/api/app.py`, `src/docqa/cli.py`, `agents/docqa_agent` | Chat UI, REST API, CLI, ADK developer UI entrypoint |
| Agent | `src/docqa/agent.py`, `src/docqa/chat.py` | ADK `Agent` (instruction + tools); `ChatService` runs it through `Runner` with per-session history |
| Engine | `src/docqa/engine.py` | Single facade for ingestion, retrieval and answering; thread-safe; process-wide singleton via `get_engine()` |
| Retrieval | `src/docqa/retrieval.py` | `HybridRetriever` (LlamaIndex `BaseRetriever`): vector + BM25 + Reciprocal Rank Fusion |
| Generation | `src/docqa/prompts.py` | Numbered-context prompt, refusal detection, citation parsing |
| Ingestion | `src/docqa/documents.py` | File discovery, `SimpleDirectoryReader` per file, stable ids, failure isolation |
| Persistence | `src/docqa/storage.py` | Versioned index folders, atomic `CURRENT` pointer, Qdrant collections (server or embedded), cleanup |
| Config | `src/docqa/config.py`, `src/docqa/providers.py` | Env-based settings; Gemini or Ollama model factories |

## Query path

```text
question
  └─ DocQAEngine.query
       ├─ ensure_ready()            load/reload current index version (build if missing/incompatible)
       ├─ HybridRetriever.search    vector top-8 ┐
       │                            bm25  top-8  ┴─ RRF → top-4 chunks (+ best cosine score)
       ├─ similarity gate           best cosine < MIN_SIMILARITY → not_found (no LLM call)
       ├─ LLM.complete(QA_PROMPT)   numbered excerpts, cite [n], refuse when unsupported
       └─ post-process              refusal → not_found; [n] → sources[].cited; latency per stage
```

## Agent path (agentic RAG)

```text
user message ─ ChatService.chat(session_id)          RunConfig(max_llm_calls=AGENT_MAX_LLM_CALLS)
  └─ ADK Runner (InMemorySessionService keeps the conversation)
       └─ agent (Gemini), ReAct-style loop driven by the instruction:
            ├─ plan: standalone query; split multi-part questions into sub-questions
            ├─ search_passages(query, document?) → DocQAEngine.search_passages → evidence + ids
            │     (hybrid retrieval, optional per-document filter, similarity gate, no generation)
            ├─ judge evidence → re-search (≤ 3 per sub-question) or read_context(passage_id)
            │     read_context → DocQAEngine.read_context → neighbouring chunks (PREVIOUS/NEXT,
            │     crossing into the adjacent page when needed)
            ├─ draft with [#id] citations → check_answer(draft) → DocQAEngine.check_answer
            │     (unknown ids + uncited sentences checked in code; unsupported claims by an LLM judge)
            └─ list_documents() / sync_documents()
  ChatService post-processing:
     steps        ← every function_call / function_response pair (tool, args, result summary)
     llm_calls    ← model events in the turn
     answer       ← [#id] renumbered to [1], [2]; ids never retrieved are dropped
     sources      ← the cited passages
```

The tools are `async` and run the engine in a worker thread (`asyncio.to_thread`): this keeps the event loop free and avoids LlamaIndex's sync Gemini client calling `asyncio.run()` inside ADK's running loop. Tools return plain dicts and never raise; errors become `{"status": "error"}` so the agent can react instead of crashing the turn.

## Ingestion and persistence

```text
data/** ─ discover_files ─ SimpleDirectoryReader (one file at a time)
        └─ Document(id="<relative path>::<part>", metadata={source, file_name, page_label})
sync():
  known   = index.ref_doc_info          (documents currently indexed)
  added   = ids not in known
  updated = ids whose content hash changed
  deleted = known ids missing on disk (except files that failed to parse)
  1. split + embed added/updated chunks     ← network calls happen BEFORE any mutation
  2. delete updated/deleted ref docs, insert new nodes, store hashes
  3. commit: persist to storage/versions/<new>, write manifest, atomically switch CURRENT
  on failure: drop in-memory index → next call reloads the last committed version
```

`manifest.json` stores the *index fingerprint* (embedding provider and model, chunk size and overlap, vector store). If the current settings differ from it, the engine rebuilds instead of mixing incompatible vectors.

## Concurrency model

- One `RLock` per engine serializes ingestion (sync, rebuild, upload, delete).
- Queries take a snapshot of the current retriever object; a sync swaps the retriever atomically after commit.
- Several processes (API, `adk web`, CLI) can share `storage/`. Each process compares the `CURRENT` version with what it has loaded and reloads when it changes.

## Testing strategy

All tests run offline:

- `FakeEmbedding` makes hashed bag-of-words vectors, so semantically related Vietnamese texts really are closer.
- `FakeLLM` answers from the best-matching numbered excerpt or refuses.
- `ScriptedLlm` is an ADK `BaseLlm` that issues a real function call. This exercises the real ADK `Runner`, tool execution and session history.
