# Design decisions

Short records of the main choices, the alternatives considered and the trade-offs.

## 1. Agent (ADK) on top of a RAG engine, not RAG inside the prompt

The agent owns the conversation: memory, follow-up rewriting, choosing tools. The engine owns grounding: retrieval, the not-found policy and citations. This split means:

- The RAG engine is usable without the agent (`/query`, CLI, evaluation), so its quality can be measured in isolation.
- The not-found behaviour is enforced in code (similarity gate and refusal detection), not only by the agent prompt.
- Cost: an agent turn uses two LLM calls (agent and answer generation). `/query` is the single-call path.

## 2. Hybrid retrieval with Reciprocal Rank Fusion

Policy documents are full of exact tokens: amounts (`1.200.000 đồng`), times (`8:30`), codes (`P1`, `MFA`). Dense embeddings capture paraphrases but can miss exact tokens. BM25 does the opposite. RRF combines the two rankings without calibrating their incompatible score scales:

```
score(d) = Σ 1 / (k + rank_i(d)),  k = 60
```

BM25 is configured for Vietnamese: no English stemmer, plus a small Vietnamese stop-word list. `docqa eval` compares the `vector`, `bm25` and `hybrid` modes on the same questions, so the choice is backed by data rather than assumed.

## 3. "Not found" is a first-class result

The original prototype used a hand-written keyword-overlap heuristic with a special case for one test question. It was replaced with two general mechanisms:

1. A **similarity gate**: if the best cosine score is below `MIN_SIMILARITY`, return `not_found` without calling the LLM, which saves cost and latency. The threshold is calibrated from the evaluation set (`calibration` section of the report). The default is deliberately conservative, because a false rejection is worse than an extra LLM call.
2. A **strict prompt and refusal detection**: the LLM must answer with the exact not-found sentence when the excerpts do not support an answer, and the engine converts that into `status="not_found"`.

The evaluator tracks both directions: the correct-rejection rate (unanswerable questions) and the false-rejection rate (answerable questions).

## 4. Incremental sync with content hashes

Re-embedding a whole corpus on every change is slow and costs money. Each document part has a stable id (`relative/path::part`) and a hash of its text plus content metadata. Volatile metadata such as modification times is dropped, so touching a file does not trigger re-embedding. A sync embeds only new or changed parts and deletes removed ones. Files that fail to parse are reported and keep their old version, instead of being treated as deleted.

## 5. Crash-safe, versioned persistence

Writes never modify the serving index in place on disk. Every commit goes to `storage/versions/<new>/`, then `CURRENT` is replaced atomically (`os.replace`). Embedding calls, the most failure-prone step, happen before any in-memory mutation. If anything fails, the in-memory index is dropped and reloaded from the last committed version. The previous version is kept for manual rollback.

The same `CURRENT` pointer lets several processes share one storage folder and hot-reload after another process writes.

## 6. Index fingerprint

Vectors from different embedding models, or chunks from different chunking settings, must never be mixed. The manifest records a fingerprint of the settings that affect the index. A mismatch triggers a rebuild automatically.

## 7. Least-privilege tools

The LLM can call `search_passages`, `read_context`, `check_answer`, `list_documents` and `sync_documents` (read-only or cheap and idempotent). `sync_documents` is cheap and idempotent. A full `rebuild` re-embeds everything, so it is only available to operators via the CLI or the API, where it is protected by `X-API-Key`. Uploads are validated for extension, size and file name, and deletes are confined to the data folder (path-traversal check).

## 8. Prompt-injection hygiene

Retrieved text is untrusted. The prompt labels excerpts as data and tells the model to ignore instructions inside them. Answers must cite excerpt numbers, which makes unsupported claims easier to spot.

## 9. Testability through dependency injection

`DocQAEngine` accepts `llm` and `embed_model`, and the API factory accepts the engine and chat providers. The whole stack, including the real ADK runtime, is tested offline with deterministic fakes. CI needs no secrets.

## 10. Lessons from real-world testing (uploaded slide deck)

Asking *"RAG là gì"* after uploading a 44-page slide deck returned "not found", although page 3 defines RAG. Three issues were found and fixed, each with a regression test:

1. **Tiny chunks.** Slide pages that contain only a title ("Quy trình của RAG", 17 characters) became separate chunks. BM25's length normalization over-scores them and they carry no answer. *Fix:* pages shorter than `MIN_PART_CHARS` (200) are merged into the following page (`documents.merge_short_parts`), which also gives the next page its title as context. Result: 46 → 30 chunks for the deck, and the shortest chunk went from 8 to 152 characters.
2. **RRF drops single-retriever winners.** Dense search ranked the definition #1 (cosine 0.78), but BM25 missed it because it mentions "RAG" only once. Chunks found by both retrievers at mediocre ranks (~2/62) outscored it (1/61). *Fix:* `protect_leaders` guarantees each retriever's #1 a slot in the fused top-k. Retrieval metrics on the demo set were unchanged (hybrid Hit@1 94.1%, Hit@4 100%).
3. **The agent refused without searching.** The instruction told it to decline "general knowledge" questions, so it decided on its own that "RAG là gì" was out of scope. An agent cannot know what the corpus contains without searching. *Fix:* informational questions must always go through the search tool (now `search_passages`), and refusals happen only after a `not_found` result.

The broader lesson is that unit tests with fake models proved the plumbing, but these failures only appeared with real data and a real model. That is why the evaluation set and manual exploratory testing both matter.

## 11. Qdrant as the vector database (replacing Chroma)

- **Why Qdrant:** it is a widely used production vector database with a client/server architecture, HNSW approximate search, payload filtering (deletes by `doc_id` when a file changes) and on-disk storage. Chroma was also dropped because its OpenTelemetry requirements conflicted with google-adk's version pins.
- **Two modes, one code path:** `QDRANT_URL` set → server mode (Docker Compose starts `qdrant/qdrant`), so several processes can share the index. `QDRANT_URL` empty → embedded mode in `storage/qdrant`, convenient for development but locked to one process at a time. One client per location is shared per process and closed at exit.
- **Fits the versioning design:** every full rebuild writes to a new collection (`docqa_<id>`), so the serving collection is never modified half-way. Incremental syncs update the current collection in place. Collections not referenced by the current or previous version are dropped. If the manifest points to a collection that no longer exists (for example a new server), the engine rebuilds instead of failing.
- **Same scores:** Qdrant reports cosine similarity, like the `simple` store. The retrieval benchmark is identical on both backends (Hit@1 vector 91.2%, BM25 97.1%, hybrid 94.1%), so `MIN_SIMILARITY` needs no recalibration.

## 12. Agentic RAG: tools return evidence, the agent reasons, the code enforces grounding

- **Evidence tools instead of an answer tool.** The first version gave the agent `search_documents`, which retrieved *and* generated in one shot, so the agent could not see or judge the evidence. The agent now gets `search_passages` (evidence with short ids, optional per-document filter), `read_context` (neighbouring chunks, across page boundaries) and `check_answer` (verification). It plans sub-questions, re-searches when evidence is insufficient (at most 3 searches per sub-question), drafts with `[#id]` citations, and verifies once.
- **Guardrails live in code, not only in the prompt.** A smaller model does not follow instructions reliably, so:
  - a per-turn LLM-call limit (`RunConfig.max_llm_calls`) stops loops;
  - citations to passages the agent never retrieved are dropped;
  - grouped citations are normalised;
  - after a search, an answer with no valid citation becomes the standard refusal ("no citation, no claim");
  - `check_answer` validates ids and uncited sentences in code, and only the semantic "is this claim supported" check uses an LLM judge.
- **Observable.** Every tool call (tool, arguments, result summary) and the LLM-call count are returned by `/chat` and shown in the UI. This is how the paraphrased-refusal and grouped-citation bugs were found.
- **Measured, including where it does not pay off.** On the 12-chunk demo corpus, one-shot RAG already reaches 100% on multi-hop questions at `TOP_K=4`, so the agent only adds cost (~4× LLM calls, ~5× latency). With `TOP_K=2`, standing in for a large corpus, the agent's decomposition fixed a question one-shot RAG missed (100% vs 90%). Both paths stay available: `/query` is the cheap default, and `/chat` is for conversation, multi-part questions and larger corpora.
- **Lesson from a live test: verification can itself loop.** On *"Can employees on probation work remotely?"* the agent called `check_answer` 8 times and then gave up with a refusal (10 LLM calls). Three causes:
  - the judge rejected a direct logical consequence ("only staff who passed probation may" implies "staff on probation may not");
  - "check once" lived only in the prompt;
  - the model dropped the `#` from citations.

  Fixes:
  - the judge prompt accepts direct entailments;
  - `check_answer` returns `skipped` after the first call in a turn, enforced through invocation-scoped `temp:` state;
  - `[ac7acd25]` is accepted as a citation;
  - step summaries show the flagged claims.

  Result: 3 LLM calls and a correct, cited answer. Rule of thumb: every "do X at most N times" in an agent prompt needs a matching limit in code.

## 13. Structure-aware chunking, decided by a tagged evaluation

- **Problem, found by reading misses rather than guessing.** On a 30-page `.docx` proposal, every retrieval miss of the fixed-size splitter came from lost structure: `docx2txt` flattens headings into plain lines and tables into loose cells, so a chunk could start in one section and end in the next, a list lost the heading that named it, and "Chatbase … Hạn chế (1 model/bot)" no longer said which product the value belonged to.
- **Design: read into one block model, then one generic chunker.** Each format has a small reader that emits blocks (heading, paragraph, table, caption): Word heading styles and outline levels, Markdown `#`, and numbered or legal headings in plain text (`2.1`, `Chương II`, `Điều 5`). Rules are about kinds of content, not about one file:
  - captions are recognised by text ("Bảng 1.", "Hình 2.", "Table 3:") whatever their style, because some templates style captions as headings;
  - a table is rendered one row per line as `header: value` pairs, so any row can be split off without losing its columns; a captioned table is its own section named by its caption;
  - a lone top-level heading that opens the document is its title and is left out of breadcrumbs, because repeating generic title words in every chunk was measured to add BM25 noise;
  - short sections merge forward into the next one.

  The heading path goes into `section` metadata, which LlamaIndex prepends to the text that is embedded and BM25-indexed, gives to the LLM, and returns in citations.
- **Fallback keeps the worst case unchanged.** No headings, or an implausible number of them (more than twice the paragraphs and tables, e.g. bold lines misread as titles), or a parsing error: the file goes through the original loader. PDFs and unstructured `.txt` currently always do.
- **Measured on questions written before the chunker existed**, tagged by where the answer lives (text, table, section, image). Hybrid Hit@1 64.3% → 78.6%, MRR@5 71.8% → 82.4%; section questions 57% → 100% Hit@1, table questions 67% → 83%. Image questions stayed at 0%: the content was never extracted, which no chunker can fix.
- **Lessons.**
  - Inspect the chunks, not only the metrics. A merged-cell fix that compared cell *text* silently dropped one of two adjacent "Có" cells and shifted the columns of the most important comparison table; evidence-phrase metrics would still have counted those chunks as hits.
  - Fix causes, not questions. One demo question dropped from rank 1 to rank 2 under BM25 because a "business-trip registration" section genuinely shares its words; it was left alone rather than tuned away.
  - Measuring needs a budget. Three full index builds plus evaluations used the free tier's 1,000 embeddings per day, which also exposed that ingestion failed on the 100-per-minute limit. Embeddings are now batched, per-minute 429s are waited out using the provider's hint, and daily-quota errors fail fast.

## 14. Describe images at ingestion instead of hoping retrieval finds them

- **Problem.** In the lecture PDF, slides 25–31 (chunking pitfalls, context-window limits, metadata filtering, parent–child and hierarchical retrieval, graph RAG) are pictures. Text extraction returned nothing, so those pages were not in the index at all, and all seven questions about them failed under every chunking strategy. The thesis proposal has nine diagrams (architecture, ERD, module map) with the same problem.
- **Design.** Each meaningful image is sent once to the multimodal LLM with a prompt that asks for (1) every visible word verbatim, tables row by row, then (2) a short explanation, and never to identify people. The description is inserted where the image was: appended to its PDF page (so image-only slides gain text and keep their page number), or as a block in its Word section, named after the adjacent "Hình N." caption.
- **Cost is controlled before, during and after the call.**
  - Before: images repeated on more than two pages (backgrounds, logos) and tiny images are dropped without a call. In the lecture this removed 14 decorations repeated on up to 30 slides each, leaving 24 candidates.
  - During: the model answers a fixed marker for decorative images; a per-file budget caps calls.
  - After: descriptions, including "decorative", are cached on disk by image hash and prompt version, so rebuilding an index or switching chunking strategy never re-describes an image.
- **Quota never breaks ingestion.** A per-minute 429 is waited out; a daily quota stops captioning for the run and leaves the remaining images undescribed (not cached), so a later sync fills them in. This path was exercised for real: the free tier's daily LLM quota ran out after two images, and ingestion still completed.
- **First results.** The two descriptions produced before the quota ran out transcribed a mind map of eight evaluation criteria and a six-module architecture diagram almost verbatim, content that was previously unsearchable. Retrieval impact on the seven image questions is to be measured with the tagged eval set.

## 15. Semantic chunking only where there is no structure

- **Why add it.** The corpus is meant to be varied: besides headed reports there are slide decks, notes and transcripts with no headings, where sentence splitting cuts ideas at arbitrary points. Semantic chunking finds the topic shifts from the text itself.
- **Why not everywhere.** It needs one embedding per sentence just to find split points: 1,117 extra embeddings on the eval corpus, more than the free tier's daily quota, against 132 when it is used only for the files without structure. Where headings exist they are the author's own topic boundaries, free and exact. The Chroma benchmark quoted in the lecture also has Kamradt-style semantic chunking below recursive splitting on recall (83.6% vs 89.5%). And comparison tables are a weak case for it, because their rows are semantically near-identical.
- **Implementation.** Our own `MetadataAwareTextSplitter` rather than LlamaIndex's `SemanticSplitterNodeParser`:
  - Embeddings go through the engine's batched, quota-aware path; the stock parser calls the embedding model directly and would fail mid-document on a per-minute quota.
  - Sentence segmentation re-joins PDF lines broken mid-sentence and keeps list items separate, so breakpoints are not driven by layout.
  - After cutting at the 95th percentile of consecutive-window cosine distance, oversized groups are re-split with the sentence splitter and tiny ones merged, never above the chunk size. A test caught the first version producing a 103-token chunk under an 80-token limit while carrying a tiny piece forward.
  - Texts with more than 1,500 sentences fall back to sentence splitting.
- **Status.** `CHUNK_STRATEGY=semantic` exists so the three strategies can be compared on the same tagged benchmark; the comparison is pending embedding quota.

## 16. Prompt injection: structural defences first, patterns measured on someone else's attacks

- **Threat.** Indexed documents are untrusted. In RAG an injected sentence ("ignore your instructions and …") reaches the model as context; in the agent it can also ask for tool calls. Categories follow BIPIA (indirect injection in retrieved content) and AgentDojo (tool-using agents).
- **Four layers, no extra model call** (`INJECTION_DEFENSE=off|spotlight|full`):
  - spotlighting with a random per-request nonce, so an attacker cannot close the data tag;
  - an output guard that removes links absent from the excerpts and withholds answers that reproduce the system instructions;
  - a code-level permission check on the state-changing `sync_documents` tool, based on the user's own message (ADK `user_content`), not on what the model was told by a document;
  - pattern sanitisation of sentences addressing the AI.
- **The pattern layer was evaluated on data it was not developed on.** On deepset/prompt-injections, development used only the train split; the held-out test split gives 26.7% recall with 0% false positives. On 718,149 clean Vietnamese sentences (61k legal articles, Wikipedia, demo and private documents) it flags 0.0003%. The first version flagged everyday Vietnamese ("đóng vai trò", "sử dụng công cụ", "ai phải …" meaning "who must …") and caught 1.7% of held-out attacks, which is why false positives are measured on large clean corpora and recall on external data.
- **Conclusion drawn from those numbers.** A filter that misses three attacks in four cannot be the main defence. The structural layers do not depend on recognising an attack, and they carry most of the weight; the filter is a cheap, high-precision extra. End-to-end attack success rates, with accuracy under attack and on clean documents so a defence cannot win by refusing everything, come from `docqa injection-eval`.

## 17. A first CI run on Linux found a security check that Windows never triggers

All tests passed on Windows, including in a fresh virtualenv, but one failed on the GitHub runner with `LookupError`. The traceback (surfaced as a public annotation, because job logs need sign-in) showed NLTK 3.10.4 *refusing* the sentence-tokenizer data LlamaIndex bundles in `site-packages`: the runner's tool cache is writable by other accounts, and NLTK no longer loads data from such directories, since a planted file would be trusted. Windows has no such permission check, and the developer machine also had an older NLTK. The fix is environmental and applies to deployment too: Docker and CI set `NLTK_DATA` to a folder only its owner can write and pre-download `stopwords` and `punkt_tab` at build time (no runtime download), and `nltk` is now pinned. Lesson: "works on my machine" included an unpinned transitive dependency and an OS-specific security behaviour; a clean Linux CI run is what exposed both.
