# MA-RAG Phase 0 Report

## Goal

Phase 0 focused on making the original MA-RAG repository runnable as a practical local project. The upstream repo was primarily a research benchmark runner: it expected prebuilt Wikipedia/DPR embeddings, hardcoded data paths, mixed LangChain imports, and had no direct user-facing question entry point.

The Phase 0 goal was to stabilize the project enough to run an end-to-end question-answering flow locally.

## Starting State

The original repo had these issues:

- Dependency pins were brittle and not friendly to local macOS setup.
- `requirements.txt` used `faiss==1.8.0`, but the installable package is usually `faiss-cpu` or `faiss-gpu`.
- `.env.sample` used `OPENAI_API_KEY`, while the code read `API_KEY`.
- Some files used deprecated `langchain.chat_models.ChatOpenAI`; others used `langchain_openai.ChatOpenAI`.
- `main.py` only ran batch experiments over benchmark datasets.
- Dataset paths were hardcoded under `/scratch2/...`.
- There was no `ask.py` for asking one question.
- Full DPR/Wikipedia retrieval required large precomputed FAISS shards.
- Local PDF/document ingestion did not exist.

## Phase 0 Changes

### 1. Dependency Stabilization

Updated `requirements.txt`:

- Replaced `faiss` with `faiss-cpu`.
- Added `python-dotenv`.
- Added `pypdf` for PDF ingestion.
- Kept default requirements CPU-friendly.
- Moved GPU FAISS to `requirements-gpu.txt`.
- Avoided requiring heavyweight optional packages like `vllm` for the basic local flow.

Added:

- `requirements-gpu.txt`

### 2. Environment Configuration

Added:

- `src/env.py`

This centralizes configuration:

- `OPENAI_API_KEY`
- legacy fallback `API_KEY`
- `MODEL_NAME`
- benchmark data directory
- original DPR/Wikipedia index directory
- local document index directory
- embedding model settings

Updated:

- `.env.sample`

The minimum required `.env` now is:

```env
OPENAI_API_KEY=your_openai_key
MODEL_NAME=gpt-4o-mini
```

Validated by running:

```bash
python -c "from src.env import get_openai_api_key, get_model_name; get_openai_api_key(); print('OK', get_model_name())"
```

Observed output:

```text
OK gpt-4o-mini
```

### 3. Unified LLM Client

Added:

- `src/llm.py`

This provides one helper:

```python
create_chat_llm()
```

Updated agent files to use the modern `langchain_openai.ChatOpenAI` path through this helper:

- `agents/plan.py`
- `agents/step_definer.py`
- `agents/plan_executor.py`
- `agents/rag.py`

### 4. Shared Pipeline Setup

Added:

- `src/pipeline.py`

This moved shared graph setup out of `main.py` and made it reusable by:

- `main.py`
- `ask.py`

It provides:

- question normalization
- retriever setup
- graph construction
- formatted graph output
- JSON trace writing

### 5. Interactive Question Entry Point

Added:

- `ask.py`

Supported modes:

```bash
python ask.py "Who directed Inception?"
```

```bash
python ask.py "Who directed Inception?" --llm-only
```

```bash
python ask.py "Who directed Inception?" --retrieve-only
```

```bash
python ask.py "Who directed Inception?" --output-json outputs/run.json
```

The `--llm-only` mode validates OpenAI without retrieval.

The `--retrieve-only` mode validates retrieval without calling OpenAI.

### 6. Local Document Ingestion

Added:

- `ingest.py`
- `src/local_retrieval.py`

Supported source files:

- `.pdf`
- `.txt`
- `.md`

Usage:

```bash
python ingest.py ./docs
```

This writes a local FAISS index under:

```text
local_index/
```

The first version used a neural HuggingFace embedding model, but on macOS it caused a segmentation fault / model download instability. The local ingestion path was changed to use a CPU-only hashing embedder by default.

This avoids:

- Apple MPS instability
- remote-code HuggingFace model loading
- large model downloads
- segfaults during local smoke testing

The original DPR/Wikipedia embedding path still exists separately for benchmark-style usage.

### 7. Test Document

Added:

- `docs/test_knowledge_base.md`

It contains small test facts about:

- Inception
- Interstellar
- The Matrix
- Christopher Nolan
- Leonardo DiCaprio
- DreamTech Labs
- OrbitAI
- simple multi-hop facts

Example questions:

```bash
python ask.py "Who directed Inception?"
python ask.py "Who founded DreamTech Labs?"
python ask.py "Where did the founder of DreamTech Labs study?"
python ask.py "Who played Dom Cobb in Inception?"
python ask.py "Which company was founded by someone who previously worked at NASA?"
```

### 8. Documentation Updates

Updated:

- `README.md`
- `data/README.md`

Documentation now explains:

- environment setup
- local ingestion
- single-question asking
- batch benchmark usage
- local vs original DPR/Wikipedia retrieval

## Validation Performed

### 1. OpenAI Configuration

Command:

```bash
python -c "from src.env import get_openai_api_key, get_model_name; get_openai_api_key(); print('OK', get_model_name())"
```

Result:

```text
OK gpt-4o-mini
```

### 2. LLM-Only Smoke Test

Command:

```bash
python ask.py "Who directed Inception?" --llm-only
```

Result:

```text
=== LLM-only answer (retrieval skipped) ===
Christopher Nolan
Success: Yes
Confidence: 10
```

### 3. Local Document Ingestion

Command:

```bash
python ingest.py ./docs
```

Result:

```text
Indexed 2 chunks into /Users/bhargavboyapati/Projects/MA-RAG/local_index
Now ask with:
  python ask.py "your question"
```

### 4. Retrieval-Only Test

Command:

```bash
python ask.py "Who directed Inception?" --retrieve-only
```

Result:

The retrieved chunks included:

```text
Inception is a 2010 science fiction action film directed by Christopher Nolan.
```

### 5. Full End-to-End Local RAG Test

Command:

```bash
python ask.py "Who directed Inception?"
```

Result:

```text
=== Plan ===
1. Identify the director of the film Inception.

=== Step trace ===

--- Step 1 (question-answering) ---
Task: Who is the director of the film Inception?
Answer: Christopher Nolan
Success: Yes
Confidence: 10

=== Final answer ===
Final answer: Christopher Nolan
Confidence score: 10

Summary:
Output: Successful, score: 10
```

This validates the full Phase 0 pipeline:

```text
User question
  -> Planner
  -> Step Definer
  -> Local Retriever
  -> Extractor
  -> QA Agent
  -> Summarizer
  -> Final answer
```

## Current Working Commands

Create and activate environment:

```bash
cd ~/Projects/MA-RAG
source .venv/bin/activate
```

Verify OpenAI config:

```bash
python -c "from src.env import get_openai_api_key, get_model_name; get_openai_api_key(); print('OK', get_model_name())"
```

Ingest local docs:

```bash
python ingest.py ./docs
```

Retrieve only:

```bash
python ask.py "Who directed Inception?" --retrieve-only
```

Run full local RAG:

```bash
python ask.py "Who directed Inception?"
```

Run LLM-only:

```bash
python ask.py "Who directed Inception?" --llm-only
```

## Known Non-Blocking Warnings

### urllib3 / LibreSSL warning

Observed:

```text
NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+
```

Cause:

- macOS system Python is compiled with LibreSSL.

Impact:

- Non-blocking. OpenAI calls still succeeded.

Future cleanup:

- Use a Python distribution compiled with OpenSSL, such as Homebrew Python or pyenv Python.

### LangGraph pending deprecation warning

Observed:

```text
LangChainPendingDeprecationWarning
```

Impact:

- Non-blocking.

Future cleanup:

- Pin/adjust LangGraph serializer settings or upgrade with explicit configuration.

### Pydantic serializer warnings

Observed during structured output parsing:

```text
PydanticSerializationUnexpectedValue
```

Impact:

- Non-blocking. Structured outputs were returned correctly.

Future cleanup:

- Reduce warnings by adjusting LangChain structured output parsing or serialization.

## Security Note

The OpenAI key should remain only in local `.env`.

`.env` is listed in `.gitignore`, so it should not be committed.

Do not paste API keys into chat or source files.

## Files Added

- `PHASE_0_REPORT.md`
- `ask.py`
- `ingest.py`
- `requirements-gpu.txt`
- `src/env.py`
- `src/llm.py`
- `src/pipeline.py`
- `src/local_retrieval.py`
- `data/README.md`
- `data/benchmarks/.gitkeep`
- `docs/test_knowledge_base.md`

## Files Modified

- `.env.sample`
- `README.md`
- `requirements.txt`
- `main.py`
- `agents/plan.py`
- `agents/step_definer.py`
- `agents/plan_executor.py`
- `agents/rag.py`
- `corpus/retrieve.py`
- `src/utils.py`

## Current Phase 0 Status

Phase 0 is complete.

The repo now has a working local end-to-end RAG path using:

- local document ingestion
- local FAISS retrieval
- OpenAI-powered planner / step definer / extractor / QA / summarizer
- interactive question answering via `ask.py`

The original DPR/Wikipedia benchmark path is still present, but not required for the local smoke-test workflow.

## Recommended Next Step: Phase 1

Phase 1 should convert the current function/node-based system into explicit in-process agents before moving to A2A services.

Recommended Phase 1 tasks:

1. Define message schemas:
   - `PlanRequest`
   - `PlanResponse`
   - `StepDefineRequest`
   - `StepDefineResponse`
   - `RetrieveRequest`
   - `RetrieveResponse`
   - `ExtractRequest`
   - `ExtractResponse`
   - `AnswerRequest`
   - `AnswerResponse`
   - `SummarizeRequest`
   - `SummarizeResponse`

2. Create agent classes:
   - `PlannerAgent`
   - `StepDefinerAgent`
   - `RetrievalAgent`
   - `ExtractorAgent`
   - `QAAgent`
   - `SummarizerAgent`

3. Give each agent:
   - an ID
   - a role description
   - typed input/output
   - owned tools
   - local memory/trace hooks

4. Replace shared mutable graph state with explicit message passing.

5. Keep everything in-process first.

6. After Phase 1 works, split agents into A2A-compatible services in Phase 2.
