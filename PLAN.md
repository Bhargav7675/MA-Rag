# MA-RAG — Project Continuity Plan

> **Use this file** when starting a new Cursor chat so you do not rebuild context from scratch.  
> **Last updated:** 2026-05-28 · **Branch:** `feature/track-b-agentic`

---

## 1. What this project is

**MA-RAG** — Multi-Agent Retrieval-Augmented Generation prototype for the **IEEE Talent Meets AI** task force.

- **Research lead / stakeholder:** Chandra Shekar Konda (Oracle, AI Technical Director)
- **Implementer:** Bhargav Boyapati (volunteer researcher)
- **Goal:** Enterprise-style **fully agentic RAG** with **on-prem SLM**, auditable workflows, local FAISS retrieval — aligned to Oracle 5-plane reference architecture for IEEE and potential Oracle deployment
- **Not in scope (yet):** OCI production deployment, API Gateway/IAM, HIPAA, Oracle 23ai vector store

**Repo:** `/Users/bhargavboyapati/Projects/MA-RAG`  
**Remote:** `https://github.com/Bhargav7675/M-Rag.git` (push only when explicitly requested)

---

## 2. Employer architecture mapping (honest compliance)


| Plane                   | Oracle / IEEE target                         | Local prototype today                                      | Fully agentic target                          |
| ----------------------- | -------------------------------------------- | ---------------------------------------------------------- | --------------------------------------------- |
| **1. User / Ingress**   | API Gateway, sessions, SSE to client         | **Partial** — `POST /ask`, **`POST /ask/stream` (SSE)**    | OCI API Gateway + streaming ingress           |
| **2. Agent control**    | 8 specialized agents, per-agent LLM config   | **Done** — all agents via typed contracts + LLM            | Same; optional larger SLM per agent         |
| **3. A2A**              | OCI Queue, agent registry, remote workers    | **Partial** — in-process bus + journal + **file_queue**    | OCI Queue consumers per agent               |
| **4. Workflow runtime** | Orchestrated 0→6 steps, CLARIFY/ESCALATE     | **Done** — happy path; hardcoded orchestrator in engine    | Policy-driven workflow agent                  |
| **5. Tool & evidence**  | MCP tools, vector DB, JSONL audit            | **Done** — `faiss_retrieve`, evidence + A2A journals       | Oracle 23ai + full MCP stdio/SSE server       |
| **SLM**                 | On-prem Ollama / OCI GenAI                   | **Done** — `llama3.2:3b`                                   | Hybrid SLM + cloud for planner/critic         |
| **Realtime**            | Live agent trace to UI                       | **Partial** — SSE events per agent (`/ask/stream`)         | WebSocket + OCI Monitoring                    |


### What “fully agentic” means here

| Requirement | Status | Notes |
|-------------|--------|-------|
| Every workflow step dispatches a **named agent** via A2A | ✅ | Router → Planner → Retrieval → Curator → RAG → Summarizer → Critic |
| Typed **Pydantic contracts** on all agent I/O | ✅ | `src/contracts/messages.py` |
| **Per-run audit** (evidence ledger + A2A journal) | ✅ | `data/evidence_ledger/`, `data/a2a_journal/` |
| **Realtime observability** (agent events to client) | ✅ | `POST /ask/stream` SSE |
| **Out-of-process agents** (remote A2A workers) | ✅ scaffold | `MA_RAG_A2A_TRANSPORT=file_queue` + `run_a2a_worker.py` |
| **No orchestrator bypass** of agents on happy path | ⚠️ partial | Single-step finalize skips Summarizer/Critic LLM calls (SLM quality guard); Planner uses heuristics fallback |
| **MCP server process** (stdio/SSE) | ❌ | In-process tool registry only |
| **OCI production** (Queue, IAM, 23ai) | ❌ | Documented mapping below |


### Oracle OCI mapping (future production)

| Local component | Oracle OCI service |
|-----------------|-------------------|
| FastAPI `/ask`, `/ask/stream` | API Gateway + Functions or Container Instances |
| `FileQueueA2ABus` pending/responses | **OCI Queue** (request/response topics) |
| `A2AFileJournal` | Object Storage + Logging Analytics |
| FAISS `local_index/` | **Oracle Database 23ai** vector store |
| Ollama `llama3.2:3b` | On-prem GPU or **OCI GenAI** (hybrid) |
| Evidence ledger JSONL | Object Storage + audit retention policy |
| MCP `faiss_retrieve` | OCI Functions tool endpoints or MCP on Container Instances |


### Workflow trace (`--agentic`)

```
route → init_plan → retrieve → evidence_check → context_build → generate → finalize → verify
```

---

## 3. Phase / track status


| Track       | Status                                                       |
| ----------- | ------------------------------------------------------------ |
| **Phase 0** | Complete — ingest, FAISS, `ask.py`, docs KB                  |
| **Track B** | Complete — typed agents, `WorkflowEngine`, `--agentic`       |
| **SLM**     | Complete — Ollama, `src/slm_helpers.py`, SLM prompts/parsers |
| **Track C** | Complete — Evidence Curator, Critic, JSONL evidence ledger   |
| **Router**  | Complete — `RouterAgent`, simple vs multi-hop triage         |
| **Eval**    | `eval_kb.py` — **8/8 passing**; `pytest` — **16/16**       |


### Not started (production / Oracle)

- OCI deployment (Queue, API Gateway, IAM, 23ai)
- Full MCP server process (stdio/SSE)
- CLARIFY / ESCALATE workflow branches
- Session memory / multi-turn conversations
- Remove SLM heuristic fallbacks (requires stronger model or fine-tune)

---

## 4. Agents (Plane 2)


| Agent                | File                               | Role                               |
| -------------------- | ---------------------------------- | ---------------------------------- |
| **Router**           | `agents/router_agent.py`           | `simple_rag` vs `multi_hop_rag`    |
| **Planner**          | `agents/planner_agent.py`          | Decompose question into plan steps |
| **Retrieval**        | `agents/retrieval_agent.py`        | FAISS top-k chunks                 |
| **Evidence Curator** | `agents/evidence_curator_agent.py` | Sufficiency before generate        |
| **Step definer**     | `agents/step_definer_agent.py`     | Sub-task per plan step             |
| **RAG step**         | `agents/rag_step_agent.py`         | Extract + grounded QA              |
| **Summarizer**       | `agents/summarizer_agent.py`       | Combine step answers               |
| **Critic**           | `agents/critic_agent.py`           | Verify faithfulness                |


**Orchestrator:** `src/workflow/engine.py`  
**Contracts:** `src/contracts/messages.py`

---

## 5. Key technical decisions

### SLM (Ollama `llama3.2:3b`)

- All agent LLM calls use Ollama when `MA_RAG_LLM_PROVIDER=ollama`
- Small models weak at structured JSON → **plain-text output + parsing** in `src/slm_helpers.py`
- **Single-step success:** finalize uses step answer directly (no summarize/critic overwrite)
- **Multi-step success:** `format_kb_multi_hop_answer()` + hedging reconciliation in `src/workflow/finalize_helpers.py`

### Retrieval

- **FAISS** local index — **not Pinecone**
- `python ingest.py` (default `./docs`) — **excludes `docs/wiki/` and `PROJECT_UPDATES_NOTES.md`** (use `--include-wiki` to index wiki)
- Knowledge files live in **`docs/`** only: IEEE KB, demo corpora (`greenfield_health.md`, `summit_cloud_team.txt`), etc.

### Router heuristics (`src/slm_helpers.py`)

- `classify_route()` → `simple_rag` | `multi_hop_rag`
- `canonical_kb_plan()` + `heuristic_multi_hop_plan()` for SLM planner fallback
- `is_likely_multi_hop()` — detects `" and "`, `"both"`, etc.

---

## 6. Environment (`.env` — never commit)

```env
MA_RAG_LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434

# A2A transport: in_process (default) | file_queue (remote workers)
# MA_RAG_A2A_TRANSPORT=file_queue

# Per-agent overrides (optional):
# MA_RAG_PLANNER_PROVIDER=ollama
# MA_RAG_PLANNER_OLLAMA_MODEL=llama3.2:3b

# Optional fallback (unused when provider=ollama):
OPENAI_API_KEY=...
MODEL_NAME=gpt-4o-mini
```

---

## 7. Commands (copy-paste)

```bash
cd /Users/bhargavboyapati/Projects/MA-RAG
source .venv/bin/activate

# Ollama check
curl -s http://localhost:11434/api/tags

# Build index (wiki excluded)
python ingest.py ./docs

# Main path
python ask.py "What is the current completed phase of the MA-RAG prototype?" --agentic

# Regression (expect 8/8)
python eval_kb.py

# API tests
python -m pytest tests/ -v

# API ingress (blocking)
python run_api.py
# curl -s http://127.0.0.1:8000/health
# curl -s -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
#   -d '{"question":"What is the current completed phase of the MA-RAG prototype?"}'

# API ingress (realtime SSE — agent events as they complete)
# curl -N -X POST http://127.0.0.1:8000/ask/stream -H 'Content-Type: application/json' \
#   -d '{"question":"What is the current completed phase of the MA-RAG prototype?"}'

# Distributed A2A (two terminals — fully out-of-process agents)
# Terminal 1: MA_RAG_A2A_TRANSPORT=file_queue python run_a2a_worker.py
# Terminal 2: MA_RAG_A2A_TRANSPORT=file_queue python ask.py "your question" --agentic

# MCP-style tool + A2A introspection
# curl -s http://127.0.0.1:8000/tools
# curl -s http://127.0.0.1:8000/agents

# Retrieval debug only
python ask.py "your question" --retrieve-only

# View evidence ledger (use real run_id from output)
cat data/evidence_ledger/<run_id>.jsonl
```

### Verify Ollama is used

stderr should show:

```
LLM: ollama/llama3.2:3b @ http://localhost:11434
httpx ... POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
```

---

## 8. Eval cases (`eval_kb.py`)


| #   | Type       | Question                                     | Expected                                           |
| --- | ---------- | -------------------------------------------- | -------------------------------------------------- |
| 1   | single-hop | Current completed phase?                     | Phase 0                                            |
| 2   | single-hop | Volunteer researcher?                        | Bhargav Boyapati                                   |
| 3   | single-hop | Chandra's company at Oracle?                 | Oracle                                             |
| 4   | multi-hop  | Volunteer works with whom + title at Oracle? | Chandra Shekar Konda **and** AI Technical Director |


Multi-hop eval requires **all** expected substrings.

---

## 9. Git history (feature/track-b-agentic)

Recent commits (newest first):

1. Realtime SSE `/ask/stream`, file-queue A2A worker, architecture compliance update
2. API tests, A2A journal, rag_step on bus, heuristic multi-hop
3. MCP tool gateway, A2A bus, FastAPI ingress
4. Router agent + PLAN.md
5. Multi-step finalize + KB eval tightening
6. SLM + Track C (curator, critic, evidence ledger)
7. Typed retrieval agents + WorkflowEngine + --agentic

**Policy:** Local commits OK. **No push** unless user asks. No Cursor co-author on commits.

**Gitignored:** `.env`, `.venv/`, `local_index/`, `data/evidence_ledger/`, `.cursor/`

---

## 10. Important files

```
ask.py                          # CLI entry (--agentic, --retrieve-only, --llm-only)
run_api.py                      # FastAPI ingress (POST /ask, POST /ask/stream)
run_a2a_worker.py               # Remote A2A worker (file_queue transport)
ingest.py                       # FAISS index build (.md, .txt, .pdf)
eval_kb.py                      # KB regression harness (8 cases)
src/workflow/events.py          # Realtime workflow events (SSE)
src/a2a/file_queue_bus.py       # Out-of-process A2A transport
src/tools/                      # MCP-style tool gateway (faiss_retrieve)
src/a2a/                        # In-process A2A bus + agent registry
src/api/server.py               # FastAPI routes
src/workflow/a2a_setup.py       # Wire agents onto A2A bus
src/workflow/service.py         # Shared workflow service (CLI + API)
agents/router_agent.py
agents/planner_agent.py
agents/retrieval_agent.py
agents/evidence_curator_agent.py
agents/step_definer_agent.py
agents/rag_step_agent.py
agents/summarizer_agent.py
agents/critic_agent.py
agents/rag.py                     # RAG subgraph (retrieve/extract/generate)
src/workflow/engine.py            # WorkflowEngine
src/workflow/finalize_helpers.py  # Multi-step merge, hedging fix
src/slm_helpers.py                # Ollama parsing, route classify, plan simplify
src/evidence_ledger.py            # JSONL per run_id
src/llm.py                        # openai | ollama
src/local_retrieval.py            # FAISS + LocalRetrieverTool
src/contracts/messages.py         # Pydantic agent contracts
docs/ieee_marag_project_knowledge_base.md
docs/wiki/SLM-Setup.md
docs/wiki/Roadmap.md
```

---

## 11. Known issues / limitations

- Prototype on Mac — not production/OCI
- SLM (3B) needs heuristics; quality lower than GPT for complex planning
- `canonical_kb_plan()` is pattern-specific (not fully general multi-hop)
- Case 4 answer quality depends on 2-step retrieval succeeding
- LangGraph deprecation warning on import — harmless
- OpenAI key in `.env` for optional fallback — never commit `.env`

---

## 12. Next steps (recommended order)

1. ~~Router agent~~ **Done**
2. ~~Expand `eval_kb.py`~~ **8/8 passing**
3. ~~Per-agent model config~~ **Done**
4. ~~MCP-style tool wrapper~~ **Done** — `faiss_retrieve` via `src/tools/`
5. ~~A2A scaffolding~~ **Done** — `src/a2a/` in-process bus
6. ~~FastAPI `POST /ask`~~ **Done** — `run_api.py`
7. **Push branch** — when user requests
8. ~~API tests + warning cleanup~~ **Done**
9. ~~A2A file journal + rag_step on bus~~ **Done**
10. ~~Heuristic multi-hop + evidence re-retrieve~~ **Done**
11. ~~Remote A2A worker process~~ **Done** — `run_a2a_worker.py` + `file_queue` transport
12. ~~Realtime SSE ingress~~ **Done** — `POST /ask/stream`
13. **Full MCP server** (stdio/SSE) — future
14. **OCI Queue + 23ai mapping** — production Oracle path
15. **CLARIFY / ESCALATE** workflow branches

---

## 13. Prompt for new Cursor chat

Paste this when opening a fresh session:

```
I'm continuing MA-RAG on branch feature/track-b-agentic.
Read PLAN.md in the repo root first — it has full project continuity.
Stack: Ollama llama3.2:3b, --agentic workflow, FAISS, evidence ledger, eval_kb 8/8, pytest 16/16, FastAPI POST /ask and POST /ask/stream (SSE).
Do not push to GitHub unless I ask. Do not commit unless I ask.
```

---

## 14. Stakeholder one-liner

> Local prototype of enterprise agent control + workflow runtime: Router → Planner → Retrieval → Evidence Curator → RAG → Summarizer → Critic, with on-prem SLM (Ollama), FAISS evidence layer, per-run JSONL audit ledger, realtime SSE agent trace, optional distributed A2A workers, and 8/8 KB eval passing.

