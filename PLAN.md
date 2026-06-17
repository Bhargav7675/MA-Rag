# MA-RAG — Project Continuity Plan

> **Use this file** when starting a new Cursor chat so you do not rebuild context from scratch.  
> **Last updated:** 2026-06-16 · **Branch:** `feature/track-b-agentic`

---

## 1. What this project is

**MA-RAG** — Multi-Agent Retrieval-Augmented Generation prototype for the **IEEE Talent Meets AI** task force.

- **Research lead / stakeholder:** Chandra Shekar Konda (Oracle, AI Technical Director)
- **Implementer:** Bhargav Boyapati (volunteer researcher)
- **Goal:** Enterprise-style **agentic RAG** with **on-prem SLM**, auditable workflows, local FAISS retrieval
- **Not in scope (for now):** User/Ingress plane (API gateway, Teams), OCI deployment, HIPAA production

**Repo:** `/Users/bhargavboyapati/Projects/MA-RAG`  
**Remote:** `https://github.com/Bhargav7675/M-Rag.git` (push only when explicitly requested)

---

## 2. Employer architecture mapping

| Plane | Target | Status |
|-------|--------|--------|
| **1. User / Ingress** | API, sessions | Skipped — CLI `ask.py` only |
| **2. Agent control** | Router, Planner, Retrieval, Curator, Critic, Summarizer | **Done locally** |
| **3. A2A** | Queues, registry | Not started — typed Pydantic messages today |
| **4. Workflow runtime** | 0→6 steps | **Done** |
| **5. Tool & evidence** | FAISS, ingest, evidence ledger | **Done locally** |
| **SLM** | Ollama on-prem | **Done** — `llama3.2:3b` |

### Workflow trace (`--agentic`)

```
route → init_plan → retrieve → evidence_check → context_build → generate → finalize → verify
```

---

## 3. Phase / track status

| Track | Status |
|-------|--------|
| **Phase 0** | Complete — ingest, FAISS, `ask.py`, docs KB |
| **Track B** | Complete — typed agents, `WorkflowEngine`, `--agentic` |
| **SLM** | Complete — Ollama, `src/slm_helpers.py`, SLM prompts/parsers |
| **Track C** | Complete — Evidence Curator, Critic, JSONL evidence ledger |
| **Router** | Complete — `RouterAgent`, simple vs multi-hop triage |
| **Eval** | `eval_kb.py` — **4/4 passing** |

### Not started

- Per-agent model config (hybrid SLM + OpenAI planner)
- MCP tool gateway
- A2A message bus / agent registry
- API ingress (`POST /ask`)
- OCI mapping

---

## 4. Agents (Plane 2)

| Agent | File | Role |
|-------|------|------|
| **Router** | `agents/router_agent.py` | `simple_rag` vs `multi_hop_rag` |
| **Planner** | `agents/planner_agent.py` | Decompose question into plan steps |
| **Retrieval** | `agents/retrieval_agent.py` | FAISS top-k chunks |
| **Evidence Curator** | `agents/evidence_curator_agent.py` | Sufficiency before generate |
| **Step definer** | `agents/step_definer_agent.py` | Sub-task per plan step |
| **RAG step** | `agents/rag_step_agent.py` | Extract + grounded QA |
| **Summarizer** | `agents/summarizer_agent.py` | Combine step answers |
| **Critic** | `agents/critic_agent.py` | Verify faithfulness |

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
- `python ingest.py ./docs` — **excludes `docs/wiki/` by default** (use `--include-wiki` to index demo pages)
- Knowledge base: `docs/ieee_marag_project_knowledge_base.md`

### Router heuristics (`src/slm_helpers.py`)

- `classify_route()` → `simple_rag` | `multi_hop_rag`
- `canonical_kb_plan()` — known 2-step pattern for volunteer + Oracle title question
- `is_likely_multi_hop()` — detects `" and "`, `"both"`, etc.

---

## 6. Environment (`.env` — never commit)

```env
MA_RAG_LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434

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

# Regression (expect 4/4)
python eval_kb.py

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

| # | Type | Question | Expected |
|---|------|----------|----------|
| 1 | single-hop | Current completed phase? | Phase 0 |
| 2 | single-hop | Volunteer researcher? | Bhargav Boyapati |
| 3 | single-hop | Chandra's company at Oracle? | Oracle |
| 4 | multi-hop | Volunteer works with whom + title at Oracle? | Chandra Shekar Konda **and** AI Technical Director |

Multi-hop eval requires **all** expected substrings.

---

## 9. Git history (feature/track-b-agentic)

Recent commits (newest first):

1. Router agent + route workflow step
2. Multi-step finalize + KB eval tightening
3. SLM + Track C (curator, critic, evidence ledger)
4. Typed retrieval agents + `WorkflowEngine` + `--agentic`
5. Typed contracts + PlannerAgent (Track B1)

**Policy:** Local commits OK. **No push** unless user asks. No Cursor co-author on commits.

**Gitignored:** `.env`, `.venv/`, `local_index/`, `data/evidence_ledger/`, `.cursor/`

---

## 10. Important files

```
ask.py                          # CLI entry (--agentic, --retrieve-only, --llm-only)
ingest.py                       # FAISS index build
eval_kb.py                      # KB regression harness
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
2. **Expand `eval_kb.py`** — 10–15 questions for regression
3. **Per-agent model config** — YAML: SLM for extract, optional GPT for planner
4. **MCP-style tool wrapper** — FAISS retriever as formal tool interface
5. **A2A scaffolding** — in-process bus → future queue
6. **FastAPI `POST /ask`** — minimal ingress when employer wants API
7. **Push branch** — when user requests

---

## 13. Prompt for new Cursor chat

Paste this when opening a fresh session:

```
I'm continuing MA-RAG on branch feature/track-b-agentic.
Read PLAN.md in the repo root first — it has full project continuity.
Stack: Ollama llama3.2:3b, --agentic workflow, FAISS, evidence ledger, eval_kb 4/4.
Do not push to GitHub unless I ask. Do not commit unless I ask.
```

---

## 14. Stakeholder one-liner

> Local prototype of enterprise agent control + workflow runtime: Router → Planner → Retrieval → Evidence Curator → RAG → Summarizer → Critic, with on-prem SLM (Ollama), FAISS evidence layer, per-run JSONL audit ledger, and 4/4 KB eval passing.
