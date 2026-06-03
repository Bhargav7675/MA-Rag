# Architecture

## Current pipeline (Phase 0)

1. **Planner** — decomposes the question into steps
2. **Step definer** — sub-question per step
3. **RAG** — retrieve (FAISS) → extract per chunk → QA
4. **Summarizer** — final answer + confidence

**Entry:** `ask.py`  
**Index:** `ingest.py` → `local_index/`

## Target (enterprise multi-agent RAG)

- Router, Planner, Retrieval, Evidence Curator, Critic
- A2A message bus between agents
- 9-step workflow runtime (INIT → RETRIEVE → … → FINALIZE)
- MCP tool gateway + evidence ledger

Phase 0 validates core RAG; later phases add full agentic boundaries.
