# Roadmap

## Done — Phase 0

- Dependencies, `src/env.py`, `src/llm.py`
- `ingest.py`, local FAISS, `ask.py`
- Demo knowledge under `docs/`
- `PHASE_0_REPORT.md`

## Done — Track B + SLM + Track C (local prototype)

- Typed agent contracts (`src/contracts/messages.py`)
- Agents: Planner, Retrieval, RAG step, Step definer, Summarizer
- `WorkflowEngine` + `ask.py --agentic`
- On-prem SLM via Ollama (`MA_RAG_LLM_PROVIDER=ollama`)
- Evidence Curator + Critic agents
- Per-run evidence ledger (`data/evidence_ledger/`)
- KB eval harness: `python eval_kb.py`

## Next

- Router agent (simple vs multi-hop triage) — done
- Per-agent model config (hybrid SLM + cloud planner optional)
- A2A services + MCP tool gateway (later)
- API ingress (later)
