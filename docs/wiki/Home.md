# MA-RAG Wiki

Multi-Agent Retrieval-Augmented Generation — local prototype with document ingestion and interactive Q&A.

**Paper:** [MA-RAG (arXiv:2505.20096)](https://arxiv.org/abs/2505.20096)  
**Repo:** Code, `README.md`, and `PHASE_0_REPORT.md` in the main repository.

---

## Status

| Phase | Focus | Status |
| ----- | ----- | ------ |
| Phase 0 | Local ingest, FAISS retrieval, `ask.py`, end-to-end validation | **Complete** |
| Phase 1 | Explicit agents + typed messages (in-process) | Planned |
| Phase 2 | A2A agent services | Planned |
| Phase 3 | API, observability, deployment | Planned |

---

## Quick commands

```bash
cd MA-RAG
source .venv/bin/activate
cp .env.sample .env   # set OPENAI_API_KEY, MODEL_NAME

python ingest.py ./docs
python ask.py "Your question here"
```

**Modes:** `--retrieve-only` · `--llm-only` · `--output-json path.json`

---

## Wiki index

- [Quick Start](Quick-Start)
- [Architecture](Architecture)
- [Demo Guide](Demo-Guide)
- [Roadmap](Roadmap)
- [SLM Setup](SLM-Setup) — on-prem Ollama

---

## Local-only (not in Git)

- `.env` — API keys
- `local_index/` — rebuild with `ingest.py`
- `.venv/` — Python environment

---

## Knowledge corpus

Default demo documents under `docs/`:

- `ieee_marag_project_knowledge_base.md` — project / team / phase facts
- `test_knowledge_base.md` — sample films and multi-hop examples
- `sample_company.txt` — simple company facts

Re-index after editing: `python ingest.py ./docs`
