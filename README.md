# MA-RAG: Multi-Agent Retrieval-Augmented Generation

Research implementation of [MA-RAG: Multi-Agent Retrieval-Augmented Generation via Collaborative Chain-of-Thought Reasoning](https://arxiv.org/abs/2505.20096).

![MA-RAG Architecture](img/arch.png)

## Quick start

```bash
git clone https://github.com/Bhargav7675/M-Rag.git
cd MA-RAG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env
```

Index your documents and ask a question:

```bash
python ingest.py          # default: ./docs → local_index/
python ask.py "Your question here"
```

Interactive chat: `python ask.py` (type `quit` to exit).

See [docs/wiki/Quick-Start.md](docs/wiki/Quick-Start.md) for SLM (Ollama), API, and agentic mode.

## Document ingestion

Place knowledge files in **`docs/`**. Supported formats: `.pdf`, `.txt`, `.md`, `.docx`, `.xlsx`, `.xls`, `.pptx`. Password-protected and scanned PDFs are supported — see `.env.sample`.

```bash
python ingest.py              # indexes ./docs (skips docs/wiki/ by default)
python ingest.py --include-wiki
```

## Configuration

Copy `.env.sample` to `.env`. Key variables:

| Variable | Purpose |
| -------- | ------- |
| `OPENAI_API_KEY` | Cloud LLM (optional if using Ollama) |
| `MODEL_NAME` | OpenAI model (default `gpt-4o-mini`) |
| `MA_RAG_LLM_PROVIDER` | Set to `ollama` for on-prem SLM |
| `MA_RAG_OLLAMA_MODEL` | Ollama model (e.g. `llama3.2:3b`) |

Paths and device overrides: see `.env.sample`.

## API

```bash
python run_api.py
# POST /ask          — JSON response
# POST /ask/stream   — SSE agent trace
```

## Benchmark (original paper workflow)

1. Place KILT dev JSONL files in `data/benchmarks/` ([data/README.md](data/README.md)).
2. `python corpus/embed_corpus.py`
3. `python main.py --model gpt4omini --dataset hotpotqa --exp plan_rag_extract --gpus 0`

## Project layout

| Path | Role |
| ---- | ---- |
| `agents/` | Specialized workflow agents |
| `src/workflow/` | Orchestration engine |
| `src/local_retrieval.py` | FAISS ingest and search |
| `docs/` | Knowledge corpus and wiki |
| `tests/` | Unit and integration tests |

Run tests: `PYTHONPATH=. pytest -q`

## Citation

```bibtex
@article{marag2025,
  title={MA-RAG: Multi-Agent Retrieval-Augmented Generation via Collaborative Chain-of-Thought Reasoning},
  author={Thang Nguyen, Peter Chin, Yu-Wing Tai},
  year={2025},
  journal={arXiv preprint arXiv:2505.20096},
}
```
