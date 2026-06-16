# SLM setup (Ollama — on-prem)

Employer direction: proceed with **small on-prem models** for agent LLM calls.

## 1. Install Ollama

Download from https://ollama.com and install for macOS.

## 2. Pull a small instruct model

```bash
ollama pull llama3.2:3b
```

Alternatives: `phi3:mini`, `gemma2:2b`

## 3. Configure `.env`

```env
MA_RAG_LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
```

Keep `OPENAI_API_KEY` commented out or set only if you switch back to `openai`.

## 4. Verify Ollama is running

```bash
ollama list
curl http://localhost:11434/api/tags
```

## 5. Run MA-RAG

```bash
source .venv/bin/activate
python ingest.py ./docs
python ask.py "What is the current completed phase?" --agentic
```

## Switch back to GPT

```env
MA_RAG_LLM_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

## Notes

- **Retrieval** (`local_index/`) is unchanged — still local FAISS + `docs/`.
- Small SLMs may be weaker on **multi-hop** and **structured JSON** (plans). Use scoped demo questions from Demo Guide.
- **Hybrid:** use SLM for daily dev; keep GPT for benchmark comparison on the same 10 questions.
