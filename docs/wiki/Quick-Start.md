# Quick Start

## Prerequisites

- Python 3.9+
- OpenAI API key

## Setup

```bash
git clone https://github.com/Bhargav7675/M-Rag.git
cd M-Rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your_key
MODEL_NAME=gpt-4o-mini
```

## Index documents

```bash
python ingest.py ./docs
```

## Ask a question

```bash
python ask.py "What is the current completed phase of the MA-RAG prototype?"
```

## Verify retrieval (no LLM cost)

```bash
python ask.py "your question" --retrieve-only
```

## Troubleshooting

| Issue | Fix |
| ----- | --- |
| Missing API key | Set `OPENAI_API_KEY` in `.env` |
| No index | Run `python ingest.py ./docs` |
| Low confidence | Ask scoped questions; see Demo Guide |
| Many API calls | Multi-step plans; default `top_k=3` for local retrieval |
