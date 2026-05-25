# MA-RAG data layout

## Benchmark questions (for `main.py`)

Place KILT-style dev JSONL files under `data/benchmarks/`:

| Dataset   | Filename                 |
|-----------|--------------------------|
| nq        | `nq-dev-kilt.jsonl`      |
| hotpotqa  | `hotpotqa-dev-kilt.jsonl`|
| triviaqa  | `triviaqa-dev-kilt.jsonl`|
| 2wiki     | `2WikiMultihopQA.jsonl`  |
| fever     | `fever-dev-kilt.jsonl`   |

Download from the [KILT benchmark](https://github.com/facebookresearch/KILT) or the paper authors' release.

Override the root with `MA_RAG_DATA_DIR`.

## Local document index (recommended for development)

Create a `docs/` folder with `.pdf`, `.txt`, or `.md` files, then run:

```bash
python ingest.py ./docs
```

This creates a local FAISS index under `local_index/`. `ask.py` uses this local index automatically when it exists:

```bash
python ask.py "What is this document about?"
```

Override the local index directory with `MA_RAG_LOCAL_INDEX_DIR`.

## Original DPR/Wikipedia retrieval index (for benchmarks)

1. Embed the DPR Wikipedia corpus (slow; GPU recommended):

   ```bash
   python corpus/embed_corpus.py
   ```

2. Ensure shard pickles exist under `save_embs/gte-ml-base/` matching `dpr100_*` (or set `MA_RAG_INDEX_DIR` / `MA_RAG_INDEX_DATASET`).

3. Ask a question:

   ```bash
   python ask.py "Who directed Inception?"
   ```

The Wikipedia **document text** for hits is loaded via `ir_datasets` (`dpr-w100/natural-questions/dev`) on first run.

## Medical corpus (optional)

For PubMed experiments, place `pubmed.tsv.gz` in `data/` or set `MA_RAG_PUBMED_CORPUS`.
