# Demo Guide

Run after `python ingest.py ./docs`.

## Single-hop (fast, high confidence)

- What is the current completed phase of the MA-RAG prototype?
- What is Chandra Shekar Konda's title at Oracle?
- Who is the volunteer researcher on the MA-RAG project?

## Multi-hop

- Who is the AI Technical Director at Oracle that Bhargav collaborates with on MA-RAG?
- What phase comes after the completed phase of the MA-RAG prototype?

## Film corpus (`test_knowledge_base.md`)

- Who directed Inception?
- Who played Dom Cobb in Inception?

## Avoid for demo

- Full biography questions not in `docs/` (date of birth, education, etc.)

## Good demo flow

1. `--retrieve-only` — show grounded chunks
2. Single-hop `ask.py`
3. Multi-hop `ask.py` — show `=== Plan ===` and step trace
