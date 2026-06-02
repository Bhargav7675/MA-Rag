# IEEE Task Force — MA-RAG Research Project Knowledge Base

## Project Overview

The **Talent Meets AI** initiative is an IEEE task force focused on exploring how multi-agent retrieval-augmented generation (RAG) can move from research prototypes to practical, production-oriented agentic systems. The current research track centers on **MA-RAG** (Multi-Agent Retrieval-Augmented Generation via Collaborative Chain-of-Thought Reasoning), based on the peer-reviewed research direction described in the MA-RAG paper (multi-agent planning, retrieval, extraction, and answer synthesis).

The active implementation effort in this repository targets **Phase 0**: stabilizing the upstream MA-RAG codebase for local execution, document ingestion, interactive question answering, and end-to-end validation before refactoring toward explicit in-process agents (Phase 1) and Agent-to-Agent (A2A) services (Phase 2).

---

## Team and Roles

### Chandra Shekar Konda

- **Role:** AI Technical Director at **Oracle**
- **Project role:** Research lead and employer stakeholder for the MA-RAG evaluation effort
- **Responsibilities:** Defines technical direction, reviews architecture decisions, and guides the transition from orchestrated LangGraph pipelines toward a fully agentic multi-agent design
- **Focus areas:** Agentic AI, enterprise RAG, LLM orchestration, production readiness, and IEEE task force alignment

### Bhargav Boyapati

- **Role:** Volunteer researcher on the **IEEE Talent Meets AI** task force
- **Project role:** Hands-on engineer implementing and validating the MA-RAG prototype
- **Employer / mentor on this effort:** Works with **Chandra Shekar Konda** on this research project
- **Phase 0 contributions:** Dependency stabilization, environment configuration, `ask.py` interactive entry point, `ingest.py` local document ingestion, local FAISS retrieval, and end-to-end pipeline validation
- **Next planned work:** Phase 1 — explicit in-process agents with typed message contracts; Phase 2 — A2A microservices

### Working Relationship

Bhargav Boyapati and Chandra Shekar Konda collaborate on the MA-RAG research project under the IEEE **Talent Meets AI** task force. Chandra Shekar Konda provides technical leadership as AI Technical Director at Oracle; Bhargav Boyapati contributes implementation and validation work as a volunteer researcher.

---

## Organization Context

### Oracle

Oracle is the employer organization for Chandra Shekar Konda. In this project, Oracle-related context primarily reflects his role as AI Technical Director and his enterprise perspective on scalable, governable AI systems.

### IEEE Talent Meets AI Task Force

The IEEE task force branded **Talent Meets AI** (Talent Meets) brings together practitioners and researchers to study how talent development, AI engineering, and real-world system design intersect. The MA-RAG prototype serves as a concrete engineering artifact for discussing multi-agent RAG, interpretable reasoning traces, and the path from research code to agentic production systems.

---

## Technical Architecture (Current Prototype)

The current MA-RAG pipeline orchestrates the following stages:

1. **Planner Agent** — Decomposes the user question into ordered reasoning steps
2. **Step Definer Agent** — Converts each plan step into a concrete sub-question
3. **RAG Agent** — Retrieves document chunks, extracts relevant evidence, and generates a step-level answer
4. **Summarizer Agent** — Combines step outputs into a final answer with a confidence score

Phase 0 validation used a local document corpus ingested via `ingest.py` and queried via `ask.py`. The system supports `.pdf`, `.txt`, and `.md` sources and stores a local FAISS index for retrieval.

---

## Phase Status


| Phase   | Description                                                             | Status   |
| ------- | ----------------------------------------------------------------------- | -------- |
| Phase 0 | Stabilize repo, local ingestion, interactive Q&A, end-to-end validation | Complete |
| Phase 1 | Explicit in-process agents with typed input/output contracts            | Planned  |
| Phase 2 | Independent A2A agent services                                          | Planned  |
| Phase 3 | Production API, UI, observability, deployment                           | Planned  |


---

## Validation Examples (From Phase 0 Testing)

During Phase 0 testing on a sample knowledge corpus, the system correctly answered factual questions such as film director identification (e.g., Christopher Nolan directed Inception) with visible plan steps, retrieval traces, and confidence scoring. Local ingestion and retrieval were confirmed using `--retrieve-only` and full `ask.py` runs.

---

## Frequently Useful Facts (Single-Hop)

- **Employer name (research lead):** Chandra Shekar Konda
- **Volunteer researcher:** Bhargav Boyapati
- **Bhargav's employer/stakeholder on this project:** Chandra Shekar Konda
- **Chandra's title at Oracle:** AI Technical Director
- **Chandra's company:** Oracle
- **Bhargav's participation type:** Volunteer
- **Task force name:** IEEE Talent Meets AI (Talent Meets)
- **Research system:** MA-RAG
- **Current completed phase:** Phase 0
- **Next phase:** Phase 1 (explicit in-process agents)
- **Default LLM for agents:** gpt-4o-mini
- **Local question entry point:** ask.py
- **Local document ingestion command:** ingest.py
- **Retrieval index technology:** FAISS (local_index/)

---

## Frequently Useful Facts (Multi-Hop)

Use these chained relationships for multi-step demo questions (planner should use 2+ steps):


| Hop | Fact                                                                                       |
| --- | ------------------------------------------------------------------------------------------ |
| 1   | The volunteer researcher implementing MA-RAG Phase 0 is **Bhargav Boyapati**.              |
| 2   | The research lead and employer stakeholder Bhargav works with is **Chandra Shekar Konda**. |
| 3   | **Chandra Shekar Konda** is the **AI Technical Director at Oracle**.                       |
| 4   | The IEEE task force for this effort is **Talent Meets AI** (IEEE Talent Meets AI).         |
| 5   | The research system being prototyped is **MA-RAG** (multi-agent RAG).                      |
| 6   | **Phase 0** is **complete**; **Phase 1** (explicit in-process agents) is **planned**.      |




---

