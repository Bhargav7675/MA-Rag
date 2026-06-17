"""Evidence Curator — assess retrieval sufficiency before generation."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from src.contracts.messages import (
    EvidenceReviewRequest,
    EvidenceReviewResponse,
    EvidenceSufficiency,
    RetrievedChunk,
)
from src.llm import create_chat_llm, is_agent_ollama
from src.prompt_template import (
    evidence_curator_human_message,
    evidence_curator_input_variables,
    evidence_curator_slm_output_format_addon,
    evidence_curator_system_message,
)
from src.slm_helpers import parse_evidence_review_response
from src.utils import EvidenceReviewFormat

AGENT_ID = "evidence_curator"
ROLE = "Evidence Curator — assess sufficiency and gaps in retrieved evidence"

_NO_INFO_MARKERS = (
    "no related information",
    "no relevant information",
    "not mentioned",
)


class EvidenceCuratorAgent:
    def __init__(self, *, agent_id: str = AGENT_ID, role: str = ROLE):
        self.agent_id = agent_id
        self.role = role

    def run(self, request: EvidenceReviewRequest) -> EvidenceReviewResponse:
        heuristic = self._heuristic_review(request.chunks)
        if heuristic is not None:
            return EvidenceReviewResponse(
                run_id=request.run_id,
                step_index=request.step_index,
                sufficiency=heuristic["sufficiency"],
                gaps=heuristic["gaps"],
                proceed=heuristic["proceed"],
                rationale=heuristic["rationale"],
            )

        chunks_text = self._format_chunks(request.chunks)
        system_message = evidence_curator_system_message
        if is_agent_ollama("evidence_curator"):
            system_message += evidence_curator_slm_output_format_addon
        messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(evidence_curator_human_message),
        ]
        prompt = ChatPromptTemplate(
            input_variables=evidence_curator_input_variables,
            messages=messages,
        )
        llm = create_chat_llm(agent_id="evidence_curator", temperature=0.0)
        inputs = {"question": request.question, "chunks": chunks_text}

        if is_agent_ollama("evidence_curator"):
            text = (prompt | llm | StrOutputParser()).invoke(inputs)
            parsed = parse_evidence_review_response(text)
        else:
            raw = (prompt | llm.with_structured_output(EvidenceReviewFormat)).invoke(inputs)
            parsed = {
                "sufficiency": raw.sufficiency.lower(),
                "proceed": str(raw.proceed).lower() == "yes",
                "gaps": raw.gaps,
                "rationale": raw.rationale,
            }

        sufficiency = self._normalize_sufficiency(parsed["sufficiency"])
        return EvidenceReviewResponse(
            run_id=request.run_id,
            step_index=request.step_index,
            sufficiency=sufficiency,
            gaps=parsed["gaps"],
            proceed=bool(parsed["proceed"]),
            rationale=parsed["rationale"],
        )

    def _heuristic_review(self, chunks: list[RetrievedChunk]):
        if not chunks:
            return {
                "sufficiency": EvidenceSufficiency.INSUFFICIENT,
                "gaps": ["No documents retrieved"],
                "proceed": False,
                "rationale": "Retriever returned zero chunks.",
            }

        nonempty = [c for c in chunks if c.text and c.text.strip()]
        if not nonempty:
            return {
                "sufficiency": EvidenceSufficiency.INSUFFICIENT,
                "gaps": ["Retrieved chunks are empty"],
                "proceed": False,
                "rationale": "All retrieved chunks were empty.",
            }

        lower_texts = [c.text.lower() for c in nonempty]
        if all(any(marker in text for marker in _NO_INFO_MARKERS) for text in lower_texts):
            return {
                "sufficiency": EvidenceSufficiency.INSUFFICIENT,
                "gaps": ["Chunks report no related information"],
                "proceed": False,
                "rationale": "Extracted passages contain no related information.",
            }

        return None

    @staticmethod
    def _format_chunks(chunks: list[RetrievedChunk]) -> str:
        parts = []
        for chunk in chunks:
            preview = chunk.text[:1200]
            parts.append(f"[{chunk.doc_id}]\n{preview}")
        return "\n\n".join(parts)

    @staticmethod
    def _normalize_sufficiency(value: str) -> EvidenceSufficiency:
        normalized = (value or "").strip().lower()
        if normalized == "sufficient":
            return EvidenceSufficiency.SUFFICIENT
        if normalized == "partial":
            return EvidenceSufficiency.PARTIAL
        return EvidenceSufficiency.INSUFFICIENT
