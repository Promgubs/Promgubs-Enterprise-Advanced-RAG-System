"""LLM-based query understanding: follow-up condensing, multi-query expansion and HyDE.

Every function degrades gracefully: if the LLM call fails, retrieval proceeds with the original query.
"""
import json
import logging
import re
from dataclasses import dataclass, field

from ..generation.llm import get_llm

log = logging.getLogger(__name__)

CONDENSE_SYSTEM = (
    "Rewrite the user's follow-up question as a single standalone search query, using the conversation "
    "for missing context (names, dates, 'it', 'that one'). Keep every concrete detail. "
    "Output only the query, nothing else."
)

EXPAND_SYSTEM = """You help a search engine over internal company documents. Given a search query, reply with JSON only:
{"rewrites": ["...", "...", "..."], "hypothetical_answer": "..."}
- rewrites: 3 alternative phrasings (synonyms, expanded acronyms, more formal or more casual wording).
- hypothetical_answer: a 2-3 sentence passage that could plausibly appear in a document answering the query.
Do not invent specific figures or names. No markdown fences, no commentary."""


@dataclass
class Expansion:
    rewrites: list[str] = field(default_factory=list)
    hypothetical: str | None = None


def condense_question(history: list[dict], question: str) -> str:
    convo = "\n".join(f"{t['role']}: {t['content']}" for t in history[-6:])
    try:
        out = get_llm().complete(
            CONDENSE_SYSTEM, f"Conversation:\n{convo}\n\nFollow-up question: {question}", max_tokens=200
        )
        return out.strip().strip('"') or question
    except Exception:
        log.exception("condense_question failed; using the raw question")
        return question


def expand_query(query: str) -> Expansion:
    try:
        raw = get_llm().complete(EXPAND_SYSTEM, query, max_tokens=500)
        match = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(match.group(0)) if match else {}
        rewrites = [r.strip() for r in data.get("rewrites", []) if isinstance(r, str) and r.strip()]
        hyde = data.get("hypothetical_answer")
        return Expansion(rewrites=rewrites[:4], hypothetical=hyde.strip() if isinstance(hyde, str) and hyde.strip() else None)
    except Exception:
        log.exception("expand_query failed; continuing without expansion")
        return Expansion()
