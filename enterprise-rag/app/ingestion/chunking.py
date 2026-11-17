"""Structure-aware, parent/child chunking.

Documents are split on headings into sections. Each section becomes one or more
*parents* (up to parent_max chars). Each parent is split into small *children*
(~child_chars) with a little overlap. Children are embedded and searched; the
parent is what the LLM reads, so retrieval is precise and the answer context is whole.
"""
import re
from dataclasses import dataclass

from .types import RawPage

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
PARA_SPLIT = re.compile(r"\n\s*\n")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ParentDraft:
    ordinal: int
    section: str  # "Leave policy > Annual leave"
    location: str | None
    text: str
    children: list[str]


def split_sections(text: str, stack: list[tuple[int, str]]) -> list[tuple[tuple[str, ...], str]]:
    """Split markdown-ish text into (heading_path, body). `stack` carries headings across pages."""
    sections: list[tuple[tuple[str, ...], str]] = []
    body: list[str] = []
    in_fence = False

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append((tuple(h for _, h in stack), content))
        body.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip("*_ ")))
        else:
            body.append(line)
    flush()
    return sections


def split_long(text: str, max_chars: int) -> list[str]:
    """Break an over-long paragraph on sentence boundaries, hard-splitting as a last resort."""
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    cur = ""
    for sentence in SENTENCE_SPLIT.split(text):
        while len(sentence) > max_chars:
            if cur:
                parts.append(cur)
                cur = ""
            cut = sentence.rfind(" ", 0, max_chars)  # prefer a word boundary
            if cut <= 0:
                cut = max_chars  # no whitespace at all (e.g. a long token): cut anywhere
            parts.append(sentence[:cut].rstrip())
            sentence = sentence[cut:].lstrip()
        if cur and len(cur) + len(sentence) + 1 > max_chars:
            parts.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}" if cur else sentence
    if cur:
        parts.append(cur)
    return parts


def _tail(text: str, n: int) -> str:
    """Last ~n chars of text, starting on a word boundary."""
    if n <= 0:
        return ""
    tail = text[-n:]
    cut = tail.find(" ")
    return tail[cut + 1 :] if 0 <= cut < len(tail) - 1 else tail


def pack(paragraphs: list[str], max_chars: int, overlap: int = 0) -> list[str]:
    """Greedily pack paragraphs into chunks of about max_chars (soft limit: max_chars + overlap)."""
    chunks: list[str] = []
    cur = ""
    for para in paragraphs:
        for piece in split_long(para, max_chars):
            if cur and len(cur) + len(piece) + 2 > max_chars:
                chunks.append(cur)
                carry = _tail(cur, overlap)
                cur = f"{carry}\n\n{piece}" if carry else piece
            else:
                cur = f"{cur}\n\n{piece}" if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in PARA_SPLIT.split(text) if p.strip()]


def chunk_document(
    pages: list[RawPage], *, child_chars: int = 900, overlap: int = 120, parent_max: int = 3500
) -> list[ParentDraft]:
    stack: list[tuple[int, str]] = []
    parents: list[ParentDraft] = []
    for page in pages:
        for path, body in split_sections(page.text, stack):
            for parent_text in pack(_paragraphs(body), parent_max, 0):
                parents.append(
                    ParentDraft(
                        ordinal=len(parents),
                        section=" > ".join(path),
                        location=page.location,
                        text=parent_text,
                        children=pack(_paragraphs(parent_text), child_chars, overlap),
                    )
                )
    return parents
