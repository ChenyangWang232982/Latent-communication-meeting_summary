"""Lightweight candidate retrieval used before semantic claim verification.

The retriever deliberately has no model dependency. It narrows a full meeting
transcript to a few auditable evidence windows; the caller's LLM then decides
whether a window actually entails the claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "we",
    "will", "with", "would", "you",
})


@dataclass(frozen=True)
class EvidenceCandidate:
    """A source span retrieved for one generated claim."""

    text: str
    start: int
    end: int
    score: float


def _terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9_]+", text)
        if len(token) > 1 and token.casefold() not in STOPWORDS
    }


def _units(source: str) -> list[tuple[str, int, int]]:
    """Split a transcript into auditable utterance-like units.

    Newlines are preferred because ASR transcripts commonly preserve turns or
    timestamps there. Sentence boundaries provide a fallback for plain text.
    """
    spans: list[tuple[str, int, int]] = []
    for match in re.finditer(r"[^\n]+", source):
        text = match.group().strip()
        if text:
            offset = match.start() + len(match.group()) - len(match.group().lstrip())
            spans.append((text, offset, offset + len(text)))
    if len(spans) > 1:
        return spans

    spans.clear()
    for match in re.finditer(r"[^.!?]+[.!?]?", source, flags=re.DOTALL):
        text = match.group().strip()
        if text:
            offset = match.start() + len(match.group()) - len(match.group().lstrip())
            spans.append((text, offset, offset + len(text)))
    return spans


def retrieve_evidence_candidates(
    source: str,
    claim: str,
    *,
    keywords: list[str] | None = None,
    limit: int = 3,
    neighbor_units: int = 1,
) -> list[EvidenceCandidate]:
    """Return high-overlap source windows for a claim.

    The score is intentionally lexical and deterministic. Keyword hints make
    paraphrased claims retrievable without treating a generated quote as proof.
    A semantic verifier must still approve the selected candidate.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    units = _units(source)
    if not units:
        return []
    query_terms = _terms(claim)
    for keyword in keywords or []:
        query_terms.update(_terms(keyword))
    if not query_terms:
        return []

    ranked: list[tuple[float, int]] = []
    for index, (text, _, _) in enumerate(units):
        terms = _terms(text)
        overlap = len(query_terms & terms)
        if overlap:
            # Reward coverage more than repetitions in a long transcript turn.
            score = overlap / len(query_terms) + overlap / max(len(terms), 1)
            ranked.append((score, index))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    candidates: list[EvidenceCandidate] = []
    seen_ranges: set[tuple[int, int]] = set()
    for score, index in ranked:
        left = max(0, index - neighbor_units)
        right = min(len(units), index + neighbor_units + 1)
        start = units[left][1]
        end = units[right - 1][2]
        key = (start, end)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        candidates.append(EvidenceCandidate(source[start:end], start, end, score))
        if len(candidates) == limit:
            break
    return candidates
