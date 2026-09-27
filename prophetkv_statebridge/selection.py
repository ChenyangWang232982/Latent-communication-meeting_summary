"""Lightweight query-aware chunk selection for long documents.

The selector deliberately has no embedding-model dependency: it is intended
to validate whether query-conditioned context reduction helps the downstream
StateBridge workflow before replacing it with an engine-level KV reuse backend.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+-]*")


@dataclass(frozen=True)
class Chunk:
    """A token-bounded document fragment and its stable local identifier."""

    chunk_id: int
    text: str
    token_count: int


def terms(text: str) -> list[str]:
    return [term.lower() for term in WORD_PATTERN.findall(text)]


def chunk_document(tokenizer, text: str, *, chunk_tokens: int, overlap_tokens: int) -> list[Chunk]:
    """Split a document in tokenizer space so every selected unit has a known cost."""
    if chunk_tokens < 32:
        raise ValueError("chunk_tokens must be at least 32")
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be non-negative and smaller than chunk_tokens")

    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        return []
    step = chunk_tokens - overlap_tokens
    chunks: list[Chunk] = []
    for start in range(0, len(ids), step):
        part = ids[start : start + chunk_tokens]
        if not part:
            break
        chunks.append(
            Chunk(
                chunk_id=len(chunks),
                text=tokenizer.decode(part, skip_special_tokens=True),
                token_count=len(part),
            )
        )
        if start + chunk_tokens >= len(ids):
            break
    return chunks


def rank_chunks(chunks: list[Chunk], query: str) -> list[tuple[Chunk, float]]:
    """Rank chunks by a compact BM25-style lexical score.

    Scores are calculated separately for each document. This makes selection
    stable for one meeting/transcript and avoids corpus-wide preprocessing.
    """
    if not chunks:
        return []

    query_terms = Counter(terms(query))
    if not query_terms:
        return [(chunk, 0.0) for chunk in chunks]
    document_terms = [Counter(terms(chunk.text)) for chunk in chunks]
    doc_frequency = Counter()
    for counts in document_terms:
        doc_frequency.update(counts.keys())
    average_length = sum(sum(counts.values()) for counts in document_terms) / len(document_terms)
    k1, b = 1.2, 0.75

    scored: list[tuple[Chunk, float]] = []
    for chunk, counts in zip(chunks, document_terms):
        length = max(sum(counts.values()), 1)
        score = 0.0
        for term, query_count in query_terms.items():
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            inverse_frequency = math.log(1.0 + (len(chunks) - doc_frequency[term] + 0.5) / (doc_frequency[term] + 0.5))
            denominator = frequency + k1 * (1.0 - b + b * length / max(average_length, 1.0))
            score += query_count * inverse_frequency * frequency * (k1 + 1.0) / denominator
        scored.append((chunk, score))

    return sorted(scored, key=lambda item: (-item[1], item[0].chunk_id))


def select_chunks(chunks: list[Chunk], query: str, limit: int) -> list[tuple[Chunk, float]]:
    """Return the most relevant chunks in source order for coherent reading."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    # Preserve source order after ranking: generation reads coherent context.
    best = rank_chunks(chunks, query)[:limit]
    return sorted(best, key=lambda item: item[0].chunk_id)
