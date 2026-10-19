"""Fact-centric claim parsing and deterministic evidence guards for meetings."""

from __future__ import annotations

import json
import re
from typing import Any


DECISION_CUES = ("agree", "decid", "approv", "choos", "reject", "commit", "settle")
ACTION_CUES = (" will ", "assigned", "responsible", "need to", "deadline", "due ")
FACT_DECISION_CUES = (" decided ", " agreed ", " approved ", " committed ", " settled ")
TASK_VERBS = (" prepare", " send", " review", " write", " investigate", " finalize", " create", " complete")


def _normalized(text: str) -> str:
    return re.sub(r"[^\w]+", "", text.casefold())


def _word_spans(text: str) -> list[tuple[str, int, int]]:
    """Return normalized words together with their original character spans."""
    return [(match.group().casefold(), match.start(), match.end()) for match in re.finditer(r"\w+", text)]


def _recover_source_quote(quote: str, source: str) -> str | None:
    """Locate a model-copied quote, tolerating harmless transcript formatting drift.

    Specialists are asked to copy their evidence verbatim, but ASR transcripts often
    differ in whitespace, punctuation, or omit a short filler word.  We only recover
    a source span when a substantial *contiguous* word sequence is shared; this is
    deliberately not a semantic-paraphrase match.
    """
    quote_normalized = _normalized(quote)
    source_normalized = _normalized(source)
    if quote_normalized and quote_normalized in source_normalized:
        return quote

    quote_words = [word for word, _, _ in _word_spans(quote)]
    source_words = _word_spans(source)
    if len(quote_words) < 8 or len(source_words) < 8:
        return None

    # A shared six-word anchor avoids accepting generic meeting language.  Expand
    # every anchor to its maximal exact run and keep the strongest evidence span.
    anchor_size = 6
    source_anchors: dict[tuple[str, ...], list[int]] = {}
    for start in range(len(source_words) - anchor_size + 1):
        key = tuple(word for word, _, _ in source_words[start : start + anchor_size])
        source_anchors.setdefault(key, []).append(start)

    best: tuple[int, int, int] | None = None
    for quote_start in range(len(quote_words) - anchor_size + 1):
        key = tuple(quote_words[quote_start : quote_start + anchor_size])
        for source_start in source_anchors.get(key, []):
            left = 0
            while (
                quote_start - left - 1 >= 0
                and source_start - left - 1 >= 0
                and quote_words[quote_start - left - 1] == source_words[source_start - left - 1][0]
            ):
                left += 1
            right = anchor_size
            while (
                quote_start + right < len(quote_words)
                and source_start + right < len(source_words)
                and quote_words[quote_start + right] == source_words[source_start + right][0]
            ):
                right += 1
            run_start = source_start - left
            run_length = left + right
            if best is None or run_length > best[2]:
                best = (run_start, run_start + run_length, run_length)

    required = max(anchor_size, (len(quote_words) + 1) // 2)
    if best is None or best[2] < required:
        return None
    start, end, _ = best
    # Return the enclosing utterance, rather than an arbitrary middle fragment:
    # role cues (for example "will") can precede a filler that the model skipped.
    char_start = source_words[start][1]
    char_end = source_words[end - 1][2]
    while char_start > 0 and source[char_start - 1] not in ".!?\n":
        char_start -= 1
    while char_end < len(source) and source[char_end] not in ".!?\n":
        char_end += 1
    if char_end < len(source):
        char_end += 1
    return source[char_start:char_end].strip()


def _json_value(text: str) -> Any | None:
    """Find the first JSON object or array in a model response."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def _grounded_metadata(value: str, claim: str, quote: str) -> str:
    """Keep optional metadata only when the claim or its evidence names it."""
    normalized_value = _normalized(value)
    if not normalized_value:
        return ""
    supported_by = _normalized(f"{claim} {quote}")
    return value if normalized_value in supported_by else ""


def parse_grounded_claims(message: str, *, role: str, source: str, block_id: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse claims and reject entries without recoverable source evidence."""
    value = _json_value(message)
    if isinstance(value, dict):
        value = value.get("claims", [])
    if not isinstance(value, list):
        return [], ["specialist output was not a JSON claim list"]

    accepted: list[dict[str, Any]] = []
    rejected: list[str] = []
    for entry in value:
        if not isinstance(entry, dict):
            rejected.append("claim entry was not an object")
            continue
        category = str(entry.get("category", role)).strip().lower()
        claim = str(entry.get("claim", "")).strip()
        quote = str(entry.get("evidence_quote", "")).strip()
        if category != role:
            rejected.append("claim category does not match its specialist")
            continue
        if len(claim) < 3 or len(quote) < 8:
            rejected.append("claim or evidence quote was too short")
            continue
        requested_quote = quote
        recovered_quote = _recover_source_quote(requested_quote, source)
        if recovered_quote is None:
            rejected.append("evidence quote was not found in the source block")
            continue
        quote = recovered_quote
        # Evaluate role cues from the specialist's requested span: recovery may
        # start after an ASR filler such as "uh", while the commitment cue is
        # immediately before that filler in the same source utterance.
        quote_lower = f" {requested_quote.casefold()} "
        if category == "decisions" and not any(cue in quote_lower for cue in DECISION_CUES):
            rejected.append("decision evidence lacks an explicit commitment cue")
            continue
        if category == "actions" and not any(cue in quote_lower for cue in ACTION_CUES):
            rejected.append("action evidence lacks an explicit assignment or commitment cue")
            continue
        claim_lower = f" {claim.casefold()} "
        if category == "facts" and any(cue in claim_lower for cue in FACT_DECISION_CUES):
            rejected.append("facts claim asserts a decision-like outcome; use decisions or discussion instead")
            continue
        if category == "next_steps" and any(verb in claim_lower for verb in TASK_VERBS):
            rejected.append("next_steps claim is an assigned task; it belongs to actions")
            continue
        timestamp = str(entry.get("timestamp", "")).strip()
        owner = str(entry.get("owner", "")).strip()
        deadline = str(entry.get("deadline", "")).strip()
        accepted.append(
            {
                "category": category,
                "claim": claim,
                "evidence_quote": quote,
                "block_id": block_id,
                "timestamp": _grounded_metadata(timestamp, claim, quote),
                # A speaker who asks a question is not automatically responsible
                # for resolving it.  Only action items expose an owner field.
                "owner": _grounded_metadata(owner, claim, quote) if category == "actions" else "",
                "deadline": _grounded_metadata(deadline, claim, quote),
                "confidence": str(entry.get("confidence", "high")).strip().lower(),
            }
        )
    return accepted, rejected


def claim_packet(claims: list[dict[str, Any]]) -> str:
    """Create a compact, deterministic packet for the semantic verifier."""
    return json.dumps(
        [
            {"index": index, "category": claim["category"], "claim": claim["claim"], "evidence_quote": claim["evidence_quote"]}
            for index, claim in enumerate(claims)
        ],
        ensure_ascii=False,
    )


def parse_verdicts(message: str, expected_count: int) -> set[int]:
    """Return indices explicitly marked supported by the semantic verifier."""
    value = _json_value(message)
    if isinstance(value, dict):
        value = value.get("verdicts", [])
    if not isinstance(value, list):
        return set()
    supported: set[int] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        verdict = str(item.get("verdict", "")).strip().lower()
        if isinstance(index, int) and 0 <= index < expected_count and verdict == "supported":
            supported.add(index)
    return supported


def verified_handoff(role: str, claims: list[dict[str, Any]], *, include_evidence: bool = False) -> str:
    """Render verified claims for communication, optionally with audit evidence.

    Evidence is needed to audit the extraction stage, but long verbatim quotes
    consume the small StateBridge budget and can obscure the compact conclusion.
    The communication payload therefore defaults to claims only; callers can
    request the evidence-rich version for reports and JSON inspection.
    """
    if not claims:
        return f"{role}: None explicitly stated."
    lines = []
    for claim in claims:
        metadata = []
        if claim["timestamp"]:
            metadata.append(claim["timestamp"])
        if claim["owner"]:
            metadata.append(f"owner={claim['owner']}")
        if claim["deadline"]:
            metadata.append(f"deadline={claim['deadline']}")
        suffix = f" ({'; '.join(metadata)})" if metadata else ""
        line = f"- [{role}] {claim['claim']}{suffix}"
        if include_evidence:
            line = f"- Evidence [{role}]: \"{claim['evidence_quote']}\"\n  Verified claim: {claim['claim']}{suffix}"
        lines.append(line)
    return "\n".join(lines)
