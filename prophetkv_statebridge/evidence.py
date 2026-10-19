"""Fact-centric claim parsing and deterministic evidence guards for meetings."""

from __future__ import annotations

import json
import re
from typing import Any


DECISION_CUES = ("agree", "decid", "approv", "choos", "reject", "commit", "settle")
ACTION_CUES = (" will ", "assigned", "responsible", "need to", "deadline", "due ")


def _normalized(text: str) -> str:
    return re.sub(r"[^\w]+", "", text.casefold())


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


def parse_grounded_claims(message: str, *, role: str, source: str, block_id: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse claims and reject entries without a literal source quotation."""
    value = _json_value(message)
    if isinstance(value, dict):
        value = value.get("claims", [])
    if not isinstance(value, list):
        return [], ["specialist output was not a JSON claim list"]

    source_normalized = _normalized(source)
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
        if _normalized(quote) not in source_normalized:
            rejected.append("evidence quote was not found in the source block")
            continue
        quote_lower = f" {quote.casefold()} "
        if category == "decisions" and not any(cue in quote_lower for cue in DECISION_CUES):
            rejected.append("decision evidence lacks an explicit commitment cue")
            continue
        if category == "actions" and not any(cue in quote_lower for cue in ACTION_CUES):
            rejected.append("action evidence lacks an explicit assignment or commitment cue")
            continue
        accepted.append(
            {
                "category": category,
                "claim": claim,
                "evidence_quote": quote,
                "block_id": block_id,
                "timestamp": str(entry.get("timestamp", "")).strip(),
                "owner": str(entry.get("owner", "")).strip(),
                "deadline": str(entry.get("deadline", "")).strip(),
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


def verified_handoff(role: str, claims: list[dict[str, Any]]) -> str:
    """Render only verified fact units for text and latent communication."""
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
        lines.append(f"- [{role}] {claim['claim']}{suffix}\n  Evidence: \"{claim['evidence_quote']}\"")
    return "\n".join(lines)
