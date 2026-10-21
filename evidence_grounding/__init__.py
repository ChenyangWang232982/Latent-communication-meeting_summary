"""Training-free claim-to-evidence retrieval for grounded meeting summaries."""

from .retrieval import EvidenceCandidate, retrieve_evidence_candidates

__all__ = ["EvidenceCandidate", "retrieve_evidence_candidates"]
