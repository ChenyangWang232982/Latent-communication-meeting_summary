"""Closed-form StateBridge alignment for continuous LLM messages.

This mirrors the paper's communication interface: whitened orthogonal
Procrustes alignment, norm calibration, and vocabulary anchoring.  It has no
learnable parameters and is run independently for every sender message.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional


class StateBridge:
    """Align final-layer sender states to a receiver input embedding space."""

    def __init__(
        self,
        embedding_weight: torch.Tensor,
        *,
        regularization: float = 1e-3,
        snap_ratio: float = 0.3,
        vocab_chunk_size: int = 8192,
    ) -> None:
        if embedding_weight.ndim != 2:
            raise ValueError("embedding_weight must have shape [vocab_size, hidden_size]")
        self.vocab_embeddings = embedding_weight.detach().float()
        self.target_norm = self.vocab_embeddings.norm(dim=-1).mean()
        self.regularization = regularization
        self.snap_ratio = snap_ratio
        self.vocab_chunk_size = vocab_chunk_size

    @staticmethod
    def _symmetric_power(matrix: torch.Tensor, power: float, eps: float = 1e-6) -> torch.Tensor:
        matrix = (matrix + matrix.transpose(-1, -2)) * 0.5
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        powered = eigenvalues.clamp_min(eps).pow(power)
        return (eigenvectors * powered.unsqueeze(0)) @ eigenvectors.transpose(-1, -2)

    def _nearest_vocabulary_embeddings(self, states: torch.Tensor) -> torch.Tensor:
        """Exact cosine nearest-neighbour search, chunked to control VRAM use."""
        normalized_states = functional.normalize(states, dim=-1)
        normalized_vocab = functional.normalize(self.vocab_embeddings, dim=-1)
        best_scores = torch.full(
            (states.size(0),), -torch.inf, device=states.device, dtype=states.dtype
        )
        best_ids = torch.zeros(states.size(0), device=states.device, dtype=torch.long)

        for start in range(0, normalized_vocab.size(0), self.vocab_chunk_size):
            chunk = normalized_vocab[start : start + self.vocab_chunk_size]
            scores = normalized_states @ chunk.transpose(0, 1)
            chunk_scores, chunk_ids = scores.max(dim=-1)
            replace = chunk_scores > best_scores
            best_scores = torch.where(replace, chunk_scores, best_scores)
            best_ids = torch.where(replace, chunk_ids + start, best_ids)
        return self.vocab_embeddings[best_ids]

    @torch.inference_mode()
    def align(self, sender_states: torch.Tensor, message_token_ids: torch.Tensor) -> torch.Tensor:
        """Return an aligned continuous prefix with shape ``[K, hidden_size]``.

        ``message_token_ids`` are used only on the sender side as reference
        embeddings; they are never passed to the receiver as text.
        """
        if sender_states.ndim != 2:
            raise ValueError("sender_states must have shape [K, hidden_size]")
        if message_token_ids.ndim != 1:
            raise ValueError("message_token_ids must have shape [K]")

        length = min(sender_states.size(0), message_token_ids.numel())
        if length < 2:
            return sender_states[:length]

        states = sender_states[:length].float()
        references = self.vocab_embeddings[message_token_ids[:length]].float()
        if states.size(-1) != references.size(-1):
            raise ValueError(
                "StateBridge requires sender hidden size and receiver embedding size to match."
            )

        hidden_size = states.size(-1)
        centered_states = states - states.mean(dim=0, keepdim=True)
        centered_references = references - references.mean(dim=0, keepdim=True)
        identity = torch.eye(hidden_size, device=states.device, dtype=states.dtype)
        covariance_states = centered_states.transpose(0, 1) @ centered_states / length
        covariance_references = centered_references.transpose(0, 1) @ centered_references / length
        covariance_states = covariance_states + self.regularization * identity
        covariance_references = covariance_references + self.regularization * identity

        inverse_root_states = self._symmetric_power(covariance_states, -0.5)
        inverse_root_references = self._symmetric_power(covariance_references, -0.5)
        whitened_states = centered_states @ inverse_root_states
        whitened_references = centered_references @ inverse_root_references
        left, _, right_transposed = torch.linalg.svd(
            whitened_states.transpose(0, 1) @ whitened_references, full_matrices=False
        )
        rotation = left @ right_transposed
        if torch.linalg.det(rotation) < 0:
            left[:, -1].mul_(-1)
            rotation = left @ right_transposed

        aligned = (
            centered_states
            @ inverse_root_states
            @ rotation
            @ self._symmetric_power(covariance_references, 0.5)
            + references.mean(dim=0, keepdim=True)
        )
        aligned = aligned * (self.target_norm / aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6))

        if self.snap_ratio:
            snapped = self._nearest_vocabulary_embeddings(aligned)
            aligned = (1.0 - self.snap_ratio) * aligned + self.snap_ratio * snapped
        return aligned.to(dtype=sender_states.dtype)


def random_embedding_prefix(embedding_weight: torch.Tensor, length: int) -> torch.Tensor:
    """Distribution-matched random-prefix ablation from the StateBridge paper."""
    embeddings = embedding_weight.detach()
    return torch.randn(
        length, embeddings.size(1), device=embeddings.device, dtype=embeddings.dtype
    ) * embeddings.float().std().to(dtype=embeddings.dtype) + embeddings.float().mean().to(
        dtype=embeddings.dtype
    )
