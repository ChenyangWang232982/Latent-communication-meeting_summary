"""Interlat Section 3.2 output-distribution objectives."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from interlat_paper.model import IGNORE_INDEX


def supervised_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Select supervised positions exactly as the official Interlat implementation."""
    return logits[labels.ne(IGNORE_INDEX)]


def plan_alignment_loss(
    latent_logits: torch.Tensor,
    latent_labels: torch.Tensor,
    plan_logits: torch.Tensor,
    plan_labels: torch.Tensor,
) -> torch.Tensor:
    """KL plus logit-distribution cosine alignment from Interlat Eq. (3)."""
    latent = supervised_logits(latent_logits, latent_labels)
    plan = supervised_logits(plan_logits, plan_labels).detach()
    size = min(latent.size(0), plan.size(0))
    if size == 0:
        return latent_logits.new_zeros(())
    latent, plan = latent[:size], plan[:size]
    kl = F.kl_div(F.log_softmax(latent, dim=-1), F.softmax(plan, dim=-1), reduction="batchmean")
    cosine = 1.0 - F.cosine_similarity(
        F.softmax(latent, dim=-1).reshape(-1), F.softmax(plan, dim=-1).reshape(-1), dim=0
    )
    return 0.7 * kl + 0.3 * cosine


def random_contrast_loss(
    matched_logits: torch.Tensor,
    matched_labels: torch.Tensor,
    mismatched_logits: torch.Tensor,
    mismatched_labels: torch.Tensor,
) -> torch.Tensor:
    """Official hinge loss: max(0, margin - JS) for cross-task messages."""
    matched = supervised_logits(matched_logits, matched_labels)
    mismatched = supervised_logits(mismatched_logits, mismatched_labels).detach()
    size = min(matched.size(0), mismatched.size(0))
    if size == 0:
        return matched_logits.new_zeros(())
    p = F.softmax(matched[:size], dim=-1).clamp_min(1e-8)
    q = F.softmax(mismatched[:size], dim=-1).clamp_min(1e-8)
    midpoint = 0.5 * (p + q)
    # Keep the released code's KL argument order rather than substituting a
    # mathematically equivalent-looking variant.
    js = 0.5 * F.kl_div(p.log(), midpoint, reduction="batchmean") + 0.5 * F.kl_div(
        q.log(), midpoint, reduction="batchmean"
    )
    return (0.69 - js).clamp_min(0.0)
