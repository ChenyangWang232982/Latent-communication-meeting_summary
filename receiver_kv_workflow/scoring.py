"""Receiver-query attention selection inspired by CacheBack (2026).

Implements its mean-attention/concentration score, not its complete protocol:
no latent thought generation and no continuous-row reconstruction. Queries
are processed in small batches to avoid full source x source attentions.
"""

import torch
import torch.nn.functional as F

from .cache import forward_tokens, select_cache


def pool(values):
    shape = values.shape
    return F.max_pool1d(values.reshape(-1, 1, shape[-1]), 7, stride=1, padding=3).reshape(shape)


class AttentionScore:
    def __init__(self, length, kv_heads):
        self.length = length
        self.kv_heads = kv_heads
        self.means = None
        self.squares = None
        self.count = 0

    def add(self, attentions):
        if not attentions or any(a is None for a in attentions):
            raise RuntimeError("Model did not return eager attention weights")
        batch_means, batch_squares = [], []
        count = attentions[0].shape[-2]
        for attention in attentions:
            # Average query heads sharing each KV head BEFORE nonlinear pooling.
            a = attention[0, :, :, :self.length].float()
            groups = a.reshape(self.kv_heads, -1, count, self.length).mean(dim=1)
            batch_means.append(groups.sum(dim=1))
            batch_squares.append(pool(groups).square().sum(dim=1))
        means, squares = torch.stack(batch_means), torch.stack(batch_squares)
        self.means = means if self.means is None else self.means + means
        self.squares = squares if self.squares is None else self.squares + squares
        self.count += count

    def finish(self):
        means = self.means / self.count
        q = pool(means.mean(dim=1).sum(dim=0))
        numerator = (self.squares / self.count).sum(dim=(0, 1))
        denominator = pool(means).square().sum(dim=(0, 1))
        concentration = torch.where(denominator > 0, numerator / denominator.clamp_min(1e-30), 1.0)
        return (q * concentration.square()).cpu().tolist()


@torch.inference_mode()
def query_scores(model, source_cache, query_ids, *, query_batch=8):
    length = source_cache.get_seq_length()
    cache = select_cache(source_cache, range(length))
    accumulator = AttentionScore(length, model.config.num_key_value_heads)
    old_implementation = model.config._attn_implementation
    model.set_attn_implementation("eager")
    try:
        for start in range(0, query_ids.shape[1], query_batch):
            output = forward_tokens(
                model, query_ids[:, start:start + query_batch], cache, length + start,
                output_attentions=True, logits_to_keep=1,
            )
            cache = output.past_key_values
            accumulator.add(output.attentions)
            del output
        return accumulator.finish()
    finally:
        model.set_attn_implementation(old_implementation)
