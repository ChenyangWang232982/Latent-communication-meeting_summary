"""Qwen3 embedding-based block ranking; only original KV is handed off."""
import time

# A practical per-window limit, not a source-transcript truncation limit.
# Longer turns are encoded in consecutive windows and then pooled.
MAX_WINDOW_TOKENS = 8192
MAX_BATCH_TOKENS = 8192
MAX_BATCH_SIZE = 8


def unit_scale(values):
    low, high = min(values), max(values)
    return [(v - low) / (high - low) if high > low else 0.0 for v in values]


def combine_scores(attention, semantic, weight):
    return [(1 - weight) * a + weight * s
            for a, s in zip(unit_scale(attention), unit_scale(semantic))]


class SemanticRouter:
    def __init__(self, model, device='cpu'):
        import torch
        from transformers import AutoModel, AutoTokenizer
        started = time.perf_counter()
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side='left')
        self.model = AutoModel.from_pretrained(model).to(device).eval()
        if getattr(self.model.config, 'model_type', None) != 'qwen3':
            raise ValueError('This embedding adapter requires a Qwen3 embedding model, such as Qwen/Qwen3-Embedding-0.6B')
        # Recover the single-sequence suffix from the tokenizer's public API.
        # Qwen3-Embedding appends <|endoftext|> (not necessarily eos_token_id).
        # Transformers 5 no longer exposes build_inputs_with_special_tokens.
        self.special_suffix = self.tokenizer('', add_special_tokens=True)['input_ids']
        probe = 'Meeting context: verify tokenization.'
        raw_probe = self.tokenizer(probe, add_special_tokens=False)['input_ids']
        processed_probe = self.tokenizer(probe, add_special_tokens=True)['input_ids']
        if processed_probe != raw_probe + self.special_suffix:
            raise ValueError('Unsupported Qwen3 embedding tokenizer special-token template')
        self.device = device
        self.name = model
        self.limit = min(MAX_WINDOW_TOKENS, self.tokenizer.model_max_length,
                         self.model.config.max_position_embeddings)
        self.load_seconds = time.perf_counter() - started

    def encode(self, texts):
        """Last-token pool each window, length-weight windows, then normalize.

        Both sides are speaker blocks, so encode them symmetrically without a
        retrieval-query instruction. All block tokens are processed.
        """
        torch = self.torch
        if not texts:
            raise ValueError('Cannot encode an empty list of speaker blocks')
        capacity = self.limit - len(self.special_suffix)
        if capacity < 1:
            raise ValueError('Embedding context has no space for block text')
        windows, owners, weights = [], [], []
        for owner, text in enumerate(texts):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            for start in range(0, max(1, len(ids)), capacity):
                chunk = ids[start:start + capacity]
                input_ids = chunk + self.special_suffix
                if not input_ids:
                    if self.tokenizer.eos_token_id is None:
                        raise ValueError('Embedding tokenizer requires an EOS token to encode an empty block')
                    input_ids = [self.tokenizer.eos_token_id]
                windows.append({'input_ids': input_ids})
                owners.append(owner)
                weights.append(max(1, len(chunk)))
        pooled = torch.zeros(len(texts), self.model.config.hidden_size)
        counts = torch.zeros(len(texts), 1)
        with torch.inference_mode():
            start = 0
            while start < len(windows):
                end = start
                max_length = 0
                while end < len(windows) and end - start < MAX_BATCH_SIZE:
                    next_length = max(max_length, len(windows[end]['input_ids']))
                    if end > start and next_length * (end - start + 1) > MAX_BATCH_TOKENS:
                        break
                    max_length = next_length
                    end += 1
                batch = self.tokenizer.pad(windows[start:end], padding=True, return_tensors='pt').to(self.device)
                hidden = self.model(**batch, use_cache=False).last_hidden_state
                # Qwen3's model card uses the last non-padding token.
                mask = batch['attention_mask']
                positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
                last = positions.masked_fill(~mask.bool(), -1).max(dim=1).values
                vectors = hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float().cpu()
                for j, embedding in enumerate(vectors, start):
                    pooled[owners[j]] += embedding * weights[j]
                    counts[owners[j]] += weights[j]
                start = end
        return torch.nn.functional.normalize(pooled / counts.clamp_min(1), dim=1), {
            'embedding_input_tokens': sum(len(w['input_ids']) for w in windows),
            'embedding_windows': len(windows),
            'embedding_window_limit': self.limit,
        }

    def rank(self, texts, attention, anchor_ids, weight, *, return_vectors=False):
        started = time.perf_counter()
        vectors, stats = self.encode(texts)
        semantic, matches = [], []
        # Compare each block to attention-selected seeds; exclude self-matches.
        # Matrix is N x seeds, not N x N when only a subset was selected.
        similarities = vectors @ vectors[anchor_ids].T
        for i in range(len(texts)):
            candidates = [(float(similarities[i, j]), anchor)
                          for j, anchor in enumerate(anchor_ids) if anchor != i]
            score, match = max(candidates, default=(0.0, None))
            semantic.append(score)
            matches.append(match)
        scores = combine_scores(attention, semantic, weight)
        stats.update(model=self.name, device=self.device, seconds=time.perf_counter() - started,
                     model_load_seconds=self.load_seconds, weight=weight,
                     anchor_block_ids=anchor_ids,
                     blocks=[{'block_id': i, 'attention': attention[i], 'cosine': semantic[i],
                              'nearest_seed_block': matches[i], 'combined_score': scores[i]}
                             for i in range(len(texts))])
        return (scores, stats, vectors) if return_vectors else (scores, stats)
