"""E5 semantic block ranking. No training and no text handoff to KV receiver."""
import time


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
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(device).eval()
        self.device = device
        self.name = model
        self.limit = min(512, self.tokenizer.model_max_length,
                         self.model.config.max_position_embeddings)
        self.load_seconds = time.perf_counter() - started

    def encode(self, texts):
        """Window every block, masked mean pool, length-weight then normalize.

        Symmetric E5 similarity uses query: on both sides. Every content token
        is encoded, including tails of turns longer than the encoder window.
        """
        torch = self.torch
        prefix = self.tokenizer.encode('query: ', add_special_tokens=False)
        capacity = self.limit - len(prefix) - self.tokenizer.num_special_tokens_to_add(False)
        if capacity < 1:
            raise ValueError('Embedding context has no space for block text')
        windows, owners, weights = [], [], []
        for owner, text in enumerate(texts):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            for start in range(0, max(1, len(ids)), capacity):
                chunk = ids[start:start + capacity]
                windows.append({'input_ids': self.tokenizer.build_inputs_with_special_tokens(prefix + chunk)})
                owners.append(owner)
                weights.append(max(1, len(chunk)))
        pooled = torch.zeros(len(texts), self.model.config.hidden_size)
        counts = torch.zeros(len(texts), 1)
        with torch.inference_mode():
            for start in range(0, len(windows), 16):
                batch = self.tokenizer.pad(windows[start:start + 16], padding=True, return_tensors='pt').to(self.device)
                hidden = self.model(**batch).last_hidden_state
                mask = batch['attention_mask'].unsqueeze(-1)
                vectors = ((hidden * mask).sum(1) / mask.sum(1)).float().cpu()
                for j, embedding in enumerate(vectors, start):
                    pooled[owners[j]] += embedding * weights[j]
                    counts[owners[j]] += weights[j]
        return torch.nn.functional.normalize(pooled / counts.clamp_min(1), dim=1), {
            'embedding_input_tokens': sum(len(w['input_ids']) for w in windows),
            'embedding_windows': len(windows),
        }

    def rank(self, texts, attention, anchor_ids, weight):
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
        return scores, stats
