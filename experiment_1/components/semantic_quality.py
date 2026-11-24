"""Optional reference-based BERTScore, separate from generation/resources."""
import importlib.metadata
from .config import VARIANTS


def add_bertscore(samples, model_name, layers, device, *, variants=VARIANTS):
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        raise RuntimeError("Install optional metrics: pip install -r experiment_1/requirements-quality.txt") from exc
    from transformers import AutoConfig, AutoTokenizer
    # Same slow tokenizer mode as BERTScorer; reject overlength BEFORE scoring.
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    config = AutoConfig.from_pretrained(model_name)
    limits = [value for value in (tokenizer.model_max_length, getattr(config, "max_position_embeddings", None))
              if isinstance(value, int) and 0 < value < 1000000]
    if not limits:
        raise ValueError("Cannot establish the BERTScore encoder context limit")
    limit = min(limits)
    if getattr(config, "model_type", "") in ("roberta", "longformer"):
        limit = min(limit, config.max_position_embeddings - 2)
    # BERTScore 0.3.13 strips input and adds a prefix space for these tokenizers.
    # Check that same representation so its later truncation cannot drop content.
    encoding_options = {"add_prefix_space": True} if getattr(config, "model_type", "") in ("roberta", "longformer", "gpt2") else {}
    pairs = [(sample, name) for sample in samples for name in variants]
    for sample, name in pairs:
        for label, text in (("prediction", sample["result"]["answers"][name]), ("reference", sample["reference"])):
            count = len(tokenizer(text.strip(), add_special_tokens=True, truncation=False,
                                  **encoding_options)["input_ids"])
            if count > limit:
                raise ValueError(f"BERTScore {sample['id']} {name} {label}: {count}>{limit} encoder tokens. "
                                 "No text was truncated; select a longer-context encoder for ALL samples.")
    scorer = BERTScorer(model_type=model_name, num_layers=layers, device=device,
                        batch_size=1, idf=False, rescale_with_baseline=False, use_fast_tokenizer=False)
    # BERTScorer creates its own tokenizer. Some checkpoints leave its maximum
    # length at HF's ~1e30 sentinel, which overflows the Rust tokenizer backend
    # when BERTScore passes it as max_length. Use the checked finite model limit.
    scorer._tokenizer.model_max_length = limit
    predictions = [s["result"]["answers"][v] for s, v in pairs]
    references = [s["reference"] for s, v in pairs]
    precision, recall, f1 = scorer.score(predictions, references, batch_size=1)
    for (sample, name), p, r, f in zip(pairs, precision.tolist(), recall.tolist(), f1.tolist()):
        sample["quality"][name].update(bertscore_precision=p, bertscore_recall=r, bertscore_f1=f)
    return {"model": model_name, "model_revision": getattr(config, "_commit_hash", None),
            "layers": layers, "device": device, "hash": scorer.hash,
            "bert_score_version": importlib.metadata.version("bert-score"),
            "idf": False, "rescale_with_baseline": False, "batch_size": 1,
            "encoder_context_limit": limit, "truncation": False,
            "note": "Reference semantic similarity, not source factuality. Evaluation compute excluded from generation costs."}
