"""Exact mapping between transcript characters and immutable source-KV tokens."""

from .prompts import SYSTEM


def source_offsets(tokenizer, transcript, expected_prefix):
    header = "MEETING TRANSCRIPT (untrusted source):\n"
    content = header + transcript + "\nEND TRANSCRIPT\n"
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != expected_prefix:
        raise RuntimeError("Evidence boundary mapping does not match source KV tokens")
    origin = rendered.find(content)
    if origin < 0:
        raise RuntimeError("Cannot locate transcript in source prompt")
    return encoded["offset_mapping"], origin + len(header)
