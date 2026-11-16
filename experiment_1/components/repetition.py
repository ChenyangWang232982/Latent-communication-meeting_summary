"""Lexical duplicate diagnostics only; never rewrite or stop generation."""
from collections import Counter
import re


def repetition_metrics(text):
    bullets = [re.sub(r"\s+", " ", line.strip()).casefold()
               for line in text.splitlines() if re.match(r"^\s*(?:[-*+] |\d+[.)] )", line)]
    counts = Counter(bullets)
    excess = sum(n - 1 for n in counts.values())
    return {"bullet_count": len(bullets), "duplicate_bullet_excess": excess,
            "duplicate_bullet_fraction": excess / len(bullets) if bullets else 0.0,
            "max_identical_bullet_count": max(counts.values(), default=0)}
