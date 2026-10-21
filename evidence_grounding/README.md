# Evidence Grounding

This package implements the training-free evidence layer used by the meeting
workflow:

```text
specialist claim + optional keywords
  -> deterministic candidate-span retrieval
  -> LLM entailment verification over only those spans
  -> supported claim + selected source span in an audit ledger
```

It does not require a second embedding model or a training set. Retrieval is
not treated as proof: the workflow only forwards a claim when the semantic
verifier labels one retrieved candidate as `supported`.

`prophetkv_statebridge.evidence` adapts this package to the meeting workflow.
Use `--evidence-policy retrieval` to enable it. `strict` retains the legacy
literal-quote guard; `off` remains the unconstrained ablation.
