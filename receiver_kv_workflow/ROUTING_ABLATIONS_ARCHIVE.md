# Retired Routing Ablations

The project previously exposed three alternate source-unit selectors:

- `fixed`: per-role fixed source spans, then union;
- `fixed_pooled`: pooled role scores over whole fixed spans;
- `sentence`: punctuation and paragraph-delimited units.

They were used to diagnose the effect of selection boundaries. They are no
longer executable workflow modes. The final workflow always constructs
consecutive speaker-labelled blocks, selects complete seed blocks from the
role-conditioned scores, and adds local neighbouring speaker blocks.

Historical ablation results should be reported with their original Git revision
and configuration. They are not directly comparable to the current speaker
routing workflow.
