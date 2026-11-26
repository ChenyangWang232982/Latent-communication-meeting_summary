# Obsolete local planning scratch output

This directory contains an initial 512-draw planning-only experiment, before the
length matching implementation was finalized. It contains no generated summaries.
Do not use its budgets or code-match flag as results of the current random control.

The current method exhausts 0–2 block length combinations and samples uniformly
among closest-length ties. A separate temporary-directory validation matched
197/197 windows and 24,275 supplementary positions on both sides. See
`experiment_3/RANDOM_CONTROL.md` for the final protocol and server command.

Automatic safety review blocked recursive cleanup of this local scratch directory;
it is retained as obsolete output, not as a completed experiment.
