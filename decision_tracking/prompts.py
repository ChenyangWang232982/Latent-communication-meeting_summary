"""Requests for analysis of a candidate topic's source KV."""

GROUP_REQUEST = (
    'Analyze the source passages as a candidate decision topic in the transcript language. '
    'Passages are in their original meeting order but may be incomplete and discontinuous. '
    'Determine whether they actually concern the same proposal; semantic similarity is not proof. '
    'For each substantive decision or proposal supported by these passages, describe the supported '
    'initial proposal, discussion or disagreement, later revisions, and final status in meeting order. '
    'Distinguish accepted, rejected, deferred, conditional and unresolved; do not treat the latest '
    'speaker or a suggestion as agreement. If the outcome or relation between passages is unclear, say so. '
    'Preserve supported participants and conditions, avoid invented verbatim quotes, and do not infer '
    'missing events. If no substantive decision or proposal is supported, say so briefly.'
)
