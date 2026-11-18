"""Reuse summary metrics and retain paired sample-level observations."""
from meeting_evaluation.metrics import reference_metrics, source_overlap_support
from .config import VARIANTS


def score_sample(result, record, resources):
    quality = {}
    for name in VARIANTS:
        answer = result["answers"][name]
        support = source_overlap_support(answer, record.context)
        quality[name] = {**reference_metrics(answer, record.answer), **support}
        measurement = result["variants"][name]
        measurement.update(resources.get(name, {}))
        # Include excerpt mapping and prompt preparation omitted by the legacy stage timers.
        measurement["legacy_pipeline_seconds"] = measurement["pipeline_seconds_excluding_model_load"]
        measurement["pipeline_seconds_excluding_model_load"] = (
            result["prefill_seconds"] + result["selection_seconds"] + measurement["variant_wall_seconds"])
        peaks = [v for v in (result.get("shared_prefill_selection_peak_allocated_bytes"),
                             measurement.get("cuda_peak_allocated_bytes")) if v is not None]
        measurement["pipeline_peak_allocated_bytes"] = max(peaks) if peaks else None
        if result.get('chunk_specialists_by_variant'):
            measurement['payload_definition'] = ('Sum of stage KV tensor bytes, selected evidence UTF-8 bytes, '
                'current-part UTF-8 bytes and local report UTF-8 bytes; separate logical representations, not network traffic')
        elif name == "specialist_text":
            # Actual UTF-8 length of the decoded handoff, distinct from KV tensors and token IDs.
            measurement["payload_definition"] = "UTF-8 bytes of decoded generated handoff"
            measurement["logical_payload_bytes"] = measurement["handoff_utf8_bytes"]
        else:
            measurement["payload_definition"] = "Selected KV tensor bytes; no network traffic is measured"
    return {"id": record.id, "status": "completed", "source": record.context, "reference": record.answer,
            "quality": quality, "result": result}
