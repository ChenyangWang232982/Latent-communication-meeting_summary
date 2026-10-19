from prophetkv_statebridge.evidence import parse_grounded_claims, parse_verdicts, verified_handoff


def test_grounded_action_requires_literal_quote_and_commitment():
    source = "At 10:00, Alice said: I will send the report by Friday."
    message = (
        '[{"category":"actions","claim":"Alice will send the report by Friday.",'
        '"evidence_quote":"I will send the report by Friday.","timestamp":"10:00",'
        '"owner":"Alice","deadline":"Friday","confidence":"high"}]'
    )
    claims, rejected = parse_grounded_claims(message, role="actions", source=source, block_id=2)
    assert not rejected
    assert claims[0]["block_id"] == 2
    assert "Alice will send" in verified_handoff("actions", claims)


def test_ungrounded_or_misclassified_claim_is_rejected():
    source = "We need to discuss the budget next week."
    message = '[{"category":"decisions","claim":"The budget was approved.","evidence_quote":"We need to discuss the budget next week."}]'
    claims, rejected = parse_grounded_claims(message, role="decisions", source=source, block_id=0)
    assert not claims
    assert rejected


def test_quote_recovery_tolerates_missing_filler_but_uses_source_text():
    source = "We will, uh, prepare the final autonomy level table with better justification by next Tuesday."
    message = (
        '[{"category":"actions","claim":"The team will prepare the final autonomy level table.",'
        '"evidence_quote":"We will prepare the final autonomy level table with better justification by next Tuesday."}]'
    )
    claims, rejected = parse_grounded_claims(message, role="actions", source=source, block_id=0)
    assert not rejected
    assert claims[0]["evidence_quote"] == source


def test_unsupported_owner_metadata_is_removed():
    source = "Toufique will prepare the autonomy level table by next Tuesday."
    message = (
        '[{"category":"actions","claim":"Toufique will prepare the autonomy level table.",'
        '"evidence_quote":"Toufique will prepare the autonomy level table by next Tuesday.",'
        '"owner":"Mahade Hasan","deadline":"next Tuesday"}]'
    )
    claims, rejected = parse_grounded_claims(message, role="actions", source=source, block_id=0)
    assert not rejected
    assert claims[0]["owner"] == ""
    assert claims[0]["deadline"] == "next Tuesday"


def test_only_supported_verdicts_are_kept():
    message = '[{"index":0,"verdict":"supported"},{"index":1,"verdict":"uncertain"}]'
    assert parse_verdicts(message, 2) == {0}
