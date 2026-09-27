from prophetkv_statebridge.selection import Chunk, select_chunks


def test_query_aware_selection_prefers_matching_chunk():
    chunks = [
        Chunk(0, "The deployment discussion covers a blue green rollout.", 10),
        Chunk(1, "Alice owns the migration action and the deadline is Friday.", 12),
        Chunk(2, "The budget was approved after legal review.", 9),
    ]

    selected = select_chunks(chunks, "Who owns the migration action and what is the deadline?", 1)

    assert selected[0][0].chunk_id == 1
