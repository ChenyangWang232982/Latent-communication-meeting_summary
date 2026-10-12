from prophetkv_statebridge.state_selection import state_selection_indices


def test_tail_uses_the_final_positions():
    assert state_selection_indices(100, 4, "tail") == [96, 97, 98, 99]


def test_stratified_k64_covers_head_middle_and_tail():
    indices = state_selection_indices(160, 64, "stratified")
    assert len(indices) == 64
    assert len(set(indices)) == 64
    assert indices == sorted(indices)
    assert indices[:16] == list(range(16))
    assert indices[-16:] == list(range(144, 160))
    assert 16 in indices and 143 in indices

