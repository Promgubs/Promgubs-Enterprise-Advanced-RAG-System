from app.retrieval.fusion import rrf_fuse


def test_item_in_both_lists_wins():
    fused = rrf_fuse([["a", "b", "c"], ["b", "d"]])
    assert fused[0][0] == "b"
    assert {i for i, _ in fused} == {"a", "b", "c", "d"}


def test_single_list_keeps_order():
    assert [i for i, _ in rrf_fuse([["x", "y", "z"]])] == ["x", "y", "z"]


def test_empty():
    assert rrf_fuse([]) == []
