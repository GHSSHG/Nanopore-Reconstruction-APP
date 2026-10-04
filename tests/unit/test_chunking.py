import numpy as np
import pytest

from nanorecon.signal.chunking import ChunkPlan, chunk_count

L, O = 8192, 144
H = L - O


def starts(n, l=L, h=H):
    return [ChunkPlan(n, l, h).start(j) for j in range(ChunkPlan(n, l, h).num_chunks)]


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, []),
        (1, [0]),
        (2, [0]),
        (6143, [0]),
        (6144, [0]),
        (8191, [0]),
        (8192, [0]),
        (8193, [0, 1]),
        (9000, [0, 808]),
        (16240, [0, 8048]),  # N - L == H: exact cover, no extra tail window
        (16300, [0, 8048, 8108]),
        (16384, [0, 8048, 8192]),
    ],
)
def test_boundary_starts(n, expected):
    assert starts(n) == expected


def test_16300_has_three_window_region():
    plan = ChunkPlan(16300, L, H)
    cover = np.zeros(16300, dtype=int)
    for s, v in plan:
        cover[s : s + v] += 1
    assert cover[8108:8192].tolist() == [3] * 84
    assert cover.min() == 1


@pytest.mark.parametrize("n", [1, 7, 8191, 8192, 8193, 16239, 16240, 16241, 24288, 24289, 50000, 123457])
@pytest.mark.parametrize("l,o", [(8192, 144), (8192, 0), (64, 16), (10, 9), (7, 3)])
def test_invariants(n, l, o):
    h = l - o
    plan = ChunkPlan(n, l, h)
    k = plan.num_chunks
    if n >= l:
        assert k == 1 + -(-(n - l) // h)
    assert k == chunk_count(n, l, h)
    ss = [plan.start(j) for j in range(k)]
    assert ss == sorted(ss) and len(set(ss)) == k  # ordered, never the same window twice
    cover = np.zeros(n, dtype=int)
    for s, v in plan:
        assert 0 <= s and s + v <= n and v == min(l, n - s)
        cover[s : s + v] += 1
    assert cover.min() >= 1  # no gaps
    assert ss[-1] + plan.valid_length(k - 1) == n  # last window ends at the read end
    if n >= l:
        assert all(plan.valid_length(j) == l for j in range(k))


def test_out_of_range_and_bad_geometry():
    plan = ChunkPlan(100, L, H)
    with pytest.raises(IndexError):
        plan.start(1)
    with pytest.raises(ValueError):
        ChunkPlan(10, 8, 9)
    with pytest.raises(ValueError):
        chunk_count(-1, 8, 4)
