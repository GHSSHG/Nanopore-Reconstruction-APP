"""Calibration, normalization, padding and pA-space stitching without a model."""

import numpy as np
import pytest

from nanorecon.signal.chunking import ChunkPlan
from nanorecon.signal.normalize import (
    adc_to_pa,
    checked_calibration,
    denormalize,
    normalize_window,
    pad_reflect_right,
    prepare_chunk,
    reflect_indices,
)
from nanorecon.signal.stitch import StitchAccumulator, edge_weights, pa_to_adc
from nanorecon.types import NanoReconError

L, O = 8192, 144
H = L - O
EPS = np.float32(1e-6)


def training_reflect_pad(arr, target):
    """The training repository's _reflect_pad_right_1d, verbatim in behaviour."""
    n = arr.shape[0]
    if n >= target:
        return arr[:target]
    if n == 1:
        return np.full((target,), float(arr[0]), dtype=np.float32)
    period = 2 * n - 2
    idx = np.arange(target) % period
    idx = np.where(idx < n, idx, period - idx)
    return arr[idx]


@pytest.mark.parametrize("n", [1, 2, 3, 5, 100, 4097, 8191])
def test_reflect_pad_matches_training(n):
    rng = np.random.default_rng(n)
    buf = np.zeros(L, np.float32)
    buf[:n] = rng.normal(size=n).astype(np.float32)
    ref = training_reflect_pad(buf[:n].copy(), L)
    pad_reflect_right(buf, n)
    assert np.array_equal(buf, ref)


def test_reflect_indices_empty_rejected():
    with pytest.raises(ValueError):
        reflect_indices(0, 10)


def test_normalization_matches_training_formula():
    rng = np.random.default_rng(0)
    adc = rng.integers(200, 900, size=L).astype(np.int16)
    off, sc = checked_calibration(-240.0, 0.1755)
    pa = adc_to_pa(adc, off, sc, np.empty(L, np.float32))
    # pod5's own calibration: (adc + float32(offset)) * float32(scale)
    assert np.array_equal(pa, (adc + np.float32(-240.0)) * np.float32(0.1755))
    y = np.empty(L, np.float32)
    center, half = normalize_window(pa, EPS, y)
    lo, hi = float(pa.min()), float(pa.max())
    assert center == np.float32(0.5 * (lo + hi)) and half == np.float32(0.5 * (hi - lo))
    expected = np.clip((pa - center) / half, -1.0, 1.0)
    assert np.array_equal(y, expected.astype(np.float32))
    assert y.min() >= -1 and y.max() <= 1


def test_constant_window_is_zero_and_restores_center():
    pa = np.full(1000, 123.25, np.float32)
    y = np.empty(1000, np.float32)
    center, half = normalize_window(pa, EPS, y)
    assert half < EPS and np.all(y == 0)
    out = denormalize(np.random.default_rng(1).normal(size=1000).astype(np.float32), center, half, EPS, np.empty(1000, np.float32))
    assert np.all(out == np.float32(123.25))


@pytest.mark.parametrize("offset, scale", [(float("nan"), 0.1), (0.0, 0.0), (0.0, float("inf")), (1e39, 0.1)])
def test_invalid_calibration_rejected(offset, scale):
    with pytest.raises(NanoReconError):
        checked_calibration(offset, scale)


def test_non_finite_signal_rejected():
    pa = np.array([1.0, np.inf], np.float32)
    with pytest.raises(NanoReconError):
        normalize_window(pa, EPS, np.empty(2, np.float32))


def test_short_read_normalizes_real_samples_then_pads():
    adc = np.array([400, 410, 405, 420, 399], np.int16)
    plan = ChunkPlan(5, L, H)
    out, scratch = np.empty(L, np.float32), np.empty(L, np.float32)
    center, half = prepare_chunk(adc, plan, 0, *checked_calibration(-240.0, 0.2), EPS, out, scratch)
    pa = (adc + np.float32(-240.0)) * np.float32(0.2)
    assert center == np.float32(0.5 * (float(pa.min()) + float(pa.max())))  # padding not in the stats
    assert np.array_equal(out, training_reflect_pad(out[:5].copy(), L))


def test_edge_weights_positive_and_shape():
    w = edge_weights(L, O)
    assert w.shape == (L,) and w.dtype == np.float32
    assert w.min() > 0 and w[0] == np.float32(1 / 145) and w[-1] == np.float32(1 / 145)
    assert np.all(w[144 : L - 144] == 1)
    assert np.all(edge_weights(L, 0) == 1)


def roundtrip_adc(adc, offset=-240.0, scale=0.1755, l=L, o=O):
    """Cut, normalize, denormalize and stitch with an identity 'model'."""
    h = l - o
    plan = ChunkPlan(adc.shape[0], l, h)
    off, sc = checked_calibration(offset, scale)
    acc = StitchAccumulator(plan, edge_weights(l, o), EPS)
    buf, scratch = np.empty(l, np.float32), np.empty(l, np.float32)
    order = list(range(plan.num_chunks))[::-1]  # stitching order must not matter
    for j in order:
        center, half = prepare_chunk(adc, plan, j, off, sc, EPS, buf, scratch)
        acc.add(j, buf.copy(), center, half)
    return pa_to_adc(acc.finish(), off, sc)


@pytest.mark.parametrize("n", [1, 2, 6143, 6144, 8191, 8192, 8193, 9000, 16240, 16300, 16384, 40000, 65537])
def test_identity_roundtrip_is_exact(n):
    rng = np.random.default_rng(n)
    adc = rng.integers(-500, 1500, size=n).astype(np.int16)
    out = roundtrip_adc(adc)
    assert out.shape == adc.shape and out.dtype == np.int16
    assert np.array_equal(out, adc)


@pytest.mark.parametrize("l,o", [(64, 16), (10, 9), (7, 3), (32, 0)])
def test_identity_roundtrip_other_geometries(l, o):
    rng = np.random.default_rng(l * 100 + o)
    for n in (1, l - 1, l, l + 1, 3 * l + 5, 1000):
        adc = rng.integers(-2000, 2000, size=n).astype(np.int16)
        assert np.array_equal(roundtrip_adc(adc, l=l, o=o), adc)


def test_constant_and_extreme_reads():
    assert np.array_equal(roundtrip_adc(np.full(20000, 512, np.int16)), np.full(20000, 512, np.int16))
    extreme = np.array([-32768, 32767] * 5000, np.int16)
    assert np.array_equal(roundtrip_adc(extreme, offset=0.0, scale=1.0), extreme)


def test_stitch_detects_missing_and_duplicate_chunks():
    plan = ChunkPlan(9000, L, H)
    acc = StitchAccumulator(plan, edge_weights(L, O), EPS)
    acc.add(0, np.zeros(L, np.float32), np.float32(1), np.float32(1))
    with pytest.raises(RuntimeError):
        acc.add(0, np.zeros(L, np.float32), np.float32(1), np.float32(1))
    with pytest.raises(RuntimeError):
        acc.finish()


def test_adc_conversion_rounds_and_clips():
    pa = np.array([0.24, 0.26, -0.26, 1e9, -1e9], np.float32)
    out = pa_to_adc(pa, np.float32(0.0), np.float32(0.5))
    assert out.tolist() == [0, 1, -1, 32767, -32768]
