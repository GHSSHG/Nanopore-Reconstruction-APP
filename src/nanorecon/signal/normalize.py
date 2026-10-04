"""ADC -> pA calibration and per-chunk min-max normalization (profile "minmax_pm1").

Arithmetic follows the training pipeline: pA = (adc + offset) * scale in float32; per chunk
center = (min + max) / 2 and scale = (max - min) / 2 computed from the float32 extrema,
stored as float32, and y = clip((pA - center) / scale, -1, 1) in float32. A chunk whose stored
scale is below epsilon is constant: y = 0 and denormalization returns the center.
Short windows are normalized on their real samples first, then padded ("reflect").
"""

from __future__ import annotations

import math

import numpy as np

from ..types import NanoReconError
from .chunking import ChunkPlan


def checked_calibration(offset: float, scale: float, *, read_id: str = "") -> tuple[np.float32, np.float32]:
    """Validate POD5 calibration; invalid values are rejected, never replaced by defaults."""
    with np.errstate(over="ignore"):
        off32, sc32 = np.float32(offset), np.float32(scale)
    where = f" for read {read_id}" if read_id else ""
    if not (math.isfinite(offset) and np.isfinite(off32)):
        raise NanoReconError(f"calibration offset {offset!r}{where} is not finite")
    if not (math.isfinite(scale) and np.isfinite(sc32)) or sc32 == 0:
        raise NanoReconError(f"calibration scale {scale!r}{where} is not a finite non-zero number")
    return off32, sc32


def adc_to_pa(adc: np.ndarray, offset: np.float32, scale: np.float32, out: np.ndarray) -> np.ndarray:
    np.add(adc, offset, out=out, dtype=np.float32)
    np.multiply(out, scale, out=out)
    return out


def normalize_window(pa: np.ndarray, epsilon: np.float32, out: np.ndarray) -> tuple[np.float32, np.float32]:
    """Normalize real samples `pa` into `out` (same length); returns (center, scale) as float32."""
    lo = pa.min()
    hi = pa.max()
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise NanoReconError("calibrated signal contains non-finite values")
    center = np.float32(0.5 * (float(lo) + float(hi)))
    half = np.float32(0.5 * (float(hi) - float(lo)))
    if half < epsilon:
        out[...] = 0.0
    else:
        np.subtract(pa, center, out=out)
        np.divide(out, half, out=out)
        np.clip(out, -1.0, 1.0, out=out)
    return center, half


def reflect_indices(valid: int, target: int) -> np.ndarray:
    """Source indices for positions valid..target-1 of a right reflect pad (no edge repeat);
    one real sample is repeated."""
    if valid <= 0:
        raise ValueError("cannot pad an empty window")
    if valid == 1:
        return np.zeros(target - valid, dtype=np.int64)
    period = 2 * valid - 2
    idx = np.arange(valid, target, dtype=np.int64) % period
    return np.where(idx < valid, idx, period - idx)


def pad_reflect_right(buf: np.ndarray, valid: int) -> None:
    target = buf.shape[0]
    if valid >= target:
        return
    buf[valid:] = buf[:valid][reflect_indices(valid, target)]


def prepare_chunk(
    adc: np.ndarray,
    plan: ChunkPlan,
    index: int,
    offset: np.float32,
    scale: np.float32,
    epsilon: np.float32,
    out: np.ndarray,
    scratch: np.ndarray,
) -> tuple[np.float32, np.float32]:
    """Fill `out` (float32[L]) with the model input for chunk `index`; returns (center, scale)."""
    start, valid = plan.window(index)
    pa = adc_to_pa(adc[start : start + valid], offset, scale, scratch[:valid])
    center, half = normalize_window(pa, epsilon, out[:valid])
    pad_reflect_right(out, valid)
    return center, half


def denormalize(y: np.ndarray, center: np.float32, scale: np.float32, epsilon: np.float32, out: np.ndarray) -> np.ndarray:
    """Inverse of normalize_window for decoded samples `y` (float32)."""
    if scale < epsilon:
        out[...] = center
    else:
        np.multiply(y, scale, out=out)
        np.add(out, center, out=out)
    return out
