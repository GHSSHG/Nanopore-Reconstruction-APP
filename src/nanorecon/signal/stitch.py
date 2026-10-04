"""Overlap stitching in pA space (profile stitch="linear_edges_v1") and the final ADC conversion.

Each decoded chunk is denormalized to pA on its own, multiplied by the edge weight
    w(t) = min(1, (t + 1) / (O + 1), (L - t) / (O + 1)),   t = 0 .. L-1,  O = L - H,
over its real samples only, summed, and divided by the actual weight sum at every position.
Every weight is > 0, so each covered sample has a positive sum no matter how many windows
(two, three, ...) overlap it; a single contributor reproduces its chunk exactly.
ADC conversion happens once on the merged pA signal ("rint_clip_int16").
"""

from __future__ import annotations

import numpy as np

from .chunking import ChunkPlan
from .normalize import denormalize

ADC_BLOCK_SAMPLES = 1 << 18  # pa_to_adc works in float64 blocks of this size


def edge_weights(chunk_samples: int, overlap_samples: int) -> np.ndarray:
    t = np.arange(chunk_samples, dtype=np.float64)
    ramp = float(overlap_samples + 1)
    w = np.minimum(1.0, np.minimum((t + 1.0) / ramp, (chunk_samples - t) / ramp))
    return w.astype(np.float32)


class StitchAccumulator:
    def __init__(self, plan: ChunkPlan, weights: np.ndarray, epsilon: np.float32) -> None:
        if weights.shape != (plan.chunk_samples,) or not np.all(weights > 0):
            raise ValueError("stitch weights must be positive and have one value per chunk sample")
        self.plan = plan
        self._weights = weights
        self._epsilon = epsilon
        self._pa_sum = np.zeros(plan.num_samples, dtype=np.float32)
        self._w_sum = np.zeros(plan.num_samples, dtype=np.float32)
        self._scratch = np.empty(plan.chunk_samples, dtype=np.float32)
        self._added = np.zeros(plan.num_chunks, dtype=bool)

    def add(self, index: int, decoded: np.ndarray, center: np.float32, scale: np.float32) -> None:
        """Add chunk `index` (decoded normalized float32[L])."""
        if self._added[index]:
            raise RuntimeError(f"chunk {index} stitched twice")
        start, valid = self.plan.window(index)
        pa = denormalize(decoded[:valid], center, scale, self._epsilon, self._scratch[:valid])
        w = self._weights[:valid]
        pa *= w
        self._pa_sum[start : start + valid] += pa
        self._w_sum[start : start + valid] += w
        self._added[index] = True

    def finish(self) -> np.ndarray:
        """Merged pA signal (float32[N]); every sample must have been covered."""
        if not self._added.all():
            missing = np.flatnonzero(~self._added)
            raise RuntimeError(f"chunks never stitched: {missing[:8].tolist()}")
        if self.plan.num_samples and not self._w_sum.min() > 0:  # reduction, no N-byte temporary
            raise RuntimeError("stitching left samples without positive weight")
        np.divide(self._pa_sum, self._w_sum, out=self._pa_sum)
        pa = self._pa_sum
        self._w_sum = np.empty(0, dtype=np.float32)
        return pa


def pa_to_adc(pa: np.ndarray, offset: np.float32, scale: np.float32) -> np.ndarray:
    """ADC = rint(pA / scale - offset), clipped to int16. Works in bounded blocks."""
    out = np.empty(pa.shape[0], dtype=np.int16)
    off, sc = float(offset), float(scale)
    for s in range(0, pa.shape[0], ADC_BLOCK_SAMPLES):
        seg = pa[s : s + ADC_BLOCK_SAMPLES].astype(np.float64)
        seg /= sc
        seg -= off
        np.rint(seg, out=seg)
        np.clip(seg, -32768.0, 32767.0, out=seg)
        out[s : s + seg.shape[0]] = seg
    return out
