"""Chunk plan for one read: window starts and valid lengths, derived from (N, L, H) alone.

Rules (profile tail="shift_last"):
  * N >= L: regular starts 0, H, 2H, ... while start + L <= N; if the last regular window
    does not end at N, one right-aligned window starting at N - L is added (never twice).
    K = 1 + ceil((N - L) / H).
  * 0 < N < L: a single window at 0 holding N real samples (padded to L later).
  * N == 0: no windows.
Nothing per-window is stored; any chunk's placement is computed on demand.
"""

from __future__ import annotations

from typing import Iterator


def chunk_count(num_samples: int, chunk_samples: int, hop_samples: int) -> int:
    if num_samples < 0:
        raise ValueError(f"num_samples must be >= 0, got {num_samples}")
    if num_samples == 0:
        return 0
    if num_samples <= chunk_samples:
        return 1
    return 1 + -(-(num_samples - chunk_samples) // hop_samples)


class ChunkPlan:
    __slots__ = ("num_samples", "chunk_samples", "hop_samples", "num_chunks", "_regular")

    def __init__(self, num_samples: int, chunk_samples: int, hop_samples: int) -> None:
        if chunk_samples <= 0 or not 0 < hop_samples <= chunk_samples:
            raise ValueError(f"invalid chunk geometry L={chunk_samples} H={hop_samples}")
        self.num_samples = int(num_samples)
        self.chunk_samples = int(chunk_samples)
        self.hop_samples = int(hop_samples)
        self.num_chunks = chunk_count(self.num_samples, self.chunk_samples, self.hop_samples)
        if self.num_samples >= self.chunk_samples:
            self._regular = (self.num_samples - self.chunk_samples) // self.hop_samples + 1
        else:
            self._regular = self.num_chunks

    def start(self, index: int) -> int:
        if not 0 <= index < self.num_chunks:
            raise IndexError(f"chunk {index} out of range for {self.num_chunks} chunks")
        if self.num_samples < self.chunk_samples:
            return 0
        if index < self._regular:
            return index * self.hop_samples
        return self.num_samples - self.chunk_samples

    def valid_length(self, index: int) -> int:
        return min(self.chunk_samples, self.num_samples - self.start(index))

    def window(self, index: int) -> tuple[int, int]:
        start = self.start(index)
        return start, min(self.chunk_samples, self.num_samples - start)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for index in range(self.num_chunks):
            yield self.window(index)

    def __repr__(self) -> str:
        return f"ChunkPlan(N={self.num_samples}, L={self.chunk_samples}, H={self.hop_samples}, K={self.num_chunks})"
