"""Signal rules: chunk plan, calibration/normalization, pA-space stitching. NumPy only."""

from .chunking import ChunkPlan, chunk_count
from .normalize import checked_calibration, denormalize, prepare_chunk
from .stitch import StitchAccumulator, edge_weights, pa_to_adc

__all__ = [
    "ChunkPlan",
    "chunk_count",
    "checked_calibration",
    "denormalize",
    "prepare_chunk",
    "StitchAccumulator",
    "edge_weights",
    "pa_to_adc",
]
