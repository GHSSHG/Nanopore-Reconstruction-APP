"""Shared data structures, exit codes and error types.

Runtime objects (arrays, batch buffers) live in the pipeline modules; this module only
holds plain values that cross module boundaries.
"""

from __future__ import annotations

import enum
import math
import uuid
from dataclasses import dataclass, fields

import numpy as np


class ExitCode(enum.IntEnum):
    OK = 0
    ERROR = 1
    USAGE = 2
    CANCELLED = 130


class NanoReconError(Exception):
    """An error reported to the user as a message (plus an optional hint), without a traceback."""

    exit_code = ExitCode.ERROR

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class UsageError(NanoReconError):
    exit_code = ExitCode.USAGE


class Cancelled(NanoReconError):
    exit_code = ExitCode.CANCELLED


@dataclass(frozen=True, slots=True)
class RunInfoMeta:
    """POD5 run information. Timestamps are integer milliseconds since the Unix epoch (UTC),
    the precision POD5 stores; map fields keep their entry order and duplicates."""

    acquisition_id: str
    acquisition_start_time_ms: int
    adc_max: int
    adc_min: int
    context_tags: tuple[tuple[str, str], ...]
    experiment_name: str
    flow_cell_id: str
    flow_cell_product_code: str
    protocol_name: str
    protocol_run_id: str
    protocol_start_time_ms: int
    sample_id: str
    sample_rate: int
    sequencing_kit: str
    sequencer_position: str
    sequencer_position_type: str
    software: str
    system_name: str
    system_type: str
    tracking_id: tuple[tuple[str, str], ...]


RUN_INFO_FIELDS = tuple(f.name for f in fields(RunInfoMeta))
RUN_INFO_MAP_FIELDS = ("context_tags", "tracking_id")
RUN_INFO_TIME_FIELDS = ("acquisition_start_time_ms", "protocol_start_time_ms")
RUN_INFO_INT_RANGES = {
    "acquisition_start_time_ms": (-(2**63), 2**63 - 1),
    "protocol_start_time_ms": (-(2**63), 2**63 - 1),
    "adc_max": (-(2**15), 2**15 - 1),
    "adc_min": (-(2**15), 2**15 - 1),
    "sample_rate": (0, 2**16 - 1),
}
RUN_INFO_STR_FIELDS = tuple(
    name for name in RUN_INFO_FIELDS if name not in RUN_INFO_MAP_FIELDS and name not in RUN_INFO_INT_RANGES
)


@dataclass(frozen=True, slots=True)
class ReadMeta:
    """Every POD5 read field this version preserves. Float fields hold float32 values."""

    read_id: bytes  # 16 raw UUID bytes, as stored by POD5
    num_samples: int
    read_number: int
    start_sample: int
    channel: int
    well: int
    pore_type: str
    calibration_offset: float
    calibration_scale: float
    median_before: float
    end_reason: str  # lower-case POD5 name, e.g. "signal_positive"
    end_reason_forced: bool
    run_info: RunInfoMeta
    num_minknow_events: int
    tracked_scaling_scale: float
    tracked_scaling_shift: float
    predicted_scaling_scale: float
    predicted_scaling_shift: float
    num_reads_since_mux_change: int
    time_since_mux_change: float
    open_pore_level: float

    @property
    def read_id_str(self) -> str:
        return str(uuid.UUID(bytes=self.read_id))


READ_FLOAT_FIELDS = (
    "calibration_offset",
    "calibration_scale",
    "median_before",
    "tracked_scaling_scale",
    "tracked_scaling_shift",
    "predicted_scaling_scale",
    "predicted_scaling_shift",
    "time_since_mux_change",
    "open_pore_level",
)


def same_float32(a: float, b: float) -> bool:
    """Equality for stored float32 values where NaN equals NaN."""
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return np.float32(a) == np.float32(b)


def read_meta_equal(a: ReadMeta, b: ReadMeta) -> bool:
    """Field-by-field comparison of the preserved Meta (NaN-aware)."""
    for f in fields(ReadMeta):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if f.name in READ_FLOAT_FIELDS:
            if not same_float32(va, vb):
                return False
        elif va != vb:
            return False
    return True
