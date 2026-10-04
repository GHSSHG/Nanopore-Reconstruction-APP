"""compress: POD5 -> token container.

    group consecutive reads (Meta first; signals load once a read joins the group)
      -> prepare the group's chunks in order (calibrate, normalize, pad) into a batch
      -> encode each full batch on the GPU and store the codes at the chunks' positions
      -> write the group in read order, then start the next group.

Batches span reads, so short reads do not waste GPU rows; only a group's last batch is padded.
The GPU dominates the run time, so everything runs on one thread.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Protocol

import numpy as np

from .. import __version__
from ..io.container import MAX_GROUP_BYTES, MAX_TABLE_ENTRIES, ContainerWriter, FileHeader, json_size, run_info_json_bytes
from ..io.output import OutputTransaction
from ..io.pod5_io import Pod5Source, SourceRead
from ..model_config import CodecProfile, ModelIdentity
from ..signal.chunking import ChunkPlan
from ..signal.normalize import checked_calibration, prepare_chunk
from ..types import Cancelled, NanoReconError
from .progress import Progress

log = logging.getLogger("nanorecon")


class Encoder(Protocol):
    batch_size: int

    def encode(self, batch: np.ndarray, valid_rows: int) -> np.ndarray: ...


@dataclass(frozen=True)
class CompressOptions:
    batch_size: int = 64
    # Internal limits (not command line options); tests shrink them to exercise edge cases.
    group_chunks: int = 4096  # a group closes before exceeding this many chunks (a longer read forms its own group)
    group_reads: int = 4096


@dataclass
class CompressReport:
    reads: int = 0
    empty_reads: int = 0
    chunks: int = 0
    samples: int = 0
    groups: int = 0
    max_group_reads_seen: int = 0
    max_group_chunks_seen: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    elapsed_s: float = 0.0


class ReadGroup:
    """Reads grouped together, their signals and, once encoded, their codes and chunk stats.
    The chunks of all reads are numbered consecutively: read i owns offsets[i]:offsets[i+1]."""

    def __init__(self, reads: list[SourceRead], signals: list[np.ndarray], plans: list[ChunkPlan], profile: CodecProfile) -> None:
        self.metas = [r.meta for r in reads]
        self.signals = signals
        self.plans = plans
        self.calibrations = [checked_calibration(m.calibration_offset, m.calibration_scale, read_id=m.read_id_str) for m in self.metas]
        counts = np.fromiter((p.num_chunks for p in plans), dtype=np.int64, count=len(plans))
        self.offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
        self.total_chunks = int(self.offsets[-1])
        self.codes = np.empty((self.total_chunks, profile.tokens_per_chunk), dtype=np.uint16)
        self.centers = np.empty(self.total_chunks, dtype=np.float32)
        self.scales = np.empty(self.total_chunks, dtype=np.float32)

    def encoded(self, slot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, b = int(self.offsets[slot]), int(self.offsets[slot + 1])
        return self.centers[a:b], self.scales[a:b], self.codes[a:b]


GROUP_TABLE_LIMIT = MAX_GROUP_BYTES - (64 << 10)  # headroom for the GROUP record's own keys


def admit_groups(reads: Iterator[SourceRead], profile: CodecProfile, group_chunks: int, group_reads: int) -> Iterator[ReadGroup]:
    """Consecutive reads grouped by chunk and read count; signals load only after a read joins
    a group. A group also closes before its run_info/pore/end-reason tables would outgrow a
    GROUP record."""
    pending: SourceRead | None = None
    exhausted = False
    while not exhausted:
        group: list[SourceRead] = []
        signals: list[np.ndarray] = []
        plans: list[ChunkPlan] = []
        chunks = 0
        table_bytes = 0
        runs: set = set()
        pores: set = set()
        reasons: set = set()
        while True:
            read = pending if pending is not None else next(reads, None)
            pending = None
            if read is None:
                exhausted = True
                break
            meta = read.meta
            if meta.run_info.sample_rate != profile.sample_rate_hz:
                raise NanoReconError(
                    f"read {meta.read_id_str} was sampled at {meta.run_info.sample_rate} Hz; the model expects "
                    f"{profile.sample_rate_hz} Hz and this version does not resample"
                )
            plan = ChunkPlan(meta.num_samples, profile.chunk_samples, profile.hop_samples)
            new_run = meta.run_info not in runs
            new_pore = meta.pore_type not in pores
            new_reason = meta.end_reason not in reasons
            extra = (
                (run_info_json_bytes(meta.run_info) + 1 if new_run else 0)
                + (json_size(meta.pore_type) + 1 if new_pore else 0)
                + (json_size(meta.end_reason) + 1 if new_reason else 0)
            )
            if extra > GROUP_TABLE_LIMIT:
                raise NanoReconError(f"the run information of read {meta.read_id_str} is too large for the token format")
            tables_full = (
                table_bytes + extra > GROUP_TABLE_LIMIT
                or max(len(runs) + new_run, len(pores) + new_pore, len(reasons) + new_reason) > MAX_TABLE_ENTRIES
            )
            if group and (chunks + plan.num_chunks > group_chunks or len(group) >= group_reads or tables_full):
                pending = read
                break
            signals.append(read.load_signal())
            group.append(read)
            plans.append(plan)
            chunks += plan.num_chunks
            table_bytes += extra
            runs.add(meta.run_info)
            pores.add(meta.pore_type)
            reasons.add(meta.end_reason)
        if group:
            yield ReadGroup(group, signals, plans, profile)


class ChunkEncoder:
    """Prepares a group's chunks in order into one staging batch and encodes full batches."""

    def __init__(self, engine: Encoder, profile: CodecProfile) -> None:
        self.engine = engine
        self.batch = engine.batch_size
        self.staging = np.zeros((self.batch, profile.chunk_samples), dtype=np.float32)
        self._scratch = np.empty(profile.chunk_samples, dtype=np.float32)
        self._eps = np.float32(profile.normalization_epsilon)

    def encode_group(self, group: ReadGroup) -> None:
        first = 0  # group-wide index of the chunk in staging row 0
        rows = 0
        for slot, plan in enumerate(group.plans):
            offset, scale = group.calibrations[slot]
            for seq in range(plan.num_chunks):
                center, half = prepare_chunk(group.signals[slot], plan, seq, offset, scale, self._eps,
                                             self.staging[rows], self._scratch)
                group.centers[first + rows] = center
                group.scales[first + rows] = half
                rows += 1
                if rows == self.batch:
                    self._flush(group, first, rows)
                    first += rows
                    rows = 0
        if rows:
            self.staging[rows:] = 0.0  # padding rows; their results are discarded
            self._flush(group, first, rows)

    def _flush(self, group: ReadGroup, first: int, rows: int) -> None:
        group.codes[first : first + rows] = self.engine.encode(self.staging, rows)


def compress_file(
    input_path: Path,
    output_path: Path,
    *,
    model: ModelIdentity,
    profile: CodecProfile,
    make_engine: Callable[[], Encoder],
    options: CompressOptions,
    force: bool = False,
    show_progress: bool = True,
) -> CompressReport:
    started = time.perf_counter()
    report = CompressReport()
    input_path = Path(input_path)
    try:
        report.input_bytes = input_path.stat().st_size
    except OSError as exc:
        raise NanoReconError(f"cannot read input {input_path}: {exc}") from exc

    with Pod5Source(input_path) as source:
        read_count = source.read_count()
        if read_count == 0:
            log.warning("input holds no reads; writing an empty token file")
        progress = Progress("compress", read_count) if show_progress else None
        header = FileHeader(
            model=model,
            profile=profile,
            read_count=read_count,
            producer_version=__version__,
        )
        with OutputTransaction(Path(output_path), force=force, inputs=[input_path]) as tx:
            encoder = ChunkEncoder(make_engine(), profile)
            try:
                with _writing(output_path):
                    fh = open(tx.temp_path, "xb")
                try:
                    with _writing(output_path):
                        writer = ContainerWriter(fh, header)
                    for group in admit_groups(source.iter_reads(), profile, options.group_chunks, options.group_reads):
                        report.groups += 1
                        report.max_group_reads_seen = max(report.max_group_reads_seen, len(group.metas))
                        report.max_group_chunks_seen = max(report.max_group_chunks_seen, group.total_chunks)
                        encoder.encode_group(group)
                        with _writing(output_path):
                            _write_group(writer, group)
                        for meta, plan in zip(group.metas, group.plans):
                            report.reads += 1
                            report.samples += meta.num_samples
                            report.chunks += plan.num_chunks
                            report.empty_reads += meta.num_samples == 0
                        if progress is not None:
                            progress.update(report.reads, report.samples)
                        del group
                    with _writing(output_path):
                        writer.finish()
                finally:
                    fh.close()
            except KeyboardInterrupt as exc:
                raise Cancelled("compress interrupted; no output was written") from exc
            tx.commit()
    report.output_bytes = Path(output_path).stat().st_size
    report.elapsed_s = time.perf_counter() - started
    if progress is not None:
        progress.finish(report.reads, report.samples)
    return report


@contextlib.contextmanager
def _writing(path: Path):
    try:
        yield
    except OSError as exc:
        raise NanoReconError(f"cannot write {path}: {exc}") from exc


def _write_group(writer: ContainerWriter, group: ReadGroup) -> None:
    writer.begin_group(group.metas)
    for slot, meta in enumerate(group.metas):
        centers, scales, codes = group.encoded(slot)
        writer.write_read(meta, centers, scales, codes)
