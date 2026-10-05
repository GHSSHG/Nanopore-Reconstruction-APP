"""decompress: token container -> reconstructed POD5.

The file is read sequentially. Chunks of consecutive reads are packed into full decoder
batches (each row is one independent chunk; codes are never concatenated into one long token
sequence). Each decoded chunk is denormalized to pA and added to its read's stitch
accumulator; a read whose chunks are all in is converted to ADC once and written with its Meta,
in file order. The model identity and decoding rules come from the file header.

One thread: a full batch is sent off and the previous one is collected, stitched and written
(and the next codes are read) while the GPU computes.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from ..io.container import ContainerReader, FileHeader
from ..io.output import OutputTransaction
from ..io.pod5_io import Pod5Sink, check_writable
from ..signal.normalize import checked_calibration
from ..signal.stitch import StitchAccumulator, edge_weights, pa_to_adc
from ..model_config import CodecProfile
from ..types import Cancelled, NanoReconError, ReadMeta
from .progress import Progress

log = logging.getLogger("nanorecon")


class Decoder(Protocol):
    batch_size: int

    def decode_async(self, codes: np.ndarray, valid_rows: int) -> Callable[[], np.ndarray]: ...


@dataclass(frozen=True)
class DecompressOptions:
    batch_size: int = 64


@dataclass
class DecompressReport:
    reads: int = 0
    empty_reads: int = 0
    chunks: int = 0
    samples: int = 0
    groups: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    elapsed_s: float = 0.0
    model: dict = field(default_factory=dict)


def read_header(input_path: Path) -> FileHeader:
    """Parse only the header (model identity + profile), e.g. to locate the model first."""
    try:
        with open(input_path, "rb") as fh:
            return ContainerReader(fh, name=str(input_path)).header
    except OSError as exc:
        raise NanoReconError(f"cannot read {input_path}: {exc}") from exc


@dataclass
class _PendingRead:
    group_index: int
    meta: ReadMeta
    offset: np.float32
    scale: np.float32
    centers: np.ndarray
    scales: np.ndarray
    acc: StitchAccumulator | None  # None for an empty read
    remaining: int  # chunks not decoded yet


class _BatchDecoder:
    """Packs the chunks of consecutive reads into decoder batches (two alternating buffers). A full
    batch is sent to the decoder at once and collected when the next one is sent (or at the end),
    so host work overlaps the GPU. Reads come back in file order once all their chunks are
    decoded; about two batches worth of reads are open at a time."""

    def __init__(self, engine: Decoder, profile: CodecProfile) -> None:
        self.engine = engine
        self.batch = engine.batch_size
        self._codes = [np.zeros((self.batch, profile.tokens_per_chunk), dtype=np.uint16) for _ in range(2)]
        self._fill = 0  # the buffer being filled; the other one may still be on the GPU
        self.rows: list[tuple[_PendingRead, int]] = []  # (read, chunk index) per filled row
        self._in_flight: tuple[Callable[[], np.ndarray], list[tuple[_PendingRead, int]]] | None = None
        self.pending: deque[_PendingRead] = deque()

    def add(self, read: _PendingRead, codes: np.ndarray) -> list[_PendingRead]:
        """Queue a read with its codes (uint16[K, T]); returns the reads now complete."""
        self.pending.append(read)
        i = 0
        while i < codes.shape[0]:
            row = len(self.rows)
            n = min(self.batch - row, codes.shape[0] - i)
            self._codes[self._fill][row : row + n] = codes[i : i + n]
            self.rows.extend((read, i + k) for k in range(n))
            i += n
            if len(self.rows) == self.batch:
                self._send()
        return self._finished()

    def flush(self) -> list[_PendingRead]:
        if self.rows:
            self._send()
        self._collect()
        return self._finished()

    def _send(self) -> None:
        n = len(self.rows)
        buffer = self._codes[self._fill]
        if n < self.batch:
            buffer[n:] = 0  # padding rows; their output is discarded
        wait = self.engine.decode_async(buffer, n)
        self._collect()  # the previous batch, decoded while this one was filled
        self._in_flight = (wait, self.rows)
        self.rows = []
        self._fill ^= 1

    def _collect(self) -> None:
        if self._in_flight is None:
            return
        wait, rows = self._in_flight
        self._in_flight = None
        decoded = wait()
        for r, (read, index) in enumerate(rows):
            read.acc.add(index, decoded[r], read.centers[index], read.scales[index])
            read.remaining -= 1

    def _finished(self) -> list[_PendingRead]:
        done = []
        while self.pending and self.pending[0].remaining == 0:
            done.append(self.pending.popleft())
        return done


@contextlib.contextmanager
def _writing(path: Path):
    try:
        yield
    except OSError as exc:
        raise NanoReconError(f"cannot write {path}: {exc}") from exc


def decompress_file(
    input_path: Path,
    output_path: Path,
    *,
    make_engine: Callable[[FileHeader], Decoder],
    options: DecompressOptions,
    force: bool = False,
    show_progress: bool = True,
) -> DecompressReport:
    started = time.perf_counter()
    report = DecompressReport()
    input_path = Path(input_path)
    try:
        fh = open(input_path, "rb")
        report.input_bytes = input_path.stat().st_size
    except OSError as exc:
        raise NanoReconError(f"cannot read {input_path}: {exc}") from exc
    with fh:
        reader = ContainerReader(fh, name=str(input_path))
        header = reader.header
        profile = header.profile
        report.model = {"repo_id": header.model.repo_id, "revision": header.model.revision}
        progress = Progress("decompress", header.read_count) if show_progress else None
        with OutputTransaction(Path(output_path), force=force, inputs=[input_path]) as tx:
            decoder = _BatchDecoder(make_engine(header), profile)
            weights = edge_weights(profile.chunk_samples, profile.overlap_samples)
            eps = np.float32(profile.normalization_epsilon)
            last_group = -1

            def write(read: _PendingRead) -> None:
                nonlocal last_group
                if read.group_index != last_group:
                    sink.begin_group()
                    last_group = read.group_index
                    report.groups += 1
                if read.acc is None:
                    adc = np.empty(0, dtype=np.int16)
                    report.empty_reads += 1
                else:
                    adc = pa_to_adc(read.acc.finish(), read.offset, read.scale)
                with _writing(output_path):
                    sink.write(read.meta, adc)
                report.reads += 1
                report.chunks += len(read.centers)
                report.samples += read.meta.num_samples
                if progress is not None:
                    progress.update(report.reads, report.samples)

            try:
                with _writing(output_path):
                    sink = Pod5Sink(tx.temp_path)
                completed = False
                try:
                    for group_index, stored in reader.iter_reads():
                        meta = stored.meta
                        check_writable(meta)  # before any decoding work
                        offset, scale = checked_calibration(meta.calibration_offset, meta.calibration_scale, read_id=meta.read_id_str)
                        k = stored.num_chunks
                        read = _PendingRead(group_index, meta, offset, scale, stored.centers, stored.scales,
                                            StitchAccumulator(stored.plan, weights, eps) if k else None, k)
                        for done in decoder.add(read, stored.read_codes()):
                            write(done)
                    for done in decoder.flush():
                        write(done)
                    completed = True
                finally:
                    if completed:
                        with _writing(output_path):
                            sink.close()
                    else:
                        with contextlib.suppress(Exception):
                            sink.close()  # partial file; the transaction discards it
            except KeyboardInterrupt as exc:
                raise Cancelled("decompress interrupted; no output was written") from exc
            if report.reads != header.read_count:
                raise NanoReconError(f"file announced {header.read_count} reads but held {report.reads}")
            tx.commit()
    report.output_bytes = Path(output_path).stat().st_size
    report.elapsed_s = time.perf_counter() - started
    if progress is not None:
        progress.finish(report.reads, report.samples)
    return report
