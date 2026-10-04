"""POD5 input/output with explicit Meta mapping (pod5 0.3.x, read table V4).

Reading walks the read table batch by batch. Metadata comes straight from the Arrow columns
(no ReadRecord.to_read(), which would also decode the signal); the signal of a read is loaded
only when the caller asks for it. Writing is streaming: each read is handed to pod5.Writer as
soon as it is complete.

Preserved per read: every column of the V4 read table (see ReadMeta). Preserved per run:
every RunInfo field, timestamps at their stored millisecond precision, map entries in order.
Not preserved: file-level attributes (file identifier, writing software, pod5 version) and
the exact byte layout/compression batching of the source file.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Iterator

import numpy as np
import pod5
import pyarrow as pa
import pyarrow.compute  # noqa: F401  (pa.compute)
from pod5.pod5_types import ShiftScalePair

from .. import __version__
from ..signal.normalize import checked_calibration
from .container import MAX_STRING_BYTES, MAX_TABLE_ENTRIES
from ..types import NanoReconError, ReadMeta, RunInfoMeta, RUN_INFO_FIELDS, RUN_INFO_MAP_FIELDS

_END_REASON_NAMES = {member.name.lower(): member for member in pod5.EndReasonEnum}
_TIME_FIELDS_POD5 = {
    "acquisition_start_time_ms": "acquisition_start_time",
    "protocol_start_time_ms": "protocol_start_time",
}
_RUN_INFO_CACHE_LIMIT = 256

_NUMERIC_READ_COLUMNS = (
    "read_number",
    "start",
    "median_before",
    "num_minknow_events",
    "tracked_scaling_scale",
    "tracked_scaling_shift",
    "predicted_scaling_scale",
    "predicted_scaling_shift",
    "num_reads_since_mux_change",
    "time_since_mux_change",
    "num_samples",
    "channel",
    "well",
    "calibration_offset",
    "calibration_scale",
    "end_reason_forced",
    "open_pore_level",
)


def check_writable(meta: ReadMeta) -> None:
    """Fail before any work if pod5 cannot write this read's Meta back."""
    if meta.end_reason not in _END_REASON_NAMES:
        raise NanoReconError(
            f"read {meta.read_id_str} has end reason {meta.end_reason!r}, which pod5 {pod5.__version__} cannot write",
            hint="decompress with the nanorecon/pod5 version that wrote the file, or a newer one",
        )


def _checked_text(value: str, what: str) -> str:
    if len(value.encode("utf-8")) > MAX_STRING_BYTES:
        raise NanoReconError(f"{what} is longer than {MAX_STRING_BYTES} bytes, which the token format does not store")
    return value


def _no_nulls(array: pa.Array, what: str) -> pa.Array:
    if array.null_count:
        raise NanoReconError(f"POD5 {what} contains null values, which this version cannot preserve")
    return array


def _dictionary_column(array: pa.Array, what: str) -> tuple[np.ndarray, list[str]]:
    _no_nulls(array, what)
    values = _no_nulls(array.dictionary, what).to_pylist()
    return array.indices.to_numpy(zero_copy_only=False), values


class SourceRead:
    """Meta of one input read plus the handle needed to load its signal later."""

    __slots__ = ("meta", "position", "_source", "_batch", "_row")

    def __init__(self, meta: ReadMeta, position: int, source: "Pod5Source", batch: pod5.reader.ReadRecordBatch, row: int) -> None:
        self.meta = meta
        self.position = position
        self._source = source
        self._batch = batch
        self._row = row

    def load_signal(self) -> np.ndarray:
        try:
            signal = self._batch.get_read(self._row).signal
            rows = self._batch.columns.signal[self._row].values
        except Exception as exc:  # pod5/pyarrow raise a variety of types for damaged data
            raise NanoReconError(f"cannot read signal of read {self.meta.read_id_str}: {exc}") from exc
        if len(rows):
            self._source._keep_signal_batches(int(pa.compute.min(rows).as_py()), int(pa.compute.max(rows).as_py()))
        signal = np.ascontiguousarray(signal)
        if signal.dtype != np.int16 or signal.ndim != 1:
            raise NanoReconError(f"read {self.meta.read_id_str} has signal of type {signal.dtype}/{signal.ndim}-D, expected int16 1-D")
        if signal.shape[0] != self.meta.num_samples:
            raise NanoReconError(
                f"read {self.meta.read_id_str}: signal has {signal.shape[0]} samples but num_samples says {self.meta.num_samples}"
            )
        return signal


class Pod5Source:
    def __init__(self, path: Path) -> None:
        """The file is read with ordinary reads instead of pod5's memory map: a map keeps every
        touched page of the input resident, so RSS would grow with the file size."""
        self.path = Path(path)
        previous = os.environ.get("POD5_DISABLE_MMAP_OPEN")
        os.environ["POD5_DISABLE_MMAP_OPEN"] = "1"
        try:
            self._reader = pod5.Reader(self.path)
        except Exception as exc:
            raise NanoReconError(f"cannot open POD5 file {self.path}: {exc}") from exc
        finally:
            if previous is None:
                os.environ.pop("POD5_DISABLE_MMAP_OPEN", None)
            else:
                os.environ["POD5_DISABLE_MMAP_OPEN"] = previous
        try:
            self._run_info_rows = self._index_run_infos()
        except NanoReconError:
            self.close()
            raise
        except Exception as exc:  # pyarrow/pod5 errors from a damaged run_info table
            self.close()
            raise NanoReconError(f"cannot read the run_info table of {self.path}: {exc}") from exc
        self._run_info_cache: dict[str, RunInfoMeta] = {}

    def close(self) -> None:
        reader, self._reader = getattr(self, "_reader", None), None
        if reader is not None:
            reader.close()

    def __enter__(self) -> "Pod5Source":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def read_count(self) -> int:
        table = self._reader.read_table
        return sum(table.get_batch(i).num_rows for i in range(table.num_record_batches))

    def _index_run_infos(self) -> dict[str, tuple[int, int]]:
        """acquisition_id -> first run_info row, the lookup rule pod5.Reader itself uses."""
        rows: dict[str, tuple[int, int]] = {}
        table = self._reader.run_info_table
        for b in range(table.num_record_batches):
            ids = _no_nulls(table.get_batch(b).column("acquisition_id"), "run_info.acquisition_id").to_pylist()
            for r, acq in enumerate(ids):
                rows.setdefault(acq, (b, r))
        return rows

    def _run_info(self, acquisition_id: str) -> RunInfoMeta:
        cached = self._run_info_cache.get(acquisition_id)
        if cached is not None:
            return cached
        location = self._run_info_rows.get(acquisition_id)
        if location is None:
            raise NanoReconError(f"a read references run info {acquisition_id!r}, which is missing from the run_info table")
        try:
            meta = self._read_run_info(acquisition_id, *location)
        except NanoReconError:
            raise
        except Exception as exc:
            raise NanoReconError(f"cannot read run info {acquisition_id!r}: {exc}") from exc
        if len(self._run_info_cache) >= _RUN_INFO_CACHE_LIMIT:
            self._run_info_cache.clear()
        self._run_info_cache[acquisition_id] = meta
        return meta

    def _read_run_info(self, acquisition_id: str, batch_index: int, row: int) -> RunInfoMeta:
        batch = self._reader.run_info_table.get_batch(batch_index)
        where = f"run info {acquisition_id!r}"
        values = {}
        for name in RUN_INFO_FIELDS:
            column = batch.column(_TIME_FIELDS_POD5.get(name, name))
            scalar = column[row]
            if not scalar.is_valid:
                raise NanoReconError(f"{where} has a null {name}, which this version cannot preserve")
            if name in _TIME_FIELDS_POD5:
                if not pa.types.is_timestamp(column.type) or column.type.unit != "ms":
                    raise NanoReconError(f"run_info.{_TIME_FIELDS_POD5[name]} has type {column.type}, expected timestamp[ms]")
                values[name] = int(scalar.value)
            elif name in RUN_INFO_MAP_FIELDS:
                entries = []
                for key, value in scalar.as_py():
                    if not isinstance(key, str) or not isinstance(value, str):
                        raise NanoReconError(f"{where} field {name} has a non-string entry {key!r}: {value!r}")
                    entries.append((_checked_text(key, f"{where} {name} key"), _checked_text(value, f"{where} {name} value")))
                if len(entries) > MAX_TABLE_ENTRIES:
                    raise NanoReconError(f"{where} field {name} has {len(entries)} entries; the token format stores at most {MAX_TABLE_ENTRIES}")
                values[name] = tuple(entries)
            else:
                value = scalar.as_py()
                values[name] = _checked_text(value, f"{where} {name}") if isinstance(value, str) else value
        return RunInfoMeta(**values)

    def _signal_cache(self) -> dict | None:
        # pod5.Reader memoizes every signal batch it touches, without bound. Reads are consumed
        # in order, so only the batches of the current read are worth keeping; dropping more
        # only costs a re-read, never correctness.
        cache = getattr(self._reader, "_cached_signal_batches", None)
        return cache if isinstance(cache, dict) else None

    def _drop_signal_cache(self) -> None:
        cache = self._signal_cache()
        if cache is not None:
            cache.clear()

    def _keep_signal_batches(self, first_row: int, last_row: int) -> None:
        cache = self._signal_cache()
        per_batch = self._reader.signal_batch_row_count
        if cache is None or per_batch <= 0:
            return
        lo, hi = first_row // per_batch, last_row // per_batch
        for key in [k for k in cache if not lo <= k <= hi]:
            del cache[key]

    def iter_reads(self) -> Iterator[SourceRead]:
        """Reads in file order; Meta only (signals load on request)."""
        position = 0
        for batch_index in range(self._reader.batch_count):
            self._drop_signal_cache()
            try:
                batch = self._reader.get_batch(batch_index)
                columns = batch.columns
                read_ids = _no_nulls(columns.read_id, "read_id").to_pylist()
                numeric = {
                    name: _no_nulls(getattr(columns, name), name).to_numpy(zero_copy_only=False)
                    for name in _NUMERIC_READ_COLUMNS
                }
                pore_idx, pore_values = _dictionary_column(columns.pore_type, "pore_type")
                reason_idx, reason_values = _dictionary_column(columns.end_reason, "end_reason")
                acq_idx, acq_values = _dictionary_column(columns.run_info, "run_info")
            except NanoReconError:
                raise
            except Exception as exc:  # damaged read table
                raise NanoReconError(f"cannot read read batch {batch_index} of {self.path}: {exc}") from exc
            for row in range(batch.num_reads):
                try:
                    meta = self._make_meta(row, read_ids, numeric, pore_values[pore_idx[row]],
                                           reason_values[reason_idx[row]], acq_values[acq_idx[row]])
                except NanoReconError:
                    raise
                except Exception as exc:
                    raise NanoReconError(f"cannot read the metadata of read {position} in {self.path}: {exc}") from exc
                yield SourceRead(meta, position, self, batch, row)
                position += 1
        self._drop_signal_cache()

    def _make_meta(self, row, read_ids, numeric, pore_type, end_reason, acquisition_id) -> ReadMeta:
        read_id = read_ids[row]
        if not isinstance(read_id, bytes) or len(read_id) != 16:
            raise NanoReconError(f"read {row} has a malformed read_id")
        rid = str(uuid.UUID(bytes=read_id))
        _checked_text(pore_type, f"read {rid} pore_type")
        if end_reason not in _END_REASON_NAMES:
            raise NanoReconError(f"read {rid} has end_reason {end_reason!r}, which pod5 {pod5.__version__} cannot write back")
        offset = float(numeric["calibration_offset"][row])
        scale = float(numeric["calibration_scale"][row])
        checked_calibration(offset, scale, read_id=rid)
        return ReadMeta(
            read_id=read_id,
            num_samples=int(numeric["num_samples"][row]),
            read_number=int(numeric["read_number"][row]),
            start_sample=int(numeric["start"][row]),
            channel=int(numeric["channel"][row]),
            well=int(numeric["well"][row]),
            pore_type=pore_type,
            calibration_offset=offset,
            calibration_scale=scale,
            median_before=float(numeric["median_before"][row]),
            end_reason=end_reason,
            end_reason_forced=bool(numeric["end_reason_forced"][row]),
            run_info=self._run_info(acquisition_id),
            num_minknow_events=int(numeric["num_minknow_events"][row]),
            tracked_scaling_scale=float(numeric["tracked_scaling_scale"][row]),
            tracked_scaling_shift=float(numeric["tracked_scaling_shift"][row]),
            predicted_scaling_scale=float(numeric["predicted_scaling_scale"][row]),
            predicted_scaling_shift=float(numeric["predicted_scaling_shift"][row]),
            num_reads_since_mux_change=int(numeric["num_reads_since_mux_change"][row]),
            time_since_mux_change=float(numeric["time_since_mux_change"][row]),
            open_pore_level=float(numeric["open_pore_level"][row]),
        )


def to_pod5_run_info(info: RunInfoMeta) -> pod5.RunInfo:
    # Integer millisecond timestamps pass through pod5's timestamp_to_int unchanged (a datetime
    # would go through float seconds and could lose a millisecond); list-valued maps keep order.
    return pod5.RunInfo(
        acquisition_id=info.acquisition_id,
        acquisition_start_time=info.acquisition_start_time_ms,
        adc_max=info.adc_max,
        adc_min=info.adc_min,
        context_tags=[tuple(e) for e in info.context_tags],
        experiment_name=info.experiment_name,
        flow_cell_id=info.flow_cell_id,
        flow_cell_product_code=info.flow_cell_product_code,
        protocol_name=info.protocol_name,
        protocol_run_id=info.protocol_run_id,
        protocol_start_time=info.protocol_start_time_ms,
        sample_id=info.sample_id,
        sample_rate=info.sample_rate,
        sequencing_kit=info.sequencing_kit,
        sequencer_position=info.sequencer_position,
        sequencer_position_type=info.sequencer_position_type,
        software=info.software,
        system_name=info.system_name,
        system_type=info.system_type,
        tracking_id=[tuple(e) for e in info.tracking_id],
    )


class Pod5Sink:
    """Streaming POD5 writer; `path` must not exist yet (it lives in the output transaction)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        try:
            self._writer = pod5.Writer(self.path, software_name=f"nanorecon {__version__}")
        except Exception as exc:
            raise NanoReconError(f"cannot create POD5 file {self.path}: {exc}") from exc
        self._run_infos: dict[RunInfoMeta, pod5.RunInfo] = {}

    def begin_group(self) -> None:
        """Converted RunInfo objects are only reused within one group."""
        self._run_infos.clear()

    def write(self, meta: ReadMeta, signal: np.ndarray) -> None:
        if signal.dtype != np.int16 or signal.ndim != 1 or signal.shape[0] != meta.num_samples:
            raise RuntimeError(f"reconstructed signal for {meta.read_id_str} has wrong shape/dtype")
        run_info = self._run_infos.get(meta.run_info)
        if run_info is None:
            run_info = self._run_infos[meta.run_info] = to_pod5_run_info(meta.run_info)
        read = pod5.Read(
            read_id=uuid.UUID(bytes=meta.read_id),
            pore=pod5.Pore(channel=meta.channel, well=meta.well, pore_type=meta.pore_type),
            calibration=pod5.Calibration(offset=meta.calibration_offset, scale=meta.calibration_scale),
            read_number=meta.read_number,
            start_sample=meta.start_sample,
            median_before=meta.median_before,
            end_reason=pod5.EndReason(reason=_END_REASON_NAMES[meta.end_reason], forced=meta.end_reason_forced),
            run_info=run_info,
            num_minknow_events=meta.num_minknow_events,
            tracked_scaling=ShiftScalePair(shift=meta.tracked_scaling_shift, scale=meta.tracked_scaling_scale),
            predicted_scaling=ShiftScalePair(shift=meta.predicted_scaling_shift, scale=meta.predicted_scaling_scale),
            num_reads_since_mux_change=meta.num_reads_since_mux_change,
            time_since_mux_change=meta.time_since_mux_change,
            open_pore_level=meta.open_pore_level,
            signal=signal,
        )
        try:
            self._writer.add_read(read)
        except Exception as exc:
            raise NanoReconError(f"POD5 writer rejected read {meta.read_id_str}: {exc}") from exc

    def close(self) -> None:
        writer, self._writer = getattr(self, "_writer", None), None
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:
                raise NanoReconError(f"cannot finish POD5 file {self.path}: {exc}") from exc

    def __enter__(self) -> "Pod5Sink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
