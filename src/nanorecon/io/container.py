"""NanoRecon token container (".nrpod"), format 1.0. Sequential write and read only.

    Preamble   magic b"\\x89NRECON\\n", u16 major, u16 minor, u32 flags (=0)      16 bytes
    Record*    u32 type, u64 payload length, payload                            12 + n bytes
        1 FILE_HEADER  strict UTF-8 JSON: model identity, codec profile, read count
        2 GROUP        strict UTF-8 JSON: read count + run_info / pore_type / end_reason tables
        3 READ         104-byte little-endian Meta struct, center f32[K], scale f32[K], codes u16[K*T]
        4 END          strict UTF-8 JSON: group / read / chunk totals
Order: FILE_HEADER, then per group one GROUP followed by exactly its READ records, then END,
then end of file. Chunk placement is derived from (num_samples, profile) and never stored.
The full specification is docs/format.md.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

import numpy as np

from ..model_config import CodecProfile, ModelIdentity
from ..signal.chunking import ChunkPlan, chunk_count
from ..types import (
    NanoReconError,
    ReadMeta,
    RunInfoMeta,
    RUN_INFO_FIELDS,
    RUN_INFO_INT_RANGES,
    RUN_INFO_MAP_FIELDS,
    RUN_INFO_STR_FIELDS,
)

MAGIC = b"\x89NRECON\n"
FORMAT_MAJOR = 1
FORMAT_MINOR = 0
FORMAT_NAME = "nanorecon-tokens"

_PREAMBLE = struct.Struct("<8sHHI")
_RECORD = struct.Struct("<IQ")
REC_FILE_HEADER, REC_GROUP, REC_READ, REC_END = 1, 2, 3, 4
_RECORD_NAMES = {REC_FILE_HEADER: "FILE_HEADER", REC_GROUP: "GROUP", REC_READ: "READ", REC_END: "END"}

MAX_HEADER_BYTES = 1 << 20
MAX_GROUP_BYTES = 4 << 20  # parsed JSON can take ~30x its size in memory, so keep this small
MAX_END_BYTES = 1 << 16
MAX_TABLE_ENTRIES = 32767  # POD5 dictionary indices are int16
MAX_STRING_BYTES = 1 << 20
MAX_GROUP_READS = 1 << 31
MAX_READ_SAMPLES = 1 << 40

_LEGACY_ZIP = b"PK\x03\x04"
_POD5_MAGIC = b"\x8bPOD\r\n\x1a\n"

READ_META_DTYPE = np.dtype(
    [
        ("read_id", "V16"),
        ("num_samples", "<u8"),
        ("num_chunks", "<u8"),
        ("read_number", "<u4"),
        ("start_sample", "<u8"),
        ("channel", "<u2"),
        ("well", "u1"),
        ("end_reason_forced", "u1"),
        ("calibration_offset", "<f4"),
        ("calibration_scale", "<f4"),
        ("median_before", "<f4"),
        ("num_minknow_events", "<u8"),
        ("tracked_scaling_scale", "<f4"),
        ("tracked_scaling_shift", "<f4"),
        ("predicted_scaling_scale", "<f4"),
        ("predicted_scaling_shift", "<f4"),
        ("num_reads_since_mux_change", "<u4"),
        ("time_since_mux_change", "<f4"),
        ("open_pore_level", "<f4"),
        ("run_info_index", "<u2"),
        ("pore_type_index", "<u2"),
        ("end_reason_index", "<u2"),
        ("reserved", "<u2"),
    ]
)
assert READ_META_DTYPE.itemsize == 104

_META_SCALARS = (
    "read_number",
    "start_sample",
    "channel",
    "well",
    "calibration_offset",
    "calibration_scale",
    "median_before",
    "num_minknow_events",
    "tracked_scaling_scale",
    "tracked_scaling_shift",
    "predicted_scaling_scale",
    "predicted_scaling_shift",
    "num_reads_since_mux_change",
    "time_since_mux_change",
    "open_pore_level",
)


def read_payload_bytes(num_chunks: int, tokens_per_chunk: int) -> int:
    return READ_META_DTYPE.itemsize + num_chunks * 8 + num_chunks * tokens_per_chunk * 2


# ---------------------------------------------------------------------------------------
# Header / group / end payloads
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FileHeader:
    model: ModelIdentity
    profile: CodecProfile
    read_count: int
    producer_version: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT_NAME,
            "model": self.model.to_json_dict(),
            "profile": self.profile.to_json_dict(),
            "read_count": self.read_count,
            "producer": {"name": "nanorecon", "version": self.producer_version},
        }


@dataclass(frozen=True)
class GroupTables:
    run_infos: tuple[RunInfoMeta, ...]
    pore_types: tuple[str, ...]
    end_reasons: tuple[str, ...]


def _dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _loads(payload: bytes, what: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NanoReconError(f"{what} is not valid UTF-8: {exc}") from exc

    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise NanoReconError(f"{what} repeats key {key!r}")
            obj[key] = value
        return obj

    def constant(name):
        raise NanoReconError(f"{what} contains non-standard JSON constant {name}")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except NanoReconError:
        raise
    except (ValueError, RecursionError) as exc:
        raise NanoReconError(f"{what} is not valid JSON: {exc}") from exc


def _obj(value: Any, keys: Sequence[str], what: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise NanoReconError(f"{what} must be a JSON object")
    missing = [k for k in keys if k not in value]
    extra = sorted(set(value) - set(keys))
    if missing or extra:
        raise NanoReconError(f"{what} fields mismatch: missing={missing} unexpected={extra}")
    return value


def _json_int(value: Any, what: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise NanoReconError(f"{what} must be an integer in [{lo}, {hi}], got {value!r}")
    return value


def _json_str(value: Any, what: str, *, max_bytes: int = MAX_STRING_BYTES) -> str:
    if not isinstance(value, str):
        raise NanoReconError(f"{what} must be a string")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:  # JSON "\ud800" escapes decode to lone surrogates
        raise NanoReconError(f"{what} is not valid Unicode") from exc
    if size > max_bytes:
        raise NanoReconError(f"{what} must be a string of at most {max_bytes} bytes")
    return value


def json_size(value: Any) -> int:
    """Bytes `value` takes in a record payload (compact UTF-8 JSON)."""
    return len(_dumps(value))


def run_info_json_bytes(info: RunInfoMeta) -> int:
    return json_size(run_info_to_json(info))


def run_info_to_json(info: RunInfoMeta) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in RUN_INFO_FIELDS:
        value = getattr(info, name)
        out[name] = [list(e) for e in value] if name in RUN_INFO_MAP_FIELDS else value
    return out


def run_info_from_json(value: Any, what: str) -> RunInfoMeta:
    data = _obj(value, RUN_INFO_FIELDS, what)
    parsed: dict[str, Any] = {}
    for name in RUN_INFO_STR_FIELDS:
        parsed[name] = _json_str(data[name], f"{what}.{name}")
    for name, (lo, hi) in RUN_INFO_INT_RANGES.items():
        parsed[name] = _json_int(data[name], f"{what}.{name}", lo, hi)
    for name in RUN_INFO_MAP_FIELDS:
        entries = data[name]
        if not isinstance(entries, list) or len(entries) > MAX_TABLE_ENTRIES:
            raise NanoReconError(f"{what}.{name} must be a list of [key, value] pairs")
        pairs = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, list) or len(entry) != 2:
                raise NanoReconError(f"{what}.{name}[{i}] must be a [key, value] pair")
            pairs.append((_json_str(entry[0], f"{what}.{name}[{i}]"), _json_str(entry[1], f"{what}.{name}[{i}]")))
        parsed[name] = tuple(pairs)
    return RunInfoMeta(**parsed)


def _parse_header(payload: bytes) -> FileHeader:
    data = _obj(_loads(payload, "file header"), ("format", "model", "profile", "read_count", "producer"), "file header")
    if data["format"] != FORMAT_NAME:
        raise NanoReconError(f"file header names format {data['format']!r}, expected {FORMAT_NAME!r}")
    model = _obj(data["model"], ("repo_id", "revision", "architecture", "model_type", "codebook_size"), "file header model")
    identity = ModelIdentity(
        repo_id=_json_str(model["repo_id"], "model.repo_id", max_bytes=256),
        revision=_json_str(model["revision"], "model.revision", max_bytes=256),
        architecture=_json_str(model["architecture"], "model.architecture", max_bytes=256),
        model_type=_json_str(model["model_type"], "model.model_type", max_bytes=256),
        codebook_size=_json_int(model["codebook_size"], "model.codebook_size", 1, 2**32),
    )
    try:
        profile = CodecProfile.from_json_dict(data["profile"])
    except ValueError as exc:
        raise NanoReconError(f"file header profile is invalid: {exc}") from exc
    producer = _obj(data["producer"], ("name", "version"), "file header producer")
    _json_str(producer["name"], "producer.name", max_bytes=256)
    return FileHeader(
        model=identity,
        profile=profile,
        read_count=_json_int(data["read_count"], "read_count", 0, 2**63 - 1),
        producer_version=_json_str(producer["version"], "producer.version", max_bytes=256),
    )


def _parse_group(payload: bytes, index: int) -> tuple[int, GroupTables]:
    what = f"group {index}"
    data = _obj(_loads(payload, what), ("read_count", "run_infos", "pore_types", "end_reasons"), what)
    count = _json_int(data["read_count"], f"{what}.read_count", 1, MAX_GROUP_READS)
    tables = []
    for key in ("run_infos", "pore_types", "end_reasons"):
        entries = data[key]
        if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_TABLE_ENTRIES:
            raise NanoReconError(f"{what}.{key} must be a list of 1..{MAX_TABLE_ENTRIES} entries")
        tables.append(entries)
    run_infos = tuple(run_info_from_json(v, f"{what}.run_infos[{i}]") for i, v in enumerate(tables[0]))
    pore_types = tuple(_json_str(v, f"{what}.pore_types[{i}]") for i, v in enumerate(tables[1]))
    end_reasons = tuple(_json_str(v, f"{what}.end_reasons[{i}]", max_bytes=256) for i, v in enumerate(tables[2]))
    return count, GroupTables(run_infos, pore_types, end_reasons)


# ---------------------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------------------


def _write_array(fh: BinaryIO, array: np.ndarray, dtype: str) -> None:
    arr = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    if arr.size:
        fh.write(memoryview(arr).cast("B"))


class ContainerWriter:
    """Writes one container to an already opened binary file (the output transaction owns it)."""

    def __init__(self, fh: BinaryIO, header: FileHeader) -> None:
        self._fh = fh
        self.header = header
        self._profile = header.profile
        self.groups_written = 0
        self.reads_written = 0
        self.chunks_written = 0
        self._reads_left = 0
        self._finished = False
        self._run_info_index: dict[RunInfoMeta, int] = {}
        self._pore_index: dict[str, int] = {}
        self._reason_index: dict[str, int] = {}
        payload = _dumps(header.to_json_dict())
        if len(payload) > MAX_HEADER_BYTES:
            raise ValueError("file header too large")
        fh.write(_PREAMBLE.pack(MAGIC, FORMAT_MAJOR, FORMAT_MINOR, 0))
        self._record(REC_FILE_HEADER, payload)

    def _record(self, rtype: int, payload: bytes) -> None:
        self._fh.write(_RECORD.pack(rtype, len(payload)))
        self._fh.write(payload)

    def begin_group(self, metas: Sequence[ReadMeta]) -> GroupTables:
        """Start a group holding exactly `metas` (in order); builds the de-duplicated tables."""
        if self._reads_left:
            raise RuntimeError(f"previous group still expects {self._reads_left} reads")
        if not metas:
            raise ValueError("groups must not be empty")
        run_infos: dict[RunInfoMeta, int] = {}
        pores: dict[str, int] = {}
        reasons: dict[str, int] = {}
        for meta in metas:
            run_infos.setdefault(meta.run_info, len(run_infos))
            pores.setdefault(meta.pore_type, len(pores))
            reasons.setdefault(meta.end_reason, len(reasons))
        if max(len(run_infos), len(pores), len(reasons)) > MAX_TABLE_ENTRIES:
            raise ValueError("group tables exceed the format limit")
        tables = GroupTables(tuple(run_infos), tuple(pores), tuple(reasons))
        payload = _dumps(
            {
                "read_count": len(metas),
                "run_infos": [run_info_to_json(r) for r in tables.run_infos],
                "pore_types": list(tables.pore_types),
                "end_reasons": list(tables.end_reasons),
            }
        )
        if len(payload) > MAX_GROUP_BYTES:
            raise ValueError(f"group tables need {len(payload)} bytes, above the {MAX_GROUP_BYTES} byte limit")
        self._record(REC_GROUP, payload)
        self._run_info_index, self._pore_index, self._reason_index = run_infos, pores, reasons
        self._reads_left = len(metas)
        self.groups_written += 1
        return tables

    def write_read(self, meta: ReadMeta, centers: np.ndarray, scales: np.ndarray, codes: np.ndarray) -> None:
        if self._reads_left <= 0:
            raise RuntimeError("write_read called outside a group")
        p = self._profile
        k = chunk_count(meta.num_samples, p.chunk_samples, p.hop_samples)
        if centers.shape != (k,) or scales.shape != (k,) or codes.shape != (k, p.tokens_per_chunk):
            raise ValueError(
                f"read {meta.read_id_str}: arrays {centers.shape}/{scales.shape}/{codes.shape} do not match K={k}, T={p.tokens_per_chunk}"
            )
        rec = np.zeros((), dtype=READ_META_DTYPE)
        rec["read_id"] = np.void(meta.read_id)
        rec["num_samples"] = meta.num_samples
        rec["num_chunks"] = k
        for name in _META_SCALARS:
            rec[name] = getattr(meta, name)
        rec["end_reason_forced"] = 1 if meta.end_reason_forced else 0
        rec["run_info_index"] = self._run_info_index[meta.run_info]
        rec["pore_type_index"] = self._pore_index[meta.pore_type]
        rec["end_reason_index"] = self._reason_index[meta.end_reason]
        self._fh.write(_RECORD.pack(REC_READ, read_payload_bytes(k, p.tokens_per_chunk)))
        self._fh.write(rec.tobytes())
        _write_array(self._fh, centers, "<f4")
        _write_array(self._fh, scales, "<f4")
        _write_array(self._fh, codes, "<u2")
        self._reads_left -= 1
        self.reads_written += 1
        self.chunks_written += k

    def finish(self) -> None:
        if self._reads_left:
            raise RuntimeError(f"current group still expects {self._reads_left} reads")
        if self.reads_written != self.header.read_count:
            raise RuntimeError(f"header announced {self.header.read_count} reads but {self.reads_written} were written")
        self._record(
            REC_END,
            _dumps(
                {
                    "group_count": self.groups_written,
                    "read_count": self.reads_written,
                    "chunk_count": self.chunks_written,
                }
            ),
        )
        self._finished = True


# ---------------------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------------------


def identify_format(prefix: bytes) -> str:
    """'nrecon', 'legacy-nrpod', 'pod5' or 'unknown' from the first bytes of a file."""
    if prefix.startswith(MAGIC):
        return "nrecon"
    if prefix.startswith(_LEGACY_ZIP):
        return "legacy-nrpod"
    if prefix.startswith(_POD5_MAGIC):
        return "pod5"
    return "unknown"


class StoredRead:
    """One READ record: Meta and per-chunk stats in memory, codes streamed on request."""

    def __init__(self, reader: "ContainerReader", meta: ReadMeta, plan: ChunkPlan, centers: np.ndarray, scales: np.ndarray) -> None:
        self._reader = reader
        self.meta = meta
        self.plan = plan
        self.centers = centers
        self.scales = scales
        self._next_chunk = 0

    @property
    def num_chunks(self) -> int:
        return self.plan.num_chunks

    def code_batches(self, batch_size: int, out: np.ndarray | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (first chunk index, codes[b, T]) in order; at most `batch_size` rows at a time.
        With `out` (uint16[batch_size, T]) the rows are read into it and views are yielded."""
        tokens = self._reader.header.profile.tokens_per_chunk
        if out is None:
            out = np.empty((batch_size, tokens), dtype=np.uint16)
        while self._next_chunk < self.plan.num_chunks:
            first = self._next_chunk
            rows = min(batch_size, self.plan.num_chunks - first)
            view = out[:rows]
            self._reader._read_into(view, f"codes of read {self.meta.read_id_str}")
            if self._reader._code_limit < 65536 and rows and int(view.max()) >= self._reader._code_limit:
                raise NanoReconError(f"read {self.meta.read_id_str} holds a code outside the codebook")
            self._next_chunk += rows
            yield first, view

    def read_codes(self) -> np.ndarray:
        """All (remaining) codes of this read as uint16[K, T]; about 1 KB per chunk."""
        tokens = self._reader.header.profile.tokens_per_chunk
        out = np.empty((self.plan.num_chunks - self._next_chunk, tokens), dtype=np.uint16)
        for _ in self.code_batches(max(out.shape[0], 1), out=out):
            pass
        return out

    def _skip_rest(self) -> None:
        left = self.plan.num_chunks - self._next_chunk
        if left:
            self._reader._skip(left * self._reader.header.profile.tokens_per_chunk * 2)
            self._next_chunk = self.plan.num_chunks


class ContainerReader:
    def __init__(self, fh: BinaryIO, *, name: str = "input") -> None:
        self._fh = fh
        self._name = name
        prefix = self._read_exact_or_none(_PREAMBLE.size)
        kind = identify_format(prefix or b"")
        if kind == "legacy-nrpod":
            raise NanoReconError(
                f"{name} is a legacy NanoRecon .nrpod (ZIP) file; this version does not read that format",
                hint="re-create it from the original POD5 with this version's `nanorecon compress`",
            )
        if kind == "pod5":
            raise NanoReconError(f"{name} is a POD5 file, not a NanoRecon token file", hint="did you mean `nanorecon compress`?")
        if kind != "nrecon" or prefix is None or len(prefix) < _PREAMBLE.size:
            raise NanoReconError(f"{name} is not a NanoRecon token file")
        _, major, minor, flags = _PREAMBLE.unpack(prefix)
        if major != FORMAT_MAJOR:
            raise NanoReconError(f"{name} uses container format {major}.{minor}; this version reads {FORMAT_MAJOR}.x only")
        if minor > FORMAT_MINOR:
            raise NanoReconError(
                f"{name} uses container format {major}.{minor}, newer than this version ({FORMAT_MAJOR}.{FORMAT_MINOR})",
                hint="upgrade nanorecon",
            )
        if flags != 0:
            raise NanoReconError(f"{name} sets unknown preamble flags {flags:#x}")
        rtype, payload = self._next_record(expect_json=True)
        if rtype != REC_FILE_HEADER:
            raise NanoReconError(f"{name}: first record is {_RECORD_NAMES.get(rtype, rtype)}, expected FILE_HEADER")
        self.header = _parse_header(payload)
        self._code_limit = self.header.profile.codebook_size
        self.groups_read = 0
        self.reads_read = 0
        self.chunks_read = 0
        self._iterating = False

    # -- low level ---------------------------------------------------------------------

    def _read_exact_or_none(self, n: int) -> bytes | None:
        data = self._fh.read(n)
        if not data:
            return None
        while len(data) < n:
            more = self._fh.read(n - len(data))
            if not more:
                return data
            data += more
        return data

    def _read_exact(self, n: int, what: str) -> bytes:
        data = self._read_exact_or_none(n) if n else b""
        if data is None or len(data) != n:
            raise NanoReconError(f"{self._name} is truncated inside {what}")
        return data

    def _read_into(self, array: np.ndarray, what: str) -> None:
        view = memoryview(array).cast("B")
        filled = 0
        while filled < len(view):
            n = self._fh.readinto(view[filled:])
            if not n:
                raise NanoReconError(f"{self._name} is truncated inside {what}")
            filled += n
        if not np.little_endian:
            array.byteswap(inplace=True)

    def _skip(self, n: int) -> None:
        while n > 0:
            chunk = self._fh.read(min(n, 1 << 20))
            if not chunk:
                raise NanoReconError(f"{self._name} is truncated")
            n -= len(chunk)

    def _next_record(self, *, expect_json: bool = False) -> tuple[int, bytes | int]:
        head = self._read_exact_or_none(_RECORD.size)
        if head is None:
            raise NanoReconError(
                f"{self._name} ends without an END record (the file is truncated or was never completed)"
            )
        if len(head) != _RECORD.size:
            raise NanoReconError(f"{self._name} is truncated inside a record header")
        rtype, length = _RECORD.unpack(head)
        if rtype not in _RECORD_NAMES:
            raise NanoReconError(f"{self._name} contains unknown record type {rtype}")
        if rtype == REC_READ:
            return rtype, length
        limit = {REC_FILE_HEADER: MAX_HEADER_BYTES, REC_GROUP: MAX_GROUP_BYTES, REC_END: MAX_END_BYTES}[rtype]
        if length > limit:
            raise NanoReconError(f"{self._name}: {_RECORD_NAMES[rtype]} record claims {length} bytes (limit {limit})")
        return rtype, self._read_exact(length, _RECORD_NAMES[rtype])

    # -- records -----------------------------------------------------------------------

    def _read_meta(self, length: int, tables: GroupTables) -> StoredRead:
        p = self.header.profile
        if length < READ_META_DTYPE.itemsize:
            raise NanoReconError(f"{self._name}: READ record of {length} bytes is too short")
        rec = np.frombuffer(self._read_exact(READ_META_DTYPE.itemsize, "READ meta"), dtype=READ_META_DTYPE)[0]
        read_id = rec["read_id"].tobytes()
        num_samples = int(rec["num_samples"])
        num_chunks = int(rec["num_chunks"])
        where = f"read #{self.reads_read}"
        if num_samples > MAX_READ_SAMPLES:
            raise NanoReconError(f"{where} claims {num_samples} samples")
        expected_k = chunk_count(num_samples, p.chunk_samples, p.hop_samples)
        if num_chunks != expected_k:
            raise NanoReconError(f"{where} stores {num_chunks} chunks; the profile implies {expected_k} for {num_samples} samples")
        if length != read_payload_bytes(num_chunks, p.tokens_per_chunk):
            raise NanoReconError(f"{where} record length {length} does not match its {num_chunks} chunks")
        if int(rec["reserved"]) != 0 or int(rec["end_reason_forced"]) > 1:
            raise NanoReconError(f"{where} has invalid reserved/flag bytes")
        indices = (int(rec["run_info_index"]), int(rec["pore_type_index"]), int(rec["end_reason_index"]))
        sizes = (len(tables.run_infos), len(tables.pore_types), len(tables.end_reasons))
        if any(i >= n for i, n in zip(indices, sizes)):
            raise NanoReconError(f"{where} references a group table entry that does not exist")
        centers = np.empty(num_chunks, dtype=np.float32)
        scales = np.empty(num_chunks, dtype=np.float32)
        self._read_into(centers, f"{where} centers")
        self._read_into(scales, f"{where} scales")
        if not (np.isfinite(centers).all() and np.isfinite(scales).all() and (scales >= 0).all()):
            raise NanoReconError(f"{where} has non-finite or negative normalization parameters")
        scalars = {name: rec[name].item() for name in _META_SCALARS}
        for name in ("read_number", "start_sample", "channel", "well", "num_minknow_events", "num_reads_since_mux_change"):
            scalars[name] = int(scalars[name])
        for name in ("calibration_offset", "calibration_scale"):
            if not math.isfinite(scalars[name]) or (name == "calibration_scale" and scalars[name] == 0):
                raise NanoReconError(f"{where} has invalid {name} {scalars[name]!r}")
        meta = ReadMeta(
            read_id=read_id,
            num_samples=num_samples,
            pore_type=tables.pore_types[indices[1]],
            end_reason=tables.end_reasons[indices[2]],
            end_reason_forced=bool(rec["end_reason_forced"]),
            run_info=tables.run_infos[indices[0]],
            **scalars,
        )
        return StoredRead(self, meta, ChunkPlan(num_samples, p.chunk_samples, p.hop_samples), centers, scales)

    def iter_reads(self) -> Iterator[tuple[int, StoredRead]]:
        """Yield (group index, read) in file order; validates END totals and end of file."""
        if self._iterating:
            raise RuntimeError("a container can only be iterated once")
        self._iterating = True
        while True:
            rtype, payload = self._next_record()
            if rtype == REC_END:
                self._check_end(payload)
                return
            if rtype != REC_GROUP:
                raise NanoReconError(f"{self._name}: found {_RECORD_NAMES[rtype]} where a GROUP or END record was expected")
            count, tables = _parse_group(payload, self.groups_read)
            if self.reads_read + count > self.header.read_count:
                raise NanoReconError(
                    f"{self._name}: group {self.groups_read} would bring the file to {self.reads_read + count} reads; "
                    f"the header announces {self.header.read_count}"
                )
            group_index = self.groups_read
            self.groups_read += 1
            for _ in range(count):
                rtype, length = self._next_record()
                if rtype != REC_READ:
                    raise NanoReconError(f"{self._name}: group {group_index} ended early; found {_RECORD_NAMES[rtype]}")
                stored = self._read_meta(length, tables)
                yield group_index, stored
                stored._skip_rest()
                self.reads_read += 1
                self.chunks_read += stored.num_chunks

    def _check_end(self, payload: bytes) -> None:
        data = _obj(_loads(payload, "END record"), ("group_count", "read_count", "chunk_count"), "END record")
        actual = {"group_count": self.groups_read, "read_count": self.reads_read, "chunk_count": self.chunks_read}
        for key, value in actual.items():
            if data[key] != value:
                raise NanoReconError(f"{self._name}: END record says {key}={data[key]!r} but the file holds {value}")
        if self.reads_read != self.header.read_count:
            raise NanoReconError(f"{self._name}: header announces {self.header.read_count} reads, file holds {self.reads_read}")
        if self._fh.read(1):
            raise NanoReconError(f"{self._name} has data after its END record")
