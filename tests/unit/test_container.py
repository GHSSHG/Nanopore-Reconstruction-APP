"""Token container write/read, structural validation, truncation and corruption."""

import io
import struct
import uuid
from dataclasses import replace

import numpy as np
import pytest

from nanorecon.io import container as C
from nanorecon.io.container import ContainerReader, ContainerWriter, FileHeader
from nanorecon.model_config import MODEL
from nanorecon.signal.chunking import chunk_count
from nanorecon.types import NanoReconError, ReadMeta, RunInfoMeta, read_meta_equal

RUN_A = RunInfoMeta(
    acquisition_id="acq-a", acquisition_start_time_ms=1735787045123, adc_max=4095, adc_min=-4096,
    context_tags=(("k", "v"), ("dup", "1"), ("dup", "2")), experiment_name="", flow_cell_id="FC",
    flow_cell_product_code="P", protocol_name="p", protocol_run_id="r", protocol_start_time_ms=-5,
    sample_id="☃", sample_rate=5000, sequencing_kit="k", sequencer_position="1A",
    sequencer_position_type="t", software="s", system_name="n", system_type="st", tracking_id=(),
)
RUN_B = replace(RUN_A, acquisition_id="acq-b", sample_id="other")


def meta(i, n, run=RUN_A, **kw):
    values = dict(
        read_id=uuid.UUID(int=i + 1).bytes, num_samples=n, read_number=i, start_sample=2**40 + i,
        channel=1 + i, well=1, pore_type="not_set" if i % 2 else "", calibration_offset=-240.0,
        calibration_scale=float(np.float32(0.1755)), median_before=float("nan"), end_reason="signal_positive",
        end_reason_forced=bool(i % 2), run_info=run, num_minknow_events=2**40, tracked_scaling_scale=float("inf"),
        tracked_scaling_shift=float("-inf"), predicted_scaling_scale=-0.0, predicted_scaling_shift=1.5,
        num_reads_since_mux_change=3, time_since_mux_change=0.25, open_pore_level=float("nan"),
    )
    values.update(kw)
    return ReadMeta(**values)


def header(profile, read_count):
    ident = MODEL
    return FileHeader(model=ident, profile=profile, read_count=read_count, producer_version="test")


def arrays(profile, n, seed):
    k = chunk_count(n, profile.chunk_samples, profile.hop_samples)
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=k).astype(np.float32),
        np.abs(rng.normal(size=k)).astype(np.float32),
        rng.integers(0, 65536, size=(k, profile.tokens_per_chunk)).astype(np.uint16),
    )


def build(profile, groups):
    """groups: list of lists of (meta, arrays)."""
    buf = io.BytesIO()
    w = ContainerWriter(buf, header(profile, sum(len(g) for g in groups)))
    for g in groups:
        w.begin_group([m for m, _ in g])
        for m, (c, s, k) in g:
            w.write_read(m, c, s, k)
    w.finish()
    return buf.getvalue()


def read_all(data, batch=3):
    r = ContainerReader(io.BytesIO(data))
    out = []
    for gi, stored in r.iter_reads():
        codes = [c.copy() for _, c in stored.code_batches(batch)]
        codes = np.concatenate(codes) if codes else np.empty((0, r.header.profile.tokens_per_chunk), np.uint16)
        out.append((gi, stored.meta, stored.centers.copy(), stored.scales.copy(), codes))
    return r, out


@pytest.fixture
def sample(profile):
    lengths = [[0, 1, 8192, 16300], [9000], [5, 40000]]
    groups, i = [], 0
    for g in lengths:
        items = []
        for n in g:
            items.append((meta(i, n, run=RUN_A if i % 2 == 0 else RUN_B), arrays(profile, n, i)))
            i += 1
        groups.append(items)
    return groups, build(profile, groups)


def test_roundtrip(sample, profile):
    groups, data = sample
    reader, out = read_all(data)
    flat = [(gi, m, a) for gi, g in enumerate(groups) for m, a in g]
    assert len(out) == len(flat)
    for (gi, m, (c, s, k)), (ogi, om, oc, os_, ok) in zip(flat, out):
        assert gi == ogi
        assert read_meta_equal(m, om) and om.run_info == m.run_info
        assert np.array_equal(c, oc) and np.array_equal(s, os_) and np.array_equal(k, ok)
    assert reader.header.profile == profile
    assert (reader.groups_read, reader.reads_read) == (3, 7)


def test_layout_is_deterministic(sample, profile):
    groups, data = sample
    assert build(profile, groups) == data
    assert data[:8] == C.MAGIC and struct.unpack("<HH", data[8:12]) == (1, 0)


def test_empty_file_roundtrip(profile):
    data = build(profile, [])
    reader, out = read_all(data)
    assert out == [] and reader.header.read_count == 0


def test_every_truncation_fails_cleanly(profile):
    groups = [[(meta(0, 9000), arrays(profile, 9000, 1)), (meta(1, 3), arrays(profile, 3, 2))]]
    data = build(profile, groups)
    step = 1 if len(data) < 5000 else 7
    for cut in list(range(0, 400)) + list(range(400, len(data), step)):
        with pytest.raises(NanoReconError):
            read_all(data[:cut])


def test_trailing_bytes_rejected(sample):
    with pytest.raises(NanoReconError, match="after its END"):
        read_all(sample[1] + b"\0")


@pytest.mark.parametrize(
    "prefix, message",
    [
        (b"PK\x03\x04rest-of-zip", "legacy"),
        (b"\x8bPOD\r\n\x1a\nmore", "POD5"),
        (b"hello world, not ours", "not a NanoRecon"),
        (b"", "not a NanoRecon"),
    ],
)
def test_foreign_files_identified(prefix, message):
    with pytest.raises(NanoReconError, match=message):
        ContainerReader(io.BytesIO(prefix))


def patch_preamble(data, major=1, minor=0, flags=0):
    return data[:8] + struct.pack("<HHI", major, minor, flags) + data[16:]


def test_versions(sample):
    data = sample[1]
    with pytest.raises(NanoReconError, match="reads 1.x only"):
        ContainerReader(io.BytesIO(patch_preamble(data, major=2)))
    with pytest.raises(NanoReconError, match="newer"):
        ContainerReader(io.BytesIO(patch_preamble(data, minor=1)))
    with pytest.raises(NanoReconError, match="flags"):
        ContainerReader(io.BytesIO(patch_preamble(data, flags=1)))


def records(data):
    """[(offset, type, length)] of all records."""
    out, pos = [], 16
    while pos < len(data):
        t, n = struct.unpack_from("<IQ", data, pos)
        out.append((pos, t, n))
        pos += 12 + n
    return out


def replace_record(data, index, payload, rtype=None):
    pos, t, n = records(data)[index]
    return data[:pos] + struct.pack("<IQ", rtype or t, len(payload)) + payload + data[pos + 12 + n :]


def test_json_is_strict(sample, profile):
    data = sample[1]
    hdr = data[records(data)[0][0] + 12 : records(data)[0][0] + 12 + records(data)[0][2]]
    dup = hdr[:-1] + b',"read_count":7}'
    with pytest.raises(NanoReconError, match="repeats key"):
        ContainerReader(io.BytesIO(replace_record(data, 0, dup)))
    nan = hdr.replace(b'"normalization_epsilon":1e-06', b'"normalization_epsilon":NaN')
    with pytest.raises(NanoReconError, match="non-standard"):
        ContainerReader(io.BytesIO(replace_record(data, 0, nan)))
    with pytest.raises(NanoReconError, match="UTF-8"):
        ContainerReader(io.BytesIO(replace_record(data, 0, b"\xff\xfe")))
    extra = hdr[:-1] + b',"extra":1}'
    with pytest.raises(NanoReconError, match="unexpected"):
        ContainerReader(io.BytesIO(replace_record(data, 0, extra)))


def test_structure_errors(sample, profile):
    data = sample[1]
    recs = records(data)
    read_idx = next(i for i, r in enumerate(recs) if r[1] == C.REC_READ and r[2] > C.READ_META_DTYPE.itemsize + 1000)
    pos, _, n = recs[read_idx]
    body = bytearray(data[pos + 12 : pos + 12 + n])

    def with_body(b):
        return data[:pos] + struct.pack("<IQ", C.REC_READ, len(b)) + bytes(b) + data[pos + 12 + n :]

    def meta_field(name, value):
        b = bytearray(body)
        rec = np.frombuffer(bytes(b[: C.READ_META_DTYPE.itemsize]), dtype=C.READ_META_DTYPE).copy()
        rec[name] = value
        b[: C.READ_META_DTYPE.itemsize] = rec.tobytes()
        return with_body(b)

    cases = [
        (meta_field("num_chunks", 99), "implies"),
        (meta_field("num_samples", 100000), "implies"),  # K changes
        (meta_field("run_info_index", 9), "does not exist"),
        (meta_field("pore_type_index", 9), "does not exist"),
        (meta_field("reserved", 1), "reserved"),
        (meta_field("end_reason_forced", 2), "reserved"),
        (meta_field("calibration_scale", 0.0), "calibration_scale"),
        (with_body(body[:-2]), "record length"),
        (replace_record(data, 0, b"{}", rtype=99), "unknown record type"),
    ]
    scale_at = C.READ_META_DTYPE.itemsize + 4 * ((n - C.READ_META_DTYPE.itemsize) // (8 + 2 * profile.tokens_per_chunk))
    neg = bytearray(body)
    neg[scale_at : scale_at + 4] = struct.pack("<f", -1.0)
    cases.append((with_body(neg), "negative"))
    nanc = bytearray(body)
    nanc[C.READ_META_DTYPE.itemsize : C.READ_META_DTYPE.itemsize + 4] = struct.pack("<f", float("nan"))
    cases.append((with_body(nanc), "non-finite"))
    for corrupted, message in cases:
        with pytest.raises(NanoReconError, match=message):
            read_all(corrupted)


def test_group_and_end_consistency(sample):
    data = sample[1]
    recs = records(data)
    end_i = len(recs) - 1
    with pytest.raises(NanoReconError, match="END record says"):
        read_all(replace_record(data, end_i, b'{"chunk_count":1,"group_count":3,"read_count":7}'))
    # drop the last READ record: the group ends early
    pos, _, n = recs[end_i - 1]
    with pytest.raises(NanoReconError, match="ended early"):
        read_all(data[:pos] + data[pos + 12 + n :])
    # a READ where a GROUP is expected
    g_pos = recs[1][0]
    with pytest.raises(NanoReconError, match="GROUP or END"):
        read_all(data[:g_pos] + data[recs[2][0] :])


def test_oversized_json_record_rejected(sample):
    data = sample[1]
    pos = records(data)[1][0]
    bad = data[:pos] + struct.pack("<IQ", C.REC_GROUP, C.MAX_GROUP_BYTES + 1) + data[pos + 12 :]
    with pytest.raises(NanoReconError, match="limit"):
        read_all(bad)


def test_code_batches_bounded_and_skip(sample, profile):
    data = sample[1]
    r = ContainerReader(io.BytesIO(data))
    seen = []
    for _, stored in r.iter_reads():
        buf = np.empty((2, profile.tokens_per_chunk), np.uint16)
        for first, codes in stored.code_batches(2, out=buf):
            assert codes.base is buf or codes is buf or np.shares_memory(codes, buf)
            assert codes.shape[0] <= 2
            seen.append(first)
            break  # leave the rest unread: the reader must skip it
    assert r.reads_read == 7


def test_writer_guards(profile):
    buf = io.BytesIO()
    w = ContainerWriter(buf, header(profile, 1))
    m = meta(0, 9000)
    c, s, k = arrays(profile, 9000, 0)
    with pytest.raises(RuntimeError):
        w.write_read(m, c, s, k)  # outside a group
    w.begin_group([m])
    with pytest.raises(ValueError):
        w.write_read(m, c[:1], s[:1], k[:1])  # wrong K
    w.write_read(m, c, s, k)
    w.finish()
    w2 = ContainerWriter(io.BytesIO(), header(profile, 2))
    w2.begin_group([m])
    w2.write_read(m, c, s, k)
    with pytest.raises(RuntimeError, match="announced"):
        w2.finish()
    with pytest.raises(ValueError):
        ContainerWriter(io.BytesIO(), header(profile, 0)).begin_group([])


def test_read_meta_layout_matches_spec():
    """docs/format.md section 4 lists these offsets; the struct must not drift from the spec."""
    spec = {
        "read_id": 0, "num_samples": 16, "num_chunks": 24, "read_number": 32, "start_sample": 36, "channel": 44,
        "well": 46, "end_reason_forced": 47, "calibration_offset": 48, "calibration_scale": 52, "median_before": 56,
        "num_minknow_events": 60, "tracked_scaling_scale": 68, "tracked_scaling_shift": 72,
        "predicted_scaling_scale": 76, "predicted_scaling_shift": 80, "num_reads_since_mux_change": 84,
        "time_since_mux_change": 88, "open_pore_level": 92, "run_info_index": 96, "pore_type_index": 98,
        "end_reason_index": 100, "reserved": 102,
    }
    assert {name: C.READ_META_DTYPE.fields[name][1] for name in C.READ_META_DTYPE.names} == spec
    assert C.READ_META_DTYPE.itemsize == 104
    assert all(C.READ_META_DTYPE.fields[n][0].byteorder in "<|=" for n in C.READ_META_DTYPE.names)


def test_header_profile_and_strings(sample, profile):
    data = sample[1]
    pos, _, n = records(data)[0]
    hdr = data[pos + 12 : pos + 12 + n]
    missing = hdr.replace(b'"normalization_epsilon":1e-06,', b"")
    with pytest.raises(NanoReconError, match="profile is invalid"):
        ContainerReader(io.BytesIO(replace_record(data, 0, missing)))
    surrogate = hdr.replace(b'"producer":{"name":"nanorecon"', b'"producer":{"name":"\\ud800"')
    with pytest.raises(NanoReconError, match="not valid Unicode"):
        ContainerReader(io.BytesIO(replace_record(data, 0, surrogate)))
    assert ContainerReader(io.BytesIO(data)).header.producer_version == "test"


def test_group_larger_than_header_count_rejected_before_reading(sample):
    data = sample[1]
    pos, _, n = records(data)[1]
    group = data[pos + 12 : pos + 12 + n].replace(b'"read_count":4', b'"read_count":2147483648')
    r = ContainerReader(io.BytesIO(replace_record(data, 1, group)))
    with pytest.raises(NanoReconError, match="header announces"):
        next(r.iter_reads())
