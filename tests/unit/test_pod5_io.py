"""POD5 Meta mapping in both directions, lazy signal access, zero-length reads."""

import itertools
import math
import uuid

import numpy as np
import pod5
import pytest

from nanorecon.io.pod5_io import Pod5Sink, Pod5Source
from nanorecon.types import NanoReconError, read_meta_equal

from ..helpers import make_run_info, write_pod5


def test_meta_matches_pod5_api(tmp_path):
    path = tmp_path / "in.pod5"
    written = write_pod5(path, [0, 1, 9000, 16300, 5, 20000])
    with Pod5Source(path) as src:
        assert src.read_count() == 6
        reads = [(r, r.load_signal()) for r in src.iter_reads()]
    assert [r.position for r, _ in reads] == list(range(6))
    for (src_read, signal), ref in zip(reads, written):
        m = src_read.meta
        assert m.read_id == ref.read_id.bytes
        assert m.num_samples == len(ref.signal)
        assert (m.read_number, m.start_sample, m.channel, m.well) == (ref.read_number, ref.start_sample, ref.pore.channel, ref.pore.well)
        assert m.pore_type == ref.pore.pore_type
        assert m.end_reason == ref.end_reason.reason.name.lower() and m.end_reason_forced == ref.end_reason.forced
        assert m.calibration_offset == float(np.float32(ref.calibration.offset))
        assert m.calibration_scale == float(np.float32(ref.calibration.scale))
        for ours, theirs in ((m.median_before, ref.median_before), (m.open_pore_level, ref.open_pore_level),
                             (m.tracked_scaling_shift, ref.tracked_scaling.shift), (m.tracked_scaling_scale, ref.tracked_scaling.scale)):
            assert (math.isnan(ours) and math.isnan(theirs)) or ours == float(np.float32(theirs))
        assert m.run_info.acquisition_id == ref.run_info.acquisition_id
        assert m.run_info.acquisition_start_time_ms == ref.run_info.acquisition_start_time
        assert m.run_info.context_tags == tuple(ref.run_info.context_tags)  # order + duplicates kept
        assert np.array_equal(signal, ref.signal)


def roundtrip(tmp_path, lengths, **kw):
    src_path, out_path = tmp_path / "a.pod5", tmp_path / "b.pod5"
    written = write_pod5(src_path, lengths, **kw)
    with Pod5Source(src_path) as src:
        items = [(r.meta, r.load_signal()) for r in src.iter_reads()]
    sink = Pod5Sink(out_path)
    sink.begin_group()
    for m, sig in items:
        sink.write(m, sig)
    sink.close()
    with Pod5Source(out_path) as again:
        back = [(r.meta, r.load_signal()) for r in again.iter_reads()]
    return written, items, back, out_path


def test_meta_roundtrip_is_exact(tmp_path):
    _, items, back, out_path = roundtrip(tmp_path, [0, 3, 8192, 9000, 12345, 7, 1])
    assert len(back) == len(items)
    for (m1, s1), (m2, s2) in zip(items, back):
        assert read_meta_equal(m1, m2), (m1, m2)
        assert m1.run_info == m2.run_info
        assert np.array_equal(s1, s2)
    with pod5.Reader(out_path) as r:  # the official reader agrees too
        ids = [str(x.read_id) for x in r.reads()]
    assert ids == [m.read_id_str for m, _ in items]


def test_timestamps_keep_millisecond_precision(tmp_path):
    # 1735787045.123 * 1000 == 1735787045122.9998 in float: a datetime detour would lose 1 ms.
    info = make_run_info("acq-ms", acquisition_start_time=1735787045123, protocol_start_time=1)
    _, items, back, _ = roundtrip(tmp_path, [10], run_infos=[info])
    assert back[0][0].run_info.acquisition_start_time_ms == 1735787045123
    assert back[0][0].run_info.protocol_start_time_ms == 1


def test_signal_loaded_only_on_request_and_cache_bounded(tmp_path):
    path = tmp_path / "in.pod5"
    write_pod5(path, [5000] * 30)
    with Pod5Source(path) as src:
        reads = list(itertools.islice(src.iter_reads(), 4))
        assert len(reads) == 4
        assert src._reader._cached_signal_batches == {}  # nothing decoded while listing
        reads[0].load_signal()
        assert src._reader._cached_signal_batches  # decoded on demand


def test_duplicate_ids_are_kept_in_order(tmp_path):
    path = tmp_path / "dup.pod5"
    info = make_run_info()
    rid = uuid.uuid4()
    reads = [
        pod5.Read(read_id=rid, pore=pod5.Pore(1, 1, "p"), calibration=pod5.Calibration(0.0, 1.0), read_number=i,
                  start_sample=0, median_before=0.0, end_reason=pod5.EndReason.from_reason_with_default_forced(pod5.EndReasonEnum.UNKNOWN),
                  run_info=info, signal=np.full(10 + i, i, np.int16))
        for i in range(3)
    ]
    with pod5.Writer(path) as w:
        w.add_reads(reads)
    with Pod5Source(path) as src:
        got = [(r.meta.read_number, r.load_signal()[0]) for r in src.iter_reads()]
    assert got == [(0, 0), (1, 1), (2, 2)]


@pytest.mark.parametrize("scale", [0.0, float("nan")])
def test_invalid_calibration_fails_before_signal(tmp_path, scale):
    path = tmp_path / "bad.pod5"
    info = make_run_info()
    read = pod5.Read(read_id=uuid.uuid4(), pore=pod5.Pore(1, 1, "p"), calibration=pod5.Calibration(0.0, scale),
                     read_number=0, start_sample=0, median_before=0.0,
                     end_reason=pod5.EndReason.from_reason_with_default_forced(pod5.EndReasonEnum.UNKNOWN),
                     run_info=info, signal=np.zeros(10, np.int16))
    with pod5.Writer(path) as w:
        w.add_read(read)
    with Pod5Source(path) as src, pytest.raises(NanoReconError, match="calibration"):
        list(src.iter_reads())


def test_empty_pod5(tmp_path):
    path = tmp_path / "empty.pod5"
    with pod5.Writer(path):
        pass
    with Pod5Source(path) as src:
        assert src.read_count() == 0 and list(src.iter_reads()) == []


def test_not_a_pod5(tmp_path):
    path = tmp_path / "x.pod5"
    path.write_bytes(b"not a pod5 file at all")
    with pytest.raises(NanoReconError):
        Pod5Source(path)


def test_damaged_tables_raise_input_error(tmp_path, monkeypatch):
    import pyarrow as pa

    from nanorecon.io import pod5_io

    path = tmp_path / "in.pod5"
    write_pod5(path, [10, 20])

    def broken(self):
        raise pa.ArrowInvalid("corrupt run_info")

    monkeypatch.setattr(pod5_io.Pod5Source, "_index_run_infos", broken)
    with pytest.raises(NanoReconError, match="run_info table"):
        Pod5Source(path)


def test_overlong_strings_rejected_at_source(tmp_path, monkeypatch):
    from nanorecon.io import pod5_io

    path = tmp_path / "in.pod5"
    write_pod5(path, [10])
    monkeypatch.setattr(pod5_io, "MAX_STRING_BYTES", 4)
    with Pod5Source(path) as src, pytest.raises(NanoReconError, match="longer than 4 bytes"):
        list(src.iter_reads())
