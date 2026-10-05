"""compress/decompress pipelines with stand-in engines.

The batch size must never change what a read's chunks encode to or decode from (stand-in
engines are row-wise); decompression packs chunks of consecutive reads into full batches and
still writes every read in order; failures anywhere leave no output.
"""

from __future__ import annotations

import errno
from pathlib import Path

import numpy as np
import pytest

from nanorecon.io import container as C
from nanorecon.io.pod5_io import Pod5Source
from nanorecon.model_config import MODEL
from nanorecon.pipeline import compress as P
from nanorecon.pipeline.compress import CompressOptions, compress_file
from nanorecon.pipeline.decompress import DecompressOptions, decompress_file
from nanorecon.types import Cancelled, NanoReconError, read_meta_equal

from ..helpers import ContentEngine, LookupEngine, write_pod5

LENGTHS = [0, 1, 2, 5, 6143, 8191, 8192, 8193, 9000, 16240, 16300, 16384, 30001, 3, 45000, 7, 8200]


@pytest.fixture
def pod5_file(tmp_path):
    path = tmp_path / "in.pod5"
    write_pod5(path, LENGTHS, seed=7)
    return path


def compress(tmp_path, pod5_path, profile, engine, name="out.nrpod", **opts):
    out = tmp_path / name
    options = CompressOptions(**{"batch_size": engine.batch_size, **opts})
    report = compress_file(pod5_path, out, model=MODEL, profile=profile, make_engine=lambda: engine,
                           options=options, show_progress=False)
    return out, report


def decompress(token_file, out, engine):
    return decompress_file(token_file, out, make_engine=lambda h: engine,
                           options=DecompressOptions(batch_size=engine.batch_size), show_progress=False)


def leftovers(directory: Path):
    return sorted(p.name for p in directory.iterdir() if p.name.endswith(".nanorecon-tmp"))


def reads_of(path):
    with Pod5Source(path) as source:
        return [(r.meta, r.load_signal()) for r in source.iter_reads()]


def test_batch_size_does_not_change_the_token_file(tmp_path, pod5_file, profile):
    ref, report = compress(tmp_path, pod5_file, profile, ContentEngine(4, profile), "ref.nrpod")
    assert report.reads == len(LENGTHS) and report.empty_reads == 1
    for batch in (1, 3, 7, 16, 64):
        out, _ = compress(tmp_path, pod5_file, profile, ContentEngine(batch, profile), f"b{batch}.nrpod")
        # batch composition across reads and padding never leak into a row's result
        assert out.read_bytes() == ref.read_bytes(), batch


@pytest.mark.parametrize("compress_batch, decompress_batch", [(3, 3), (5, 1), (4, 7), (16, 64)])
def test_lossless_roundtrip(tmp_path, pod5_file, profile, compress_batch, decompress_batch):
    engine = LookupEngine(compress_batch, profile)
    token_file, _ = compress(tmp_path, pod5_file, profile, engine)
    engine.batch_size = decompress_batch  # the stand-in decodes any stored chunk at any batch size
    report = decompress(token_file, tmp_path / "back.pod5", engine)
    assert report.reads == len(LENGTHS) and report.empty_reads == 1
    left, right = reads_of(pod5_file), reads_of(tmp_path / "back.pod5")
    assert len(left) == len(right)
    for (m1, s1), (m2, s2) in zip(left, right):
        assert read_meta_equal(m1, m2)
        assert s1.shape == s2.shape
        assert np.array_equal(s1, s2), m1.num_samples  # ADC exact: mapping, stitching and tails are right
    assert leftovers(tmp_path) == []


def test_decompress_packs_reads_into_full_batches(tmp_path, profile):
    """Short reads share decoder batches; empty and long reads in between keep their place."""
    lengths = [100, 0, 3000, 8192, 20000, 0, 50, 1] * 12 + [0]
    path = tmp_path / "short.pod5"
    write_pod5(path, lengths, seed=3)
    engine = LookupEngine(16, profile)
    token_file, report = compress(tmp_path, path, profile, engine)
    before = engine.decode_calls
    back = tmp_path / "back.pod5"
    out = decompress(token_file, back, engine)
    assert out.chunks == report.chunks
    assert engine.decode_calls - before == -(-report.chunks // 16)  # full batches; only the last one padded
    left, right = reads_of(path), reads_of(back)
    assert [m.read_id for m, _ in left] == [m.read_id for m, _ in right]
    for (m1, s1), (m2, s2) in zip(left, right):
        assert read_meta_equal(m1, m2) and np.array_equal(s1, s2)


def test_next_batch_is_sent_before_the_previous_is_collected(tmp_path, pod5_file, profile):
    """The host prepares, reads and writes while the GPU computes: batch n + 1 is sent before
    batch n is collected, results are collected in order, and at most two batches are out."""
    engine = LookupEngine(4, profile)
    token_file, _ = compress(tmp_path, pod5_file, profile, engine)
    decompress(token_file, tmp_path / "back.pod5", engine)
    for direction in ("encode", "decode"):
        order = [(kind, call) for kind, d, call in engine.events if d == direction]
        sends = [call for kind, call in order if kind == "send"]
        assert len(sends) >= 3 and [call for kind, call in order if kind == "collect"] == sends
        for call in sends[:-1]:
            assert order.index(("send", call + 1)) < order.index(("collect", call))
        out = 0
        for kind, _ in order:
            out += 1 if kind == "send" else -1
            assert out <= 2


def group_layout(path):
    """[[num_chunks of each read] per group]"""
    with open(path, "rb") as fh:
        groups: dict[int, list[int]] = {}
        for gi, stored in C.ContainerReader(fh).iter_reads():
            groups.setdefault(gi, []).append(stored.num_chunks)
    return list(groups.values())


def test_group_limits(tmp_path, pod5_file, profile):
    engine = ContentEngine(4, profile)
    out, report = compress(tmp_path, pod5_file, profile, engine, "small.nrpod", group_chunks=3)
    layout = group_layout(out)
    assert sum(len(g) for g in layout) == len(LENGTHS) and report.groups == len(layout) > 1
    # a group stays within the chunk limit unless one read alone exceeds it
    assert all(sum(g) <= 3 or len(g) == 1 for g in layout), layout
    assert any(len(g) == 1 and g[0] > 3 for g in layout)
    capped, _ = compress(tmp_path, pod5_file, profile, engine, "capped.nrpod", group_reads=2)
    assert max(len(g) for g in group_layout(capped)) == 2
    ref, _ = compress(tmp_path, pod5_file, profile, engine, "one.nrpod")
    assert len(group_layout(ref)) == 1
    with open(out, "rb") as a, open(ref, "rb") as b:  # same reads and codes regardless of grouping
        for (_, x), (_, y) in zip(C.ContainerReader(a).iter_reads(), C.ContainerReader(b).iter_reads()):
            assert read_meta_equal(x.meta, y.meta)
            assert np.array_equal(x.read_codes(), y.read_codes())


def test_long_read_is_not_rejected(tmp_path, profile):
    path = tmp_path / "long.pod5"
    write_pod5(path, [100, 400000, 100])
    engine = LookupEngine(8, profile)
    token_file, report = compress(tmp_path, path, profile, engine, group_chunks=4)
    assert report.reads == 3 and report.max_group_chunks_seen == 50
    decompress(token_file, tmp_path / "back.pod5", engine)
    for (_, s1), (_, s2) in zip(reads_of(path), reads_of(tmp_path / "back.pod5")):
        assert np.array_equal(s1, s2)


def test_prepare_failure_propagates(tmp_path, pod5_file, profile, monkeypatch):
    real = P.prepare_chunk
    calls = {"n": 0}

    def flaky(*args):
        calls["n"] += 1
        if calls["n"] == 6:
            raise ValueError("injected prepare failure")
        return real(*args)

    monkeypatch.setattr(P, "prepare_chunk", flaky)
    with pytest.raises(ValueError, match="injected prepare failure"):
        compress(tmp_path, pod5_file, profile, ContentEngine(2, profile))
    assert not (tmp_path / "out.nrpod").exists() and leftovers(tmp_path) == []


def test_encoder_failure_propagates(tmp_path, pod5_file, profile):
    engine = ContentEngine(2, profile, fail_on_call=3, fail_with=RuntimeError("GPU exploded"))
    with pytest.raises(RuntimeError, match="GPU exploded"):
        compress(tmp_path, pod5_file, profile, engine)
    assert leftovers(tmp_path) == []


def test_cancellation_leaves_old_output(tmp_path, pod5_file, profile):
    target = tmp_path / "out.nrpod"
    target.write_bytes(b"previous result")
    engine = ContentEngine(2, profile, fail_on_call=2, fail_with=KeyboardInterrupt())
    with pytest.raises(Cancelled):
        compress_file(pod5_file, target, model=MODEL, profile=profile, make_engine=lambda: engine,
                      options=CompressOptions(batch_size=2), force=True, show_progress=False)
    assert target.read_bytes() == b"previous result" and leftovers(tmp_path) == []


def test_writer_failure_is_reported(tmp_path, pod5_file, profile, monkeypatch):
    def disk_full(self, *a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(C.ContainerWriter, "write_read", disk_full)
    with pytest.raises(NanoReconError, match="No space left"):
        compress(tmp_path, pod5_file, profile, ContentEngine(2, profile))
    assert leftovers(tmp_path) == [] and not (tmp_path / "out.nrpod").exists()


def test_empty_input(tmp_path, profile):
    engine = LookupEngine(4, profile)
    empty_pod5 = tmp_path / "empty.pod5"
    write_pod5(empty_pod5, [])
    out, report = compress(tmp_path, empty_pod5, profile, engine, "empty.nrpod")
    assert report.reads == 0
    back = tmp_path / "empty_back.pod5"
    decompress(out, back, engine)
    with Pod5Source(back) as src:
        assert src.read_count() == 0


def test_sample_rate_mismatch_rejected(tmp_path, profile):
    path = tmp_path / "4k.pod5"
    write_pod5(path, [100, 200], sample_rate=4000)
    with pytest.raises(NanoReconError, match="does not resample"):
        compress(tmp_path, path, profile, ContentEngine(2, profile))


def test_decoder_failure_leaves_no_pod5(tmp_path, pod5_file, profile):
    engine = LookupEngine(4, profile)
    token_file, _ = compress(tmp_path, pod5_file, profile, engine)

    class Broken:
        batch_size = 4

        def __init__(self):
            self.n = 0

        def decode_async(self, codes, rows):
            self.n += 1
            if self.n == 4:
                raise RuntimeError("decode failure")
            return engine.decode_async(codes, rows)

    with pytest.raises(RuntimeError, match="decode failure"):
        decompress(token_file, tmp_path / "x.pod5", Broken())
    assert not (tmp_path / "x.pod5").exists() and leftovers(tmp_path) == []


def test_truncated_token_file_never_produces_pod5(tmp_path, pod5_file, profile):
    engine = LookupEngine(4, profile)
    token_file, _ = compress(tmp_path, pod5_file, profile, engine)
    data = token_file.read_bytes()
    cut = tmp_path / "cut.nrpod"
    cut.write_bytes(data[: len(data) - 30])
    with pytest.raises(NanoReconError, match="truncated|END"):
        decompress(cut, tmp_path / "x.pod5", engine)
    assert not (tmp_path / "x.pod5").exists() and leftovers(tmp_path) == []


def test_unknown_end_reason_fails_before_decoding_everything(tmp_path, pod5_file, profile):
    engine = LookupEngine(4, profile)
    token_file, report = compress(tmp_path, pod5_file, profile, engine)
    data = token_file.read_bytes()
    # rename one end reason inside the first GROUP table (same length keeps the record length valid)
    patched = data.replace(b'"mux_change"', b'"mux_chang3"', 1)
    assert patched != data
    bad = tmp_path / "bad.nrpod"
    bad.write_bytes(patched)
    before = engine.decode_calls
    with pytest.raises(NanoReconError, match="cannot write"):
        decompress(bad, tmp_path / "x.pod5", engine)
    assert not (tmp_path / "x.pod5").exists() and leftovers(tmp_path) == []
    assert engine.decode_calls - before < -(-report.chunks // 4)  # stopped at the offending read


def test_group_tables_stay_within_format_limit(tmp_path, profile, monkeypatch):
    from ..helpers import make_run_info

    infos = [make_run_info(f"acq-{i}", sample_id="x" * 500) for i in range(6)]
    path = tmp_path / "runs.pod5"
    write_pod5(path, [100] * 12, run_infos=infos)
    monkeypatch.setattr(P, "GROUP_TABLE_LIMIT", 3000)  # ~2 run infos per group
    out, report = compress(tmp_path, path, profile, ContentEngine(4, profile))
    assert report.groups >= 3
    with open(out, "rb") as fh:
        assert sum(1 for _ in C.ContainerReader(fh).iter_reads()) == 12
    monkeypatch.setattr(P, "GROUP_TABLE_LIMIT", 100)
    with pytest.raises(NanoReconError, match="too large for the token format"):
        compress(tmp_path, path, profile, ContentEngine(4, profile), "tiny.nrpod")
