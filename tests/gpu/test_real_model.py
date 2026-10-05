"""Released weights on a CUDA GPU. Run on the GPU server, never part of plain `pytest`.

Needs `nanorecon pull` beforehand (the model is taken from the Hugging Face cache) and network
access for the Hub query. Optional:
    NANORECON_TRAINING_REPO   training repository checkout; enables the reference comparison
    NANORECON_TEST_POD5       a small real POD5 file; enables the real-data round trip
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

from nanorecon.model_config import MODEL, PROFILE, REVISION

from ..helpers import write_pod5

TRAINING_REPO = os.environ.get("NANORECON_TRAINING_REPO")
REAL_POD5 = os.environ.get("NANORECON_TEST_POD5")


@pytest.fixture(scope="module")
def local_model():
    from nanorecon.models import load_local_model

    return load_local_model()


@pytest.fixture(scope="module")
def engine(local_model):
    from nanorecon.runtime.engine import JaxCodecEngine

    return JaxCodecEngine(local_model.network, PROFILE, local_model.weights_path, batch_size=4)


def normalized_chunks(count, seed=0):
    from nanorecon.signal.chunking import ChunkPlan
    from nanorecon.signal.normalize import prepare_chunk

    from ..helpers import squiggle

    rng = np.random.default_rng(seed)
    L = PROFILE.chunk_samples
    out = np.empty((count, L), np.float32)
    for i in range(count):
        n = L if i % 3 else int(rng.integers(100, L))
        adc = squiggle(rng, n)
        prepare_chunk(adc, ChunkPlan(n, L, PROFILE.hop_samples), 0, np.float32(-240.0), np.float32(0.1755),
                      np.float32(1e-6), out[i], np.empty(L, np.float32))
    return out


def test_engine_shapes_and_padding_independence(engine):
    x = normalized_chunks(4)
    codes = engine.encode(x, 4)
    assert codes.shape == (4, PROFILE.tokens_per_chunk) and codes.dtype == np.uint16
    y = x.copy()
    y[2:] = np.random.default_rng(5).uniform(-1, 1, size=(2, PROFILE.chunk_samples)).astype(np.float32)
    assert np.array_equal(engine.encode(y, 2), codes[:2])  # padding rows never change real rows
    wave = engine.decode(codes.copy(), 4)
    assert wave.shape == (4, PROFILE.chunk_samples) and np.isfinite(wave).all()
    other = codes.copy()
    other[3] = 12345
    assert np.allclose(engine.decode(other, 3), wave[:3], atol=1e-6)
    assert float(np.abs(wave - x).mean()) < 0.2  # the model actually reconstructs


@pytest.mark.skipif(not TRAINING_REPO, reason="set NANORECON_TRAINING_REPO to compare with the training model")
def test_matches_training_model(engine, local_model):
    """Encoder and decoder are compared separately: the decoder always gets the same codes in
    both implementations, so a few differing encoder choices never skip the decoder check."""
    import jax
    from flax.serialization import msgpack_restore

    sys.path.insert(0, TRAINING_REPO)
    from codec.models import build_audio_model

    cfg = json.loads(local_model.config_path.read_text())["model"]
    if engine.attention != "cudnn":
        cfg["decoder_pos_net_attention_backend"] = "xla"
    ref_model = build_audio_model(cfg)
    variables = msgpack_restore(local_model.weights_path.read_bytes())
    x = normalized_chunks(4, seed=9)
    out = jax.jit(lambda v, a: ref_model.apply(v, a, train=False, offset=0, rng=jax.random.PRNGKey(0), collect_codebook_stats=False))(variables, x)
    ref_codes = np.asarray(out["enc"]["indices"]).astype(np.uint16)
    ref_wave = np.asarray(out["wave_hat"])

    agreement = float((engine.encode(x, 4) == ref_codes).mean())
    print(f"encoder code agreement with the training model: {agreement:.6f}")
    assert agreement >= 0.999  # 1.0 measured on the A100 server (2026-10-04); near-ties may flip on other GPUs

    max_diff = float(np.abs(engine.decode(ref_codes.copy(), 4) - ref_wave).max())
    print(f"decoder max abs difference for identical codes: {max_diff:.3e}")
    # fp16 operands in the ConvNeXt pointwise layers sum in another order than the training model's
    # TF32 kernels: ~2e-4 on an RTX 3080 Ti (2026-10-05); batch 64 vs 256 alone differ by ~8e-4.
    assert max_diff < 1e-3


def test_cli_roundtrip_preserves_meta_and_lengths(tmp_path, local_model):
    from nanorecon.cli import main
    from nanorecon.io.pod5_io import Pod5Source
    from nanorecon.types import read_meta_equal

    lengths = [0, 1, 5, 5000, 8192, 16300, 30001]
    src, token, back = tmp_path / "in.pod5", tmp_path / "t.nrpod", tmp_path / "back.pod5"
    write_pod5(src, lengths, seed=3)
    assert main(["compress", str(src), "-o", str(token), "--batch-size", "4"]) == 0
    assert main(["decompress", str(token), "-o", str(back), "--batch-size", "4"]) == 0
    with Pod5Source(src) as a, Pod5Source(back) as b:
        left = [(r.meta, r.load_signal()) for r in a.iter_reads()]
        right = [(r.meta, r.load_signal()) for r in b.iter_reads()]
    assert [m.read_id for m, _ in left] == [m.read_id for m, _ in right]
    for (m1, s1), (m2, s2) in zip(left, right):
        assert read_meta_equal(m1, m2) and s1.shape == s2.shape and s2.dtype == np.int16
        if s1.size > 16:
            pa1 = (s1 + np.float32(m1.calibration_offset)) * np.float32(m1.calibration_scale)
            pa2 = (s2 + np.float32(m2.calibration_offset)) * np.float32(m2.calibration_scale)
            spread = float(pa1.max() - pa1.min()) or 1.0
            assert float(np.abs(pa1 - pa2).mean()) / spread < 0.1, m1.num_samples
    assert token.stat().st_size < src.stat().st_size
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["back.pod5", "in.pod5", "t.nrpod"]  # no side files


@pytest.mark.skipif(not REAL_POD5, reason="set NANORECON_TEST_POD5 to a small real POD5 file")
def test_real_pod5_roundtrip(tmp_path, local_model):
    from nanorecon.cli import main
    from nanorecon.io.pod5_io import Pod5Source
    from nanorecon.types import read_meta_equal

    token, back = tmp_path / "t.nrpod", tmp_path / "back.pod5"
    assert main(["compress", REAL_POD5, "-o", str(token)]) == 0
    assert main(["decompress", str(token), "-o", str(back)]) == 0
    with Pod5Source(REAL_POD5) as a, Pod5Source(back) as b:
        pairs = list(zip(a.iter_reads(), b.iter_reads(), strict=True))
        assert pairs
        for ra, rb in pairs:
            assert read_meta_equal(ra.meta, rb.meta)
            assert rb.load_signal().shape == (ra.meta.num_samples,)


def test_file_decode_equals_in_memory_reference(tmp_path, engine):
    """Hard-quantization reconstruction computed in memory == compress -> file -> decompress."""
    from nanorecon.io.pod5_io import Pod5Source
    from nanorecon.pipeline.compress import CompressOptions, compress_file
    from nanorecon.pipeline.decompress import DecompressOptions, decompress_file
    from nanorecon.signal.chunking import ChunkPlan
    from nanorecon.signal.normalize import checked_calibration, prepare_chunk
    from nanorecon.signal.stitch import StitchAccumulator, edge_weights, pa_to_adc

    p = PROFILE
    src = tmp_path / "in.pod5"
    write_pod5(src, [16300, 700], seed=21)
    token, back = tmp_path / "t.nrpod", tmp_path / "b.pod5"
    compress_file(src, token, model=MODEL, profile=p, make_engine=lambda: engine,
                  options=CompressOptions(batch_size=4), show_progress=False)
    decompress_file(token, back, make_engine=lambda h: engine, options=DecompressOptions(batch_size=4), show_progress=False)
    with Pod5Source(src) as a, Pod5Source(back) as b:
        for ra, rb in zip(a.iter_reads(), b.iter_reads()):
            adc = ra.load_signal()
            plan = ChunkPlan(adc.shape[0], p.chunk_samples, p.hop_samples)
            off, sc = checked_calibration(ra.meta.calibration_offset, ra.meta.calibration_scale)
            batch = np.zeros((4, p.chunk_samples), np.float32)
            stats = []
            for j in range(plan.num_chunks):
                stats.append(prepare_chunk(adc, plan, j, off, sc, np.float32(p.normalization_epsilon), batch[j], np.empty(p.chunk_samples, np.float32)))
            codes = engine.encode(batch, plan.num_chunks)
            padded = np.zeros((4, p.tokens_per_chunk), np.uint16)
            padded[: plan.num_chunks] = codes
            wave = engine.decode(padded, plan.num_chunks)
            acc = StitchAccumulator(plan, edge_weights(p.chunk_samples, p.overlap_samples), np.float32(p.normalization_epsilon))
            for j, (center, half) in enumerate(stats):
                acc.add(j, wave[j], center, half)
            expected = pa_to_adc(acc.finish(), off, sc)
            assert np.array_equal(rb.load_signal(), expected)


def test_hub_status(capsys):
    from nanorecon.cli import main

    assert main(["ls-remote"]) == 0
    out = capsys.readouterr().out
    assert REVISION in out and "available" in out
