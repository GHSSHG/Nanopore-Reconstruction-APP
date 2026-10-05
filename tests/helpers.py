"""Test helpers: the release config, synthetic POD5 files, stand-in engines."""

from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pod5

from nanorecon.model_config import CodecProfile

RELEASE_CONFIG = {
    "architectures": ["SimVQAudioModel"],
    "model_type": "nanorecon-simvq-audio-codec",
    "framework": "flax",
    "library_name": "jax",
    "input_signal": {"sample_rate_hz": 5000.0, "segment_samples": 8192, "segment_seconds": 1.6384,
                     "expected_shape": ["batch", "samples"], "dtype": "float32"},
    "model": {
        "variant": "foundation_v1",
        "enc_channels": [32, 64, 128, 256, 512],
        "enc_down_strides": [2, 2, 2, 2],
        "enc_stage_num_res_blocks": [2, 2, 3, 3],
        "enc_kernel_size": 7,
        "latent_dim": 512,
        "quantizer_dim": 512,
        "codebook_size": 65536,
        "dec_channels": [512],
        "decoder_dim": 768,
        "decoder_intermediate_dim": 2304,
        "decoder_num_layers": 12,
        "decoder_pos_net_enabled": True,
        "decoder_pos_net_dropout": 0.0,
        "decoder_pos_net_attention_heads": 12,
        "decoder_pos_net_attention_backend": "jax_cudnn",
        "istft_n_fft": 512,
        "istft_hop_length": 16,
        "istft_sample_rate": 5000,
        "istft_band_edges_hz": [200, 500, 1000],
        "istft_band_gain_init": [1.0, 0.8, 0.45, 0.12],
        "istft_dynamic_gate_scale": 0.5,
        "residual_correction": {"enabled": True, "alpha_init": 0.0, "alpha_max": 0.1, "hidden_dim": 192},
        "latent_bilstm_layers": 2,
        "latent_bilstm_hidden_dim": 256,
        "cnn_compute_dtype": "fp32",
        "param_dtype": "fp32",
        "diveq_sigma2": 0.001,
        "search_chunk_size": 8192,
        "quant_conv_kernel_size": 7,
        "post_quant_conv_kernel_size": 7,
    },
    "weights": {"format": "flax_msgpack", "file": "flax_model.msgpack", "variables": ["params", "vq"]},
}


# ------------------------------------------------------------------ synthetic POD5


def make_run_info(acquisition_id: str = "acq-1", **overrides) -> pod5.RunInfo:
    values = dict(
        acquisition_id=acquisition_id,
        acquisition_start_time=1735787045123,  # integer ms; 1735787045.123 * 1000 is inexact in float
        adc_max=4095,
        adc_min=-4096,
        context_tags=[("sample_frequency", "5000"), ("exp", "é✓"), ("dup", "1"), ("dup", "2")],
        experiment_name="",
        flow_cell_id="FAX00001",
        flow_cell_product_code="FLO-PRO114M",
        protocol_name="sequencing/sequencing_PRO114_DNA_e8_2_400K",
        protocol_run_id="run-1",
        protocol_start_time=1735787000001,
        sample_id="sample ☃",
        sample_rate=5000,
        sequencing_kit="SQK-LSK114",
        sequencer_position="1A",
        sequencer_position_type="PromethION",
        software="MinKNOW 24.x",
        system_name="host",
        system_type="linux",
        tracking_id=[("run_id", "abc"), ("empty", "")],
    )
    values.update(overrides)
    return pod5.RunInfo(**values)


def squiggle(rng: np.random.Generator, n: int, *, offset: float = -240.0, scale: float = 0.1755) -> np.ndarray:
    """Nanopore-like ADC samples: piecewise levels with noise."""
    if n == 0:
        return np.empty(0, dtype=np.int16)
    parts, total = [], 0
    while total < n:
        dwell = int(rng.integers(5, 50))
        parts.append(np.full(dwell, rng.normal(90.0, 15.0)) + rng.normal(0.0, 2.0, dwell))
        total += dwell
    pa = np.concatenate(parts)[:n]
    return np.clip(np.round(pa / scale - offset), -32768, 32767).astype(np.int16)


def write_pod5(path: Path, lengths, *, seed: int = 0, run_infos=None, special: bool = True, sample_rate: int = 5000) -> list[pod5.Read]:
    """Write reads with the given signal lengths; returns the pod5.Read objects written."""
    rng = np.random.default_rng(seed)
    infos = run_infos or [make_run_info("acq-1", sample_rate=sample_rate), make_run_info("acq-2", sample_id="", sample_rate=sample_rate)]
    reasons = list(pod5.EndReasonEnum)
    reads = []
    for i, n in enumerate(lengths):
        offset, scale = float(np.float32(-240.0 + i % 3)), float(np.float32(0.1755 + 0.001 * (i % 2)))
        reads.append(
            pod5.Read(
                read_id=uuid.UUID(int=(0x1234 << 100) + i),
                pore=pod5.Pore(channel=1 + i % 3000, well=1 + i % 4, pore_type="not_set" if i % 2 else "R10 ☃"),
                calibration=pod5.Calibration(offset=offset, scale=scale),
                read_number=100000 + i,
                start_sample=(1 << 40) + 5000 * i,
                median_before=float("nan") if (special and i % 5 == 1) else 200.5 + i,
                end_reason=pod5.EndReason(reason=reasons[i % len(reasons)], forced=bool(i % 2)),
                run_info=infos[i % len(infos)],
                num_minknow_events=7 * i,
                tracked_scaling=pod5.pod5_types.ShiftScalePair(shift=float("nan") if i % 2 else 1.5, scale=float("inf") if (special and i == 2) else 2.5),
                predicted_scaling=pod5.pod5_types.ShiftScalePair(shift=-0.0, scale=3.25),
                num_reads_since_mux_change=i,
                time_since_mux_change=0.125 * i,
                open_pore_level=float("nan") if i % 3 == 0 else 220.25,
                signal=squiggle(rng, int(n), offset=offset, scale=scale),
            )
        )
    with pod5.Writer(path) as writer:
        writer.add_reads(reads)
    return reads


# ------------------------------------------------------------------ stand-in engines


class LookupEngine:
    """Lossless stand-in for the model: encode stores each chunk under a fresh id (written into
    the first two tokens), decode returns the stored chunk. A compress -> decompress round trip
    through it must reproduce the ADC signal, so any read/chunk/order mistake is visible.

    Like a GPU, the async calls read their input only when the result is collected, so a
    pipeline that reused a buffer still in flight would break the round trip. `events` records
    ("send" | "collect", direction, call number)."""

    def __init__(self, batch_size: int, profile: CodecProfile) -> None:
        self.batch_size = batch_size
        self.L, self.T = profile.chunk_samples, profile.tokens_per_chunk
        self._store: dict[int, np.ndarray] = {}
        self.encode_calls = 0
        self.decode_calls = 0
        self.events: list[tuple[str, str, int]] = []

    def encode_async(self, batch: np.ndarray, valid_rows: int):
        assert batch.shape == (self.batch_size, self.L) and batch.dtype == np.float32
        self.encode_calls += 1
        call = self.encode_calls
        self.events.append(("send", "encode", call))

        def collect() -> np.ndarray:
            self.events.append(("collect", "encode", call))
            codes = np.zeros((valid_rows, self.T), dtype=np.uint16)
            for r in range(valid_rows):
                key = len(self._store) + 1
                self._store[key] = batch[r].copy()
                codes[r, 0], codes[r, 1] = key & 0xFFFF, key >> 16
            return codes

        return collect

    def decode_async(self, codes: np.ndarray, valid_rows: int):
        assert codes.shape == (self.batch_size, self.T)
        self.decode_calls += 1
        call = self.decode_calls
        self.events.append(("send", "decode", call))

        def collect() -> np.ndarray:
            self.events.append(("collect", "decode", call))
            out = np.empty((valid_rows, self.L), dtype=np.float32)
            for r in range(valid_rows):
                out[r] = self._store[int(codes[r, 0]) | (int(codes[r, 1]) << 16)]
            return out

        return collect

    def encode(self, batch: np.ndarray, valid_rows: int) -> np.ndarray:
        return self.encode_async(batch, valid_rows)()

    def decode(self, codes: np.ndarray, valid_rows: int) -> np.ndarray:
        return self.decode_async(codes, valid_rows)()


class ContentEngine:
    """Deterministic content-derived codes (so runs with different batch sizes must agree byte for
    byte); decode is a piecewise-constant approximation. Optionally fails when the result of a
    given call is collected. Inputs are read at collection time, as with LookupEngine."""

    def __init__(self, batch_size: int, profile: CodecProfile, *, fail_on_call: int | None = None, fail_with: BaseException | None = None) -> None:
        self.batch_size = batch_size
        self.L, self.T = profile.chunk_samples, profile.tokens_per_chunk
        self.fail_on_call = fail_on_call
        self.fail_with = fail_with
        self.calls = 0
        self._weights = (np.arange(self.L) % 251 + 1).astype(np.float64)

    def encode_async(self, batch: np.ndarray, valid_rows: int):
        self.calls += 1
        call = self.calls

        def collect() -> np.ndarray:
            if self.fail_on_call is not None and call == self.fail_on_call:
                raise self.fail_with or RuntimeError("injected encode failure")
            seg = batch[:valid_rows].reshape(valid_rows, self.T, self.L // self.T).mean(axis=2)
            codes = np.clip(np.rint((seg + 1.0) * 0.5 * 65535.0), 0, 65535).astype(np.uint16)
            for r in range(valid_rows):  # position-weighted sum, so distinct chunks get distinct first tokens
                codes[r, 0] = int(np.rint(np.dot(batch[r].astype(np.float64), self._weights) * 1000.0)) % 65536
            return codes

        return collect

    def decode_async(self, codes: np.ndarray, valid_rows: int):
        def collect() -> np.ndarray:
            vals = codes[:valid_rows].astype(np.float32) / 65535.0 * 2.0 - 1.0
            return np.repeat(vals, self.L // self.T, axis=1)

        return collect

    def encode(self, batch: np.ndarray, valid_rows: int) -> np.ndarray:
        return self.encode_async(batch, valid_rows)()

    def decode(self, codes: np.ndarray, valid_rows: int) -> np.ndarray:
        return self.decode_async(codes, valid_rows)()


def small_profile(profile: CodecProfile, **changes) -> CodecProfile:
    return replace(profile, **changes)
