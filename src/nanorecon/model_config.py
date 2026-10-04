"""The one supported model: its identity, the network described by its config.json, and the
fixed inference profile (CodecProfile) that every token file records.

The released config.json only states the network; the profile (overlap, normalization,
padding, tail, quantization, stitching) is this app's inference contract for that release.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

REPO_ID = "GHSSHG/NanoRecon"
REVISION = "92be5d5aa05990eccc9edca33c263ee5ffc72739"
ARCHITECTURE = "SimVQAudioModel"
MODEL_TYPE = "nanorecon-simvq-audio-codec"
CONFIG_FILE = "config.json"
WEIGHTS_FILE = "flax_model.msgpack"


# --------------------------------------------------------------------------------------
# Network description (mirrors the training factory's interpretation of config["model"])
# --------------------------------------------------------------------------------------

# Values the training factory uses when a key is absent from config["model"]. The release
# weights were produced under this interpretation, so it must be reproduced exactly.
_FOUNDATION_DEFAULTS: dict[str, Any] = {
    "enc_channels": (32, 64, 128, 256, 512),
    "enc_down_strides": (2, 2, 2, 3),
    "enc_stage_num_res_blocks": (2, 2, 3, 3),
    "enc_kernel_size": 7,
    "latent_dim": 512,
    "quantizer_dim": 512,
    "codebook_size": 16384,
    "dec_channels": (512,),
    "decoder_dim": 512,
    "decoder_intermediate_dim": 1536,
    "decoder_num_layers": 12,
    "decoder_pos_net_enabled": True,
    "decoder_pos_net_dropout": 0.0,
    "decoder_pos_net_attention_heads": 12,
    "decoder_pos_net_attention_backend": "jax_cudnn",
    "istft_n_fft": 512,
    "istft_hop_length": 24,
    "istft_sample_rate": 5000.0,
    "istft_band_edges_hz": (200.0, 500.0, 1000.0),
    "istft_band_gain_init": (1.0, 0.8, 0.45, 0.12),
    "istft_dynamic_gate_scale": 0.5,
    "residual_correction_enabled": True,
    "residual_correction_alpha_max": 0.1,
    "residual_correction_hidden_dim": 128,
    "latent_bilstm_layers": 2,
    "latent_bilstm_hidden_dim": 256,
    "diveq_sigma2": 1e-3,
    "search_chunk_size": 2048,
    "quant_conv_kernel_size": 7,
    "post_quant_conv_kernel_size": 7,
    "encoder_use_block_norm": True,
    "encoder_use_input_norm": True,
    "encoder_use_transition_norm": True,
}
_ATTENTION_BACKENDS = {"jax_cudnn": "cudnn", "cudnn": "cudnn", "xla": "xla"}


@dataclass(frozen=True)
class NetworkConfig:
    enc_channels: tuple[int, ...]
    enc_down_strides: tuple[int, ...]
    enc_stage_num_res_blocks: tuple[int, ...]
    enc_kernel_size: int
    encoder_use_block_norm: bool
    encoder_use_input_norm: bool
    encoder_use_transition_norm: bool
    latent_dim: int
    latent_bilstm_layers: int
    latent_bilstm_hidden_dim: int
    quantizer_dim: int
    codebook_size: int
    quant_conv_kernel_size: int
    post_quant_conv_kernel_size: int
    search_chunk_size: int
    dec_channels: tuple[int, ...]
    decoder_dim: int
    decoder_intermediate_dim: int
    decoder_num_layers: int
    decoder_pos_net_enabled: bool
    decoder_pos_net_attention_heads: int
    attention_backend: str  # "cudnn" | "xla"
    istft_n_fft: int
    istft_hop_length: int
    istft_sample_rate: float
    istft_band_edges_hz: tuple[float, ...]
    istft_band_gain_init: tuple[float, ...]
    istft_dynamic_gate_scale: float
    residual_correction_enabled: bool
    residual_correction_alpha_max: float
    residual_correction_hidden_dim: int
    # Training-only fact, shown by `info` but not used: inference is hard nearest-codeword.
    diveq_sigma2: float

    @property
    def downsample_factor(self) -> int:
        return math.prod(self.enc_down_strides)

    def tokens_for(self, chunk_samples: int) -> int:
        """Encoder output length for an input of `chunk_samples` (ceil per strided conv)."""
        length = int(chunk_samples)
        for stride in self.enc_down_strides:
            length = -(-length // stride)
        return length

    def band_ids(self) -> tuple[int, ...]:
        """STFT bin -> band index, exactly as the trained decoder derives it."""
        bins = self.istft_n_fft // 2 + 1
        bin_hz = float(self.istft_sample_rate) / float(self.istft_n_fft)
        stops = [min(bins, max(1, int(math.floor(edge / bin_hz)) + 1)) for edge in self.istft_band_edges_hz]
        ids = []
        for idx in range(bins):
            band = 0
            for stop in stops:
                if idx < stop:
                    break
                band += 1
            ids.append(band)
        return tuple(ids)


def parse_network_config(model_cfg: Mapping[str, Any]) -> NetworkConfig:
    """config.json["model"] of the release -> NetworkConfig."""
    cfg = {**_FOUNDATION_DEFAULTS, **dict(model_cfg)}
    residual = cfg.get("residual_correction") or {}

    def ints(key: str) -> tuple[int, ...]:
        return tuple(int(v) for v in cfg[key])

    def floats(key: str) -> tuple[float, ...]:
        return tuple(float(v) for v in cfg[key])

    return NetworkConfig(
        enc_channels=ints("enc_channels"),
        enc_down_strides=ints("enc_down_strides"),
        enc_stage_num_res_blocks=ints("enc_stage_num_res_blocks"),
        enc_kernel_size=int(cfg["enc_kernel_size"]),
        encoder_use_block_norm=bool(cfg["encoder_use_block_norm"]),
        encoder_use_input_norm=bool(cfg["encoder_use_input_norm"]),
        encoder_use_transition_norm=bool(cfg["encoder_use_transition_norm"]),
        latent_dim=int(cfg["latent_dim"]),
        latent_bilstm_layers=int(cfg["latent_bilstm_layers"]),
        latent_bilstm_hidden_dim=int(cfg["latent_bilstm_hidden_dim"]),
        quantizer_dim=int(cfg["quantizer_dim"]),
        codebook_size=int(cfg["codebook_size"]),
        quant_conv_kernel_size=int(cfg["quant_conv_kernel_size"]),
        post_quant_conv_kernel_size=int(cfg["post_quant_conv_kernel_size"]),
        search_chunk_size=int(cfg["search_chunk_size"]),
        dec_channels=ints("dec_channels"),
        decoder_dim=int(cfg["decoder_dim"]),
        decoder_intermediate_dim=int(cfg["decoder_intermediate_dim"]),
        decoder_num_layers=int(cfg["decoder_num_layers"]),
        decoder_pos_net_enabled=bool(cfg["decoder_pos_net_enabled"]),
        decoder_pos_net_attention_heads=int(cfg["decoder_pos_net_attention_heads"]),
        attention_backend=_ATTENTION_BACKENDS[str(cfg["decoder_pos_net_attention_backend"]).lower()],
        istft_n_fft=int(cfg["istft_n_fft"]),
        istft_hop_length=int(cfg["istft_hop_length"]),
        istft_sample_rate=float(cfg["istft_sample_rate"]),
        istft_band_edges_hz=floats("istft_band_edges_hz"),
        istft_band_gain_init=floats("istft_band_gain_init"),
        istft_dynamic_gate_scale=float(cfg["istft_dynamic_gate_scale"]),
        residual_correction_enabled=bool(residual.get("enabled", cfg["residual_correction_enabled"])),
        residual_correction_alpha_max=float(residual.get("alpha_max", cfg["residual_correction_alpha_max"])),
        residual_correction_hidden_dim=int(residual.get("hidden_dim", cfg["residual_correction_hidden_dim"])),
        diveq_sigma2=float(cfg["diveq_sigma2"]),
    )


# --------------------------------------------------------------------------------------
# Model identity and codec profile, both recorded in every token file header
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelIdentity:
    repo_id: str
    revision: str
    architecture: str
    model_type: str
    codebook_size: int

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CodecProfile:
    sample_rate_hz: int
    chunk_samples: int  # L
    overlap_samples: int  # O
    hop_samples: int  # H = L - O
    tokens_per_chunk: int  # T
    codebook_size: int
    code_dtype: str
    normalization: str
    normalization_epsilon: float
    padding: str
    tail: str
    stitch: str
    quantization: str
    adc_conversion: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "CodecProfile":
        names = {f.name for f in fields(cls)}
        if not isinstance(data, Mapping) or set(data) != names:
            raise ValueError(f"profile must hold exactly the fields {sorted(names)}")
        return cls(**data)


MODEL = ModelIdentity(
    repo_id=REPO_ID,
    revision=REVISION,
    architecture=ARCHITECTURE,
    model_type=MODEL_TYPE,
    codebook_size=65536,
)

PROFILE = CodecProfile(
    sample_rate_hz=5000,
    chunk_samples=8192,
    overlap_samples=144,
    hop_samples=8048,
    tokens_per_chunk=512,
    codebook_size=65536,
    code_dtype="uint16",
    normalization="minmax_pm1",  # per chunk: center/half-range of the real samples, clip to [-1, 1]
    normalization_epsilon=1e-6,  # half-range below this -> the chunk is constant, normalized to 0
    padding="reflect",  # short chunks: normalize first, then reflect to L (N=1 repeats)
    tail="shift_last",  # an extra right-aligned last window when the hops leave a tail
    stitch="linear_edges_v1",  # pA-space weights min(1, (t+1)/(O+1), (L-t)/(O+1))
    quantization="hard_v1",  # exact nearest projected codeword, no DiVeQ noise
    adc_conversion="rint_clip_int16",  # after stitching: rint(pA/scale - offset), clip to int16
)
