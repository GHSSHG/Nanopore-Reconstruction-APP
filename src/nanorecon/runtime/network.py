"""Inference-only NanoRecon network (Flax linen).

Module and parameter names mirror the training repository so the released flax msgpack loads
unchanged:
    params/encoder, params/latent_bilstm, params/quant_conv, params/quantizer/{W,proj_bias},
    params/post_quant_conv, params/decoder, vq/quantizer/codebook
Only the hard-codeword inference path exists here. There are no train flags, dropout, DiVeQ
noise, random keys or monitoring outputs; codes are chosen by exact nearest-neighbour search
against the projected codebook and decoding only looks codes up in that codebook.

Matmuls run at the training precision (TF32 inputs, fp32 sums) except the two heaviest ones, the
codebook search and the decoder's ConvNeXt pointwise layers: they take fp16 operands, which have
the same 10-bit mantissa as TF32 and twice its tensor-core rate. Operands are scaled by powers of
two (exact) so that small values stay out of fp16's subnormal range, and rounded like the GPU's
TF32 conversion, so the inputs equal the TF32 inputs; sums and outputs stay fp32.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pl_triton

from ..model_config import NetworkConfig

F32 = jnp.float32
F16 = jnp.float16


def pow2_scale(bound: jnp.ndarray) -> jnp.ndarray:
    """2^k with bound * 2^k < 2^15: fits fp16 (max 65504) with headroom; applying and undoing it
    is exact."""
    _, exp = jnp.frexp(jnp.maximum(jnp.asarray(bound, F32), jnp.finfo(F32).tiny))
    return jnp.ldexp(F32(1.0), jnp.clip(15 - exp, -100, 100))


def to_f16(x: jnp.ndarray) -> jnp.ndarray:
    """fp32 -> fp16 rounded to nearest with ties away from zero, as Ampere GPUs convert fp32 to
    TF32 (measured for cuBLAS and cuDNN); exact for fp16's normal range."""
    bits = jax.lax.bitcast_convert_type(x.astype(F32), jnp.uint32)
    bits = (bits + jnp.uint32(0x1000)) & jnp.uint32(0xFFFFE000)
    return jax.lax.bitcast_convert_type(bits, F32).astype(F16)


def _resolve_groups(channels: int, max_groups: int) -> int:
    """GroupNorm group count: the largest divisor of `channels` not above `max_groups`."""
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return max(1, groups)


def _residual_dilations(num_blocks: int) -> tuple[int, ...]:
    count = max(1, int(num_blocks))
    if count <= 3:
        return (1, 2, 4)[:count]
    return tuple(2**i for i in range(count))


class ReflectConv1d(nn.Module):
    """'Same'-length (per stride) convolution with reflect padding, as in training."""

    features: int
    kernel: int
    stride: int = 1
    dilation: int = 1
    use_bias: bool = False
    feature_group_count: int = 1
    dtype: Any = F32
    param_dtype: Any = F32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        length = x.shape[1]
        effective = (self.kernel - 1) * self.dilation + 1
        out_length = (length + self.stride - 1) // self.stride
        total = max(0, (out_length - 1) * self.stride + effective - length)
        left, right = total // 2, total - total // 2
        if total > 0:
            max_pad = max(left, right)
            if length <= max_pad:
                x = jnp.pad(x, ((0, 0), (0, max_pad - length + 1), (0, 0)), mode="edge")
            x = jnp.pad(x, ((0, 0), (left, right), (0, 0)), mode="reflect")
            if length <= max_pad:
                x = x[:, : length + total, :]
        return nn.Conv(
            features=self.features,
            kernel_size=(self.kernel,),
            strides=(self.stride,),
            kernel_dilation=(self.dilation,),
            padding="VALID",
            use_bias=self.use_bias,
            feature_group_count=self.feature_group_count,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv",
        )(x)


class GroupNorm1D(nn.Module):
    channels: int
    max_groups: int = 32
    epsilon: float = 1e-6
    dtype: Any = F32
    param_dtype: Any = F32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.GroupNorm(
            num_groups=_resolve_groups(self.channels, self.max_groups),
            epsilon=self.epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)


# ------------------------------------------------------------------------------ encoder


class EncoderResBlock(nn.Module):
    channels: int
    dilation: int
    use_norm: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hidden = max(1, self.channels // 2)
        h = ReflectConv1d(hidden, 3, dilation=max(1, self.dilation), name="conv1")(x)
        if self.use_norm:
            h = GroupNorm1D(hidden, max_groups=1, name="norm1")(h)
        h = nn.elu(h)
        h = ReflectConv1d(self.channels, 1, name="conv2")(h)
        if self.use_norm:
            h = GroupNorm1D(self.channels, max_groups=1, name="norm2")(h)
        return x + h


class EncoderStage(nn.Module):
    in_ch: int
    out_ch: int
    num_blocks: int
    stride: int
    use_block_norm: bool
    use_transition_norm: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = x
        for i, dilation in enumerate(_residual_dilations(self.num_blocks)):
            h = EncoderResBlock(self.in_ch, dilation, self.use_block_norm, name=f"block_{i}")(h)
        h = h.astype(F32)
        kernel = 3 if self.stride == 1 else max(4, self.stride * 2)
        h = ReflectConv1d(self.out_ch, kernel, stride=self.stride, name="transition")(h)
        if self.use_transition_norm:
            h = GroupNorm1D(self.out_ch, max_groups=1, name="transition_norm")(h)
        return nn.elu(h)


class Encoder(nn.Module):
    cfg: NetworkConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        cfg = self.cfg
        channels = cfg.enc_channels
        h = x[:, :, None].astype(F32)
        h = ReflectConv1d(channels[0], cfg.enc_kernel_size, name="conv_in")(h)
        if cfg.encoder_use_input_norm:
            h = GroupNorm1D(channels[0], max_groups=1, name="conv_in_norm")(h)
        h = nn.elu(h)
        for i, (stride, blocks) in enumerate(zip(cfg.enc_down_strides, cfg.enc_stage_num_res_blocks)):
            h = EncoderStage(
                channels[i], channels[i + 1], blocks, stride,
                cfg.encoder_use_block_norm, cfg.encoder_use_transition_norm, name=f"stage_{i}",
            )(h)
        return h


LSTM_UNROLL = 8  # time steps per loop iteration of the recurrence


def _input_gates(x: jnp.ndarray, x_kernel, x_bias) -> jnp.ndarray:
    """(B, T, D) -> time-major input contributions to the gates (T, B, 4H)."""
    return jnp.matmul(jnp.swapaxes(x, 0, 1), x_kernel.astype(F32)) + x_bias.astype(F32)


def _lstm_cell(state, gates_x: jnp.ndarray, h_kernel: jnp.ndarray, fb: jnp.ndarray):
    h_prev, c_prev = state
    gates = gates_x + jnp.matmul(h_prev, h_kernel)
    i, f, g, o = jnp.split(gates, 4, axis=-1)
    i = nn.sigmoid(i)
    f = nn.sigmoid(f + fb)
    g = jnp.tanh(g)
    o = nn.sigmoid(o)
    c = f * c_prev + i * g
    h = o * jnp.tanh(c)
    return h, c


def _bilstm(x: jnp.ndarray, p: dict, hidden: int, forget_bias: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Both directions of a layer in one scan: step t runs the forward cell on x[:, t] and the
    backward cell on x[:, T-1-t]. Each cell computes exactly what a scan per direction would;
    sharing the loop halves the sequential iterations, which bound the time on a GPU."""
    gates_f = _input_gates(x, p["fwd_x_kernel"], p["fwd_x_bias"])
    gates_b = _input_gates(jnp.flip(x, axis=1), p["bwd_x_kernel"], p["bwd_x_bias"])
    h_kernel_f, h_kernel_b = p["fwd_h_kernel"].astype(F32), p["bwd_h_kernel"].astype(F32)
    fb = jnp.asarray(forget_bias, dtype=F32)
    zeros = jnp.zeros((x.shape[0], hidden), dtype=F32)

    def step(carry, gates):
        fwd = _lstm_cell(carry[0], gates[0], h_kernel_f, fb)
        bwd = _lstm_cell(carry[1], gates[1], h_kernel_b, fb)
        return (fwd, bwd), (fwd[0], bwd[0])

    _, (y_f, y_b) = jax.lax.scan(step, ((zeros, zeros), (zeros, zeros)), (gates_f, gates_b), unroll=LSTM_UNROLL)
    return jnp.swapaxes(y_f, 0, 1), jnp.flip(jnp.swapaxes(y_b, 0, 1), axis=1)


class BiLSTMLayer(nn.Module):
    dim: int
    hidden: int
    forget_bias: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gates = 4 * self.hidden
        zeros = nn.initializers.zeros
        p = {
            name: self.param(name, zeros, shape, F32)
            for name, shape in (
                ("fwd_x_kernel", (self.dim, gates)),
                ("fwd_x_bias", (gates,)),
                ("fwd_h_kernel", (self.hidden, gates)),
                ("bwd_x_kernel", (self.dim, gates)),
                ("bwd_x_bias", (gates,)),
                ("bwd_h_kernel", (self.hidden, gates)),
            )
        }
        h = x.astype(F32)
        fwd, bwd = _bilstm(h, p, self.hidden, self.forget_bias)
        y = nn.Dense(self.dim, use_bias=True, name="out_proj")(jnp.concatenate((fwd, bwd), axis=-1))
        return (h + y).astype(F32)


class BiLSTM(nn.Module):
    dim: int
    num_layers: int
    hidden: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = x
        for i in range(self.num_layers):
            h = BiLSTMLayer(self.dim, self.hidden, name=f"layer_{i}")(h)
        return h


class CodebookProjection(nn.Module):
    """SimVQ codebook: fixed base embeddings ('vq' collection) and a learned affine map."""

    codebook_size: int
    code_dim: int

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        base = self.variable("vq", "codebook", lambda: jnp.zeros((self.codebook_size, self.code_dim), F32))
        w = self.param("W", nn.initializers.zeros, (self.code_dim, self.code_dim), F32)
        bias = self.param("proj_bias", nn.initializers.zeros, (self.code_dim,), F32)
        return jnp.dot(base.value.astype(F32), w) + bias


# ------------------------------------------------------------------------------ decoder


def _swish(x: jnp.ndarray) -> jnp.ndarray:
    return x * nn.sigmoid(x)


class F16Dense(nn.Module):
    """Dense layer (same parameters as nn.Dense) on fp16 operands with fp32 sums and output.
    `bound` caps |inputs|; the kernel is scaled by its own largest value."""

    features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, bound: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param("kernel", nn.initializers.zeros, (x.shape[-1], self.features), F32)
        bias = self.param("bias", nn.initializers.zeros, (self.features,), F32)
        sx, sk = pow2_scale(bound), pow2_scale(jnp.max(jnp.abs(kernel)))
        y = jnp.dot(to_f16(x * sx), to_f16(kernel * sk), preferred_element_type=F32)
        return y * (1.0 / (sx * sk)) + bias


class ConvNeXtBlock(nn.Module):
    dim: int
    intermediate: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = ReflectConv1d(self.dim, 7, feature_group_count=self.dim, use_bias=True, name="dwconv")(x)
        norm = nn.LayerNorm(dtype=F32, param_dtype=F32, name="norm")
        h = norm(h)
        # Input bounds from the weights, so the fp16 scales can never overflow: a layer-normalized
        # value is at most sqrt(C - 1) in magnitude before scale and bias; tanh-GELU of y is at
        # most max(|y|, 0.17).
        p = norm.variables["params"]
        per_input = jnp.abs(p["scale"]) * math.sqrt(self.dim - 1) + jnp.abs(p["bias"])
        pw1 = F16Dense(self.intermediate, name="pwconv1")
        h = pw1(h, jnp.max(per_input))
        q = pw1.variables["params"]
        bound2 = jnp.max(per_input @ jnp.abs(q["kernel"]) + jnp.abs(q["bias"]))
        h = nn.gelu(h, approximate=True)
        h = F16Dense(self.dim, name="pwconv2")(h, jnp.maximum(bound2, 0.17))
        gamma = self.param("gamma", nn.initializers.zeros, (self.dim,), F32)
        return x + h * gamma.astype(h.dtype)


class DecoderResBlock(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = _swish(GroupNorm1D(self.dim, name="norm1")(x))
        h = ReflectConv1d(self.dim, 3, use_bias=True, name="conv1")(h)
        h = _swish(GroupNorm1D(self.dim, name="norm2")(h))
        h = ReflectConv1d(self.dim, 3, use_bias=True, name="conv2")(h)
        return x + h


ATTENTION_IMPLS = {"cudnn": ("cudnn", jnp.bfloat16), "xla": ("xla", F32)}


class SelfAttention(nn.Module):
    dim: int
    heads: int
    attention: str  # key of ATTENTION_IMPLS

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = GroupNorm1D(self.dim, name="norm")(x)
        batch, length, _ = h.shape
        head_dim = self.dim // self.heads
        q, k, v = (
            ReflectConv1d(self.dim, 1, use_bias=True, name=n)(h).reshape(batch, length, self.heads, head_dim)
            for n in ("q", "k", "v")
        )
        implementation, qkv_dtype = ATTENTION_IMPLS[self.attention]
        h = jax.nn.dot_product_attention(
            q.astype(qkv_dtype), k.astype(qkv_dtype), v.astype(qkv_dtype), implementation=implementation
        )
        h = h.reshape(batch, length, self.dim).astype(F32)
        h = ReflectConv1d(self.dim, 1, use_bias=True, name="proj_out")(h)
        return x + h


class PosNet(nn.Module):
    dim: int
    heads: int
    attention: str

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = DecoderResBlock(self.dim, name="resnet_0")(x)
        h = DecoderResBlock(self.dim, name="resnet_1")(h)
        h = SelfAttention(self.dim, self.heads, self.attention, name="attn")(h)
        h = DecoderResBlock(self.dim, name="resnet_2")(h)
        h = DecoderResBlock(self.dim, name="resnet_3")(h)
        return GroupNorm1D(self.dim, name="norm")(h)


def hann_window(length: int) -> jnp.ndarray:
    if length <= 1:
        return jnp.ones((max(1, length),), dtype=F32)
    n = jnp.arange(length, dtype=F32)
    return 0.5 - 0.5 * jnp.cos((2.0 * jnp.pi * n) / float(length - 1))


def overlap_add(frames: jnp.ndarray, hop: int) -> jnp.ndarray:
    """Sum frames (B, F, N) placed every `hop` samples -> (B, (F - 1) * hop + N)."""
    batch, num_frames, frame_len = frames.shape
    if frame_len % hop == 0:
        # Deterministic: split each frame into hop-long segments and add them with shifts.
        r_count = frame_len // hop
        seg = frames.reshape(batch, num_frames, r_count, hop)
        acc = None
        for r in range(r_count):
            part = jnp.pad(seg[:, :, r, :], ((0, 0), (r, r_count - 1 - r), (0, 0)))
            acc = part if acc is None else acc + part
        return acc.reshape(batch, (num_frames + r_count - 1) * hop)
    out_length = (num_frames - 1) * hop + frame_len
    idx = (jnp.arange(num_frames)[:, None] * hop + jnp.arange(frame_len)[None, :]).reshape(-1)
    return jax.vmap(lambda f: jnp.zeros((out_length,), frames.dtype).at[idx].add(f.reshape(-1)))(frames)


def istft_same(spec: jnp.ndarray, n_fft: int, hop: int) -> jnp.ndarray:
    """Vocos-style 'same' ISTFT of (B, frames, bins) complex spectra -> (B, frames * hop)."""
    num_frames = spec.shape[1]
    target = num_frames * hop
    window = hann_window(n_fft)
    frames = jnp.fft.irfft(spec, n=n_fft, axis=-1).astype(F32) * window[None, None, :]
    y = overlap_add(frames, hop)
    envelope = overlap_add(jnp.broadcast_to(jnp.square(window)[None, None, :], (1, num_frames, n_fft)), hop)[0]
    crop_left = max(0, (y.shape[-1] - target) // 2)
    y = y[:, crop_left : crop_left + target]
    envelope = envelope[crop_left : crop_left + target]
    return y / jnp.maximum(envelope[None, :], 1e-8)


class Decoder(nn.Module):
    cfg: NetworkConfig
    attention: str

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        cfg = self.cfg
        h = ReflectConv1d(cfg.decoder_dim, 7, use_bias=True, name="embed")(z.astype(F32))
        if cfg.decoder_pos_net_enabled:
            h = PosNet(cfg.decoder_dim, cfg.decoder_pos_net_attention_heads, self.attention, name="pos_net")(h)
        for i in range(cfg.decoder_num_layers):
            h = ConvNeXtBlock(cfg.decoder_dim, cfg.decoder_intermediate_dim, name=f"convnext_{i}")(h)
        h = nn.LayerNorm(dtype=F32, param_dtype=F32, name="final_norm")(h).astype(F32)

        batch, frames, _ = h.shape
        bins = cfg.istft_n_fft // 2 + 1
        num_bands = len(cfg.istft_band_edges_hz) + 1
        coeff_raw = ReflectConv1d(bins * 2, 1, use_bias=True, name="coeff_proj")(h).reshape(batch, frames, bins, 2)
        gate_raw = ReflectConv1d(num_bands, 1, use_bias=True, name="gate_proj")(h)
        gates = 1.0 + float(cfg.istft_dynamic_gate_scale) * jnp.tanh(gate_raw.astype(F32))
        band_ids = jnp.asarray(cfg.band_ids(), dtype=jnp.int32)
        log_band_gain = self.param("log_band_gain", nn.initializers.zeros, (num_bands,), F32)
        gain_by_bin = jnp.exp(log_band_gain.astype(F32))[band_ids]
        coeff = coeff_raw * gain_by_bin[None, None, :, None] * jnp.take(gates, band_ids, axis=-1)[..., None]
        real = coeff[..., 0]
        imag = coeff[..., 1].at[..., 0].set(0.0).at[..., -1].set(0.0)
        spec = real.astype(jnp.complex64) + 1j * imag.astype(jnp.complex64)
        wave = istft_same(spec, cfg.istft_n_fft, cfg.istft_hop_length)

        raw_alpha = self.param("residual_raw_alpha", nn.initializers.zeros, (), F32)
        if cfg.residual_correction_enabled:
            r = ReflectConv1d(cfg.residual_correction_hidden_dim, 1, use_bias=True, name="residual_proj1")(h)
            r = nn.gelu(r, approximate=True)
            r = ReflectConv1d(cfg.istft_hop_length, 1, use_bias=True, name="residual_proj2")(r)
            r = r.reshape(batch, frames * cfg.istft_hop_length)
            alpha = float(cfg.residual_correction_alpha_max) * nn.sigmoid(raw_alpha.astype(F32))
            wave = wave + alpha * r
        return wave.astype(F32)


# ------------------------------------------------------------------------------ model


class NanoReconNet(nn.Module):
    cfg: NetworkConfig
    attention: str

    def setup(self) -> None:
        cfg = self.cfg
        self.encoder = Encoder(cfg)
        self.latent_bilstm = BiLSTM(cfg.latent_dim, cfg.latent_bilstm_layers, cfg.latent_bilstm_hidden_dim)
        self.quant_conv = ReflectConv1d(cfg.quantizer_dim, cfg.quant_conv_kernel_size)
        self.quantizer = CodebookProjection(cfg.codebook_size, cfg.quantizer_dim)
        self.post_quant_conv = ReflectConv1d(cfg.dec_channels[0], cfg.post_quant_conv_kernel_size)
        self.decoder = Decoder(cfg, self.attention)

    def latents(self, x: jnp.ndarray) -> jnp.ndarray:
        """Normalized chunks (B, L) -> pre-quantization latents (B, T, D)."""
        h = self.encoder(x).astype(F32)
        h = self.latent_bilstm(h).astype(F32)
        return self.quant_conv(h)

    def projected_codebook(self) -> jnp.ndarray:
        return self.quantizer()

    def reconstruct(self, z_q: jnp.ndarray) -> jnp.ndarray:
        """Quantized latents (B, T, D) -> normalized waveform (B, L)."""
        return self.decoder(self.post_quant_conv(z_q.astype(F32)))

    def __call__(self, x: jnp.ndarray):
        # Used for initialization/shape inference only: touches every submodule once.
        return self.reconstruct(self.latents(x)), self.projected_codebook()


# Codebook search tile: latent rows, codewords, dimensions per step, then Triton warps and
# pipeline stages. Tuned on an RTX 3080 Ti, within 6% of the best tile tried on an A100; changing
# it never changes the codes.
SEARCH_TILE = (64, 256, 32, 4, 3)


class SearchCodebook(NamedTuple):
    """The projected codebook prepared once for the search."""

    codes: jnp.ndarray  # fp16 (K, D): codebook * 2^k, zero-padded to whole search tiles
    inverse_scale: jnp.ndarray  # 2^-k
    norms: jnp.ndarray  # fp32 (K,): ||e||^2 of the fp32 codebook, inf for padding rows


def search_codebook(codebook: jnp.ndarray) -> SearchCodebook:
    _, tile_codes, tile_dims, _, _ = SEARCH_TILE
    num_codes, dim = codebook.shape
    scale = pow2_scale(jnp.max(jnp.abs(codebook)))
    codes = jnp.pad(to_f16(codebook * scale), ((0, -num_codes % tile_codes), (0, -dim % tile_dims)))
    norms = jnp.pad(jnp.sum(codebook.astype(F32) ** 2, axis=1), (0, -num_codes % tile_codes), constant_values=jnp.inf)
    return SearchCodebook(codes, 1.0 / scale, norms)


def nearest_codeword(z: jnp.ndarray, codebook: SearchCodebook) -> jnp.ndarray:
    """Exact nearest neighbour (squared L2) of each row of z (N, D) in the codebook.

    Distances ||z||^2 + ||e||^2 - 2 z.e as in training, with z.e on fp16 operands (TF32-equal
    inputs, fp32 sums). One GPU kernel computes the distances tile by tile and keeps only each
    row's minimum per tile with its first index, so the N x K distances never reach GPU memory;
    the tile minima are then compared in codebook order, so ties resolve to the lowest index.
    """
    rows, codes, dims, warps, stages = SEARCH_TILE
    num_rows = z.shape[0]
    num_codes, dim = codebook.codes.shape
    z = z.astype(F32)
    z_scale = pow2_scale(jnp.max(jnp.abs(z), axis=1))  # per row: rows never affect each other
    pad = (0, -num_rows % rows)
    z16 = jnp.pad(to_f16(z * z_scale[:, None]), (pad, (0, dim - z.shape[1])))
    unscale = jnp.pad(codebook.inverse_scale / z_scale, pad)
    z_norm = jnp.pad(jnp.sum(z**2, axis=1), pad)

    def kernel(z_ref, codes_ref, unscale_ref, z_norm_ref, norms_ref, low_ref, first_ref):
        def step(k, acc):
            cols = pl.ds(k * dims, dims)
            return acc + pl.dot(z_ref[:, cols], codes_ref[:, cols], trans_b=True)

        dots = jax.lax.fori_loop(0, dim // dims, step, jnp.zeros((rows, codes), F32))
        dists = z_norm_ref[...][:, None] + norms_ref[...][None, :] - 2.0 * (dots * unscale_ref[...][:, None])
        low = jnp.min(dists, axis=1)
        first = jnp.min(jnp.where(dists == low[:, None], jnp.arange(codes, dtype=jnp.int32)[None, :], codes), axis=1)
        low_ref[0, :] = low
        first_ref[0, :] = first + pl.program_id(1) * codes

    tiles = (z16.shape[0] // rows, num_codes // codes)
    out = jax.ShapeDtypeStruct((tiles[1], z16.shape[0]), F32)
    lows, firsts = pl.pallas_call(
        kernel,
        out_shape=(out, out.update(dtype=jnp.int32)),
        grid=tiles,
        in_specs=(
            pl.BlockSpec((rows, dim), lambda i, j: (i, 0)),
            pl.BlockSpec((codes, dim), lambda i, j: (j, 0)),
            pl.BlockSpec((rows,), lambda i, j: (i,)),
            pl.BlockSpec((rows,), lambda i, j: (i,)),
            pl.BlockSpec((codes,), lambda i, j: (j,)),
        ),
        out_specs=(pl.BlockSpec((1, rows), lambda i, j: (j, i)),) * 2,
        compiler_params=pl_triton.CompilerParams(num_warps=warps, num_stages=stages),
        interpret=jax.default_backend() == "cpu",  # unit tests on a machine without a GPU
        name="nearest_codeword",
    )(z16, codebook.codes, unscale, z_norm, codebook.norms)
    best_tile = jnp.argmin(lows, axis=0)  # the first tile holding the minimum
    return jnp.take_along_axis(firsts, best_tile[None, :], axis=0)[0, :num_rows]
