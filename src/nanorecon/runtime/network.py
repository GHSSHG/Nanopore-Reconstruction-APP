"""Inference-only NanoRecon network (Flax linen).

Module and parameter names mirror the training repository so the released flax msgpack loads
unchanged:
    params/encoder, params/latent_bilstm, params/quant_conv, params/quantizer/{W,proj_bias},
    params/post_quant_conv, params/decoder, vq/quantizer/codebook
Only the hard-codeword inference path exists here. There are no train flags, dropout, DiVeQ
noise, random keys or monitoring outputs; codes are chosen by exact nearest-neighbour search
against the projected codebook and decoding only looks codes up in that codebook.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..model_config import NetworkConfig

F32 = jnp.float32


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


def _lstm_direction(x: jnp.ndarray, x_kernel, x_bias, h_kernel, hidden: int, forget_bias: float) -> jnp.ndarray:
    batch = x.shape[0]
    x_time = jnp.swapaxes(x, 0, 1)
    h0 = jnp.zeros((batch, hidden), dtype=F32)
    c0 = jnp.zeros((batch, hidden), dtype=F32)
    x_gates = jnp.matmul(x_time, x_kernel.astype(F32)) + x_bias.astype(F32)
    fb = jnp.asarray(forget_bias, dtype=F32)

    def step(carry, gates_x):
        h_prev, c_prev = carry
        gates = gates_x + jnp.matmul(h_prev, h_kernel.astype(F32))
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = nn.sigmoid(i)
        f = nn.sigmoid(f + fb)
        g = jnp.tanh(g)
        o = nn.sigmoid(o)
        c = f * c_prev + i * g
        h = o * jnp.tanh(c)
        return (h, c), h

    _, y_time = jax.lax.scan(step, (h0, c0), x_gates)
    return jnp.swapaxes(y_time, 0, 1)


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
        fwd = _lstm_direction(h, p["fwd_x_kernel"], p["fwd_x_bias"], p["fwd_h_kernel"], self.hidden, self.forget_bias)
        bwd = _lstm_direction(jnp.flip(h, axis=1), p["bwd_x_kernel"], p["bwd_x_bias"], p["bwd_h_kernel"], self.hidden, self.forget_bias)
        bwd = jnp.flip(bwd, axis=1)
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


class ConvNeXtBlock(nn.Module):
    dim: int
    intermediate: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = ReflectConv1d(self.dim, 7, feature_group_count=self.dim, use_bias=True, name="dwconv")(x)
        h = nn.LayerNorm(dtype=F32, param_dtype=F32, name="norm")(h)
        h = nn.Dense(self.intermediate, use_bias=True, name="pwconv1")(h)
        h = nn.gelu(h, approximate=True)
        h = nn.Dense(self.dim, use_bias=True, name="pwconv2")(h)
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


def nearest_codeword(z: jnp.ndarray, codebook: jnp.ndarray, block_size: int) -> jnp.ndarray:
    """Exact nearest neighbour (squared L2) of each row of z (N, D) in codebook (K, D).

    Same arithmetic and tie rule as training: distances ||z||^2 + ||e||^2 - 2 z.e per codebook
    block, first minimum within a block, strictly smaller distance needed to replace a winner
    from an earlier block (so ties resolve to the lowest index). Peak memory is N x block.
    """
    num_codes, dim = codebook.shape
    block = max(1, int(block_size))
    z = z.astype(codebook.dtype)
    z_norm = jnp.sum(z**2, axis=1, keepdims=True)
    if block >= num_codes:
        dists = z_norm + jnp.sum(codebook**2, axis=1)[None, :] - 2.0 * jnp.dot(z, codebook.T)
        return jnp.argmin(dists, axis=1).astype(jnp.int32)
    n_blocks = -(-num_codes // block)
    pad = n_blocks * block - num_codes
    if pad:
        codebook = jnp.pad(codebook, ((0, pad), (0, 0)))
    blocks = codebook.reshape(n_blocks, block, dim)
    arange_b = jnp.arange(block, dtype=jnp.int32)
    inf = jnp.asarray(jnp.inf, dtype=z_norm.dtype)

    def step(carry, xs):
        best_dist, best_idx = carry
        codes, block_id = xs
        offset = block_id * block
        dists = z_norm + jnp.sum(codes**2, axis=1)[None, :] - 2.0 * jnp.dot(z, codes.T)
        dists = jnp.where(((offset + arange_b) < num_codes)[None, :], dists, inf)
        arg = jnp.argmin(dists, axis=1).astype(jnp.int32)
        dist = jnp.take_along_axis(dists, arg[:, None], axis=1)[:, 0]
        better = dist < best_dist
        return (jnp.where(better, dist, best_dist), jnp.where(better, offset + arg, best_idx)), None

    init = (jnp.full((z.shape[0],), jnp.inf, dtype=z_norm.dtype), jnp.zeros((z.shape[0],), dtype=jnp.int32))
    (_, best_idx), _ = jax.lax.scan(step, init, (blocks, jnp.arange(n_blocks, dtype=jnp.int32)))
    return best_idx
