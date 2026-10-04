"""Batch inference engine: encode(float32[B, L]) -> uint16[B, T], decode(uint16[B, T]) -> float32[B, L].

One fixed batch shape is compiled per direction; the caller pads the last batch and only the
first `valid_rows` rows are returned. Every call waits for the device result before returning,
so the caller may reuse its host buffers immediately and at most one call is in flight.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..model_config import CodecProfile, NetworkConfig
from ..types import NanoReconError
from .device import configure_jax, prepare_environment, select_gpu

log = logging.getLogger("nanorecon")


@dataclass
class EngineTimings:
    load_s: float = 0.0
    compile_s: dict[str, float] = field(default_factory=dict)
    compute_s: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [f"load {self.load_s:.1f} s"]
        for direction, seconds in self.compile_s.items():
            parts.append(f"compile {direction} {seconds:.1f} s")
        for direction, seconds in self.compute_s.items():
            parts.append(f"{direction} {seconds:.1f} s in {self.calls.get(direction, 0)} calls")
        return ", ".join(parts)


class JaxCodecEngine:
    def __init__(
        self,
        network: NetworkConfig,
        profile: CodecProfile,
        weights_path: Path,
        *,
        batch_size: int,
        attention: str | None = None,  # None: the model's backend (cuDNN); tests may pass "xla"
        compilation_cache: Path | None = None,
    ) -> None:
        prepare_environment()
        try:
            import jax
            import jax.numpy as jnp
        except Exception as exc:  # broken CUDA plugin installs fail at import time
            raise NanoReconError(f"cannot initialize JAX: {exc}") from exc
        from .network import NanoReconNet, nearest_codeword
        from .weights import check_structure, load_variables

        configure_jax(compilation_cache)
        self._jax = jax
        self.batch_size = int(batch_size)
        self.chunk_samples = profile.chunk_samples
        self.tokens_per_chunk = profile.tokens_per_chunk
        self.codebook_size = profile.codebook_size
        self.timings = EngineTimings()
        self.device = select_gpu()
        self.attention = attention or network.attention_backend

        t0 = time.perf_counter()
        net = NanoReconNet(network, self.attention)
        sample = jax.ShapeDtypeStruct((1, self.chunk_samples), jnp.float32)
        expected = jax.eval_shape(lambda x: net.init(jax.random.PRNGKey(0), x), sample)
        host_vars = load_variables(Path(weights_path))
        check_structure(host_vars, dict(expected))
        self._variables = jax.device_put(host_vars, self.device)
        del host_vars
        self._codebook = jax.jit(lambda v: net.apply(v, method=NanoReconNet.projected_codebook))(self._variables)
        self._codebook.block_until_ready()
        self.timings.load_s = time.perf_counter() - t0

        block = network.search_chunk_size
        tokens, dim = self.tokens_per_chunk, network.quantizer_dim

        def encode_fn(variables, codebook, x):
            z = net.apply(variables, x, method=NanoReconNet.latents)
            if z.shape[1] != tokens:
                raise ValueError(f"encoder produced {z.shape[1]} tokens, profile says {tokens}")
            idx = nearest_codeword(z.reshape(-1, dim), codebook, block)
            return idx.reshape(x.shape[0], tokens)

        def decode_fn(variables, codebook, idx):
            z_q = jnp.take(codebook, idx.reshape(-1), axis=0).reshape(idx.shape[0], tokens, dim)
            return net.apply(variables, z_q, method=NanoReconNet.reconstruct)

        self._fns = {"encode": jax.jit(encode_fn), "decode": jax.jit(decode_fn)}
        self._compiled: dict[str, object] = {}
        self._input_specs = {
            "encode": jax.ShapeDtypeStruct((self.batch_size, self.chunk_samples), jnp.float32),
            "decode": jax.ShapeDtypeStruct((self.batch_size, tokens), jnp.int32),
        }

    def _executable(self, direction: str):
        compiled = self._compiled.get(direction)
        if compiled is None:
            log.info("compiling %s for batch %d (first use)", direction, self.batch_size)
            t0 = time.perf_counter()
            try:
                compiled = self._fns[direction].lower(self._variables, self._codebook, self._input_specs[direction]).compile()
            except Exception as exc:
                hint = "this GPU/cuDNN may not support cuDNN attention" if direction == "decode" and self.attention == "cudnn" else None
                raise NanoReconError(f"cannot compile {direction}: {exc}", hint=hint) from exc
            self.timings.compile_s[direction] = time.perf_counter() - t0
            self._compiled[direction] = compiled
        return compiled

    def _run(self, direction: str, host_input: np.ndarray) -> np.ndarray:
        executable = self._executable(direction)
        t0 = time.perf_counter()
        device_input = self._jax.device_put(host_input, self.device)
        result = np.asarray(executable(self._variables, self._codebook, device_input))
        self.timings.compute_s[direction] = self.timings.compute_s.get(direction, 0.0) + time.perf_counter() - t0
        self.timings.calls[direction] = self.timings.calls.get(direction, 0) + 1
        return result

    def encode(self, batch: np.ndarray, valid_rows: int) -> np.ndarray:
        if batch.shape != (self.batch_size, self.chunk_samples) or batch.dtype != np.float32:
            raise ValueError(f"encode expects float32{[self.batch_size, self.chunk_samples]}, got {batch.dtype}{list(batch.shape)}")
        return self._run("encode", batch)[:valid_rows].astype(np.uint16)

    def decode(self, codes: np.ndarray, valid_rows: int) -> np.ndarray:
        if codes.shape != (self.batch_size, self.tokens_per_chunk):
            raise ValueError(f"decode expects {[self.batch_size, self.tokens_per_chunk]} codes, got {list(codes.shape)}")
        wave = self._run("decode", codes.astype(np.int32))
        out = np.array(wave[:valid_rows], dtype=np.float32, copy=True)
        if not np.isfinite(out).all():
            raise NanoReconError("decoder produced non-finite samples")
        return out
