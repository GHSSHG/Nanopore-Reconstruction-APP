"""Batch inference engine: encode(float32[B, L]) -> uint16[B, T], decode(uint16[B, T]) -> float32[B, L].

One fixed batch shape is compiled per direction; the caller pads the last batch and only the
first `valid_rows` rows are returned. encode/decode wait for the result. encode_async and
decode_async return as soon as the batch is on its way to the GPU, with a function that waits for
the result, so the host can prepare the next batch meanwhile; the input array must stay unchanged
until that function has returned.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..model_config import CodecProfile, NetworkConfig
from ..types import NanoReconError
from .device import configure_jax, prepare_environment, select_gpu

log = logging.getLogger("nanorecon")


@dataclass
class EngineTimings:
    load_s: float = 0.0
    compile_s: dict[str, float] = field(default_factory=dict)
    compute_s: dict[str, float] = field(default_factory=dict)  # host time spent dispatching and waiting
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
        from .network import NanoReconNet, nearest_codeword, search_codebook
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
        codebook = jax.jit(lambda v: net.apply(v, method=NanoReconNet.projected_codebook))(self._variables)
        self._operands = {"encode": jax.jit(search_codebook)(codebook), "decode": codebook}
        jax.block_until_ready(self._operands)
        self.timings.load_s = time.perf_counter() - t0

        tokens, dim = self.tokens_per_chunk, network.quantizer_dim

        def encode_fn(variables, search, x):
            z = net.apply(variables, x, method=NanoReconNet.latents)
            if z.shape[1] != tokens:
                raise ValueError(f"encoder produced {z.shape[1]} tokens, profile says {tokens}")
            idx = nearest_codeword(z.reshape(-1, dim), search)
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
                compiled = self._fns[direction].lower(self._variables, self._operands[direction], self._input_specs[direction]).compile()
            except Exception as exc:
                hint = "this GPU/cuDNN may not support cuDNN attention" if direction == "decode" and self.attention == "cudnn" else None
                raise NanoReconError(f"cannot compile {direction}: {exc}", hint=hint) from exc
            self.timings.compile_s[direction] = time.perf_counter() - t0
            self._compiled[direction] = compiled
        return compiled

    def _submit(self, direction: str, host_input: np.ndarray) -> Callable[[], np.ndarray]:
        executable = self._executable(direction)
        t0 = time.perf_counter()
        device_result = executable(self._variables, self._operands[direction], self._jax.device_put(host_input, self.device))
        self._account(direction, time.perf_counter() - t0)
        self.timings.calls[direction] = self.timings.calls.get(direction, 0) + 1

        def wait() -> np.ndarray:
            t1 = time.perf_counter()
            result = np.asarray(device_result)
            self._account(direction, time.perf_counter() - t1)
            return result

        return wait

    def _account(self, direction: str, seconds: float) -> None:
        self.timings.compute_s[direction] = self.timings.compute_s.get(direction, 0.0) + seconds

    def encode_async(self, batch: np.ndarray, valid_rows: int) -> Callable[[], np.ndarray]:
        if batch.shape != (self.batch_size, self.chunk_samples) or batch.dtype != np.float32:
            raise ValueError(f"encode expects float32{[self.batch_size, self.chunk_samples]}, got {batch.dtype}{list(batch.shape)}")
        wait = self._submit("encode", batch)
        return lambda: wait()[:valid_rows].astype(np.uint16)

    def decode_async(self, codes: np.ndarray, valid_rows: int) -> Callable[[], np.ndarray]:
        if codes.shape != (self.batch_size, self.tokens_per_chunk):
            raise ValueError(f"decode expects {[self.batch_size, self.tokens_per_chunk]} codes, got {list(codes.shape)}")
        wait = self._submit("decode", codes.astype(np.int32))

        def result() -> np.ndarray:
            out = np.array(wait()[:valid_rows], dtype=np.float32, copy=True)
            if not np.isfinite(out).all():
                raise NanoReconError("decoder produced non-finite samples")
            return out

        return result

    def encode(self, batch: np.ndarray, valid_rows: int) -> np.ndarray:
        return self.encode_async(batch, valid_rows)()

    def decode(self, codes: np.ndarray, valid_rows: int) -> np.ndarray:
        return self.decode_async(codes, valid_rows)()
