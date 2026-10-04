"""JAX process setup and GPU selection, kept in one place.

The environment must be prepared before the first `import jax`: GPU memory is not
preallocated and XLA/matmul settings mirror the training run. The first visible GPU is used;
choose another with CUDA_VISIBLE_DEVICES. There is no CPU fallback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..types import NanoReconError

_TRAINING_XLA_FLAGS = ("--xla_gpu_autotune_level=2", "--xla_gpu_enable_triton_gemm=false")
MATMUL_PRECISION = "high"  # training sets jax_default_matmul_precision="high"


def prepare_environment() -> None:
    if "jax" in sys.modules:
        return  # already configured by an earlier engine in this process (tests)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    flags = os.environ.get("XLA_FLAGS", "")
    for flag in _TRAINING_XLA_FLAGS:
        name = flag.split("=")[0]
        if name not in flags:
            flags = f"{flags} {flag}".strip()
    os.environ["XLA_FLAGS"] = flags


def configure_jax(compilation_cache: Path | None) -> None:
    import jax

    jax.config.update("jax_default_matmul_precision", MATMUL_PRECISION)
    if compilation_cache is not None:
        try:
            compilation_cache.mkdir(parents=True, exist_ok=True)
            jax.config.update("jax_compilation_cache_dir", str(compilation_cache))
        except OSError:
            pass  # the cache only saves compile time


def select_gpu():
    import jax

    try:
        return jax.devices("gpu")[0]
    except (RuntimeError, IndexError) as exc:
        raise NanoReconError(
            f"no CUDA GPU is available to JAX ({exc})".strip(),
            hint="run on a Linux host with an NVIDIA GPU and install `nanorecon[cuda12]`",
        ) from exc
