"""Release weights (flax msgpack): load for inference and compare with the network's variables."""

from __future__ import annotations

import mmap
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..types import NanoReconError


def flatten(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten(value, path))
        else:
            out[path] = value
    return out


def load_variables(path: Path) -> dict[str, Any]:
    """Load the released variables as a nested dict of numpy arrays."""
    from flax.serialization import msgpack_restore

    try:
        with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            tree = msgpack_restore(mapped)
    except Exception as exc:
        raise NanoReconError(f"cannot load weights {path}: {exc}", hint="run `nanorecon pull` again") from exc
    if not isinstance(tree, dict):
        raise NanoReconError(f"weights file {path} does not hold a variable tree")
    return tree


def check_structure(loaded: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Every expected variable present with identical shape/dtype, and nothing extra."""
    have = flatten(loaded)
    want = flatten(expected)
    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    mismatched = [
        f"{k}: file {np.asarray(have[k]).dtype}{list(np.shape(have[k]))} vs model {want[k].dtype}{list(want[k].shape)}"
        for k in sorted(set(want) & set(have))
        if tuple(np.shape(have[k])) != tuple(want[k].shape) or np.asarray(have[k]).dtype != np.dtype(want[k].dtype)
    ]
    if missing or extra or mismatched:
        parts = []
        if missing:
            parts.append(f"missing {len(missing)} (e.g. {missing[:3]})")
        if extra:
            parts.append(f"unexpected {len(extra)} (e.g. {extra[:3]})")
        if mismatched:
            parts.append(f"shape/dtype mismatch {len(mismatched)} (e.g. {mismatched[:3]})")
        raise NanoReconError("weights do not match the network described by config.json: " + "; ".join(parts))
