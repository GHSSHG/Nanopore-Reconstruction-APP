"""Getting the one supported model (model_config.REPO_ID at REVISION).

`pull` downloads config.json and the weights into the standard Hugging Face cache (HF_HOME,
HF_TOKEN and proxy settings apply; interrupted downloads resume). Everything else finds the
files in that cache without network access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .model_config import CONFIG_FILE, REPO_ID, REVISION, WEIGHTS_FILE, NetworkConfig, parse_network_config
from .types import NanoReconError

PULL_HINT = "run `nanorecon pull` once with network access"


@dataclass(frozen=True)
class LocalModel:
    config_path: Path
    weights_path: Path
    network: NetworkConfig


def _load(config_path: Path, weights_path: Path) -> LocalModel:
    with open(config_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)
    return LocalModel(config_path, weights_path, parse_network_config(config["model"]))


def cached_files() -> dict[str, Path | None]:
    """Where the model files are in the HF cache (None: not downloaded)."""
    from huggingface_hub import try_to_load_from_cache

    found = {}
    for name in (CONFIG_FILE, WEIGHTS_FILE):
        path = try_to_load_from_cache(REPO_ID, name, revision=REVISION)
        found[name] = Path(path) if isinstance(path, str) else None
    return found


def load_local_model() -> LocalModel:
    files = cached_files()
    if files[CONFIG_FILE] is None or files[WEIGHTS_FILE] is None:
        raise NanoReconError(f"model {REPO_ID}@{REVISION[:7]} is not downloaded", hint=PULL_HINT)
    return _load(files[CONFIG_FILE], files[WEIGHTS_FILE])


def _first_line(exc: Exception) -> str:
    text = str(exc).strip().splitlines()
    return text[0] if text else exc.__class__.__name__


def pull() -> LocalModel:
    from huggingface_hub import hf_hub_download

    paths = []
    for name in (CONFIG_FILE, WEIGHTS_FILE):
        try:
            paths.append(Path(hf_hub_download(REPO_ID, name, revision=REVISION)))
        except Exception as exc:
            raise NanoReconError(
                f"cannot download {name} of {REPO_ID}@{REVISION[:7]}: {_first_line(exc)}",
                hint="check network access to huggingface.co (HTTPS_PROXY/ALL_PROXY are honoured)",
            ) from exc
    return _load(*paths)


@dataclass(frozen=True)
class RemoteStatus:
    main_revision: str | None  # commit the Hub's main branch points to
    pinned_available: bool  # REVISION can still be downloaded


def remote_status() -> RemoteStatus:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RevisionNotFoundError

    api = HfApi()
    try:
        refs = api.list_repo_refs(REPO_ID)
        main = next((b.target_commit for b in refs.branches if b.name == "main"), None)
        try:
            api.model_info(REPO_ID, revision=REVISION)
            pinned = True
        except RevisionNotFoundError:
            pinned = False
    except Exception as exc:
        raise NanoReconError(f"cannot query {REPO_ID} on the Hugging Face Hub: {_first_line(exc)}",
                             hint="check network access to huggingface.co") from exc
    return RemoteStatus(main, pinned)
