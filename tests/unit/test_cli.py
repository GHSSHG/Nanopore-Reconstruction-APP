"""CLI surface: options, exit codes, messages; model commands never import the JAX runtime."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import zipfile
from dataclasses import replace

import pytest

from nanorecon import __version__
from nanorecon.cli import main
from nanorecon.io.container import ContainerWriter, FileHeader
from nanorecon.model_config import MODEL, PROFILE, REVISION
from nanorecon.types import ExitCode

from ..helpers import write_pod5
from .test_models import hf_cache, put_model  # noqa: F401  (fixture)


def run(capsys, *argv):
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_model_commands_do_not_import_jax(tmp_path):
    code = (
        "import sys\n"
        "from nanorecon.cli import main\n"
        "rc = main(['ls'])\n"
        "main(['--version'])\n"
        "assert 'jax' not in sys.modules and 'flax' not in sys.modules, 'runtime imported'\n"
        "sys.exit(rc)\n"
    )
    env = {**os.environ, "HF_HUB_CACHE": str(tmp_path)}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "not downloaded" in proc.stdout


def test_version_and_help(capsys):
    code, out, _ = run(capsys, "--version")
    assert code == 0 and out.strip() == f"nanorecon {__version__}"
    code, out, _ = run(capsys, "compress", "--help")
    assert code == 0 and "--batch-size" in out and "default 64" in out


@pytest.mark.parametrize(
    "argv",
    [
        ("compress", "x.pod5"),  # missing -o
        ("compress", "x.pod5", "-o", "y", "--batch-size", "0"),
        ("compress", "x.pod5", "-o", "y", "--no-such-option"),
        ("frobnicate",),
    ],
)
def test_usage_errors_exit_2(capsys, argv):
    code, out, err = run(capsys, *argv)
    assert code == ExitCode.USAGE and out == "" and "error" in err


def test_no_command_prints_help(capsys):
    code, out, err = run(capsys)
    assert code == ExitCode.USAGE and "COMMAND" in err and out == ""


def test_compress_missing_input(capsys, tmp_path):
    code, _, err = run(capsys, "compress", str(tmp_path / "nope.pod5"), "-o", str(tmp_path / "o.nrpod"))
    assert code == ExitCode.ERROR and "does not exist" in err


def test_compress_without_model_hints_pull(capsys, tmp_path, hf_cache):  # noqa: F811
    pod5_path = tmp_path / "in.pod5"
    write_pod5(pod5_path, [100])
    code, _, err = run(capsys, "compress", str(pod5_path), "-o", str(tmp_path / "o.nrpod"))
    assert code == ExitCode.ERROR and "not downloaded" in err and "nanorecon pull" in err
    assert not (tmp_path / "o.nrpod").exists()


def test_decompress_rejects_legacy_zip_and_pod5(capsys, tmp_path):
    legacy = tmp_path / "old.nrpod"
    with zipfile.ZipFile(legacy, "w") as z:
        z.writestr("manifest.json", "{}")
    code, _, err = run(capsys, "decompress", str(legacy), "-o", str(tmp_path / "o.pod5"))
    assert code == ExitCode.ERROR and "legacy" in err
    pod5_path = tmp_path / "in.pod5"
    write_pod5(pod5_path, [10])
    code, _, err = run(capsys, "decompress", str(pod5_path), "-o", str(tmp_path / "o.pod5"))
    assert code == ExitCode.ERROR and "compress" in err


def token_file(path, model=MODEL, profile=PROFILE):
    buf = io.BytesIO()
    ContainerWriter(buf, FileHeader(model, profile, 0, "test")).finish()
    path.write_bytes(buf.getvalue())
    return path


def test_decompress_checks_the_files_model(capsys, tmp_path, hf_cache):  # noqa: F811
    other = token_file(tmp_path / "other.nrpod", model=replace(MODEL, revision="2" * 40))
    code, _, err = run(capsys, "decompress", str(other), "-o", str(tmp_path / "o.pod5"))
    assert code == ExitCode.ERROR and "2" * 40 in err and REVISION in err
    odd = token_file(tmp_path / "odd.nrpod", profile=replace(PROFILE, overlap_samples=256, hop_samples=7936))
    code, _, err = run(capsys, "decompress", str(odd), "-o", str(tmp_path / "o.pod5"))
    assert code == ExitCode.ERROR and "profile" in err
    ok = token_file(tmp_path / "ok.nrpod")
    code, _, err = run(capsys, "decompress", str(ok), "-o", str(tmp_path / "o.pod5"))
    assert code == ExitCode.ERROR and "nanorecon pull" in err  # passes the checks, then needs the model


def test_ls_and_info(capsys, hf_cache):  # noqa: F811
    code, out, _ = run(capsys, "ls")
    assert code == 0 and "not downloaded" in out
    code, _, err = run(capsys, "info")
    assert code == ExitCode.ERROR and "nanorecon pull" in err
    put_model(hf_cache)
    code, out, _ = run(capsys, "ls")
    assert code == 0 and REVISION in out and "ready" in out
    code, out, _ = run(capsys, "info")
    assert code == 0 and "overlap_samples" in out and "144" in out and "not used" in out


def test_output_already_exists(capsys, tmp_path, hf_cache):  # noqa: F811
    put_model(hf_cache)  # the check happens before the GPU runtime is loaded
    token = token_file(tmp_path / "t.nrpod")
    existing = tmp_path / "back.pod5"
    existing.write_bytes(b"keep")
    code, _, err = run(capsys, "decompress", str(token), "-o", str(existing))
    assert code == ExitCode.ERROR and "already exists" in err and "--force" in err
    assert existing.read_bytes() == b"keep"
