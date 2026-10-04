"""Output transaction: temporary file next to the target, replace on commit, cleanup, refusals."""

import os

import pytest

from nanorecon.io.output import OutputTransaction
from nanorecon.types import NanoReconError


def leftovers(directory):
    return sorted(p.name for p in directory.iterdir() if p.name.endswith(".nanorecon-tmp"))


def test_commit_creates_target(tmp_path):
    target = tmp_path / "out.nrpod"
    with OutputTransaction(target, force=False) as tx:
        assert tx.temp_path.parent == tmp_path
        tx.temp_path.write_bytes(b"new")
        assert not target.exists()
        tx.commit()
    assert target.read_bytes() == b"new" and leftovers(tmp_path) == []


def test_existing_target_refused_without_force(tmp_path):
    target = tmp_path / "out.nrpod"
    target.write_bytes(b"old")
    with pytest.raises(NanoReconError, match="already exists") as info:
        OutputTransaction(target, force=False)
    assert "--force" in info.value.hint
    assert target.read_bytes() == b"old"


def test_force_replaces_only_on_commit(tmp_path):
    target = tmp_path / "out.nrpod"
    target.write_bytes(b"old")
    with OutputTransaction(target, force=True) as tx:
        tx.temp_path.write_bytes(b"new")
        assert target.read_bytes() == b"old"
        tx.commit()
    assert target.read_bytes() == b"new"


def test_failure_keeps_old_output_and_cleans_up(tmp_path):
    target = tmp_path / "out.nrpod"
    target.write_bytes(b"old")
    with pytest.raises(RuntimeError):
        with OutputTransaction(target, force=True) as tx:
            tx.temp_path.write_bytes(b"partial")
            raise RuntimeError("boom")
    assert target.read_bytes() == b"old" and leftovers(tmp_path) == []


def test_same_file_refused(tmp_path):
    src = tmp_path / "in.pod5"
    src.write_bytes(b"data")
    with pytest.raises(NanoReconError, match="same file"):
        OutputTransaction(src, force=True, inputs=[src])
    link = tmp_path / "hard.pod5"
    os.link(src, link)
    with pytest.raises(NanoReconError, match="same file"):
        OutputTransaction(link, force=True, inputs=[src])
    sym = tmp_path / "sym.pod5"
    sym.symlink_to(src)
    with pytest.raises(NanoReconError, match="same file"):
        OutputTransaction(sym, force=True, inputs=[src])
    assert src.read_bytes() == b"data"


def test_bad_targets(tmp_path):
    with pytest.raises(NanoReconError, match="directory"):
        OutputTransaction(tmp_path, force=True)
    with pytest.raises(NanoReconError, match="does not exist"):
        OutputTransaction(tmp_path / "missing" / "x.nrpod", force=False)


def test_commit_without_product_fails(tmp_path):
    tx = OutputTransaction(tmp_path / "x.nrpod", force=False)
    with pytest.raises(NanoReconError, match="cannot move"):
        tx.commit()
    tx.abort()
    assert leftovers(tmp_path) == [] and not (tmp_path / "x.nrpod").exists()
