"""Output transaction: the product is written to a temporary file next to the target and moved
into place (os.replace) only after it is complete and flushed to disk. On failure or
cancellation the temporary file is removed and an existing target is left untouched.

The target is checked once at start-up. If another process creates the same output path while
this run is writing, it is replaced; that race is accepted for this tool.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from ..types import NanoReconError


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


class OutputTransaction:
    def __init__(self, target: Path, *, force: bool, inputs: Sequence[Path] = ()) -> None:
        self.target = Path(os.path.abspath(target))
        self._committed = False
        for source in inputs:
            source = Path(os.path.abspath(source))
            if source == self.target or _same_file(source, self.target):
                raise NanoReconError(f"output {target} is the same file as input {source}")
        if self.target.is_dir():
            raise NanoReconError(f"output {target} is a directory")
        if not self.target.parent.is_dir():
            raise NanoReconError(f"output directory {self.target.parent} does not exist")
        if os.path.lexists(self.target) and not force:
            raise NanoReconError(f"output {target} already exists", hint="pass --force to replace it")
        self.temp_path = self.target.parent / f".{self.target.name}.{os.getpid()}.nanorecon-tmp"

    def commit(self) -> None:
        try:
            fd = os.open(self.temp_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(self.temp_path, self.target)
        except OSError as exc:
            raise NanoReconError(f"cannot move the finished output to {self.target}: {exc}") from exc
        self._committed = True

    def abort(self) -> None:
        if not self._committed:
            try:
                os.unlink(self.temp_path)
            except FileNotFoundError:
                pass

    def __enter__(self) -> "OutputTransaction":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.abort()
