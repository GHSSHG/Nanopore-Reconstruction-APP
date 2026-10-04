"""Rate-limited progress lines on stderr (through logging). Never writes to stdout."""

from __future__ import annotations

import logging
import time

log = logging.getLogger("nanorecon")


class Progress:
    def __init__(self, label: str, total_reads: int, *, interval_s: float = 5.0) -> None:
        self.label = label
        self.total = total_reads
        self.interval = interval_s
        self._start = time.perf_counter()
        self._last = self._start

    def update(self, reads: int, samples: int) -> None:
        now = time.perf_counter()
        if now - self._last < self.interval:
            return
        self._last = now
        self._emit(reads, samples, now)

    def finish(self, reads: int, samples: int) -> None:
        self._emit(reads, samples, time.perf_counter(), final=True)

    def _emit(self, reads: int, samples: int, now: float, final: bool = False) -> None:
        elapsed = max(now - self._start, 1e-9)
        share = f" ({100.0 * reads / self.total:.0f}%)" if self.total else ""
        log.info(
            "%s: %s%d/%d reads%s, %.2f Msamples/s",
            self.label, "done, " if final else "", reads, self.total, share, samples / elapsed / 1e6,
        )
