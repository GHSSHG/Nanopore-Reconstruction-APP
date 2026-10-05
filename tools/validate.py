"""Development validation on real data. Runs on the GPU server; not part of the app.

    python tools/validate.py INPUT.pod5 [--chunks N] [--batch-sizes 16,64,256]
                             [--decompress-batch-sizes 16,64] [--workdir DIR] [--report report.json]

For each batch size a child process loads the model, compresses once and (for the decompress
batch sizes) decompresses once, reporting wall time, GPU wait (time the host spent sending
batches and waiting for results; host work done while the GPU computes is not in it), padding
rows, peak GPU memory and peak host RSS. The parent then checks the first reconstruction against
the input: Meta, order and length, and the pA error by region (read interior, regular seams, the
overlap of the right-aligned last window, read edges, short reads), by read length and by signal
level.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REGIONS = ("interior", "seam", "tail", "edge", "short")
INTERIOR, SEAM, TAIL, EDGE, SHORT = range(5)


# ------------------------------------------------------------------------------ child: one batch size


def run_batch_size(input_path: Path, workdir: Path, batch: int, decompress: bool) -> dict:
    from nanorecon.model_config import MODEL, PROFILE
    from nanorecon.models import load_local_model
    from nanorecon.pipeline.compress import CompressOptions, compress_file
    from nanorecon.runtime.engine import JaxCodecEngine

    model = load_local_model()
    engine = JaxCodecEngine(model.network, PROFILE, model.weights_path, batch_size=batch)  # no compile cache
    engine.encode(np.zeros((batch, PROFILE.chunk_samples), np.float32), batch)
    engine.decode(np.zeros((batch, PROFILE.tokens_per_chunk), np.uint16), batch)
    t = engine.timings
    result = {"batch": batch, "load_s": t.load_s, "compile_s": dict(t.compile_s), "decompress": None}

    out = workdir / f"b{batch}.nrpod"
    g0, c0 = t.compute_s.get("encode", 0.0), t.calls.get("encode", 0)
    start = time.perf_counter()
    report = compress_file(input_path, out, model=MODEL, profile=PROFILE, make_engine=lambda: engine,
                           options=CompressOptions(batch_size=batch), force=True, show_progress=False)
    wall = time.perf_counter() - start
    result["compress"] = {
        "wall_s": wall, "gpu_s": t.compute_s.get("encode", 0.0) - g0, "calls": t.calls.get("encode", 0) - c0,
        "chunks": report.chunks, "samples": report.samples, "msamples_per_s": report.samples / wall / 1e6,
        "output_bytes": report.output_bytes,
    }
    if decompress:
        result["decompress"] = run_decompress(engine, out, workdir / f"b{batch}.pod5", batch)
    stats = engine.device.memory_stats() or {}
    result["gpu_peak_bytes"] = stats.get("peak_bytes_in_use")
    result["host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return result


def run_decompress(engine, token: Path, back: Path, batch: int) -> dict:
    from nanorecon.pipeline.decompress import DecompressOptions, decompress_file

    t = engine.timings
    g0, c0 = t.compute_s.get("decode", 0.0), t.calls.get("decode", 0)
    start = time.perf_counter()
    report = decompress_file(token, back, make_engine=lambda h: engine,
                             options=DecompressOptions(batch_size=batch), force=True, show_progress=False)
    wall = time.perf_counter() - start
    calls = t.calls.get("decode", 0) - c0
    return {
        "wall_s": wall, "gpu_s": t.compute_s.get("decode", 0.0) - g0, "calls": calls, "chunks": report.chunks,
        "samples": report.samples, "msamples_per_s": report.samples / wall / 1e6,
        "padding_share": 1.0 - report.chunks / (calls * batch) if calls else 0.0,
    }


# ------------------------------------------------------------------------------ parent


def environment() -> dict:
    env = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        query = "name,driver_version,memory.total"
        env["nvidia_smi"] = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                                           capture_output=True, text=True, check=True).stdout.strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        env["nvidia_smi"] = repr(exc)
    from importlib.metadata import version

    for pkg in ("jax", "jaxlib", "jax-cuda12-plugin", "nvidia-cudnn-cu12", "flax", "numpy", "pod5"):
        try:
            env[pkg] = version(pkg)
        except Exception:
            env[pkg] = None
    return env


def write_subset(src: Path, dst: Path, max_chunks: int) -> None:
    """The first reads of `src` up to about `max_chunks` chunks (read lengths vary a lot between
    experiments, so the subset is sized by work, not by read count)."""
    from nanorecon.io.pod5_io import Pod5Sink, Pod5Source
    from nanorecon.model_config import PROFILE
    from nanorecon.signal.chunking import chunk_count

    chunks = 0
    with Pod5Source(src) as source, Pod5Sink(dst) as sink:
        for read in source.iter_reads():
            if chunks >= max_chunks:
                break
            sink.write(read.meta, read.load_signal())
            chunks += chunk_count(read.meta.num_samples, PROFILE.chunk_samples, PROFILE.hop_samples)


def region_map(n: int, L: int, H: int, O: int) -> np.ndarray:
    r = np.full(n, INTERIOR, np.uint8)
    if n < L:
        r[:] = SHORT
        return r
    regular = (n - L) // H + 1
    for k in range(1, regular):
        r[k * H : k * H + O] = SEAM
    if (n - L) % H:  # a right-aligned last window overlaps the last regular one
        r[n - L : (regular - 1) * H + L] = TAIL
    r[:O] = EDGE
    r[n - O :] = EDGE
    return r


def quality(input_path: Path, back: Path) -> dict:
    from nanorecon.io.pod5_io import Pod5Source
    from nanorecon.model_config import PROFILE
    from nanorecon.types import read_meta_equal

    L, H, O = PROFILE.chunk_samples, PROFILE.hop_samples, PROFILE.overlap_samples
    abs_sum = np.zeros(len(REGIONS))
    sq_sum = np.zeros(len(REGIONS))
    count = np.zeros(len(REGIONS), np.int64)
    per_read = []  # (length, mae, pA std)
    mismatches = {"meta": 0, "length": 0, "order": 0}
    with Pod5Source(input_path) as a, Pod5Source(back) as b:
        for ra, rb in zip(a.iter_reads(), b.iter_reads(), strict=True):
            mismatches["order"] += ra.meta.read_id != rb.meta.read_id
            mismatches["meta"] += not read_meta_equal(ra.meta, rb.meta)
            x, y = ra.load_signal(), rb.load_signal()
            if x.shape != y.shape:
                mismatches["length"] += 1
                continue
            if x.size == 0:
                continue
            off, scale = float(ra.meta.calibration_offset), float(ra.meta.calibration_scale)
            pa = (x.astype(np.float64) + off) * scale
            err = (y.astype(np.float64) + off) * scale - pa
            regions = region_map(x.size, L, H, O)
            abs_sum += np.bincount(regions, weights=np.abs(err), minlength=len(REGIONS))
            sq_sum += np.bincount(regions, weights=err * err, minlength=len(REGIONS))
            count += np.bincount(regions, minlength=len(REGIONS))
            per_read.append((x.size, float(np.abs(err).mean()), float(pa.std())))
    by_region = {
        name: {"samples": int(count[i]), "mae_pa": abs_sum[i] / count[i], "rmse_pa": float(np.sqrt(sq_sum[i] / count[i]))}
        for i, name in enumerate(REGIONS) if count[i]
    }
    lengths = np.array([p[0] for p in per_read])
    mae = np.array([p[1] for p in per_read])
    std = np.array([p[2] for p in per_read])
    out = {"mismatches": mismatches, "by_region": by_region, "reads_compared": len(per_read)}
    if per_read:
        total = count.sum()
        out["overall_mae_pa"] = float(abs_sum.sum() / total)
        out["per_read_mae_pa_percentiles"] = dict(zip(("p5", "p25", "p50", "p75", "p95"), np.percentile(mae, [5, 25, 50, 75, 95]).round(3).tolist()))
        bins = [(0, 8192), (8192, 50_000), (50_000, 200_000), (200_000, None)]
        out["by_length"] = {
            f"{lo}-{hi or 'inf'}": {"reads": int(m.sum()), "mean_read_mae_pa": float(mae[m].mean())}
            for lo, hi in bins
            if (m := (lengths >= lo) & ((lengths < hi) if hi else True)).any()
        }
        edges = np.percentile(std, [25, 50, 75])
        level = np.digitize(std, edges)
        out["by_signal_std_quartile"] = {
            f"q{q + 1}": {"pa_std_mean": float(std[level == q].mean()), "mean_read_mae_pa": float(mae[level == q].mean())}
            for q in range(4) if (level == q).any()
        }
        out["mae_over_pa_std"] = float(np.mean(mae / np.maximum(std, 1e-9)))
    return out


def code_differences(a: Path, b: Path) -> dict:
    """Token-level comparison of two token files of the same input."""
    from nanorecon.io.container import ContainerReader

    tokens = differing = chunks = chunks_differing = 0
    stats_equal = True
    with open(a, "rb") as fa, open(b, "rb") as fb:
        for (_, x), (_, y) in zip(ContainerReader(fa).iter_reads(), ContainerReader(fb).iter_reads(), strict=True):
            stats_equal &= bool(np.array_equal(x.centers, y.centers) and np.array_equal(x.scales, y.scales))
            for (_, cx), (_, cy) in zip(x.code_batches(256), y.code_batches(256), strict=True):
                d = cx != cy
                tokens += d.size
                differing += int(d.sum())
                chunks += d.shape[0]
                chunks_differing += int(d.any(axis=1).sum())
    return {"tokens": tokens, "differing_tokens": differing, "differing_token_share": differing / max(tokens, 1),
            "chunks_with_differences": chunks_differing, "chunks": chunks, "centers_scales_identical": stats_equal}


def input_stats(path: Path) -> dict:
    from nanorecon.io.pod5_io import Pod5Source
    from nanorecon.model_config import PROFILE
    from nanorecon.signal.chunking import chunk_count

    with Pod5Source(path) as source:
        lengths = np.array([r.meta.num_samples for r in source.iter_reads()], np.int64)
    chunks = np.array([chunk_count(int(n), PROFILE.chunk_samples, PROFILE.hop_samples) for n in lengths])
    return {
        "bytes": path.stat().st_size, "reads": int(lengths.size), "samples": int(lengths.sum()), "chunks": int(chunks.sum()),
        "length_percentiles": dict(zip(("p5", "p50", "p95"), np.percentile(lengths, [5, 50, 95]).astype(int).tolist())) if lengths.size else {},
        "short_read_share": float((lengths < PROFILE.chunk_samples).mean()) if lengths.size else 0.0,
        "chunks_per_read_mean": float(chunks.mean()) if lengths.size else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("--chunks", type=int, help="use only the first reads, up to about N chunks (written to a subset POD5)")
    parser.add_argument("--batch-sizes", default="16,64,256")
    parser.add_argument("--decompress-batch-sizes", default="16,64")
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/nanorecon-validate"))
    parser.add_argument("--report", type=Path, default=Path("report.json"))
    parser.add_argument("--child", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    decompress_batches = [int(b) for b in args.decompress_batch_sizes.split(",")]

    if args.child:
        print(json.dumps(run_batch_size(args.input, args.workdir, args.child, args.child in decompress_batches)))
        return 0

    args.workdir.mkdir(parents=True, exist_ok=True)
    source = args.input
    if args.chunks:
        source = args.workdir / "input.pod5"
        if source.exists():
            source.unlink()
        write_subset(args.input, source, args.chunks)
    report = {"input": str(args.input), "subset_chunks": args.chunks, "environment": environment(), "input_stats": input_stats(source)}
    print(json.dumps(report["input_stats"]), flush=True)

    runs = []
    for batch in [int(b) for b in args.batch_sizes.split(",")]:
        cmd = [sys.executable, __file__, str(source), "--child", str(batch),
               "--decompress-batch-sizes", args.decompress_batch_sizes, "--workdir", str(args.workdir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            runs.append({"batch": batch, "error": proc.stderr.strip().splitlines()[-5:]})
        else:
            runs.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        print(json.dumps(runs[-1]), flush=True)
    report["runs"] = runs

    done = [r for r in runs if "error" not in r]
    if done:
        first = done[0]["batch"]
        token = args.workdir / f"b{first}.nrpod"
        report["codes_vs_first_batch_size"] = {
            str(r["batch"]): code_differences(token, args.workdir / f"b{r['batch']}.nrpod") for r in done[1:]
        }
        report["compression_ratio"] = report["input_stats"]["bytes"] / done[0]["compress"]["output_bytes"]
        decoded = [r for r in done if r["decompress"]]
        if decoded:
            report["quality"] = quality(source, args.workdir / f"b{decoded[0]['batch']}.pod5")
            report["overall_mae_pa_by_batch"] = {
                str(r["batch"]): (report["quality"]["overall_mae_pa"] if r is decoded[0]
                                  else quality(source, args.workdir / f"b{r['batch']}.pod5")["overall_mae_pa"])
                for r in decoded
            }
    args.report.write_text(json.dumps(report, indent=2))
    print(f"report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
