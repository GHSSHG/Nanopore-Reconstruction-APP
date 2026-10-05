"""nanorecon command line: compress, decompress, ls-remote, pull, ls, info.

stdout gets a one-line result; logs, progress and errors go to stderr. Model management
commands never import the JAX runtime; compress/decompress import it only after cheap checks.
Exit codes: 0 ok, 1 error, 2 usage, 130 interrupted.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .model_config import CONFIG_FILE, MODEL, PROFILE, REPO_ID, REVISION, WEIGHTS_FILE
from .types import Cancelled, ExitCode, NanoReconError, UsageError

log = logging.getLogger("nanorecon")

BATCH_SIZE = 64  # chunks per GPU call; ~1.4 GiB of GPU memory (256: ~3.4 GiB, faster)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # report through main() instead of sys.exit inside parsing
        raise UsageError(f"{self.prog}: {message}", hint=f"see `{self.prog} --help`")


def _int_at_least(lo: int):
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError("must be an integer") from None
        if value < lo:
            raise argparse.ArgumentTypeError(f"must be at least {lo}")
        return value

    return parse


def build_parser() -> argparse.ArgumentParser:
    common = _Parser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="debug logging, tracebacks on errors")

    parser = _Parser(prog="nanorecon", description="Compress POD5 signal into NanoRecon codebook tokens and reconstruct POD5 from them.")
    parser.add_argument("--version", action="version", version=f"nanorecon {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", parser_class=_Parser)

    p = sub.add_parser("compress", parents=[common], help="POD5 -> token file")
    p.add_argument("input", type=Path, help="input .pod5")
    p.add_argument("-o", "--output", type=Path, required=True, help="output token file (.nrpod)")
    p.add_argument("--batch-size", type=_int_at_least(1), default=BATCH_SIZE, help=f"chunks per GPU call (default {BATCH_SIZE})")
    p.add_argument("--force", action="store_true", help="replace an existing output (after the new one is complete)")

    p = sub.add_parser("decompress", parents=[common], help="token file -> reconstructed POD5")
    p.add_argument("input", type=Path, help="input token file (.nrpod)")
    p.add_argument("-o", "--output", type=Path, required=True, help="output .pod5")
    p.add_argument("--batch-size", type=_int_at_least(1), default=BATCH_SIZE, help=f"chunks per GPU call (default {BATCH_SIZE})")
    p.add_argument("--force", action="store_true", help="replace an existing output (after the new one is complete)")

    sub.add_parser("ls-remote", parents=[common], help="show the model on the Hugging Face Hub")
    sub.add_parser("pull", parents=[common], help="download the model into the Hugging Face cache")
    sub.add_parser("ls", parents=[common], help="show whether the model is downloaded")
    sub.add_parser("info", parents=[common], help="show the model's network and inference profile")
    return parser


def _setup_logging(verbose: bool) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("nanorecon: %(message)s"))
    log.handlers[:] = [handler]
    log.propagate = False
    log.setLevel(logging.DEBUG if verbose else logging.INFO)


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base).expanduser() if base else Path.home() / ".cache") / "nanorecon"


def _check_input_file(path: Path) -> None:
    if not path.is_file():
        raise NanoReconError(f"input {path} does not exist or is not a file")


def _engine_factory(model, batch_size: int, holder: dict):
    def make(*_):
        from .runtime.engine import JaxCodecEngine

        log.info("loading %s@%s on the GPU", REPO_ID, REVISION[:7])
        engine = JaxCodecEngine(model.network, PROFILE, model.weights_path, batch_size=batch_size,
                                compilation_cache=_cache_dir() / "jax")
        holder["engine"] = engine
        return engine

    return make


def _log_timings(holder: dict) -> None:
    if "engine" in holder:
        log.debug("model timings: %s", holder["engine"].timings.summary())


# ------------------------------------------------------------------------------ commands


def cmd_compress(args) -> None:
    from .models import load_local_model
    from .pipeline.compress import CompressOptions, compress_file

    batch = args.batch_size
    _check_input_file(args.input)
    model = load_local_model()
    holder: dict = {}
    report = compress_file(
        args.input, args.output, model=MODEL, profile=PROFILE, make_engine=_engine_factory(model, batch, holder),
        options=CompressOptions(batch_size=batch), force=args.force,
    )
    _log_timings(holder)
    ratio = report.input_bytes / report.output_bytes if report.output_bytes else 0.0
    print(f"{args.output}: {report.reads} reads, {report.chunks} chunks, {report.output_bytes} bytes "
          f"({ratio:.2f}x smaller than the input POD5) in {report.elapsed_s:.1f} s")


def cmd_decompress(args) -> None:
    from .models import load_local_model
    from .pipeline.decompress import DecompressOptions, decompress_file, read_header

    batch = args.batch_size
    _check_input_file(args.input)
    header = read_header(args.input)
    if (header.model.repo_id, header.model.revision) != (REPO_ID, REVISION):
        raise NanoReconError(
            f"{args.input} was encoded with {header.model.repo_id}@{header.model.revision}; "
            f"nanorecon {__version__} decodes files of {REPO_ID}@{REVISION} only"
        )
    if header.model != MODEL or header.profile != PROFILE:
        raise NanoReconError(f"{args.input} records a model description or inference profile that differs from this version's")
    model = load_local_model()
    holder: dict = {}
    report = decompress_file(
        args.input, args.output, make_engine=_engine_factory(model, batch, holder),
        options=DecompressOptions(batch_size=batch), force=args.force,
    )
    _log_timings(holder)
    print(f"{args.output}: {report.reads} reads reconstructed ({report.samples} samples) in {report.elapsed_s:.1f} s")


def cmd_ls_remote(args) -> None:
    from .models import remote_status

    status = remote_status()
    pinned = "available" if status.pinned_available else "NOT AVAILABLE on the Hub"
    print(f"{REPO_ID}  {REVISION}  (used by nanorecon {__version__}, {pinned})")
    if status.main_revision and status.main_revision != REVISION:
        print(f"note: the Hub's main branch now points to {status.main_revision}; this version keeps using {REVISION[:7]}")


def cmd_pull(args) -> None:
    from .models import pull

    model = pull()
    size = model.weights_path.stat().st_size
    print(f"{REPO_ID}@{REVISION[:7]} is ready ({size / 1e6:.1f} MB weights in {model.weights_path.parent})")


def cmd_ls(args) -> None:
    from .models import cached_files

    files = cached_files()
    if all(files.values()):
        size = files[WEIGHTS_FILE].stat().st_size
        print(f"{REPO_ID}  {REVISION}  ready  ({size / 1e6:.1f} MB, {files[WEIGHTS_FILE].parent})")
    else:
        missing = [name for name in (CONFIG_FILE, WEIGHTS_FILE) if files[name] is None]
        print(f"{REPO_ID}  {REVISION}  not downloaded (missing {', '.join(missing)}; run `nanorecon pull`)")


def cmd_info(args) -> None:
    from .models import load_local_model

    model = load_local_model()
    net = model.network
    print(f"{REPO_ID} @ {REVISION}")
    print(f"  files           {model.config_path.parent}")
    print(f"  architecture    {MODEL.architecture} ({MODEL.model_type})")
    print(f"  codebook        {net.codebook_size} x {net.quantizer_dim}, ids stored as {PROFILE.code_dtype}")
    print(f"  encoder         downsample x{net.downsample_factor} -> {PROFILE.tokens_per_chunk} tokens per chunk")
    print(f"  decoder         dim {net.decoder_dim}, {net.decoder_num_layers} ConvNeXt layers, attention {net.attention_backend}")
    print(f"  quantization    exact nearest codeword, no noise (training DiVeQ sigma^2 {net.diveq_sigma2} is not used)")
    print("  inference profile:")
    for key, value in PROFILE.to_json_dict().items():
        print(f"    {key:<22} {value}")


COMMANDS = {
    "compress": cmd_compress,
    "decompress": cmd_decompress,
    "ls-remote": cmd_ls_remote,
    "pull": cmd_pull,
    "ls": cmd_ls,
    "info": cmd_info,
}


def _on_sigterm(signum, frame):
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except UsageError as exc:
        return _report_error(exc)
    except SystemExit as exc:  # --help / --version printed their text
        return int(exc.code or 0)
    if args.command is None:
        parser.print_help(sys.stderr)
        return int(ExitCode.USAGE)
    _setup_logging(args.verbose)
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except ValueError:
        pass  # not the main thread
    try:
        COMMANDS[args.command](args)
    except NanoReconError as exc:
        log.debug("error details", exc_info=True)
        return _report_error(exc)
    except KeyboardInterrupt:
        return _report_error(Cancelled(f"{args.command} interrupted; no output was written"))
    except Exception as exc:
        log.debug("unexpected error", exc_info=True)
        return _report_error(NanoReconError(f"internal error: {exc!r}", hint="rerun with -v for a traceback"))
    return int(ExitCode.OK)


def _report_error(exc: NanoReconError) -> int:
    sys.stderr.write(f"nanorecon: error: {exc.message}\n")
    if exc.hint:
        sys.stderr.write(f"nanorecon: hint: {exc.hint}\n")
    return int(exc.exit_code)
