# nanorecon

`nanorecon` compresses the raw signal of an Oxford Nanopore POD5 file into codebook tokens of
the [NanoRecon](https://huggingface.co/GHSSHG/NanoRecon) model, and rebuilds a POD5 file from
those tokens.

* **Lossy signal, lossless bookkeeping.** The reconstructed signal is the model's approximation
  of the original. Read ids, read order, read count, every read's length and the POD5 read and
  run metadata are preserved exactly (see [What is preserved](#what-is-preserved)).
* **One file each way.** `compress` writes one token file (`.nrpod`), `decompress` writes one
  POD5 file. No metrics or side files are produced.
* **One model.** This version uses `GHSSHG/NanoRecon` at commit
  `92be5d5aa05990eccc9edca33c263ee5ffc72739`; every token file records it.

## Requirements

* Linux x86_64 with glibc 2.27 or newer (Ubuntu 18.04+, RHEL/Rocky 8+).
* An NVIDIA GPU of the Ampere generation or newer (A100, RTX 30 series or later) with a driver
  that supports CUDA 12 (525 or newer). CUDA itself is installed with the Python dependencies.
* Python 3.11-3.13 with the `venv` module (Ubuntu 24.04 ships 3.12).
* Network access to PyPI while installing (about 3 GB of dependencies) and to huggingface.co
  for the model (420 MB). About 8 GB of free disk space.

## Install

Download the installer from the
[Releases page](https://github.com/GHSSHG/Nanopore-Reconstruction-APP/releases) and run it:

```bash
curl -LO https://github.com/GHSSHG/Nanopore-Reconstruction-APP/releases/download/v1.0.0/nanorecon-1.0.0-linux-x86_64.sh
bash nanorecon-1.0.0-linux-x86_64.sh
```

The installer (about 70 KB) checks the system, creates its own Python environment in
`~/.local/share/nanorecon/1.0.0`, downloads the dependencies into it, links
`~/.local/bin/nanorecon` and downloads the model. It takes a few minutes; the full output goes
to `~/.local/share/nanorecon/install-1.0.0.log`. If `~/.local/bin` is not on your `PATH`, the
installer says how to add it.

* **Upgrade:** run the installer of the new version. Each version has its own directory and the
  `nanorecon` link moves to the newest; older versions stay usable as
  `~/.local/share/nanorecon/<version>/bin/nanorecon`.
* **Uninstall:** `rm -rf ~/.local/share/nanorecon ~/.local/bin/nanorecon`
* **Manual install:** `bash nanorecon-1.0.0-linux-x86_64.sh --extract DIR` unpacks the wheel;
  install it into any environment with `pip install "DIR/nanorecon-1.0.0-py3-none-any.whl[cuda12]"`.

## Use

```bash
nanorecon compress reads.pod5 -o reads.nrpod
nanorecon decompress reads.nrpod -o reconstructed.pod5
```

| Command | Purpose |
|---|---|
| `compress IN.pod5 -o OUT.nrpod [--batch-size N] [--force]` | POD5 -> token file |
| `decompress IN.nrpod -o OUT.pod5 [--batch-size N] [--force]` | token file -> reconstructed POD5 |
| `pull` | download the model into the Hugging Face cache (the installer already did this) |
| `ls` | whether the model is downloaded, and where |
| `ls-remote` | the model on the Hugging Face Hub, and whether its commit is still available |
| `info` | network summary and the inference rules |

* `--batch-size N`: chunks per GPU call (a chunk is 8 192 samples; default 64). Larger batches
  compress faster and use more GPU memory; see [Recommended settings](#recommended-settings).
  The result does not depend on it in any meaningful way.
* `--force`: replace an existing output, only after the new file is complete.
* `-v`: debug logging, model timings, and a traceback on errors.
* Choose the GPU with `CUDA_VISIBLE_DEVICES`, e.g. `CUDA_VISIBLE_DEVICES=1 nanorecon compress ...`.
  The first visible GPU is used; there is no CPU fallback.

Results go to stdout as one line; logs, progress and errors go to stderr. Exit codes: 0 success,
1 error, 2 usage error, 130 interrupted.

## Recommended settings

Measured with 1.0.0 on the lab's A100-class GPU (32 GB) with real data, about 10 000 chunks
per file (one chunk is 8 192 samples; model loading and compilation not included):

| `--batch-size` | Compression | Decompression | GPU memory |
|---|---|---|---|
| 16 | 490-510 chunks/s | 870-880 chunks/s | 1.2 GiB |
| 64 (default) | 1 010-1 080 chunks/s | 1 030 chunks/s | 1.7 GiB |
| 256 | 1 440-1 560 chunks/s | 1 070 chunks/s | 4.1 GiB |
| 512 | 1 550-1 640 chunks/s | 1 060-1 080 chunks/s | 6.6 GiB |

A 2 GB long-read POD5 (about 290 000 chunks) thus compresses in about 3 minutes at batch 512 and
decompresses in about 5 minutes. On an RTX 3080 Ti (12 GB) compression is about half as fast
and decompression about 2.6 times slower; batch 512 does not fit there. Details:
[docs/validation.md](docs/validation.md).

* **Lab server (32 GB GPUs, 512 GB RAM):** compress with `--batch-size 512`. Decompress with the
  default 64: larger batches make decompression only about 4% faster. Host memory is no
  concern: a run needs about 2.5 GB of RAM.
* **Many files:** the GPU is the bottleneck, so run one process per GPU, each on its own share of
  the files:

  ```bash
  mkdir -p out
  for gpu in 0 1 2 3; do
    (
      for f in $(ls pod5/*.pod5 | awk -v g=$gpu 'NR % 4 == g'); do
        CUDA_VISIBLE_DEVICES=$gpu nanorecon compress "$f" -o "out/$(basename "$f" .pod5).nrpod" --batch-size 512
      done
    ) &
  done
  wait
  ```

* **Smaller or shared GPUs (other jobs on the same card):** batch 512 needs about 7 GiB of GPU
  memory. Batch 256 needs about 4 GiB and compresses 5-8% slower; use the default 64 when less
  is free.
  Batch 512 did not fit on a 12 GB card.
* Long runs keep the GPUs at full load; keep an eye on their temperature (`nvidia-smi`).

## Files and caches

* The model lives in the standard Hugging Face cache (`HF_HOME` / `HF_HUB_CACHE`); `HF_TOKEN`
  and `HTTPS_PROXY` / `ALL_PROXY` (SOCKS included) are honoured by `pull` and `ls-remote`.
  `compress` and `decompress` never access the network; to use a machine without network
  access, copy the cache directory there and point `HF_HOME` at it.
* The JAX compilation cache in `~/.cache/nanorecon/jax` shortens later start-ups (the first run
  of a batch size compiles for about 10 seconds).

## Output files and failures

* An existing output is not replaced without `--force`. Input and output may not be the same
  file (hard and symbolic links included).
* Products are written to a temporary file next to the target and moved into place only when
  complete. On an error, Ctrl-C or SIGTERM the temporary file is removed and an existing output
  stays as it was.
* A token file is only accepted if it is complete (it ends with an END record). Truncated or
  inconsistent files and files of another model or profile are rejected with a message. `.nrpod`
  files of the earlier ZIP-based tool are not readable.

## What is preserved

Per read: read id, signal length, read number, start sample, channel, well, pore type,
calibration offset/scale, median before, end reason and its forced flag, number of MinKNOW
events, tracked/predicted scaling, reads/time since mux change, open pore level (all float
values bit-exact, NaN/inf included). Per run: every RunInfo field, timestamps at POD5's
millisecond precision, context tags and tracking ids with their order and duplicate keys.
Reads keep their order; duplicate read ids are kept as they are; zero-length and short reads
are kept.

Not preserved: the signal samples themselves (reconstructed), and file-level attributes of the
source POD5 (file identifier, writing software - the output names `nanorecon` and its version - and its
internal batching). Inputs must be sampled at 5 kHz; other rates are rejected, not resampled.
Inputs with metadata POD5 cannot write back (null values, unknown end reasons) are rejected at
compress time.

## Documentation

* [docs/format.md](docs/format.md) - token file specification and the inference rules
* [docs/validation.md](docs/validation.md) - test results, speed and GPU memory

## Development

```bash
pip install -e ".[dev]"
pytest                                   # unit tests: no GPU, network or model needed
nanorecon pull && pytest tests/gpu -v    # on a GPU machine
tools/build_installer.sh                 # builds dist/nanorecon-<version>-linux-x86_64.sh for a GitHub Release
```
