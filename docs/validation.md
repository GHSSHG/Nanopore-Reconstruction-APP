# Validation

Results so far and how they were obtained. GPU work runs on the lab server (`ssh
bioinformatics`, host potato): the repository is copied with rsync to `~/Work/nanopore-app`
and run with the server's `~/myenv` (Python 3.12.3; jax/jaxlib/jax-cuda12-plugin 0.8.2,
nvidia-cudnn-cu12 9.10.2.21, flax 0.12.3, numpy 2.3.5, pod5 0.3.35). GPU: NVIDIA
DRIVE-PG199-PROD (A100 class, compute capability 8.0, 32 GB), driver 570.195.03 (CUDA 12.8).

## How to rerun

```bash
rsync -a --exclude __pycache__ nanopore-app/ bioinformatics:Work/nanopore-app/
ssh bioinformatics
cd ~/Work/nanopore-app && export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0
~/myenv/bin/python -m pytest                       # unit tests
~/myenv/bin/python -m nanorecon pull
NANORECON_TRAINING_REPO=~/Work/Nanopore-Reconstruction NANORECON_TEST_POD5=<small real POD5> \
    ~/myenv/bin/python -m pytest tests/gpu -v -s   # GPU tests
~/myenv/bin/python tools/validate.py <real.pod5> --chunks 10000 --batch-sizes 16,64,256 \
    --decompress-batch-sizes 16,64 --report report.json
```

`tools/validate.py` is a development tool (not part of the app output). It sizes subsets by
chunk count, because read lengths differ by two orders of magnitude between experiments.

## Results (2026-10-04, 0.3.0)

**Tests.** Unit tests: 191 passed on the server (and on macOS). GPU tests: 7 passed. Against the
training model (same weights, cuDNN attention): encoder codes 100.0% identical; decoder max
abs difference 5.4e-7 for identical codes (normalized units).

**Datasets** (the first reads of each file, up to about 10 000 chunks):

| Dataset | Reads | Samples | Read length p50 | Short reads (< 8192) | Size vs POD5 |
|---|---:|---:|---:|---:|---:|
| hereditary_cancer_positive_control_2025.11, FC01 `_139` (whole file) | 101 | 1.18 M | 9 268 | 12% | 4.4x smaller |
| chrom_acc_2025.06, PAY22766 `_585` | 107 | 81.0 M | 711 829 | 0% | 6.7x smaller |
| hereditary_cancer_2025.09, FC01 `_123` | 6 906 | 56.0 M | 7 763 | 63% | 4.5x smaller |

Read ids, order, lengths and all Meta were identical after the round trip in every dataset.
Short reads compress less because a chunk always stores 512 codes, however few real samples it
holds.

**Reconstruction error** (pA, mean absolute error; regions as in `tools/validate.py`):

| Dataset | Overall | Interior | Seams (144) | Tail overlap | Read edges | Short reads | MAE / signal std |
|---|---:|---:|---:|---:|---:|---:|---:|
| positive control | 3.24 | 3.40 | 3.94 | 3.09 | 4.06 | 3.09 | 0.16 |
| chrom_acc | 2.63 | 2.63 | 2.91 | 2.71 | 3.59 | - | 0.12 |
| hereditary_cancer | 2.47 | 2.50 | 2.71 | 2.44 | 2.29 | 2.48 | 0.10 |

Per-read MAE (chrom_acc): p5 2.26, p50 2.58, p95 3.22 pA. Seams are 8-16% worse than read
interiors; the 144-sample overlap with `linear_edges_v1` keeps them close. The model's own
quality was evaluated when the model was developed; these numbers show that the app reproduces
it on real data.

**Speed** (one GPU; model load 1.4 s, compile 5 s encode + 4 s decode at batch 16, 7 s each at
batch 256, without the compilation cache):

| | batch 16 | batch 64 | batch 256 |
|---|---:|---:|---:|
| compress, chunks/s (chrom_acc) | 330 | 738 | 1 087 |
| compress, share of wall time on the GPU | 97% | 95% | 94% |
| decompress, Msamples/s, long reads (chrom_acc) | 5.2 (7% padding rows) | 4.6 (29%) | - |
| decompress, Msamples/s, short reads (hereditary_cancer) | 0.38 (91% padding rows) | - | - |
| peak GPU memory | 1.1 GiB | 1.7 GiB | 4.9 GiB |

Host peak RSS 1.8-2.1 GiB. On a larger chrom_acc subset (3 000 reads, 1.21 G samples, 152 k
chunks) batch-16 compression took 460 s with chunk preparation on the main thread and 469 s with
4 worker threads: the workers brought nothing, the GPU is the bottleneck (the threads were
removed afterwards). The decompression figures above were measured while decompression batched
chunks within one read only, so short-read data wasted most decoder work on padding rows; it
now packs chunks of consecutive reads into full batches (only the last batch of a file is
padded), and its speed on the GPU has not been measured since.

**Codes and batch size.** For a fixed batch size results are deterministic (repeated runs and
different worker counts give byte-identical files). Across batch sizes the GPU uses different
kernels, so a few near-tie codeword choices flip: 0.60% of codes differ between batch 16 and 64,
0.78% between 16 and 256 (chrom_acc), per-chunk normalization is identical, and the
reconstruction error is unchanged (2.6341 vs 2.6340 pA). The same applies across GPU models or
library versions.

## Not run

* **GPU 0 fell off the bus** (Xid 79, most likely overheating) at 16:02 during the
  hereditary_cancer batch-64 decompression, after about 45 minutes of heavy load with short
  breaks; the server needs a reboot.
* Further runs were judged unnecessary: the teloseq dataset, whole-file CLI runs with SIGTERM,
  and a speed comparison with the training code. Model quality evaluation (e.g. basecalling)
  is outside the app's scope.
