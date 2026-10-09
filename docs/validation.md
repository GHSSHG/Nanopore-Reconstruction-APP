# Validation

How the app was checked and what was measured.

* **1.0.0** was measured on the lab's A100-class GPU with real data, and before that on an
  NVIDIA RTX 3080 Ti (12 GB), rented while the lab server was down, with simulated data.
* **0.3.0** was measured on the lab's A100-class GPU with the same real data.

## How to rerun

```bash
rsync -a --exclude __pycache__ nanopore-app/ bioinformatics:Work/nanopore-app/
ssh bioinformatics
cd ~/Work/nanopore-app && export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0
~/myenv/bin/python -m pytest                       # unit tests
~/myenv/bin/python -m nanorecon pull
NANORECON_TRAINING_REPO=~/Work/Nanopore-Reconstruction NANORECON_TEST_POD5=<small real POD5> \
    ~/myenv/bin/python -m pytest tests/gpu -v -s   # GPU tests
~/myenv/bin/python tools/validate.py <real.pod5> --chunks 10000 --batch-sizes 16,64,256,512 \
    --decompress-batch-sizes 16,64,256,512 --report report.json
```

`tools/validate.py` is a development tool (not part of the app output). It sizes subsets by
chunk count, because read lengths differ by two orders of magnitude between experiments.

## 1.0.0 (2026-10-09, A100 class, real data)

**Environment.** The lab server (`ssh bioinformatics`, host potato), using `~/myenv`: Python
3.12.3; jax/jaxlib/jax-cuda12-plugin 0.8.2, nvidia-cudnn-cu12 9.10.2.21, flax 0.12.3, numpy
2.3.5, pod5 0.3.35. GPU: NVIDIA DRIVE-PG199-PROD (A100 class, compute capability 8.0, 32 GB),
driver 570.195.03 (CUDA 12.8). One GPU, not shared with other jobs during the runs.

**Tests.** All 193 unit tests and all 6 GPU tests passed. Against the training model (same
weights, cuDNN attention): encoder codes were 100.0% identical; for identical codes, the decoder
output differed by at most 6.9e-5 (normalized units), as on the RTX 3080 Ti.

**GPU arithmetic.** As on the RTX 3080 Ti, cuBLAS (four matmul shapes) and cuDNN (three
convolutions) round fp32 inputs to TF32 to nearest, ties away from zero. The fp16 operands of the
codebook search and the decoder's pointwise layers are rounded the same way, so they equal the
TF32 inputs.

**Codebook search.** At batch 256 the fused search took 69 ms per batch, against 96-161 ms
(two runs) for the blocked XLA search it replaced, with 0.38 instead of 4.14 GiB of temporary GPU
memory and identical codes. Of 18 other kernel tiles the fastest took 65 ms, with 0.63 GiB. The
tile tuned on the RTX 3080 Ti was kept: the faster tile would save about 2% of the compression
time.

**Datasets** (the first reads of each file, up to about 10 000 chunks):

| Dataset | Reads | Samples | Read length p50 | Short reads (< 8192) | Size vs POD5 |
|---|---:|---:|---:|---:|---:|
| hereditary_cancer_positive_control_2025.11, FC01 `_139` (whole file) | 101 | 1.18 M | 9 268 | 12% | 4.4x smaller |
| chrom_acc_2025.06, PAY22766 `_585` | 107 | 81.0 M | 711 829 | 0% | 6.7x smaller |
| hereditary_cancer_2025.09, FC01 `_123` | 6 906 | 56.0 M | 7 763 | 63% | 4.5x smaller |

* Read ids, order, lengths and all Meta were identical after the round trip in every dataset.
* Short reads compress less, because a chunk always stores 512 codes, however few real samples
  it holds.

**Reconstruction error** (pA, mean absolute error, batch 16; regions as in `tools/validate.py`):

| Dataset | Overall | Interior | Seams (144) | Tail overlap | Read edges | Short reads | MAE / signal std |
|---|---:|---:|---:|---:|---:|---:|---:|
| positive control | 3.24 | 3.40 | 3.94 | 3.09 | 4.06 | 3.09 | 0.16 |
| chrom_acc | 2.63 | 2.63 | 2.91 | 2.71 | 3.59 | - | 0.12 |
| hereditary_cancer | 2.47 | 2.50 | 2.71 | 2.44 | 2.29 | 2.48 | 0.10 |

* These are the 0.3.0 figures to two decimals; the overall error changed by at most 0.0004 pA
  between batch sizes.
* Per-read MAE (chrom_acc): p5 2.27, p50 2.58, p95 3.22 pA.
* Seams are 8-16% worse than read interiors; the 144-sample overlap with `linear_edges_v1`
  keeps them close.
* The model's own quality was evaluated when the model was developed. These numbers show that
  the app reproduces it on real data.

**Speed and memory** (`tools/validate.py`; model loaded and compiled before timing; chunks/s,
chrom_acc / hereditary_cancer; the positive control is too small to time):

| | batch 16 | batch 64 | batch 256 | batch 512 |
|---|---:|---:|---:|---:|
| compress, 0.3.0 (chrom_acc) | 330 | 738 | 1 087 | - |
| compress, 1.0.0 | 507 / 491 | 1 080 / 1 011 | 1 564 / 1 442 | 1 644 / 1 553 |
| decompress, 1.0.0 | 873 / 880 | 1 034 / 1 031 | 1 075 / 1 065 | 1 081 / 1 064 |
| peak GPU memory, 1.0.0 | 1.2 GiB | 1.7 GiB | 4.1 GiB | 6.6 GiB |

* Compression is 1.4-1.5x faster than 0.3.0 on this GPU.
* Decompression gains little above batch 64 (4%).
* Against the RTX 3080 Ti below (other data, so only roughly comparable), decompression is
  2.3-2.8x faster, and compression 1.1x at batch 16 up to 1.9x at batch 256.
* Batch 512 compresses 5-8% faster than 256. It runs here with the app's memory settings.
* The GPU was busy 77-96% of the wall time; the rest is host work not hidden behind the GPU,
  most for the short-read data.
* Model load took 1.7 s. Compilation (without the compilation cache) took 6 s per network at
  batch 16, 8.5 s at batch 256 and 11 s at batch 512.
* Host peak RSS was 1.9-2.4 GiB. The GPU reached 66 °C in these runs.

**Codes and batch size.** 0.7-0.9% of codes differ between batch 16 and the larger batches;
batch 256 and 512 gave identical codes. The per-chunk normalization is identical.

## 1.0.0 (2026-10-06, RTX 3080 Ti)

**Environment.** NVIDIA GeForce RTX 3080 Ti (Ampere, 12 GB), driver 595.71.05; Python 3.12;
jax/jaxlib/jax-cuda12-plugin 0.8.2, nvidia-cudnn-cu12 9.27.0.42, flax 0.12.3, numpy 2.3.5,
pod5 0.3.35.

**Tests.** All 193 unit tests passed, on the GPU machine and on macOS. All 5 GPU tests passed.
Against the training model (same weights, cuDNN attention):

* encoder codes were 100.0% identical;
* for identical codes, the decoder output differed by at most 6.9e-5 (normalized units). For
  0.3.0 this was 5.4e-7; the difference comes from the fp16 operands.

**Data.** Two simulated POD5 files with nanopore-like signal (current levels with noise, not
real reads). Their size and error figures say nothing about real data; they serve to compare
versions and batch sizes.

| Dataset | Reads | Samples | Chunks | Short reads (< 8192) | Size vs POD5 |
|---|---:|---:|---:|---:|---:|
| sim_long | 139 | 23.8 M | 3 025 | 0% | 6.5x smaller |
| sim_short | 2 007 | 16.2 M | 3 002 | 62% | 4.3x smaller |

**Compared with 0.3.0** (same files and GPU):

* About 0.01% of codes differ. These come from the fp16 operands of the codebook search; the
  fused search kernel itself gives identical codes for the same latents.
* Decoding identical codes, 0.15-0.17% of samples differ by one ADC unit, never more.
* The reconstruction error is unchanged: overall MAE 2.848-2.849 / 2.872 pA (sim_short / sim_long),
  for both versions and every batch size.
* Read ids, order, lengths and all Meta were identical after the round trip.

For comparison, changing the batch size flips 0.48-0.56% of codes, in both versions.

**Speed and memory** (`tools/validate.py`; model loaded and compiled before timing; chunks/s,
sim_short / sim_long):

| | batch 16 | batch 64 | batch 256 |
|---|---:|---:|---:|
| compress, 0.3.0 | 304 / 315 | 430 / 445 | 474 / 498 |
| compress, 1.0.0 | 441 / 457 | 669 / 699 | 751 / 820 |
| decompress, 0.3.0 | 266 / 279 | 287 / 294 | 288 / 297 |
| decompress, 1.0.0 | 383 / 381 | 399 / 392 | 381 / 383 |
| peak GPU memory, 0.3.0 | 1.0 GiB | 1.7 GiB | 4.9 GiB |
| peak GPU memory, 1.0.0 | 1.0 GiB | 1.4 GiB | 3.4 GiB |

* Compression is 1.45-1.65x faster and decompression 1.29-1.44x faster than in 0.3.0. The
  gains come from these changes:
  * fp16 operands (with TF32-equal inputs) for the codebook search and the decoder's pointwise
    layers;
  * the codebook search fused into one GPU kernel;
  * host work overlapping the GPU;
  * a shorter LSTM loop.
* Decompression does not get faster with larger batches: the decoder's cost per chunk hardly
  depends on the batch size.
* Model load took 2 s. Compilation (without the compilation cache) took 8.5-11 s for the encoder
  and 7.5-10 s for the decoder.
* Host peak RSS was 1.8-2.1 GiB.
* Batch 512 does not run on this 12 GB card: the encoder needs one 4.5 GiB allocation, which
  failed with the app's memory settings (it ran only with all GPU memory allocated up front).

**Reproducibility.** Repeated runs give byte-identical files only while they reuse the same
compiled program (the compilation cache). A fresh compilation lets XLA's autotuner pick other
GPU kernels, and that alone can flip up to about 0.5% of codes, as a change of batch size does.
This applies to 0.3.0 as well.

## 0.3.0 (2026-10-04, A100 class, real data)

**Environment.** The lab server (`ssh bioinformatics`, host potato), using `~/myenv`: Python
3.12.3; jax/jaxlib/jax-cuda12-plugin 0.8.2, nvidia-cudnn-cu12 9.10.2.21, flax 0.12.3, numpy
2.3.5, pod5 0.3.35. GPU: NVIDIA DRIVE-PG199-PROD (A100 class, compute capability 8.0, 32 GB),
driver 570.195.03 (CUDA 12.8).

**Tests.** All 191 unit tests passed on the server (and on macOS). All 7 GPU tests passed.
Against the training model (same weights, cuDNN attention): encoder codes were 100.0%
identical; for identical codes, the decoder output differed by at most 5.4e-7 (normalized
units).

**Data and reconstruction error.** The same three datasets as for 1.0.0 above, with the same
sizes, error figures and identical Meta, order and lengths.

**Speed** (one GPU; model load 1.4 s; compilation without the cache 5 s for the encoder and 4 s
for the decoder at batch 16, 7 s each at batch 256):

| | batch 16 | batch 64 | batch 256 |
|---|---:|---:|---:|
| compress, chunks/s (chrom_acc) | 330 | 738 | 1 087 |
| compress, share of wall time on the GPU | 97% | 95% | 94% |
| peak GPU memory | 1.1 GiB | 1.7 GiB | 4.9 GiB |

* On this GPU, 0.3.0 compressed 1.7-2.2 times faster than on the RTX 3080 Ti above (batch 64 /
  256).
* Decompression was measured before it packed chunks of several reads into one batch, so it is
  not shown here.
* Host peak RSS was 1.8-2.1 GiB.

**Codes and batch size.** Across batch sizes the GPU uses different kernels, so a few near-tie
codeword choices flip:

* 0.60% of codes differ between batch 16 and 64, and 0.78% between 16 and 256 (chrom_acc);
* the per-chunk normalization is identical;
* the reconstruction error is unchanged (2.6341 vs 2.6340 pA).

The same applies across GPU models or library versions.

## Not measured

* Judged unnecessary: the teloseq dataset, whole-file CLI runs with SIGTERM, and a speed
  comparison with the training code. Model quality evaluation (e.g. basecalling) is outside the
  app's scope.
