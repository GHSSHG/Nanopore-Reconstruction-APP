# NanoRecon token container, format 1.0

A token container (conventionally `.nrpod`) holds everything needed to rebuild a POD5 file
with a NanoRecon model: POD5 metadata, and for every read its per-chunk normalization and
codebook indices. It holds no signal samples, continuous latents, noise or random seeds.
Section 7 describes how reads are cut, normalized, quantized and stitched (the inference
profile recorded in every file).

This is a new format. Files written by the earlier NanoRecon app (a ZIP archive with
`manifest.json`, first bytes `PK\x03\x04`) are recognised and rejected with an explicit
message; they are not migrated.

## 1. Layout

All integers and floats are little-endian. The file is read and written strictly
sequentially; there is no index.

```text
Preamble       16 bytes
FILE_HEADER    record, exactly one, first
( GROUP        record
  READ         record  x GROUP.read_count )*      zero or more groups
END            record, exactly one, last; nothing may follow it
```

### Preamble

| Offset | Size | Field | Value |
|---|---|---|---|
| 0 | 8 | magic | `89 4E 52 45 43 4F 4E 0A` (`\x89NRECON\n`) |
| 8 | 2 | major | 1 |
| 10 | 2 | minor | 0 |
| 12 | 4 | flags | 0 (any other value is rejected) |

Readers accept their own major version and minors up to their own; a newer minor is rejected
("written by a newer version"), a different major is rejected. Adding fields requires a minor
bump at least; changing the meaning of a field requires a major bump.

### Records

Every record is `u32 type`, `u64 length`, then `length` payload bytes.

| Type | Name | Payload | Length limit |
|---|---|---|---|
| 1 | FILE_HEADER | UTF-8 JSON object | 1 MiB |
| 2 | GROUP | UTF-8 JSON object | 4 MiB |
| 3 | READ | binary (section 4) | exactly determined by the read |
| 4 | END | UTF-8 JSON object | 64 KiB |

Unknown record types are an error.

## 2. JSON rules

JSON payloads are strict: valid UTF-8, a single object, no duplicate keys, no `NaN` /
`Infinity` constants, exactly the listed keys (missing or extra keys are errors), integers
where integers are required (booleans are not integers). Writers emit compact, key-sorted
JSON, so identical inputs produce identical files. Individual strings are limited to 1 MiB.

JSON never carries floating-point signal metadata; those fields are binary (section 4), so
NaN and infinities never need a textual encoding.

## 3. FILE_HEADER, GROUP, END

### FILE_HEADER

```json
{
  "format": "nanorecon-tokens",
  "model": {
    "repo_id": "GHSSHG/NanoRecon",
    "revision": "92be5d5aa05990eccc9edca33c263ee5ffc72739",
    "architecture": "SimVQAudioModel",
    "model_type": "nanorecon-simvq-audio-codec",
    "codebook_size": 65536
  },
  "profile": { ...CodecProfile, section 7... },
  "read_count": 4000,
  "producer": {"name": "nanorecon", "version": "0.2.0"}
}
```

* `model.revision` is always a full 40-character commit hash, never a branch name.
* `profile` holds every field of the codec profile (section 7). nanorecon 0.3 decodes only files
  whose model identity and profile equal its own; anything else is rejected with a message.
* `read_count` is the number of READ records in the file.
* No local paths, host names, user names, timestamps or credentials are stored.

### GROUP

```json
{
  "read_count": 812,
  "run_infos":   [ {RunInfo}, ... ],
  "pore_types":  ["not_set", ...],
  "end_reasons": ["signal_positive", ...]
}
```

A group is a run of consecutive reads (the compressor's working set). The three
tables are de-duplicated within the group by value and referenced by index from READ records;
each table has 1..32767 entries. Groups are never empty. Group boundaries carry no meaning
beyond scoping the tables. The compressor closes a group early rather than let its tables
outgrow the 4 MiB record limit (parsed JSON can need ~30x its size in memory, so the limit is
kept small), and it rejects source metadata that no group could hold (a string over 1 MiB, a map
with more than 32767 entries).

`RunInfo` objects have exactly these keys (POD5 RunInfo fields):

| Key | JSON type | Notes |
|---|---|---|
| acquisition_id, experiment_name, flow_cell_id, flow_cell_product_code, protocol_name, protocol_run_id, sample_id, sequencing_kit, sequencer_position, sequencer_position_type, software, system_name, system_type | string | empty strings are kept as empty strings |
| acquisition_start_time_ms, protocol_start_time_ms | integer | milliseconds since the Unix epoch, UTC (POD5 stores `timestamp[ms, tz=UTC]`) |
| adc_max, adc_min | integer | int16 range |
| sample_rate | integer | uint16 range |
| context_tags, tracking_id | array of `[key, value]` string pairs | order and duplicate keys preserved |

### END

```json
{"group_count": 5, "read_count": 4000, "chunk_count": 61233}
```

The reader checks all three totals and `FILE_HEADER.read_count`, then requires end of file.
A file without END (for example an interrupted write) is rejected as truncated; end of file is
never accepted in place of END.

## 4. READ record

```text
meta     104 bytes (table below)
centers  float32[K]       per-chunk normalization center (pA)
scales   float32[K]       per-chunk normalization half range (pA), >= 0
codes    uint16[K * T]    codebook indices, chunk-major
```

`K` must equal the chunk count implied by `num_samples` and the profile (section 7.2), and the record length must be exactly `104 + 8K + 2KT`. Chunk start positions and
valid lengths are derived from `num_samples` and the profile; they are not stored.

| Offset | Type | Field | POD5 source |
|---|---|---|---|
| 0 | 16 bytes | read_id | `read_id` (raw UUID bytes) |
| 16 | u64 | num_samples | `num_samples` (original signal length N) |
| 24 | u64 | num_chunks | K |
| 32 | u32 | read_number | `read_number` |
| 36 | u64 | start_sample | `start` |
| 44 | u16 | channel | `channel` |
| 46 | u8 | well | `well` |
| 47 | u8 | end_reason_forced | `end_reason_forced` (0/1) |
| 48 | f32 | calibration_offset | `calibration_offset` |
| 52 | f32 | calibration_scale | `calibration_scale` (finite, non-zero) |
| 56 | f32 | median_before | `median_before` |
| 60 | u64 | num_minknow_events | `num_minknow_events` |
| 68 | f32 | tracked_scaling_scale | `tracked_scaling_scale` |
| 72 | f32 | tracked_scaling_shift | `tracked_scaling_shift` |
| 76 | f32 | predicted_scaling_scale | `predicted_scaling_scale` |
| 80 | f32 | predicted_scaling_shift | `predicted_scaling_shift` |
| 84 | u32 | num_reads_since_mux_change | `num_reads_since_mux_change` |
| 88 | f32 | time_since_mux_change | `time_since_mux_change` |
| 92 | f32 | open_pore_level | `open_pore_level` |
| 96 | u16 | run_info_index | index into GROUP.run_infos |
| 98 | u16 | pore_type_index | index into GROUP.pore_types |
| 100 | u16 | end_reason_index | index into GROUP.end_reasons |
| 102 | u16 | reserved | 0 |

Float fields are stored as the original float32 values, so NaN, infinities and -0.0 survive
exactly (NaN payload bits may be canonicalized). Reads of length 0 are stored with K = 0 and
reproduced as empty reads. Read ids are stored per read in input order; duplicates are not
merged. All numeric POD5 read columns of read table version 4 are covered; a POD5 column with
null values is rejected at compress time because it could not be written back.

Readers check before allocating per-chunk arrays: known record type, JSON limits, that a
group's read count fits the header's total, N bound (2^40), K against the profile, exact record
length, table indices, reserved bytes, then finiteness of centers/scales and the codebook range
of codes. Strings must be valid Unicode (JSON escapes of lone surrogates are rejected).

## 5. What the checks do not cover

The structural checks detect truncation, wrong versions, inconsistent lengths, invalid
references and out-of-range values. There is intentionally no content checksum: a byte flip
that keeps the structure valid (for example inside a code) is not detected.

## 6. Recognising other files

| First bytes | Meaning | Reader behaviour |
|---|---|---|
| `\x89NRECON\n` | this format | read |
| `PK\x03\x04` | legacy NanoRecon ZIP container | "legacy format, re-create from the original POD5" |
| `\x8bPOD\r\n\x1a\n` | POD5 | "this is a POD5 file, did you mean compress?" |
| anything else | unknown | "not a NanoRecon token file" |

## 7. Inference profile

The profile is the complete set of rules that turn a read into model inputs and model outputs
back into a read. It is fixed in the app for the one supported model, printed by
`nanorecon info`, and stored in every token file.

### 7.1 Fields

| Field | Value | Meaning |
|---|---|---|
| sample_rate_hz | 5000 | Required input sample rate. Other rates are rejected; nothing is resampled. |
| chunk_samples (L) | 8192 | Model input length. |
| overlap_samples (O) | 144 | Overlap of regular neighbouring windows. |
| hop_samples (H) | 8048 | L - O. |
| tokens_per_chunk (T) | 512 | Codes per chunk (encoder downsampling 16). |
| codebook_size | 65536 | Number of codewords. |
| code_dtype | uint16 | Storage type of a code. |
| normalization | minmax_pm1 | Per-chunk min-max to [-1, 1] (7.3). |
| normalization_epsilon | 1e-6 | Below this half range a chunk counts as constant. |
| padding | reflect | Right padding of windows shorter than L. |
| tail | shift_last | Placement of the last window (7.2). |
| stitch | linear_edges_v1 | Weighted merge of overlapping windows (7.5). |
| quantization | hard_v1 | Exact nearest codeword, no noise (7.4). |
| adc_conversion | rint_clip_int16 | pA -> ADC after stitching (7.5). |

The released `config.json` states only the network (and sample rate, chunk length, codebook
size); the other values are this app's inference contract for that release. Normalization,
padding and tail follow the training data pipeline; the 144-sample overlap and hard
quantization are app decisions.

### 7.2 Windows (`tail = shift_last`)

For a read of N samples:

* N >= L: regular starts 0, H, 2H, ... while start + L <= N. If the last regular window
  does not end at N, one more window starts at N - L (never duplicated when it would coincide).
  K = 1 + ceil((N - L) / H).
* 0 < N < L: one window at 0 with N real samples.
* N = 0: no windows (K = 0); the read is still stored and reproduced with an empty signal.

The right-aligned last window can overlap its neighbour by far more than O, and up to three
windows can cover the same samples (N = 16300: starts 0, 8048, 8108). There is no minimum read
length: short reads are never dropped (the training-time length filter does not apply).

### 7.3 Calibration and normalization (`minmax_pm1`, `reflect`)

Per chunk, on the real samples only (float32 arithmetic):

```text
pA     = (adc + float32(calibration_offset)) * float32(calibration_scale)
center = float32((min(pA) + max(pA)) / 2)          computed from the float32 extrema
scale  = float32((max(pA) - min(pA)) / 2)
y      = clip((pA - center) / scale, -1, 1)        if scale >= epsilon
y      = 0                                          otherwise (constant chunk)
```

Then, for a window with n < L real samples, positions n..L-1 are filled by reflection without
repeating the edge (index pattern 0,1,..,n-1,n-2,..,1,0,1,..); a single sample is repeated.
Padding never enters the statistics. Invalid calibration (non-finite, zero scale) is rejected.

center and scale are stored per chunk as float32. Denormalization: `pA = y * scale + center`,
or `pA = center` when `scale < epsilon`.

### 7.4 Quantization (`hard_v1`)

```text
projected_codebook = base_codebook @ W + proj_bias          (vq/quantizer/codebook, params/quantizer/*)
compress:   latents (B, T, D) -> index of the nearest projected codeword (squared L2)
decompress: index -> projected_codebook[index] -> post_quant_conv -> decoder
```

The search computes `|z|^2 + |e|^2 - 2 z.e` as in training and checks every codeword; ties go to
the lowest index. One GPU kernel computes the distances tile by tile (256 codewords) and keeps only
each row's minimum, so the full distance matrix is never stored (the training config's
`search_chunk_size` is not used). The dot products take fp16 operands and sum in
fp32: each latent and the codebook are first scaled by a power of two and rounded to nearest with
ties away from zero, which is how the GPU forms the TF32 operands used in training, so the operands
are the same and only the summation order differs. Norms, sums and comparisons are fp32. The
training config's `diveq_sigma2 = 0.001` is not used: inference has no noise, dropout or random
keys.

### 7.5 Stitching (`linear_edges_v1`) and ADC conversion (`rint_clip_int16`)

Each decoded chunk is denormalized to pA on its own and weighted over its real samples by

```text
w(t) = min(1, (t + 1) / (O + 1), (L - t) / (O + 1)),   t = 0 .. L - 1
```

Contributions and weights are summed per sample and the sum is divided by the actual weight
sum, so every covered sample has a positive total no matter how many windows overlap, and a
sample covered once reproduces its chunk exactly.

The merged pA signal is converted once: `adc = clip(rint(pA / scale - offset), -32768, 32767)`
in float64 (round half to even), using the read's own calibration. Chunks are never converted
to integers before merging. A future change of the weighting gets a new stitch name.

### 7.6 Numerics

* Matmul precision `high` (as in training): TF32 inputs, fp32 sums. Two exceptions take fp16
  operands (the same 10-bit mantissa as TF32, scaled by powers of two, fp32 sums and outputs):
  the codebook search (7.4) and the decoder's ConvNeXt pointwise layers, whose input scales follow
  from the weights so they cannot overflow. Compared with TF32 throughout, about 0.01% of codes and
  0.15-0.17% of reconstructed samples (by one ADC unit) changed on an RTX 3080 Ti, less than a
  change of batch size causes. The decoder attention uses cuDNN with bfloat16 q/k/v, as in
  training.
* The ISTFT overlap-add is computed deterministically (shifted segment sums) instead of the
  training code's scatter-add; for identical codes the decoder output differed from the training
  model by at most ~7e-7 (normalized units, measured on CPU).
* The codes do not depend on which chunks share a batch (padding rows never change real rows).
  Repeated runs give identical files as long as they reuse the same compiled program (the JAX
  compilation cache in `~/.cache/nanorecon/jax`). A new compilation lets XLA's autotuner choose the
  fastest GPU kernels again; kernels that sum in a different order can flip near-tie codewords.
  Between separate compilations for the same batch size up to about 0.5% of codes differed (RTX
  3080 Ti), between batch 16, 64 and 256 0.6-0.8% (A100-class GPU), with unchanged reconstruction
  error. Batch size, GPU model and library versions act the same way. The decoder is affected
  alike: the same codes decoded with batch 64 and with batch 256 differed in about 1% of the output
  samples by one ADC unit.
