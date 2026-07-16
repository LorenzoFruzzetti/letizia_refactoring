# REFERENCE — Technical map of `wfci`

Canonical technical reference for the repository. See [README.md](README.md)
for usage and [CLAUDE.md](CLAUDE.md) for operating conventions.

## Data conventions

- Arrays are `[y, x, time]` (2-D+time) or `[y, x, time, trial]` (4-D), matching
  MATLAB's `(rows, cols, frames, trials)`.
- All computation is float64 (MATLAB `double`). 16-bit TIFFs are promoted to
  float64 on load.
- ROI column order everywhere: `[Laterale_L, Verme_L, Laterale_R, Verme_R]`.

## Module reference (`src/wfci/`)

The two run axes are orthogonal: **storage format** (how channels are stored on
disk) and **memory strategy** (full-load vs streaming). Each `load_*` function is
the full-load reader for one storage format; each `*_frame_source` is the
streaming reader for the same format, so any format can be run either way.

- `load_stack(path) -> [y, x, time]` — multi-page TIFF via `tifffile` (pages =
  time). Single-page files load as one frame. Loads the whole stack into RAM.
- `load_frame_folder(folder, pattern="*.tif") -> [y, x, time]` — assembles a
  folder of single-page TIFFs sorted by filename into a time stack.
- `interleaved_channel_files(folder, pattern="*.tif", top_percent=10.0) ->
  (gcamp_files, emo_files)` — the split-and-identify half of the interleaved
  layout, **without loading frames**: sorted files split into odd- vs
  even-positioned groups; the brighter group (by `_top_percent_mean` of the first
  image in each group — only those two images are read) becomes `gcamp`, the
  dimmer `emo`; the two file lists are truncated to equal length. Shared by the
  full-load and streaming interleaved paths; the odd/even split it uses is what
  `src/inspect_channel_intensity.py` inspects.
- `load_interleaved_folder(folder, pattern="*.tif", top_percent=10.0) ->
  (gcamp, emo)` — full-load variant: `interleaved_channel_files` + stack each
  group into `[y, x, time]`.
- `tiff_frame_count(path) -> int` — page count from the IFD headers only (no
  pixel data read); used to size the streaming baseline window cheaply.
- `iter_tiff_frames(path) -> Iterator[[y, x]]` — lazy, one-frame-at-a-time
  reader (float64); the constant-memory alternative to `load_stack`.
- `iter_folder_frames(files) -> Iterator[[y, x]]` — lazy per-file reader for an
  ordered list of single-page TIFFs (the folder counterpart of
  `iter_tiff_frames`).
- `FrameSource(count, _open)` — a **re-iterable** frame sequence (`count` known
  without decoding pixels; `open()` returns a fresh iterator per pass). This is
  the storage-agnostic input the streaming pipeline consumes, decoupling storage
  format from the streaming math. Constructors:
  - `tiff_frame_source(path)` — one multi-page TIFF.
  - `frame_folder_source(folder, pattern="*.tif")` — a folder of single-page
    TIFFs (streaming counterpart of `load_frame_folder`).
  - `folder_frame_source(files)` — an explicit ordered file list (used for one
    channel of an interleaved folder, from `interleaved_channel_files`).

### `streaming.py` (constant-memory steps 1 + 3)
Same math as `correction.py` + `roi.py`, but never holds a full stack in RAM —
for recordings too large to load (the MATLAB `>4gb non lo legge` case).
- `stream_trial_roi(gcamp, emo, cfg, baseline_slice, trim=20, downsample=0.5) ->
  [time, 4]` — streams one trial to its ROI trace in **two passes**: pass 1
  accumulates the baseline-window running sum of the half-res frames →
  `MIf`/`MIr`; pass 2 re-reads, applies ΔF/F per frame, downsamples again, and
  reduces each frame to four ROI `nanmean` values. Only a couple of small images
  are ever resident. `gcamp`/`emo` are `FrameSource` objects (any storage
  format); a bare TIFF path is also accepted and treated as a multi-page file
  (original call style).
- `run_streaming(trial_sources, cfg, baseline_slice, corr_window, trim,
  downsample) -> StreamingResult` — streams each `(gcamp, emo)` FrameSource pair
  to its `[time, 4]` trace, stacks to `[time, 4, trial]`, then runs the usual
  `functional_connectivity`.
- `run_streaming_resting_state` / `run_streaming_stimulated` — window presets
  matching `run_resting_state` / `run_stimulated`.
- `StreamingResult(temp_roi, R, R_mean, averaged_traces)` — like
  `PipelineResult` but **without `dff_stack`** (it is streamed away, so step-2
  visualization requires the in-memory path).
- Trade-offs: two disk passes instead of one; agreement with the in-memory path
  is exact up to baseline-mean summation order (~1e-16). Verified in
  `tests/test_streaming.py`.

### `resize.py`
- `imresize_box(arr, scale)` — MATLAB-equivalent `imresize(arr, scale, 'box')`;
  resizes the first two axes only, leaving time/trial axes untouched.
- `_contributions(in_length, scale)` — reproduces MATLAB's `contributions`
  routine for the box kernel: output length `ceil(in_length*scale)`, pixel-center
  inverse map `u = x/scale + 0.5*(1 - 1/scale)`, antialiasing kernel widened to
  `1/scale` when downsampling, weights row-normalised, edge indices clamped
  (replicates end pixels). For even dims at scale 0.5 this equals a 2×2 block
  mean; the general path handles odd dims. **Verified exact vs MATLAB.**

### `correction.py` (step 1)
- `hemodynamic_correction(gcamp, emo, baseline_slice) -> dff` — per-pixel:
  `MIf=mean(gcamp[...,baseline]); MIr=mean(emo[...,baseline]);
  dff=(gcamp/MIf)/(emo/MIr) - 1) * 100`.
- `build_dff_stack(trials, baseline_slice, trim=20, downsample=0.5) ->
  [y,x,time,trial]` — per trial: trim first `trim` frames → `imresize_box ×0.5`
  → correction → `imresize_box ×0.5`; stack trials. Equals MATLAB
  `t_TEMP_resize1`.

### `config.py`
- `Box(row_start, row_end, col_start, col_end)` — one ROI as MATLAB 1-based
  **inclusive** offsets from Bregma `(y_1, x_2)`.
- `ROIConfig(y_1, x_2, boxes)` — per-animal geometry. Default `boxes` copy
  `matlab/step3_ROI_functional_connectivity.m` verbatim.
  `ROIConfig.from_bregma(bregma_row, bregma_col)` applies `floor(.../2)`.

### `roi.py` (step 3)
- `_box_slices(box, y_1, x_2)` — MATLAB `y_1+a : y_1+b` (1-based inclusive) →
  Python `slice(y_1+a-1, y_1+b)`.
- `extract_roi_timeseries(dff_stack, cfg) -> [time, 4, trial]` — `nanmean` over
  rows and cols of each ROI box → `TEMP_ROI`.
- `functional_connectivity(temp_roi, window) -> (R, R_mean, averaged_traces)` —
  per-trial `np.corrcoef` (columns = regions) over `window`, then trial means.
  `np.corrcoef` matches MATLAB `corr` (N vs N−1 normalisation cancels).

### `visualize.py` (step 2)
- `overlay_rois(frame, cfg, fill=1.0)` — paint the four ROI boxes onto a frame.
- `show_roi_placement(dff_stack, cfg, frame_index=302, average_trials=False,
  clim=(0.3, 3.0), ax=None)` — display overlay (matplotlib).

### `pipeline.py`
- `run_pipeline(trials, cfg, baseline_slice, corr_window, trim, downsample) ->
  PipelineResult` — steps 1 + 3.
- `run_resting_state(trials, cfg)` — baseline full, correlation full.
- `run_stimulated(trials, cfg, baseline_slice=slice(0,278),
  corr_window=slice(279,300))` — MATLAB baseline `1:278`, window `280:300`.
- `PipelineResult(dff_stack, temp_roi, R, R_mean, averaged_traces)`.

## MATLAB ↔ Python variable map

| MATLAB | Python | Shape |
|--------|--------|-------|
| `t_TEMP_resize1` | `PipelineResult.dff_stack` | `[y,x,time,trial]` |
| `TEMP_ROI` | `PipelineResult.temp_roi` | `[time,4,trial]` |
| `R` | `PipelineResult.R` | `[4,4,trial]` |
| `R_mean` | `PipelineResult.R_mean` | `[4,4]` |
| `averaged_traces` | `PipelineResult.averaged_traces` | `[time,4]` |
| `y_1`, `x_2` | `ROIConfig.y_1`, `ROIConfig.x_2` | scalars |

## Windowing (MATLAB inclusive → Python slice)

| Meaning | MATLAB | Python |
|---------|--------|--------|
| Frame trim | `21:end` | `slice(20, None)` / `trim=20` |
| RS baseline / correlation | full | `slice(None)` |
| Stim baseline | `1:278` | `slice(0, 278)` |
| Stim correlation window | `280:300` | `slice(279, 300)` |

## Validation

- Generator: `tests/matlab_reference/gen_reference.m` → `reference.mat`.
- Test: `tests/test_parity.py` (pytest). Tolerances `atol=1e-7`, `rtol=1e-9`;
  observed max diffs 0 (resize, ΔF/F) to ~1e-16 (correlation).
- Streaming: `tests/test_streaming.py` — asserts the streaming path reproduces
  the (MATLAB-validated) in-memory path on synthesized paired multi-page TIFFs;
  observed max diff ~9e-16, so streaming inherits MATLAB parity.

## Modality benchmark (`benchmarks/`)

Compares the four layouts (`stack` | `folder` | `stream` | `interleaved`) on the
sample dataset for **RAM**, **time**, and **numerical agreement**, and
cross-checks against MATLAB. Writes `outputs/modality_comparison.txt`.

- `bench_common.py` — the single source of truth is the interleaved split of
  `data/` (`load_interleaved_folder`); `prepare_inputs` materialises that same
  float64 `(gcamp, emo)` trial into each layout's on-disk form (multi-page TIFFs
  for `stack`/`stream`, single-page folders for `folder`, the raw `data/` folder
  for `interleaved`) so any output difference is the layout's doing alone.
  Multi-page TIFFs are written `photometric="minisblack"` so a 3-frame stack is
  not mistaken for a 3-sample RGB image (which would collapse to one IFD page and
  break the streaming reader). `run_layout` runs one layout end-to-end. Fixed
  params: RS, `trim=0`, `downsample=0.5`, Bregma `(121,134)`.
- `bench_worker.py` — runs one layout in a **fresh subprocess** (so the peak-RAM
  high-water mark is not cross-contaminated), timing load+compute over `--repeats`
  runs and recording two memory figures: OS peak working set
  (`GetProcessMemoryInfo` via ctypes) and the `tracemalloc` peak of the timed
  region. Emits a `METRICS <json>` line and saves the result arrays to `.npz`.
- `benchmark_modalities.py` — orchestrator (`RUN_CONFIG` block + argparse):
  prepares inputs, benchmarks each layout via the worker, invokes MATLAB
  (`step_interleaved_intermingle.m`) for the interleaved reference, compares all
  outputs (NaN-aware max-|diff|), and writes the report. Transient files go to
  `temporary_files/modality_bench/`.

## MATLAB scripts (`matlab/`)

- `step1_*` / `step2_*` / `step3_*` — the original per-step pipeline scripts.
- `step_interleaved_intermingle.m` — MATLAB implementation of the `interleaved`
  ("intermingle") modality: reads `data/`, splits odd/even frames, picks the
  brighter group (mean of top-10% pixels of each group's first image) as gcamp,
  runs the RS pipeline (`trim=0`, `ds=0.5`, `y_1=60`, `x_2=67`) as a single
  trial, and saves `TEMP_ROI`/`R`/`R_mean`/`averaged_traces`. Data dir and output
  path come from env vars `WFCI_DATA` / `WFCI_INTERLEAVED_OUT` (defaults relative
  to the script). Used by the modality benchmark for the MATLAB cross-check.

## Entry points

- CLI/editor: `run_pipeline.py` (`RUN_CONFIG` block + argparse override). Two
  orthogonal knobs: `--source {stack_file,frame_folder,interleaved_folder}` and
  `--streaming/--no-streaming` (any source × either memory strategy).
- Example: `examples/run_example.py`.
- Benchmark: `benchmarks/benchmark_modalities.py` (`RUN_CONFIG` block + argparse).
