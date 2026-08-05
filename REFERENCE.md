# REFERENCE — Technical map of `wfci`

Canonical technical reference for the repository. See [README.md](README.md)
for usage, [GUIDE.md](GUIDE.md) for a plain-language tour of the scripts,
[LIBRARY.md](LIBRARY.md) for the full self-contained API + editing rules, and
[CLAUDE.md](CLAUDE.md) for operating conventions.

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
  even-positioned groups; the dimmer group (by `_top_percent_mean` of the first
  image in each group — only those two images are read) becomes `gcamp`, the
  brighter `emo`; the two file lists are truncated to equal length. Shared by the
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

### `streaming.py` (constant-memory pipeline)
Same math as `correction.py` + `roi.py` (+ `mask.py`/`gsr.py`), but never holds a
full stack in RAM — for recordings too large to load (the MATLAB
`>4gb non lo legge` case).
- `stream_trial_roi(gcamp, emo, cfg, baseline_slice, trim=20, downsample=0.5,
  mask=None, gsr=None, mask_threshold=0.0) -> [time, n_rois]` — streams one trial
  to its ROI trace in **two passes**: pass 1 accumulates the baseline-window
  running sum of the half-res frames → `MIf`/`MIr`; pass 2 re-reads, applies ΔF/F
  per frame, downsamples again, and reduces each frame to `n_rois` `nanmean`
  values. Only a couple of small images are ever resident. `gcamp`/`emo` are
  `FrameSource` objects (any storage format); a bare TIFF path is also accepted
  and treated as a multi-page file (original call style).
- `run_streaming(trial_sources, cfg, baseline_slice, corr_window, trim,
  downsample, mask=None, gsr=None) -> StreamingResult` — streams each
  `(gcamp, emo)` FrameSource pair to its `[time, n_rois]` trace, stacks to
  `[time, n_rois, trial]`, then runs the usual `functional_connectivity`.
- `run_streaming_profile(trial_sources, cfg, profile, mask=None)` — streaming
  counterpart of `run_profile`. Same profile, same numbers, constant memory.
- `run_streaming_resting_state` / `run_streaming_stimulated` — window presets
  matching `run_resting_state` / `run_stimulated`.
- `StreamingResult(temp_roi, R, R_mean, averaged_traces)` — like
  `PipelineResult` but **without `dff_stack`** (it is streamed away, so step-2
  visualization requires the in-memory path).

**GSR under streaming** (`_stream_pass2_gsr`) — GSR regresses each pixel's *whole*
time-series against the global signal, which looks like it needs `[y,x,time]`
resident. It does not, for two reasons:
1. Per-pixel OLS needs only **sufficient statistics**, all accumulable one frame
   at a time: `N`, `Σg`, `Σg²` (scalars) and `Σp`, `Σgp` (two `[y,x]` images). The
   global signal `g(t)` is a *spatial* mean, so it is known at frame `t` — no
   lookahead.
2. The ROI mean is linear and `g(t)` is one scalar per frame, so
   `T_B(t) = mean_B(p(t)) − g(t)·mean_B(a) − mean_B(b)` — the regressed *stack* is
   never needed.

  Cost: two extra `[y,x]` images and two `[time]` vectors. **Still two passes.**
- `_check_static_nans` — the precondition: `drop_pixel` decides validity from a
  pixel's whole time-series, so a pixel that is NaN in *some* frames would be
  treated differently by the two paths. Mask-only NaNs are static (fine); anything
  else **raises** and points the caller at `--no-streaming`, rather than silently
  returning numbers that differ from the in-memory reference.
- Trade-offs: two disk passes instead of one; agreement with the in-memory path
  is exact up to summation order (~1e-16 without GSR, ~1e-14 with). Verified in
  `tests/test_streaming.py` and `tests/test_streaming_gsr.py`. The two-pass /
  constant-memory guarantees are enforced by `tests/test_efficiency_invariants.py`.

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
- `ROIConfig(y_1, x_2, boxes)` — per-animal geometry. Default `boxes` is
  `atlases.CEREBELLUM_4`. `boxes` order **is** the column order of `TEMP_ROI` and
  the row/column order of `R`. `ROIConfig.from_bregma(bregma_row, bregma_col)`
  applies `floor(.../2)`. Properties: `.labels` (derived from `boxes`, never
  stored separately), `.n_rois`.

### `atlases.py`
- `Atlas(name, boxes, grid=None, source="")` — an ordered `{label: Box}` mapping
  that knows where it is valid. Read-only `Mapping`, so it is a drop-in for the
  plain dict it replaced (`dict(atlas)`, `list(atlas)`, `atlas["V1R"]`,
  `len(atlas)`, `atlas == {...}`). Copies its boxes, so a preset cannot be mutated
  process-wide by a caller. `.labels` = column order.
  - `grid` = `(rows, cols)` of the **final** analysis frame the offsets were drawn
    for, or `None` for "unknown, do not check". **The boxes are not anatomy** —
    they are anatomy projected through one optical setup, so they do not transfer
    between FOVs. `grid` is the only thing that can catch a *scaled* atlas whose
    boxes all still fit.
  - `source` = free-text provenance.
- `CEREBELLUM_4` — the 4 cerebellar boxes, verbatim from
  `matlab/step3_ROI_functional_connectivity.m`. MATLAB-validated. `grid=(128,128)`.
  (NB `step2_area_location_RS.m` draws `Laterale_L` 3 columns from where step3
  averages it — the MATLAB pair has drifted. step3 wins; `wfci.visualize` derives
  the overlay from the same atlas so it cannot drift again.)
- `CORTEX_22` — the 22 cortical boxes, transcribed from `Antea_scripts/(3)` and
  `(4)` (which duplicate each other and were cross-checked — **identical**).
  Order = MATLAB's `ALL = cat(2, regioni_L, regioni_R)`: 11 left, then 11 right.
  `grid=(128,128)`. **No MATLAB reference** — pinned to the source scripts by
  `tests/test_atlas_transcription.py`, which re-parses the .txt files each run.
- `ATLASES` — name → atlas. `atlas_labels(atlas)` — labels in column order.
- `as_atlas(boxes, name="custom")` — coerce a plain dict to an `Atlas` (`grid=None`).
- `load_atlas(path)` / `save_atlas(atlas, path)` — YAML or JSON, by extension. How
  a study owns its geometry without editing the library. Round-trips; box order is
  preserved (it *is* the column order of `R`). YAML is `safe_load`ed — an atlas is
  data and must never execute. Unknown box keys are an error, not ignored.
- Presets, not a closed set: any `{label: Box}` mapping is a valid atlas.

### `mask.py` (cortical stage 1)
- `load_mask(path) -> [y,x]` — explicit path in, float64 out. No `uiopen`.
- `resize_mask(mask, scale=0.5)` — the **same** `imresize_box` as the data, so
  mask and data land on one grid. Box-resizing a binary mask gives *fractional*
  boundary values.
- `valid_from_mask(mask, threshold=0.0) -> bool[y,x]` — keep `mask > threshold`.
  `0.0` reproduces MATLAB's `if Mask_resized(i,j)==0` (partial coverage kept).
- `apply_mask(stack, mask, threshold=0.0)` — outside → `NaN` across all frames and
  trials. Shape mismatch is an error, not a broadcast.

### `gsr.py` (cortical stage 2)
- `GSRConfig(nan_policy="drop_pixel", min_variance=0.0)` — `"drop_pixel"` = MATLAB's
  `if ~isnan(data(row,col,:))` (any NaN frame ⇒ pixel dropped everywhere).
  `"per_frame"` is reserved and raises `NotImplementedError`.
- `global_signal(stack) -> [time]` — spatial mean over each frame's **finite**
  pixels. Purely spatial ⇒ knowable at frame `t` ⇒ streamable. (Stricter than
  MATLAB's `nanmean`, which propagates `inf`.)
- `regress_global(stack, cfg=None, g=None) -> [y,x,time]` — per-pixel OLS in
  closed form: `a = cov(g,p)/var(g)`, `b = mean(p) − a·mean(g)`, residual
  `p − a·g − b`. Replaces 16 384 `fitlm` calls per trial.

### `profiles.py`
- `Profile(name, atlas, trim, baseline, corr_window, use_mask, gsr,
  mask_downsample, overlay_frame, channel_order, downsample)` — the per-pipeline
  bundle. Frozen; copies its atlas. Properties `.labels`, `.n_rois`.
- `CEREBELLAR_RS` — 4 ROIs, trim 20, full baseline/window.
- `CEREBELLAR_STIM` — 4 ROIs, trim 20, baseline `slice(0,278)`, window `slice(279,300)`.
- `CORTICAL_GSR` — 22 ROIs, trim 0, `use_mask=True`, `gsr=GSRConfig()`.
- `PROFILES` / `get_profile(name)`. Presets, not a closed set.

### `roi.py` (step 3)
- `_box_slices(box, y_1, x_2)` — MATLAB `y_1+a : y_1+b` (1-based inclusive) →
  Python `slice(y_1+a-1, y_1+b)`. **Unvalidated**; prefer `box_slices_for`.
- `box_slices_for(cfg, frame_shape, expected_grid=None) -> [(label, rs, cs), ...]` —
  resolve every box against a real frame, **or raise**. Used by the in-memory
  path, the streaming path and the overlay, so there is one definition of "does
  this geometry fit".
  - A box outside the frame does **not** fail loudly in NumPy: a negative index
    reads from the opposite edge (a left ROI averages the right hemisphere and
    returns a normal-looking number), and an over-long one truncates or empties.
    MATLAB raises on a negative index — the port was more permissive than its
    source. This raises, listing every offending ROI with its coordinates.
  - `expected_grid` catches what bounds cannot: an atlas drawn for another FOV
    whose boxes all still fit. Pass `atlas.grid`.
- `extract_roi_timeseries(dff_stack, cfg, expected_grid=None) -> [time, n_rois, trial]`
  — `nanmean` over rows and cols of each ROI box → `TEMP_ROI`. Sized from
  `len(cfg.boxes)`; nothing is fixed at 4.
- `functional_connectivity(temp_roi, window) -> (R, R_mean, averaged_traces)` —
  per-trial `np.corrcoef` (columns = regions) over `window`, then trial means.
  `np.corrcoef` matches MATLAB `corr` (N vs N−1 normalisation cancels).

### `visualize.py` (step 2)
- `overlay_rois(frame, cfg, fill=1.0)` — paint the four ROI boxes onto a frame.
- `show_roi_placement(dff_stack, cfg, frame_index=302, average_trials=False,
  clim=(0.3, 3.0), ax=None)` — display overlay (matplotlib).

### `pipeline.py`
The stage chain is `correction → [mask] → [GSR] → ROI → connectivity`; the
bracketed stages are optional and configured, not branched on.
- `run_pipeline(trials, cfg, baseline_slice, corr_window, trim, downsample,
  mask=None, gsr=None, mask_threshold=0.0) -> PipelineResult`. `mask` must already
  be at the corrected stack's resolution. Order matters: mask before GSR (the
  global signal is the mean over *brain* pixels), both before the ROI means.
- `run_profile(trials, cfg, profile, mask=None)` — the usual entry point; `cfg`
  supplies only the Bregma, the profile supplies the atlas and everything else.
  `mask` is passed **as loaded**; the profile's `mask_downsample` puts it on the
  data's grid.
- `prepare_mask(mask, profile)` — resize + enforce the profile's mask rules
  (required when `use_mask`, rejected otherwise). Shared with the streaming path.
- `run_resting_state(trials, cfg)` — baseline full, correlation full.
- `run_stimulated(trials, cfg, baseline_slice=slice(0,278),
  corr_window=slice(279,300))` — MATLAB baseline `1:278`, window `280:300`.
- `PipelineResult(dff_stack, temp_roi, R, R_mean, averaged_traces)`.

### `cohort.py` (group layer — MATLAB step 5)
Generic: **no study knowledge** (P2 — enforced by `tests/test_cohort.py`, which
greps `src/wfci/`). Stacks per-animal matrices, selects subsets by study-defined
metadata, averages and differences them.
- `AnimalResult(matrix, labels, metadata={}, source="")` — one animal's
  `[n_roi, n_roi]` matrix + the tags the study attaches (`group`, `animal`, …).
  Validated square and label-matched.
- `LabeledMatrix(matrix, labels)` — a matrix that keeps its labels through `-`/`+`;
  `DIFF = healthy - disease` is one. Combining mismatched labels raises.
- `load_results(paths, metadata_from=None, matrix_key="R_mean",
  labels_key="roi_labels")` — read per-animal `.npz` (as `run_pipeline.py` writes).
  `metadata_from(path)->dict` is the seam: the library reads the matrix, the study
  says what it is.
- `CohortTable(results)` — requires one shared label order (else not poolable).
  - `.select(**criteria)` — AND over equality; empty selection raises.
  - `.filter(predicate)`, `.groupby(key)`, `.values(key)`.
  - `.stack() -> [n_roi, n_roi, n_animal]` (MATLAB `cat(3,…)`); `.mean() ->
    LabeledMatrix` (`nanmean` over animals); `.to_dataframe()` (lazy pandas).

### `significance.py` (figure layer — MATLAB step 6)
Generic. **Ingests** an adjacency (significant edges, e.g. from NBS); does **not**
compute the network statistic (non-goal). Colours/positions are arguments.
- `mask_by_adjacency(value_matrix, adjacency, symmetrize=True)` — keep values where
  `adjacency>0`, else 0; symmetrize (`triu+triu'`). NaN→0.
- `node_strength(masked) -> [n]` — per-node weighted degree (an honest vector, not
  the MATLAB's matrix-valued `node_sizes`).
- `count_significant_edges(masked, sign) -> [n]` — per-node counts (`"negative"` /
  `"positive"` / `"both"`); the bar-plot input.
- `circular_layout(n)`, `hemispheric_layout(n_left, n_right)` — default positions.
- `network_figure(masked, node_positions=None, node_values=None, labels=None,
  cmap="RdBu_r", norm=None, …) -> Axes` — node-link plot; symmetric diverging
  colour by default; every edge colour computed locally (not the MATLAB's reused
  `COLOR_EDGE`).
- `significance_barplot(masked, labels, sign="negative", ax=None) -> Axes`.

## MATLAB ↔ Python variable map

| MATLAB | Python | Shape |
|--------|--------|-------|
| `t_TEMP_resize1` | `PipelineResult.dff_stack` | `[y,x,time,trial]` |
| `TEMP_ROI` | `PipelineResult.temp_roi` | `[time,n_rois,trial]` |
| `R` | `PipelineResult.R` | `[n_rois,n_rois,trial]` |
| `R_mean` | `PipelineResult.R_mean` | `[n_rois,n_rois]` |
| `averaged_traces` | `PipelineResult.averaged_traces` | `[time,n_rois]` |
| `y_1`, `x_2` | `ROIConfig.y_1`, `ROIConfig.x_2` | scalars |

Cortical pipeline (`Antea_scripts/`) only:

| MATLAB | Python | Shape |
|--------|--------|-------|
| `t_TEMP` | `build_dff_stack(..., trim=0)` output | `[y,x,time,trial]` |
| `Mask_resized` | `resize_mask(load_mask(path), 0.5)` | `[y,x]` |
| `t_TEMP_resized` (masked) | `apply_mask(dff, mask)` | `[y,x,time,trial]` |
| `global_signal` | `gsr.global_signal(stack)` | `[time]` |
| `t_TEMP_regressed` | `gsr.regress_global(stack, cfg)` per trial | `[y,x,time]` |
| `ALL` / `TEMP` | `temp_roi` (22 cols: 11 left, then 11 right) | `[time,22,trial]` |

`n_rois` = `len(cfg.boxes)`: 4 for `CEREBELLUM_4`, 22 for `CORTEX_22`, or whatever
a custom atlas defines.

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
  dimmer group (mean of top-10% pixels of each group's first image) as gcamp,
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
- Study scripts (root, `RUN_CONFIG` block + argparse): `scan_botox_dataset.py`
  (dataset → `manifests/*.csv`), `run_intermingle_rs.py` (interleaved folders:
  `discover_recordings` walks the input for folders holding ≥2 images and analyses
  each separately, mirroring the tree under `--output-dir`),
  `run_botox_batch.py` (manifest batch; one analysis per `t#` recording by
  default, or one concatenated trial per animal with `--merge-recordings`).
- ROI geometry utility: `roi_editor.py` — interactive placement of the ROI boxes and
  the two landmarks on the first image of each recording (rendered on the final
  analysis grid), writing `roi_sets/<key>.yaml`. Built on **pyqtgraph/Qt**
  (`pg.RectROI` per box, `pg.TargetItem` per landmark, whole-pixel snapping); Qt is
  imported inside `ROIEditor._build_ui`, never at module level, so importing the module
  for `load_roi_set` stays GUI-free and works headless. Covered by
  `tests/test_roi_editor.py` under `QT_QPA_PLATFORM=offscreen`.
  **Bregma** is the origin: moving it translates the layout rigidly (the offsets are
  not touched). **Lambda** is a midline landmark `lambda_row_offset` rows posterior;
  its distance from Bregma is the *scale* the offsets are in, so moving it rescales
  every box about Bregma via `roi_editor.scale_boxes` — centres only by default, sizes
  held fixed so each ROI mean keeps averaging the same number of pixels
  (`lambda_scales_box_size` opts into the full similarity transform). Rescaling is
  always derived from `Session.base_boxes` at `Session.base_lambda`, never from its own
  last output, so it round-trips exactly; a direct box edit `rebase()`s that reference.
  The file is an atlas file plus `bregma_row` / `bregma_col` / `lambda_row_offset`, so
  `wfci.load_atlas` reads it unchanged; `roi_editor.load_roi_set` returns
  `(atlas, bregma_row, bregma_col)` and the Lambda distance is read separately by
  `load_lambda_offset` — the saved boxes are already scaled, so the pipeline must not
  apply it again. The run scripts take it via
  `--roi-set` (one file) or `--roi-set-dir` (per `<day>_<animal>`), applied with
  `run_intermingle_rs.apply_roi_set`, which `dataclasses.replace`s the profile's
  atlas. See `docs/ROI_EDITOR.md`.
