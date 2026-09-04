# LIBRARY.md — `wfci` as a whole, for an agent or a new maintainer

This is the **single self-contained document** for working on or with the `wfci`
package. If you have only this file, you have enough to use the library correctly
and to edit it without breaking the things that matter. It is exhaustive on
purpose.

Companion docs (don't duplicate them, reach for them):
- [README.md](README.md) — copy-paste commands, environment setup, I/O contract.
- [REFERENCE.md](REFERENCE.md) — the same API as a terse module-by-module map.
- [GUIDE.md](GUIDE.md) — a human-oriented tour: what every script is, when to use it.
- [MERGING_PLAN.md](MERGING_PLAN.md) — why the package is shaped the way it is.

> **Signatures below are generated from the real code**, not hand-copied. If you
> change a signature, regenerate this section (see [Keeping this doc true](#12-keeping-this-doc-true)).

---

## 1. What `wfci` is

A Python port of two MATLAB wide-field calcium-imaging pipelines, unified into one
package. From dual-channel recordings (a GCaMP fluorescence channel + an `emo`
hemodynamic/reflectance channel) it computes:

1. **hemodynamic correction** to ΔF/F (%),
2. optional **brain masking** and **global signal regression** (GSR),
3. **ROI** time-series by averaging ΔF/F inside anatomical boxes,
4. **functional connectivity** — the Pearson correlation matrix between ROIs,
5. (separately) a **group layer** to average/difference per-animal matrices and
   draw significance figures.

Two pipelines ship as presets: **cerebellar** (4 ROIs, no mask/GSR;
MATLAB-validated to machine precision) and **cortical** (22 ROIs, mask + GSR; no
MATLAB reference — see [§10](#10-what-is-and-isnt-validated)).

---

## 2. The mental model — three orthogonal axes + a fixed stage chain

Every run is described by three **independent** choices. They combine freely: any
value of one works with any value of the others, and the memory axis never changes
the numbers.

```
            WHICH PIPELINE                HOW STORED                  HOW MUCH IN RAM
            (profile)                     (source)                    (streaming)
        ┌───────────────────┐    ┌──────────────────────────┐   ┌────────────────────┐
        │ cerebellar_rs     │    │ stack_file (multipage)   │   │ in-memory (default)│
        │ cerebellar_stim   │  × │ frame_folder             │ × │ streaming (2-pass) │
        │ cortical_gsr      │    │ interleaved_folder       │   │                    │
        │ …your own Profile │    │ …or FrameSource objects  │   │                    │
        └───────────────────┘    └──────────────────────────┘   └────────────────────┘
```

- **Profile** = *what analysis*: the ROI atlas, the time windows, whether to mask,
  whether to regress the global signal. A `Profile` bundles these; see
  [§5.7](#57-profiles--profilespy).
- **Source** = *how the two channels sit on disk*. Decoupled from memory by the
  `FrameSource` abstraction. See [§5.2](#52-io--iopy).
- **Streaming** = *how much is resident*. In-memory keeps the full stack (and the
  `dff_stack`, so the overlay works); streaming makes two passes in constant
  memory and keeps no stack. **Streaming and in-memory agree to float roundoff.**

Underneath, the pipeline is always the same **stage chain**, with two optional,
profile-configured stages in brackets:

```
correction  →  [mask]  →  [GSR]  →  ROI means  →  connectivity
   step 1                              step 3        step 3
```

Order is load-bearing: the mask precedes GSR (the global signal is a mean over
*brain* pixels), and both precede the ROI means (which must average the regressed
data). This is not branched on — the bracketed stages are simply `None` or not.

---

## 3. Array-shape & indexing conventions — READ THIS FIRST

This is the single easiest thing to get wrong. Every rule here is a real trap.

- **All arrays are `float64`.** MATLAB computed in doubles; the port matches it,
  and parity depends on it. Do not introduce `float32` in the numeric path.
- **Spatial-first, MATLAB axis order: `[y, x, time, trial]`.**
  - a raw or corrected stack for one trial is `[y, x, time]`;
  - `build_dff_stack` / `PipelineResult.dff_stack` is `[y, x, time, trial]`;
  - `tifffile` decodes a multipage TIFF as `[time, y, x]`, so the loaders
    `np.moveaxis(..., 0, -1)` it to `[y, x, time]`. If you touch I/O, preserve
    that.
- **ROI outputs are time-first: `[time, n_roi, trial]`** (`temp_roi` / `TEMP_ROI`).
- **Connectivity: `R` is `[n_roi, n_roi, trial]`, `R_mean` is `[n_roi, n_roi]`.**
  Symmetric, unit diagonal, entries in `[-1, 1]` (Pearson).
- **`n_roi` is never hard-coded.** It is `len(cfg.boxes)` — 4 for cerebellum, 22
  for cortex, whatever a custom atlas defines. Any literal `4` in the code is in a
  comment.
- **Column order = atlas key order.** The order of `cfg.boxes` / `atlas` keys *is*
  the column order of `temp_roi` and the row/column order of `R`. Reorder the
  atlas and you silently permute every matrix. The cortical order is
  `[all 11 left, then all 11 right]` (MATLAB `cat(2, regioni_L, regioni_R)`).
- **Box offsets are MATLAB 1-based, inclusive**, relative to Bregma:
  `Box(row_start, row_end, col_start, col_end)` means MATLAB
  `img(y_1+row_start : y_1+row_end, x_2+col_start : x_2+col_end)`. Conversion to
  Python 0-based half-open slices happens in `roi._box_slices`; **do not** apply
  offsets by hand.
- **Bregma is downsampled.** `ROIConfig.y_1 / x_2` are `floor(full_res / 2)`.
  `ROIConfig.from_bregma(row, col)` does the `//2`. The boxes live on the final
  (twice-downsampled) grid.
- **`grid` is the final frame size** an atlas was drawn for (e.g. `(128, 128)` for
  512² raw after two 0.5× downsamples). Offsets are pixels, so they do **not**
  transfer between fields of view — see [§5.1](#51-geometry--configpy--atlasespy).
- **One exception, for output files only: the pixel dumps are `[time, y, x]`.**
  `wfci.dump.PixelDump` writes frame-major because each frame is then one
  contiguous write into a preallocated memmap (`[y, x, time]` would stride every
  frame across the whole file), and because that is `tifffile`'s and ImageJ's
  order. This applies to the written `.npy` only — nothing in the numeric path
  changes. Their storage dtype (float16/float32) is likewise a file format
  choice, not a relaxation of the float64 rule above.
- **Units:** ΔF/F is a **percentage**; `R` is dimensionless.
- **NaN is the "no data here" marker.** Masked-out pixels are `NaN` for their whole
  time-series; every reduction downstream uses `nanmean`. GSR's `drop_pixel` drops
  a pixel from *all* frames if it is NaN in *any* (MATLAB's `~isnan(...)`).

---

## 4. Install & run in one breath

```bash
CONDA="$USERPROFILE/miniconda3/condabin/conda.bat"      # this machine
"$CONDA" env create -f .env/environment.yml             # env `letizia`, py3.11
"$CONDA" run -n letizia pip install -e .                # editable install
"$CONDA" run -n letizia python -m pytest tests/ -q      # 156 tests, all green
```

Everything importable as `import wfci`. The CLI/editor entrypoint is
`run_pipeline.py` (see [GUIDE.md](GUIDE.md) for the flags and [README.md](README.md)
for copy-paste commands).

---

## 5. The public API

64 names are exported from `wfci`. Grouped by concern below. Shapes in `[...]`,
all `float64` unless stated. `→` gives the return.

### 5.1 Geometry — `config.py` / `atlases.py`

```python
Box(row_start:int, row_end:int, col_start:int, col_end:int)
    # One ROI as MATLAB 1-based inclusive offsets from Bregma. Frozen.

ROIConfig(y_1:int, x_2:int, boxes:dict[str,Box] = <CEREBELLUM_4>)
    .from_bregma(bregma_row:int, bregma_col:int, **kw) -> ROIConfig   # applies //2
    .labels   -> list[str]     # box keys in column order (derived, not stored)
    .n_rois   -> int
    # Per-animal Bregma + the boxes. Default boxes = the cerebellar atlas.

Atlas(name:str, boxes:dict[str,Box], grid:tuple[int,int]|None=None, source:str="")
    .labels   -> list[str]
    # A read-only Mapping (dict-drop-in) that ALSO carries `grid` (the final FOV
    # the offsets are valid for) and `source` (provenance). Copies its boxes.

CEREBELLUM_4 : Atlas   # 4 boxes, grid=(128,128), MATLAB-validated
CORTEX_22    : Atlas   # 22 boxes, grid=(128,128), NO MATLAB reference
ATLASES      : dict[str, Atlas]

atlas_labels(atlas:Mapping) -> list[str]
as_atlas(boxes:Mapping, name:str="custom") -> Atlas   # wrap a plain dict (grid=None)
load_atlas(path:str|Path) -> Atlas                    # YAML/JSON, safe_load
save_atlas(atlas:Mapping, path:str|Path) -> Path      # round-trips load_atlas
```

`grid` is what makes a wrong field of view an error instead of a silent
wrong-anatomy result — see the bounds check in [§5.5](#55-roi-means--connectivity--roipy).

### 5.2 I/O — `io.py` (**do not touch lightly; invariants I1–I9 live here**)

```python
# Whole-stack loaders (in-memory path)
load_stack(path) -> [y,x,time]
load_frame_folder(folder, pattern="*.tif") -> [y,x,time]
load_interleaved_folder(folder, pattern="*.tif", top_percent=10.0,
                        channel_order="auto") -> (gcamp[y,x,time], emo[y,x,time])
interleaved_channel_files(folder, pattern="*.tif", top_percent=10.0,
                          channel_order="auto") -> (gcamp_files, emo_files)
    # Splits ONE interleaved folder by odd/even position; decodes exactly 2 images
    # to decide which group is dimmer (=gcamp), unless channel_order forces it.

# Lazy, constant-memory building blocks (streaming path)
tiff_frame_count(path) -> int                   # IFD headers only, decodes 0 pixels
iter_tiff_frames(path) -> Iterator[[y,x]]
iter_folder_frames(files) -> Iterator[[y,x]]

FrameSource(count:int, _open:Callable)
    .open() -> Iterator[[y,x]]                   # a FRESH iterator each call
tiff_frame_source(path) -> FrameSource           # one multipage TIFF
folder_frame_source(files) -> FrameSource        # an explicit ordered file list
frame_folder_source(folder, pattern="*.tif") -> FrameSource   # a folder
```

`FrameSource` is the abstraction that makes **storage format orthogonal to memory
strategy**: `count` (known without decoding) + a re-iterable `open()` (two passes
need a fresh iterator). `channel_order ∈ {"auto","gcamp_first","emo_first"}`.

### 5.3 Resize — `resize.py`

```python
imresize_box(arr, scale) -> arr   # MATLAB imresize(arr, scale, 'box'), exact,
                                  # resizes only the first two axes (y, x)
```

Bit-exact with MATLAB, including odd dimensions. The 0.5×-on-even fast path is a
pure block mean. Per-frame resize ≡ whole-stack resize (invariant I11) — this is
why streaming can match in-memory exactly.

### 5.4 Correction, mask, GSR — `correction.py` / `mask.py` / `gsr.py`

```python
# Step 1: hemodynamic correction
hemodynamic_correction(gcamp[y,x,time], emo[y,x,time], baseline_slice=slice(None))
    -> dff[y,x,time]                             # ((g/mean_g)/(e/mean_e) - 1)*100
build_dff_stack(trials:list[(gcamp,emo)], baseline_slice=slice(None),
                trim=20, downsample=0.5) -> [y,x,time,trial]
    # per trial: drop `trim` frames -> imresize 0.5 -> correct -> imresize 0.5

# Brain mask (cortical stage 1)
load_mask(path) -> [y,x]                          # explicit path in, no uiopen
resize_mask(mask, scale=0.5) -> [y,x]             # SAME imresize_box as the data
valid_from_mask(mask, threshold=0.0) -> bool[y,x] # keep mask > threshold
apply_mask(stack, mask, threshold=0.0) -> stack   # outside -> NaN, all frames/trials

# Global signal regression (cortical stage 2)
GSRConfig(nan_policy="drop_pixel", min_variance=0.0)
    # "drop_pixel": any NaN frame -> pixel NaN everywhere (MATLAB's ~isnan).
    # "per_frame" is reserved and raises NotImplementedError.
global_signal(stack[y,x,time]) -> [time]          # spatial mean over FINITE pixels
regress_global(stack[y,x,time], cfg=None, g=None) -> [y,x,time]
    # closed-form per-pixel OLS: a=cov(g,p)/var(g), b=mean(p)-a*mean(g);
    # residual = p - a*g - b. Replaces 16384 fitlm calls/trial.
```

`global_signal` excludes `inf` as well as `NaN` (MATLAB's `nanmean` propagates
`inf`, and ΔF/F can produce it by dividing by a zero `emo` pixel).

### 5.5 ROI means & connectivity — `roi.py`

```python
box_slices_for(cfg, frame_shape:(rows,cols), expected_grid=None)
    -> [(label, row_slice, col_slice), ...]
    # Resolves every box against a REAL frame, or RAISES. A box off the left edge
    # becomes a negative NumPy index (reads the opposite hemisphere, silently);
    # this refuses instead, naming the ROI. `expected_grid` also catches a
    # right-shape-but-wrong-FOV atlas. The single source of "does the geometry fit".

extract_roi_timeseries(dff_stack[y,x,time,trial], cfg, expected_grid=None)
    -> temp_roi[time, n_roi, trial]               # nanmean inside each box

functional_connectivity(temp_roi[time,n_roi,trial], window=slice(None))
    -> (R[n_roi,n_roi,trial], R_mean[n_roi,n_roi], averaged_traces[time,n_roi])
    # per-trial explicit float64 Pearson reductions over `window`, then trial
    # means. Matches MATLAB corr without dispatching the small matrix to BLAS.
```

### 5.6 Overlay (step 2) — `visualize.py`

```python
overlay_rois(frame[y,x], cfg, fill=1.0, expected_grid=None) -> frame[y,x]
show_roi_placement(dff_stack, cfg, frame_index=302, average_trials=False,
                   clim=(0.3,3.0), ax=None) -> Axes    # needs matplotlib
```

The overlay derives boxes from the same `cfg` the pipeline averages, so the
picture can't disagree with the numbers (the cerebellar MATLAB pair drifted here).

### 5.7 Profiles — `profiles.py`

```python
Profile(name, atlas, trim=20, baseline=slice(None), corr_window=slice(None),
        use_mask=False, gsr=None, mask_downsample=0.5, overlay_frame=302,
        channel_order="auto", downsample=0.5)
    .labels -> list[str]     .n_rois -> int
    # Frozen; normalises `atlas` to an Atlas and copies it.

CEREBELLAR_RS    # 4 ROIs, trim 20, full baseline & window
CEREBELLAR_STIM  # 4 ROIs, trim 20, baseline slice(0,278), window slice(279,300)
CORTICAL_GSR     # 22 ROIs, trim 0, use_mask=True, gsr=GSRConfig()
PROFILES : dict[str, Profile]
get_profile(name) -> Profile
```

Presets, not a closed set. Build your own or `dataclasses.replace` one.

### 5.8 Orchestration — `pipeline.py` / `streaming.py`

```python
# In-memory
run_pipeline(trials, cfg, baseline_slice=slice(None), corr_window=slice(None),
             trim=20, downsample=0.5, mask=None, gsr=None, mask_threshold=0.0,
             expected_grid=None) -> PipelineResult
run_profile(trials, cfg, profile, mask=None) -> PipelineResult
run_resting_state(trials, cfg, **kw) -> PipelineResult
run_stimulated(trials, cfg, baseline_slice=slice(0,278),
               corr_window=slice(279,300), **kw) -> PipelineResult
PipelineResult(dff_stack, temp_roi, R, R_mean, averaged_traces)

# Streaming (constant memory, two passes; agrees to roundoff; no dff_stack)
stream_trial_roi(gcamp, emo, cfg, baseline_slice=slice(None), trim=20,
                 downsample=0.5, mask=None, gsr=None, mask_threshold=0.0,
                 expected_grid=None) -> temp_roi[time, n_roi]
run_streaming(trial_sources, cfg, ...same windows/stages...) -> StreamingResult
run_streaming_profile(trial_sources, cfg, profile, mask=None) -> StreamingResult
run_streaming_resting_state(trial_sources, cfg, **kw) -> StreamingResult
run_streaming_stimulated(trial_sources, cfg, ...) -> StreamingResult
StreamingResult(temp_roi, R, R_mean, averaged_traces)      # NO dff_stack
```

`run_profile` / `run_streaming_profile` take `cfg` for the **Bregma only** — the
profile supplies the atlas. `mask` is passed *as loaded*; the profile's
`mask_downsample` puts it on the data grid. `gcamp`/`emo` in the streaming
functions accept a `FrameSource` or a bare path (coerced — invariant I7).

**GSR under streaming still makes exactly two passes**, via sufficient statistics
(`Σg, Σg², Σp, Σgp, N` + the `[time]` vectors), because the regressed ROI mean is
`T_B(t) = mean_B(p(t)) − g(t)·mean_B(a) − mean_B(b)` — no regressed stack needed.
It refuses (raises) if a pixel is NaN in some frames but not all (the static-NaN
precondition), pointing you to `--no-streaming`.

### 5.8b Per-pixel dumps — `dump.py`

```python
PixelDump(out_dir, suffix, n_time, region=None, downsample=0.5,
          dff_dtype=np.float16, f_dtype=np.float32, metadata=None)
    .baselines(mean_f, mean_r)          # once, after pass 1
    .frame(idx, g_half, e_half, dff_q)  # once per frame, from inside pass 2
    .close()                            # flush + write pixels_meta_<suffix>.npz
```

The pipeline reduces every frame to `n_roi` box means and discards the pixels.
`PixelDump` is an optional **observer** that writes them out on the way past, so
questions that are not one of the predefined boxes (a different atlas, a
seed-pixel map, a check on the correction itself) can be answered without
re-reading the raw TIFFs.

Attach it with `stream_trial_roi(..., pixel_dump=dump)`, or across trials with
`run_streaming(..., pixel_dump_factory=lambda i: dump_for(i))`. It writes three
volumes — `pixels_dff_*.npy` (the corrected ΔF/F the ROI means are taken from)
and `pixels_f_gcamp_*.npy` / `pixels_f_emo_*.npy` (raw fluorescence per channel,
brought onto the final grid) — plus a sidecar `.npz` holding `mean_f`/`mean_r`
and the crop origin.

Non-negotiables it is built around:

* **it must not change the numbers.** It only reads pass 2's locals; the ROI
  traces are byte-identical with and without it.
* **it must not cost a pass.** Writing happens inside the existing pass 2, so
  I10 (exactly two passes) and I2 (one frame resident) still hold. Measured on a
  real 3000-frame recording: 300.9 s with the dump, 301.0 s without.
* **it never clips and never overflows silently.** A `region` outside the grid
  raises; a value too large for the target dtype raises rather than writing `inf`.
* **GSR raises.** `_stream_pass2_gsr` never materialises a post-GSR per-pixel
  frame (that is the point of accumulating `a`/`b`/`g_vec`), and producing one
  would need a third pass. Dumping the pre-GSR pixels under that name would be
  worse than refusing.

`dff` is dumped rather than derived on purpose: the pipeline corrects at half
resolution and downsamples the *result*, so a ratio recomputed from the
already-downsampled F volumes is close but not equal. The raw-F volumes are
nonetheless exact, and `F / mean` gives the per-channel MATLAB `If2`/`Ir2`.

Which region, which dtypes and which directory are **policy** and live in the
study script (P2) — see `run_botox_batch.py`'s `save_data*` keys.

---

### 5.9 Group layer — `cohort.py` (**generic; no study knowledge, P2**)

```python
AnimalResult(matrix[n,n], labels:tuple[str,...], metadata:dict={}, source:str="")
LabeledMatrix(matrix[n,n], labels)          # supports -, +; keeps labels
load_results(paths, *, metadata_from=None, matrix_key="R_mean",
             labels_key="roi_labels") -> list[AnimalResult]
    # paths: glob str | Path | iterable. metadata_from(Path)->dict is the seam:
    # the library reads the matrix, the STUDY says what it is.

CohortTable(results:Iterable[AnimalResult])
    .select(**criteria) -> CohortTable       # AND of equality; empty -> raises
    .filter(predicate) -> CohortTable
    .groupby(key) -> dict[value, CohortTable]
    .values(key) -> set
    .stack() -> [n, n, n_animal]             # MATLAB cat(3,...)
    .mean() -> LabeledMatrix                 # nanmean over animals
    .labels -> tuple[str,...]                .to_dataframe()  # lazy pandas
```

`DIFF = table.select(group="healthy").mean() - table.select(group="disease").mean()`.

### 5.10 Significance & figures — `significance.py` (**generic; ingests, not computes**)

```python
mask_by_adjacency(value_matrix[n,n], adjacency[n,n], *, symmetrize=True) -> [n,n]
    # keep value where adjacency>0 else 0; symmetrize. NaN->0.
node_strength(masked) -> [n]                       # per-node |weighted degree|
count_significant_edges(masked, sign="negative") -> [n]   # "positive"|"both"
circular_layout(n, radius=1.0) -> [n,2]
hemispheric_layout(n_left, n_right) -> [n,2]       # left column, then right
network_figure(masked, *, node_positions=None, node_values=None, labels=None,
               cmap="RdBu_r", norm=None, node_size_range=(60,600),
               edge_width_range=(0.5,6.0), annotate=True, ax=None) -> Axes
significance_barplot(masked, labels=None, *, sign="negative", ax=None, color=None)
    -> Axes
```

`wfci` does **not** compute NBS — you pass an adjacency in. Colours/positions are
arguments (the study's choice), with sensible diverging defaults.

---

## 6. Worked recipes

### 6.1 Cerebellar resting-state (the original pipeline)

```python
from wfci import ROIConfig, load_stack, run_resting_state

trials = [(load_stack("t1_gcamp.tif"), load_stack("t1_emo.tif"))]
cfg = ROIConfig.from_bregma(bregma_row=121, bregma_col=134)
res = run_resting_state(trials, cfg)          # 4-ROI R_mean, dff_stack kept
print(res.R_mean.shape)                        # (4, 4)
```

### 6.2 Cortical GSR pipeline, streaming a large interleaved folder

```python
from wfci import (ROIConfig, CORTICAL_GSR, load_mask,
                  interleaved_channel_files, folder_frame_source,
                  run_streaming_profile)

gcamp_files, emo_files = interleaved_channel_files("/big/folder")   # decodes 2 images
sources = [(folder_frame_source(gcamp_files), folder_frame_source(emo_files))]
cfg  = ROIConfig.from_bregma(126, 126)         # profile supplies the atlas
mask = load_mask("mask.tif")                   # 256x256 for 512x512 raw
res  = run_streaming_profile(sources, cfg, CORTICAL_GSR, mask=mask)
print(res.R_mean.shape)                        # (22, 22), constant memory
```

### 6.3 A custom atlas (in Python or a file a study owns)

```python
from dataclasses import replace
from wfci import Atlas, Box, CORTICAL_GSR, save_atlas, load_atlas

atlas = Atlas("motor_only",
              {"M1L": Box(-17,-12,-32,-27), "M1R": Box(-17,-12,27,32)},
              grid=(128,128), source="drawn by AB 2026-03")
profile = replace(CORTICAL_GSR, name="motor", atlas=atlas)

save_atlas(CORTEX_22, "study/atlas.yaml")      # export a preset, edit, then:
atlas = load_atlas("study/atlas.yaml")         # one line per ROI in the file
```

### 6.4 A custom profile

```python
from dataclasses import replace
from wfci import CEREBELLAR_STIM
my = replace(CEREBELLAR_STIM, name="my_stim", trim=10,
             corr_window=slice(300, 340))       # different stimulus window
```

### 6.5 Group difference across animals + a network figure

```python
from wfci import CohortTable, load_results, mask_by_adjacency, network_figure

def whose(path): return {"group": "healthy" if "PV" in path.stem else "disease"}
table = CohortTable(load_results("outputs/*.npz", metadata_from=whose))
diff  = table.select(group="healthy").mean() - table.select(group="disease").mean()
network_figure(mask_by_adjacency(diff.matrix, nbs_adjacency), labels=diff.labels)
```

### 6.6 A study script

Copy `experiments/healthy_vs_disease_day4.py`, edit the `COHORT` table and the
`.select(...)` lines, point it at your `.npz` folder. You never touch `src/wfci/`.

### 6.7 Verify a change end to end

```bash
"$CONDA" run -n letizia python -m pytest tests/ -q          # all 156 green
"$CONDA" run -n letizia python examples/run_example.py      # exit 0
"$CONDA" run -n letizia python experiments/healthy_vs_disease_day4.py
```

---

## 7. The do-not-break list — import-efficiency invariants I1–I11

These make the streaming path worth having. **Every one is invisible to a
correctness test** — a refactor could break any of them and leave the numbers
right, just slower and fatter, until someone points the pipeline at a recording
too large for RAM. They are pinned by `tests/test_efficiency_invariants.py` (15
tests) and the RAM-slope assert in `benchmarks/benchmark_scaling.py`. Do not
"simplify" past them.

| # | Invariant | Why it matters |
|---|-----------|----------------|
| I1 | `tiff_frame_count` decodes **0** pixels (IFD headers only) | size the baseline window on a multi-GB file for free |
| I2 | frames decode **lazily**, one at a time | never more than one frame resident |
| I3 | `FrameSource.open()` returns a **fresh** iterator | two passes need to restart the sequence |
| I4 | storage × memory stay **orthogonal** | any source streams or loads; no matrix of special cases |
| I5 | interleaved split decodes **exactly 2** images | a folder too big for RAM can still be split |
| I6 | the **same split** feeds full-load and streaming | one code path, no drift |
| I7 | a bare path is **coerced** to a FrameSource | the old call style keeps working |
| I8 | `--debug-max-frames k` reads **2k** frames, not 2N | debug stays lazy (`islice` over `open()`) |
| I9 | debug full-load routes through the **lazy** sources | a debug in-memory run doesn't read the whole folder |
| I10 | a streaming run is **exactly two passes** (2N decodes/channel) | the constant-memory guarantee itself |
| I11 | per-frame resize ≡ whole-stack resize | why streaming matches in-memory exactly |

The guard is proven to bite: five plausible regressions (cached `open()`, decoding
in `tiff_frame_count`, reading every image in the split, dropping path coercion,
an extra pass) each fail it — while the rest of the suite stays green.

---

## 8. The library / experiment boundary (P2)

**Rule an agent can apply:** *would the next study, with different animals, still
want this unchanged?* If **no**, it belongs in `experiments/` (a study script the
user owns). If **yes**, it belongs in `src/wfci/`.

- **Mechanism** (average a box, correlate, stream, mask, regress, select, figure)
  → library.
- **Reference data** (`CEREBELLUM_4`, `CORTEX_22`) → library, as *overridable
  presets* — the next cerebellar study wants those exact boxes.
- **Policy** (which animals, which groups, sex/day splits, significance
  thresholds, node coordinates, the per-animal Bregma) → `experiments/`, never the
  library.

This is **enforced as a test**: `tests/test_cohort.py` greps `src/wfci/` for
`sani|pd_|macchi|day4|sex` and fails if any appears. The cohort/significance
layers are pure primitives *because* of this rule; the worked study script is
where the group names live.

---

## 9. What each source file is (one line each)

```
io.py          TIFF loading + FrameSource (I1–I9 live here; touch with care)
resize.py      exact MATLAB imresize('box')
correction.py  step 1: hemodynamic ΔF/F
config.py      Box, ROIConfig (Bregma + boxes)
atlases.py     Atlas + CEREBELLUM_4/CORTEX_22 + load/save
mask.py        brain mask -> NaN
gsr.py         global signal regression (closed-form OLS)
roi.py         step 3: box_slices_for (bounds check), ROI means, connectivity
profiles.py    Profile presets = which pipeline
streaming.py   constant-memory pipeline + streaming GSR (2 passes)
dump.py        optional per-pixel .npy dumps written from inside pass 2
visualize.py   step 2: ROI overlay
pipeline.py    orchestration: correction -> [mask] -> [GSR] -> ROI -> connectivity
cohort.py      generic group layer (stack/select/mean/DIFF)
significance.py generic figures (mask by adjacency, network, bars)
```

---

## 10. What is and isn't validated

- **P1 — the existing Python is the reference.** The cerebellar path is validated
  against MATLAB to machine precision (`tests/test_parity.py`, diffs `0`–`1e-16`).
  The port's outputs must not change; the README's documented commands are
  **byte-identical** across every phase of the merge.
- **P4 — the cortical path has NO MATLAB reference.** Its 22 boxes are *pinned to
  the source scripts* (`tests/test_atlas_transcription.py` re-parses the `.txt`
  each run), and its GSR is validated as *arithmetic* (vectorised OLS ≡ per-pixel
  `lstsq` to ~1e-15). But nothing proves a real MATLAB run would produce these
  numbers. Treat cortical numbers as carefully-read, not machine-verified. Phase 8
  (deferred) is the escape hatch if they are ever questioned.

---

## 11. Test map — how to know you didn't break anything

| File | Guards |
|------|--------|
| `test_parity.py` | cerebellar ≡ MATLAB (machine precision) |
| `test_streaming.py` | streaming ≡ in-memory (cerebellar) |
| `test_efficiency_invariants.py` | I1–I11 (decode/open counts, constant memory) |
| `test_atlas_transcription.py` | CORTEX_22 pinned to the MATLAB `.txt` |
| `test_atlas_generality.py` | pipeline sized by the atlas, not by "4" |
| `test_atlas_files.py` | load/save a study-owned atlas |
| `test_roi_bounds.py` | geometry that doesn't fit **raises** |
| `test_mask.py` | masking → NaN, thresholds, resolution checks |
| `test_gsr.py` | vectorised OLS ≡ per-pixel `lstsq` |
| `test_streaming_gsr.py` | streaming GSR ≡ in-memory GSR, still 2 passes |
| `test_cli.py` | `--profile` / `--mode` alias / mask rules |
| `test_cohort.py` | group means/DIFF vs hand computation + **P2 grep** |
| `test_significance.py` | masking, per-node reductions, headless figures |
| `test_dump.py` | pixel dumps: inert on the numbers, still 2 passes, files reproduce `temp_roi` |

Run all: `"$CONDA" run -n letizia python -m pytest tests/ -q`.

---

## 12. Keeping this doc true

The API in [§5](#5-the-public-api) is derived from real signatures. If you add,
remove or re-sign an exported name, update `wfci.__all__` **and** this section (and
[REFERENCE.md](REFERENCE.md)). A quick way to re-derive the signatures:

```python
import inspect, wfci
for n in wfci.__all__:
    o = getattr(wfci, n)
    if inspect.isroutine(o) or inspect.isclass(o):
        print(n, inspect.signature(o))
```

If you change numeric behaviour on the cerebellar path, `test_parity.py` must
still pass byte-for-byte — that is P1, and it is not negotiable.
