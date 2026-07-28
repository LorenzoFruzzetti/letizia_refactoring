# Wide-field Calcium Imaging — ROI Functional Connectivity (`wfci`)

Python package for analysing dual-channel wide-field imaging data (GCaMP
fluorescence + hemodynamic reflectance) acquired in mouse brain. It performs
hemodynamic correction (ΔF/F), ROI placement relative to Bregma, optional brain
masking and global signal regression, and region-to-region
functional-connectivity analysis.

It runs **two pipelines**, selected with `--profile`:

| Profile | Regions | Stages | MATLAB origin |
|---------|---------|--------|---------------|
| `cerebellar_rs` | 4 cerebellar | correction → ROI → connectivity | [`matlab/`](matlab/) resting-state |
| `cerebellar_stim` | 4 cerebellar | correction → ROI → connectivity | [`matlab/`](matlab/) stimulated |
| `cortical_gsr` | 22 cortical | correction → **mask** → **GSR** → ROI → connectivity | [`Antea_scripts/`](Antea_scripts/) |

The cerebellar path is a port of [`matlab/`](matlab/) and is **numerically
validated against MATLAB to machine precision** — see
[Validation against MATLAB](#validation-against-matlab). The cortical path has
**no MATLAB reference**: its ROI coordinates and GSR maths are a careful reading
of [`Antea_scripts/`](Antea_scripts/), pinned to those scripts by
`tests/test_atlas_transcription.py` and validated as *arithmetic* (the vectorised
OLS matches a per-pixel `lstsq` fit), but not proven to reproduce a MATLAB run.
See [MERGING_PLAN.md](MERGING_PLAN.md) Phase 8.

> **Other docs:** [GUIDE.md](GUIDE.md) — a plain-language tour of every script and
> how to run it; [LIBRARY.md](LIBRARY.md) — the full programming API and the rules
> for editing the package; [REFERENCE.md](REFERENCE.md) — the terse canonical
> technical map.

The cerebellar ROIs are the left/right **vermis** (`Verme_L/R`) and the left/right
**lateral hemispheres** (`Laterale_L/R`). The cortical ROIs are 22 regions (11 per
hemisphere: motor, barrel field, trunk, limb, retrosplenial and visual areas) —
see [`src/wfci/atlases.py`](src/wfci/atlases.py). Neither set is built in: the ROI
layout is an argument, so a study can supply its own. Background: Falcicchia et
al., "Microglial extracellular vesicles induce Alzheimer's disease-like changes",
*Brain Communications* 2023 (open access) — not redistributed here; see
`.gitignore`.

---

## Entrypoints

Commands assume the repository root and the `letizia` conda environment (see
[Environment setup](#environment-setup)). On this machine `conda` is invoked
via its full path:

```bash
CONDA="$USERPROFILE/miniconda3/condabin/conda.bat"   # Git Bash
```

| Task | Command |
|------|---------|
| Run the pipeline (edit `RUN_CONFIG` first, or pass flags) | `conda run -n letizia python run_pipeline.py` |
| Run the pipeline from CLI | `conda run -n letizia python run_pipeline.py --source stack_file --no-streaming --profile cerebellar_rs --trial gcamp.tif,emo.tif --bregma-row 121 --bregma-col 134` |
| Run on a large recording (constant memory) | `conda run -n letizia python run_pipeline.py --source stack_file --streaming --profile cerebellar_rs --trial gcamp.tif,emo.tif --bregma-row 121 --bregma-col 134` |
| Run on one folder of interleaved (odd/even) channels | `conda run -n letizia python run_pipeline.py --source interleaved_folder --no-streaming --profile cerebellar_rs --trial data --bregma-row 121 --bregma-col 134` |
| Stream a large interleaved folder (constant memory) | `conda run -n letizia python run_pipeline.py --source interleaved_folder --streaming --profile cerebellar_rs --trial /path/big_folder --bregma-row 121 --bregma-col 134` |
| **Cortical pipeline (22 ROIs, mask + GSR)** | `conda run -n letizia python run_pipeline.py --source interleaved_folder --streaming --profile cortical_gsr --mask mask.tif --trial /path/folder --bregma-row 126 --bregma-col 126` |
| Force the interleaved channel order (skip the brightness guess) | `conda run -n letizia python run_pipeline.py --source interleaved_folder --profile cortical_gsr --mask mask.tif --channel-order emo_first --trial /path/folder --bregma-row 126 --bregma-col 126` |
| Debug run: only the first N frames per channel | `conda run -n letizia python run_pipeline.py --source interleaved_folder --no-streaming --profile cerebellar_rs --trial data --bregma-row 121 --bregma-col 134 --debug-max-frames 60` |
| **Place the ROIs / Bregma by eye** (interactive window) | `conda run --no-capture-output -n letizia python roi_editor.py` |
| Run one interleaved folder with the ROIs you drew | `conda run -n letizia python run_intermingle_rs.py --roi-set roi_sets/260611_R1.yaml --full` |
| Run every recording under a day/animal folder, each separately | `conda run -n letizia python run_intermingle_rs.py --folder "\\\\146.48.88.209\\share2\\BOTOX_RESTANI\\260611" --output-dir outputs/intermingle_260611` |
| Run the manifest batch, one ROI set per animal | `conda run -n letizia python run_botox_batch.py --roi-set-dir roi_sets --full` |
| Run the manifest batch, the same ROIs for every session | `conda run -n letizia python run_botox_batch.py --roi-set roi_sets/shared_roi_set.yaml --full` |
| Run the manifest batch as ONE concatenated trial per animal (instead of one per `t#`) | `conda run -n letizia python run_botox_batch.py --merge-recordings --full` |
| Runnable example on sample data | `conda run -n letizia python examples/run_example.py` |
| Worked group-contrast study (cohort → DIFF → figures) | `conda run -n letizia python experiments/healthy_vs_disease_day4.py` |
| Benchmark all 4 layouts (RAM, time, cross-layout + MATLAB parity) | `conda run -n letizia python benchmarks/benchmark_modalities.py` |
| Estimate time + RAM for a full folder from short debug runs | `conda run -n letizia python benchmarks/benchmark_scaling.py --folder "\\\\server\\share\\animal\\t1" --limits 200,300,400` |
| Parity test vs MATLAB | `conda run -n letizia python -m pytest tests/ -s -v` |
| Regenerate MATLAB reference | `matlab -batch "run('tests/matlab_reference/gen_reference.m')"` |

**From the editor (VS Code):** open the repo, select the `letizia` interpreter,
and use the launch configs in [.vscode/launch.json](.vscode/launch.json)
("Run pipeline", "Run example", "Debug current file"). `run_pipeline.py` runs
with no CLI flags by editing the `RUN_CONFIG` block at the top of the file.

Programmatic use:

```python
from wfci import ROIConfig, load_stack, run_resting_state, run_stimulated

trials = [(load_stack("gcamp.tif"), load_stack("emo.tif"))]   # one (gcamp, emo) per trial
cfg = ROIConfig.from_bregma(bregma_row=121, bregma_col=134)   # per-animal Bregma
result = run_resting_state(trials, cfg)                       # or run_stimulated(...)
print(result.R_mean)                                          # 4x4 connectivity matrix
```

The cortical pipeline is the same call with a profile and a mask:

```python
from wfci import CORTICAL_GSR, ROIConfig, load_mask, load_stack, run_profile

trials = [(load_stack("gcamp.tif"), load_stack("emo.tif"))]
cfg = ROIConfig.from_bregma(bregma_row=126, bregma_col=126)   # profile supplies the atlas
mask = load_mask("mask.tif")                                  # 256x256 for 512x512 raw
result = run_profile(trials, cfg, CORTICAL_GSR, mask=mask)
print(result.R_mean.shape)                                    # (22, 22)
print(CORTICAL_GSR.labels)                                    # the row/column order
```

A custom atlas or profile needs no library change:

```python
from dataclasses import replace
from wfci import CORTICAL_GSR, Atlas, Box

my_atlas = Atlas(
    name="my_study",
    boxes={"left_M1": Box(-17, -12, -32, -27), "right_M1": Box(-17, -12, 27, 32)},
    grid=(128, 128),          # the FINAL frame these offsets were drawn for
    source="drawn by AB, 2026-03, 4x objective",
)
my_profile = replace(CORTICAL_GSR, name="my_study", atlas=my_atlas)
```

Or keep the geometry in a file the study owns — see
[Bringing your own ROI atlas](#bringing-your-own-roi-atlas).

---

## Expected input

Each **trial** consists of two channels: a GCaMP fluorescence stack and an
`emo` (hemodynamic / reflectance) stack. A run is described by **three
independent knobs**: `--profile` (which pipeline to run), `--source` (how the
channels are stored on disk) and `--streaming` (whether to load into RAM or read
frame-by-frame in constant memory). All three combine freely — any profile runs
from any source, streaming or not, and produces the same numbers either way.

**`--profile` (which pipeline):**

- **`cerebellar_rs`** (default): 4 cerebellar ROIs, 20-frame trim,
  full-recording baseline and correlation. The original behaviour of this script.
- **`cerebellar_stim`**: as above but with a pre-stimulus baseline (MATLAB
  `1:278`) and a stimulus-window correlation (MATLAB `280:300`).
- **`cortical_gsr`**: 22 cortical ROIs, **no** trim, full-recording baseline,
  plus two extra stages — a **brain mask** and **global signal regression** —
  between the correction and the ROI means. Requires `--mask PATH`.

Profiles are presets, not a closed set: build your own `wfci.Profile` (or
`dataclasses.replace` one of these) for a study-specific setup, without editing
the package. `--mode resting_state|stimulated` still works as a **deprecated
alias** for the two cerebellar profiles.

**`--atlas PATH` (optional, any profile):**

A YAML/JSON ROI layout to use **instead of** the profile's built-in one — see
[Bringing your own ROI atlas](#bringing-your-own-roi-atlas). Everything else about
the profile (windows, trim, mask/GSR stages) still applies, so this stays "the
cortical pipeline, on my ROIs".

**`--mask PATH` (required by `cortical_gsr`, rejected by the others):**

A single 2-D TIFF marking the brain: non-zero inside, zero outside. It must be
drawn on the **once-downsampled** FOV — e.g. **256×256** for 512×512 raw frames —
matching the MATLAB's `imresize(Mask,0.5,'box')` in
`Antea_scripts/(2)_Global_Signal_Regression_SCRIPT.txt`; the pipeline applies
that same 0.5× box resize to land it on the final 128×128 grid. Masked-out pixels
become `NaN` for every frame and trial, and every later average ignores them. A
wrong-resolution mask is an error, not a broadcast.

**`--channel-order` (interleaved folders only):**

- **`auto`** (default): identify the channels by brightness — the **dimmer**
  group is GCaMP, the brighter one is the reflectance (emo) channel.
- **`gcamp_first` / `emo_first`**: assign by position instead, reading no pixels
  at all. The MATLAB scripts do this implicitly (cerebellar: GCaMP first;
  cortical: emo first), and getting it wrong **silently swaps the channels** and
  inverts the hemodynamic correction. Use these only when you know the layout and
  the brightness heuristic misfires (e.g. an unusually bright GCaMP recording).

**`--source` (storage format):**

- **`stack_file`** (default): each channel is one **multi-page TIFF** (one page
  per time frame). Full-load uses `load_stack(path)`. Each trial is a
  `(gcamp, emo)` pair.
- **`frame_folder`**: each channel is a **folder of single-page TIFFs** sorted by
  filename (e.g. `R11_00001.tif …`). Full-load uses `load_frame_folder(path)`.
  Each trial is a `(gcamp_folder, emo_folder)` pair.
- **`interleaved_folder`**: a **single folder** whose single-page TIFFs hold
  **both channels acquired alternately** — odd-positioned images (1st, 3rd, 5th,
  …) are one channel, even-positioned (2nd, 4th, …) the other. The split assigns
  the **dimmer** group (by mean of the top-10% pixels of the first image in
  each group) to **gcamp**, the brighter to **emo**. Full-load uses
  `load_interleaved_folder(folder)`. Here each trial is a **single folder path**,
  not a `(gcamp, emo)` pair. The sample `data/` folder is of this kind; inspect
  the split with
  [src/inspect_channel_intensity.py](src/inspect_channel_intensity.py).

**`--streaming` (memory strategy):**

- **`--no-streaming`** (default): load each trial's full `[y, x, time]` stacks
  into RAM. Keeps `dff_stack`, so the step-2 overlay is available.
- **`--streaming`**: read **frame-by-frame in constant memory** (two passes over
  each channel). Use this for large recordings that do not fit in RAM (the MATLAB
  `>4 GB` case), with **any** `--source` — e.g. `--source interleaved_folder
  --streaming` splits the folder up front and streams each channel without ever
  stacking it. No `dff_stack` is retained, so the step-2 overlay is unavailable.
  See [Large recordings](#large-recordings-streaming). Programmatically, build
  `wfci.FrameSource` objects (`tiff_frame_source`, `frame_folder_source`,
  `folder_frame_source`) and pass them to `wfci.run_streaming(...)`.

**`--debug-max-frames N` (debug mode, optional):**

Use only the **first N frames per channel** of every trial, for a fast smoke run
over a subset of a recording; omit it (or set `RUN_CONFIG["debug_max_frames"] =
None`) for a full run. Works with any `--source` and either memory strategy, and
stays lazy — only the frames actually used are read off disk. For
`--source interleaved_folder` a limit of N reads the folder's **first 2N images**,
since they alternate between the two channels, giving N frames per channel.

N must exceed the 20-frame `trim` (the run raises otherwise); `--mode stimulated`
uses a fixed `0:278` baseline and `279:300` correlation window, so N must be
≳ 320 there for the result to be meaningful. The run summary prints the active
limit so a truncated run is not mistaken for a full one.

All loaders return a float64 array shaped `[y, x, time]` (MATLAB convention).
Images are 16-bit; math is done in float64 to match MATLAB's `double`.

**Per-animal parameters** you must set (as in the MATLAB scripts):
- **Bregma** `bregma_row`, `bregma_col` (full resolution; `floor(.../2)` is
  applied internally to get the downsampled `y_1`, `x_2`).
- ROI box offsets — defaults live in `wfci.config.ROIConfig` and match
  `matlab/step3_ROI_functional_connectivity.m`; override per animal if needed.
- Frame `trim` (default 20 → MATLAB `21:end`), baseline window, and correlation
  window — the resting-state vs stimulated defaults are wired into
  `run_resting_state` / `run_stimulated`.

**Sample data:** `data/R11_00001.tif … R11_00007.tif` — 7 single-page 512×512
frames of one channel, used by the parity test and example.

---

## Bringing your own ROI atlas

ROI geometry is **experimental design**: which regions you care about, and where
they land, depends on your preparation and your rig — not on the analysis maths.
So the library never *requires* its presets. `CEREBELLUM_4` and `CORTEX_22` are
validated defaults that reproduce the two MATLAB pipelines; a study can bring its
own layout and keep it in a file it owns, next to its Bregma values and trial
list.

**Drawing it by eye.** [`roi_editor.py`](roi_editor.py) shows the first image of each
recording on the final analysis grid and lets you drag the boxes and Bregma onto the
anatomy, then writes a **ROI set** — an atlas file plus the Bregma the boxes were
drawn from — that `run_intermingle_rs.py` (`--roi-set`) and `run_botox_batch.py`
(`--roi-set-dir` per animal, or `--roi-set` for one shared layout) load directly. Full
walkthrough: [docs/ROI_EDITOR.md](docs/ROI_EDITOR.md).

Export a preset as a starting point, edit it, run it:

```python
from wfci import CORTEX_22, save_atlas
save_atlas(CORTEX_22, "my_study/atlas.yaml")     # one line per ROI, ready to edit
```

```yaml
name: motor_only
grid: [128, 128]          # the FINAL frame these offsets were drawn for
source: derived from CORTEX_22, motor regions only, 2026-03
boxes:                    # ORDER IS THE COLUMN ORDER OF R
  M2L_alta:  {row_start: -28, row_end: -23, col_start: -15, col_end: -10}
  M1L_alta:  {row_start: -17, row_end: -12, col_start: -32, col_end: -27}
  M2R_alta:  {row_start: -28, row_end: -23, col_start:  10, col_end:  15}
  M1R_alta:  {row_start: -17, row_end: -12, col_start:  27, col_end:  32}
```

```bash
conda run -n letizia python run_pipeline.py --profile cortical_gsr \
    --atlas my_study/atlas.yaml --mask mask.tif \
    --source interleaved_folder --trial /path/folder \
    --bregma-row 126 --bregma-col 126
```

Offsets are **MATLAB-style: 1-based and inclusive**, relative to Bregma — written
exactly as in the original scripts, so a file can be diffed against them by eye.
Selecting fewer ROIs is a *view*, not a different analysis: everything upstream
(correction, masking, the global signal) is computed over the whole brain, so the
surviving ROIs come out bit-for-bit unchanged.

### Why `grid` matters

**The boxes are not anatomy.** They are anatomy projected through one optical
setup — a field of view, a magnification, a downsampling. The same region sits at
different pixel offsets on a different rig. `grid` declares the **final** frame
(after both 0.5× downsamples) the offsets were drawn for, and the pipeline refuses
to run against anything else.

That check is not bureaucracy. Without it the failure is silent:

- a box running off the **left** edge is a *negative* index, and NumPy reads
  negative indices **from the far end** — so a left-hemisphere ROI quietly
  averages the **right** side of the brain and returns a perfectly ordinary
  number. (MATLAB raises here; the Python port was more permissive than its
  source.)
- an atlas drawn for a **different FOV** whose boxes all still fit is wrong by a
  scale factor while every ROI lands on real pixels. Bounds-checking cannot see
  this; only a declared `grid` can.

Both now raise, naming the ROI and the coordinate. Omit `grid` (or set it to
`None`) if you genuinely don't know — you keep the bounds check and lose only the
wrong-FOV one.

---

## Expected output

`run_pipeline.py` prints the profile, the stage chain it ran, and the `R_mean`
connectivity matrix; unless `output_path` is `None`, it writes an `.npz` (default
`outputs/connectivity.npz`) containing:

| Array | Shape | MATLAB name |
|-------|-------|-------------|
| `dff_stack` | `[y, x, time, trial]` | `t_TEMP_resize1` (in-memory runs only) |
| `temp_roi` | `[time, n_rois, trial]` | `TEMP_ROI` |
| `R` | `[n_rois, n_rois, trial]` | `R` |
| `R_mean` | `[n_rois, n_rois]` | `R_mean` |
| `averaged_traces` | `[time, n_rois]` | `averaged_traces` |
| `roi_labels` | `[n_rois]` | — (the row/column order of `R`) |
| `profile` | scalar string | — (which pipeline produced this) |

`n_rois` is 4 for the cerebellar profiles and 22 for `cortical_gsr`. Column order
is the profile atlas's own order — `[Laterale_L, Verme_L, Laterale_R, Verme_R]`
for the cerebellar profiles, and for `cortical_gsr` all 11 **left** regions then
all 11 **right** (matching the MATLAB's `ALL = cat(2, regioni_L, regioni_R)`).
`roi_labels` is saved alongside the matrix because a 22×22 matrix is not
something you can label from memory afterwards.

`examples/run_example.py` additionally writes `examples/output/roi_overlay.png`
(step-2 ROI-placement overlay).

`benchmarks/benchmark_modalities.py` writes `outputs/modality_comparison.txt` —
a plain-text report comparing all four layouts (`stack` | `folder` | `stream` |
`interleaved`) on the sample dataset: per-layout **wall time** and **peak RAM**
(OS working set + algorithm allocations, each measured in an isolated
subprocess), a **cross-layout correctness** check (all layouts are fed identical
pixel data, so outputs must match to roundoff), and a **cross-check against the
MATLAB interleaved script** (`matlab/step_interleaved_intermingle.m`).
Intermediate inputs and per-layout `.npz` results are written under
`temporary_files/modality_bench/`.

---

## Group analysis across animals

`run_pipeline.py` produces one `R_mean` per animal. Comparing **groups** of
animals — the `DIFF = healthy − disease` of MATLAB steps 5–6 — is a separate,
generic layer (`wfci.cohort`, `wfci.significance`):

```python
from wfci import CohortTable, load_results, mask_by_adjacency, network_figure

# The study's callback says which group/condition each file is — the ONLY place
# that knowledge lives (the library never learns it).
def whose(path): return {"group": "healthy" if "PV" in path.stem else "disease"}

table   = CohortTable(load_results("outputs/*.npz", metadata_from=whose))
diff    = table.select(group="healthy").mean() - table.select(group="disease").mean()
network_figure(mask_by_adjacency(diff.matrix, nbs_adjacency), labels=diff.labels)
```

The **library holds no study knowledge** — no group name, animal, sex or day
(enforced by a grep test). Which animals exist and how they split lives in a
study script under [`experiments/`](experiments/), the *policy* layer you own and
edit. See [experiments/README.md](experiments/README.md) and the worked example
[`experiments/healthy_vs_disease_day4.py`](experiments/healthy_vs_disease_day4.py).
The network-based statistic (NBS) itself is **not** re-implemented — `wfci`
consumes an adjacency matrix; it does not compute one.

---

## Pipeline overview

The three original steps map to the package as follows:

- **Step 1 — hemodynamic correction → ΔF/F** (`wfci.correction`): trim the first
  20 frames, downsample ×0.5 (box), normalise each channel by its temporal mean,
  divide GCaMP ratio by `emo` ratio, express as `(ratio-1)*100` %, downsample
  ×0.5 again → `t_TEMP_resize1` `[y, x, time, trial]`. Resting-state uses the
  full-recording baseline; stimulated uses the pre-stimulus window (`1:278`).
- **Step 2 — ROI placement check** (`wfci.visualize`): overlay the four ROI
  boxes on a representative ΔF/F frame to confirm Bregma-relative placement.
- **Step 3 — ROI functional connectivity** (`wfci.roi`): average ΔF/F inside
  each ROI box (`nanmean`) → one trace per region, compute the per-trial 4×4
  Pearson correlation matrix, average across trials. Resting-state correlates
  over the full recording; stimulated over the stimulus window (`280:300`).

---

## Large recordings (streaming)

The MATLAB script (and the default `--no-streaming` mode) reads an entire
multi-page TIFF into memory as `double` before processing — at 512×512 × 8 bytes
≈ 2 MB per frame, a multi-GB recording is 10+ GB per channel. That is exactly the
case the MATLAB comment *"ricorda che se pesa più di 4 gb non lo legge"* warns
about.

The **streaming path** (`--streaming`, or `wfci.run_streaming`) avoids this, and
works with any `--source`.
Nothing in the analysis needs the whole stack at once: the baseline image is a
temporal mean (a running sum suffices), the ΔF/F correction and both ×0.5
downsamples are per-frame, and step 3 reduces each frame to four ROI averages.
So each trial is streamed from disk in **two passes** —

1. accumulate the baseline-window running sum → mean images `MIf`, `MIr`;
2. re-read frame by frame → ΔF/F → downsample → four ROI `nanmean` values,

keeping only a couple of small images resident regardless of recording length.
Results match the in-memory (MATLAB-validated) path to ~1e-16 (baseline-mean
summation order); see `tests/test_streaming.py`.

**Trade-offs:** two disk passes instead of one, and `dff_stack` is not retained
(streamed away), so the step-2 ROI overlay needs the in-memory (`--no-streaming`)
mode.

```python
from wfci import ROIConfig, run_streaming_resting_state

cfg = ROIConfig.from_bregma(bregma_row=121, bregma_col=134)
trials = [("gcamp.tif", "emo.tif")]            # file PATHS, not loaded arrays
result = run_streaming_resting_state(trials, cfg)
print(result.R_mean)                            # StreamingResult (no dff_stack)
```

---

## Directory map

```
letizia/
├── README.md                 ← this file
├── REFERENCE.md              ← canonical technical map
├── CLAUDE.md                 ← agent/operating conventions
├── pyproject.toml            ← package metadata (installs `wfci` from src/)
├── run_pipeline.py           ← editor/CLI entrypoint (RUN_CONFIG block)
├── roi_editor.py             ← interactive ROI/Bregma editor → roi_sets/*.yaml
├── run_intermingle_rs.py     ← RS connectivity on interleaved folders, one analysis
│                               per recording folder found below the input (study script)
├── run_botox_batch.py        ← BOTOX manifest batch, one analysis per t# recording
├── scan_botox_dataset.py     ← builds manifests/botox_restani_manifest.csv
├── roi_sets/                 ← ROI sets drawn with roi_editor.py (boxes + Bregma)
├── manifests/                ← dataset manifests (CSV)
├── docs/                     ← study write-ups (ROI_EDITOR.md, INTERMINGLE_RS_R1_t1.md)
├── .env/                     ← environment contract
│   ├── environment.yml       ← conda env `letizia`
│   ├── requirements.txt      ← pip deps
│   ├── install_linux.sh      ← one-command Linux installer (conda or venv)
│   ├── .envVariables         ← env vars
│   └── ENVIRONMENT_SETUP.md  ← setup notes
├── .vscode/                  ← interpreter + launch/debug configs
├── src/wfci/                 ← the package
│   ├── io.py                 ← TIFF loading (stack / frame-folder / frame iterator)
│   ├── resize.py             ← MATLAB-equivalent imresize(...,'box')
│   ├── correction.py         ← step 1: hemodynamic ΔF/F
│   ├── config.py             ← ROIConfig (Bregma + ROI boxes)
│   ├── atlases.py            ← Atlas + presets (CEREBELLUM_4 / CORTEX_22), load/save
│   ├── mask.py               ← brain mask → NaN (cortical stage)
│   ├── gsr.py                ← global signal regression (cortical stage)
│   ├── profiles.py           ← Profile presets: which pipeline to run
│   ├── roi.py                ← step 3: ROI traces + connectivity
│   ├── streaming.py          ← constant-memory pipeline for large files
│   ├── visualize.py          ← step 2: ROI overlay
│   ├── cohort.py             ← generic group layer (stack/select/mean/DIFF)
│   ├── significance.py       ← generic figures (mask by adjacency, network, bars)
│   └── pipeline.py           ← orchestration (correction → [mask] → [GSR] → ROI → conn.)
├── tests/
│   ├── test_parity.py        ← Python-vs-MATLAB numerical parity (cerebellar)
│   ├── test_streaming.py     ← streaming vs in-memory equivalence
│   ├── test_efficiency_invariants.py ← I1–I11: decode/open counts, constant memory
│   ├── test_atlas_transcription.py   ← CORTEX_22 pinned to the MATLAB source
│   ├── test_atlas_generality.py      ← pipeline sized by the atlas, not by "4"
│   ├── test_atlas_files.py           ← load/save an atlas a study owns
│   ├── test_roi_bounds.py            ← geometry that does not fit must raise
│   ├── test_cohort.py                ← group means/DIFF + the P2 boundary grep
│   ├── test_significance.py          ← masking, per-node reductions, headless figures
│   ├── test_mask.py          ← masking → NaN, thresholds, resolution checks
│   ├── test_gsr.py           ← vectorised OLS ≡ per-pixel lstsq
│   ├── test_streaming_gsr.py ← streaming GSR ≡ in-memory GSR, still 2 passes
│   ├── test_cli.py           ← --profile / --mode alias / mask rules
│   └── matlab_reference/
│       ├── gen_reference.m   ← generates reference.mat from sample data
│       └── reference.mat     ← MATLAB outputs (git-ignored; regenerate)
├── examples/
│   ├── run_example.py        ← end-to-end demo on sample data
│   ├── README.md
│   └── output/               ← example outputs
├── experiments/             ← study scripts (POLICY: groups, splits, figures)
│   ├── healthy_vs_disease_day4.py  ← worked group contrast (Antea steps 5-6)
│   ├── README.md
│   └── output/               ← study figures
├── benchmarks/               ← modality comparison (RAM / time / parity)
│   ├── benchmark_modalities.py  ← orchestrator + report writer (entrypoint)
│   ├── bench_worker.py          ← runs one layout in an isolated subprocess
│   └── bench_common.py          ← input prep + per-layout run helpers
├── outputs/                  ← modality_comparison.txt (benchmark report)
├── data/                     ← sample TIFFs (R11_00001..7.tif)
├── Antea_scripts/            ← original cortical MATLAB scripts (no parity reference)
├── matlab/                   ← original MATLAB scripts (kept for reference/parity)
│   └── step_interleaved_intermingle.m  ← interleaved modality reference
└── Microglial extracellular vesicles induce.pdf
```

---

## Validation against MATLAB

The port is checked against the original MATLAB pipeline numerically:

1. `tests/matlab_reference/gen_reference.m` reads the sample frames, builds a
   deterministic 2-trial two-channel dataset, and runs the pipeline math using
   **MATLAB's own** `imresize('box')`, `nanmean` and `corr`, saving both the
   exact inputs and all outputs to `reference.mat`.
2. `tests/test_parity.py` loads those identical inputs, runs the `wfci` package,
   and asserts every output array matches MATLAB's.

Result (MATLAB R2024a):

| Output | Max absolute difference |
|--------|------------------------|
| `imresize` box, incl. odd dims (3×3, 5×5, 7×5, 31×29) | **0** (exact) |
| `dff_stack` (ΔF/F, 128×128×7×2) | **0** (exact) |
| `TEMP_ROI`, `R`, `R_mean`, `averaged_traces` | ~1e-16 (float roundoff) |

Run it yourself:

```bash
matlab -batch "run('tests/matlab_reference/gen_reference.m')"   # once, to (re)build reference.mat
conda run -n letizia python -m pytest tests/ -s -v
```

### What is *not* validated this way: the cortical pipeline

The table above covers the **cerebellar** profiles only. There is **no MATLAB
reference for `cortical_gsr`**, and this is a deliberate, documented gap
([MERGING_PLAN.md](MERGING_PLAN.md) P4 / Phase 8) rather than an oversight. What
*is* checked:

| Claim | How | Result |
|-------|-----|--------|
| The 22 ROI boxes match the MATLAB source | `tests/test_atlas_transcription.py` parses `Antea_scripts/(3)` and `(4)` and compares every coordinate | exact |
| The scripts' two duplicate box lists agree | same test, re-run every time | identical |
| The column order matches `ALL = cat(2, regioni_L, regioni_R)` | same test | exact |
| GSR's vectorised OLS ≡ a per-pixel `lstsq` fit | `tests/test_gsr.py` | ~8e-15 |
| Streaming GSR ≡ in-memory GSR | `tests/test_streaming_gsr.py` | ~1e-14 |

So the *arithmetic* is pinned and the *transcription* is pinned to the scripts —
but nothing proves that running the original MATLAB on real data would produce
these numbers. Treat the cortical path as carefully-read, not machine-verified.

---

## Environment setup

### Windows (this machine)

```bash
CONDA="$USERPROFILE/miniconda3/condabin/conda.bat"
"$CONDA" env create -f .env/environment.yml     # creates env `letizia` (python 3.11)
"$CONDA" run -n letizia pip install -e .         # install the wfci package (editable)
"$CONDA" run -n letizia python -m pytest tests/ -s -v   # verify
```

### Linux — one command

[`.env/install_linux.sh`](.env/install_linux.sh) creates the environment,
installs `wfci` in editable mode, verifies every import and runs the test suite:

```bash
bash .env/install_linux.sh                # conda/mamba/micromamba → env `letizia`
bash .env/install_linux.sh --mode venv    # no conda: creates .venv/ from requirements.txt
bash .env/install_linux.sh --force        # recreate from scratch
bash .env/install_linux.sh --help         # all flags
```

Then `conda activate letizia` (or `source .venv/bin/activate`).

Dependencies: `numpy`, `scipy`, `pandas`, `tifffile`, `matplotlib`, `pyyaml`,
`pytest`. [`roi_editor.py`](roi_editor.py) additionally needs Tk — bundled with
conda-forge Python, but in venv mode install it system-wide
(`sudo apt install python3-tk`). All other scripts are headless. See
[.env/ENVIRONMENT_SETUP.md](.env/ENVIRONMENT_SETUP.md) for details.

---

## Notes & caveats

- **Per-animal tuning.** Bregma (`y_1`, `x_2`) and ROI box offsets must be set
  per animal (step 2 is the visual check). Frame trimming (`21:end`), baseline
  window (`1:278`), and stimulus window (`280:300`) are acquisition-timing
  dependent — the RS/stimulated defaults follow the original scripts.
- **imresize 'box'.** For even dimensions and scale 0.5 this is a 2×2 block
  mean; for odd dimensions the general MATLAB antialiasing algorithm is
  reproduced (`wfci.resize`). Both paths are validated above.
- **No 4 GB TIFF limit.** The MATLAB reader mis-read files >4 GB; `tifffile`
  does not have this constraint. For recordings too large to hold in RAM, use
  the [streaming path](#large-recordings-streaming) (`--streaming`), which
  reads frame-by-frame in constant memory.
- The original MATLAB scripts are preserved under `matlab/` (not deleted) — they
  are required to regenerate the parity reference.
