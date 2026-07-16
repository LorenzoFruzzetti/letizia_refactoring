# Wide-field Calcium Imaging — Cerebellar ROI Functional Connectivity (`wfci`)

Python package for analysing dual-channel wide-field imaging data (GCaMP
fluorescence + hemodynamic reflectance) acquired in mouse cerebellum. It
performs hemodynamic correction (ΔF/F), ROI placement relative to Bregma, and
region-to-region functional-connectivity analysis, for both **resting-state**
and **stimulated** (e.g. optogenetic) recordings.

This is a Python port of the original MATLAB pipeline (now under
[`matlab/`](matlab/)). The port is **numerically validated against MATLAB** to
machine precision — see [Validation against MATLAB](#validation-against-matlab).

> Canonical technical map of the repository: **[REFERENCE.md](REFERENCE.md)**.

The four ROIs analysed are the left/right **cerebellar vermis** (`Verme_L/R`)
and the left/right **lateral hemispheres** (`Laterale_L/R`). Background:
Falcicchia et al., "Microglial extracellular vesicles induce Alzheimer's
disease-like changes", *Brain Communications* 2023 (open access) — not
redistributed here; see `.gitignore`.

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
| Run the pipeline from CLI | `conda run -n letizia python run_pipeline.py --source stack_file --no-streaming --mode resting_state --trial gcamp.tif,emo.tif --bregma-row 121 --bregma-col 134` |
| Run on a large recording (constant memory) | `conda run -n letizia python run_pipeline.py --source stack_file --streaming --mode resting_state --trial gcamp.tif,emo.tif --bregma-row 121 --bregma-col 134` |
| Run on one folder of interleaved (odd/even) channels | `conda run -n letizia python run_pipeline.py --source interleaved_folder --no-streaming --mode resting_state --trial data --bregma-row 121 --bregma-col 134` |
| Stream a large interleaved folder (constant memory) | `conda run -n letizia python run_pipeline.py --source interleaved_folder --streaming --mode resting_state --trial /path/big_folder --bregma-row 121 --bregma-col 134` |
| Debug run: only the first N frames per channel | `conda run -n letizia python run_pipeline.py --source interleaved_folder --no-streaming --mode resting_state --trial data --bregma-row 121 --bregma-col 134 --debug-max-frames 60` |
| Runnable example on sample data | `conda run -n letizia python examples/run_example.py` |
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

---

## Expected input

Each **trial** consists of two channels: a GCaMP fluorescence stack and an
`emo` (hemodynamic / reflectance) stack. A run is described by **two independent
knobs**: `--source` (how the channels are stored on disk) and `--streaming`
(whether to load into RAM or read frame-by-frame in constant memory). Any
`--source` can be combined with either memory strategy.

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
  the **brighter** group (by mean of the top-10% pixels of the first image in
  each group) to **gcamp**, the dimmer to **emo**. Full-load uses
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

## Expected output

`run_pipeline.py` prints the 4×4 `R_mean` connectivity matrix and, unless
`output_path` is `None`, writes an `.npz` (default `outputs/connectivity.npz`)
containing:

| Array | Shape | MATLAB name |
|-------|-------|-------------|
| `dff_stack` | `[y, x, time, trial]` | `t_TEMP_resize1` |
| `temp_roi` | `[time, 4, trial]` | `TEMP_ROI` |
| `R` | `[4, 4, trial]` | `R` |
| `R_mean` | `[4, 4]` | `R_mean` |
| `averaged_traces` | `[time, 4]` | `averaged_traces` |

Region/column order is `[Laterale_L, Verme_L, Laterale_R, Verme_R]`.
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
├── .env/                     ← environment contract
│   ├── environment.yml       ← conda env `letizia`
│   ├── requirements.txt      ← pip deps
│   ├── .envVariables         ← env vars
│   └── ENVIRONMENT_SETUP.md  ← setup notes
├── .vscode/                  ← interpreter + launch/debug configs
├── src/wfci/                 ← the package
│   ├── io.py                 ← TIFF loading (stack / frame-folder / frame iterator)
│   ├── resize.py             ← MATLAB-equivalent imresize(...,'box')
│   ├── correction.py         ← step 1: hemodynamic ΔF/F
│   ├── config.py             ← ROIConfig (Bregma + ROI boxes)
│   ├── roi.py                ← step 3: ROI traces + connectivity
│   ├── streaming.py          ← constant-memory steps 1+3 for large files
│   ├── visualize.py          ← step 2: ROI overlay
│   └── pipeline.py           ← orchestration (RS / stimulated)
├── tests/
│   ├── test_parity.py        ← Python-vs-MATLAB numerical parity
│   ├── test_streaming.py     ← streaming vs in-memory equivalence
│   └── matlab_reference/
│       ├── gen_reference.m   ← generates reference.mat from sample data
│       └── reference.mat     ← MATLAB outputs (git-ignored; regenerate)
├── examples/
│   ├── run_example.py        ← end-to-end demo on sample data
│   ├── README.md
│   └── output/               ← example outputs
├── benchmarks/               ← modality comparison (RAM / time / parity)
│   ├── benchmark_modalities.py  ← orchestrator + report writer (entrypoint)
│   ├── bench_worker.py          ← runs one layout in an isolated subprocess
│   └── bench_common.py          ← input prep + per-layout run helpers
├── outputs/                  ← modality_comparison.txt (benchmark report)
├── data/                     ← sample TIFFs (R11_00001..7.tif)
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

---

## Environment setup

```bash
CONDA="$USERPROFILE/miniconda3/condabin/conda.bat"
"$CONDA" env create -f .env/environment.yml     # creates env `letizia` (python 3.11)
"$CONDA" run -n letizia pip install -e .         # install the wfci package (editable)
"$CONDA" run -n letizia python -m pytest tests/ -s -v   # verify
```

Dependencies: `numpy`, `scipy`, `pandas`, `tifffile`, `matplotlib`, `pytest`. See
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
