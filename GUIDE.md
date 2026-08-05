# GUIDE.md — a guided tour of the project (for humans)

This is the friendly companion to the reference docs. It answers three questions:

1. **What can I run?** — the entry points, and exactly how to use each.
2. **What does every file do?** — a plain-language map of every script and module.
3. **How do I do X?** — a task → command cheat-sheet.

Where you'll want the other docs instead:
- [README.md](README.md) — the canonical copy-paste commands and I/O contract.
- [LIBRARY.md](LIBRARY.md) — the full programming API, shapes, and the rules for
  editing the package safely.
- [REFERENCE.md](REFERENCE.md) — the terse module-by-module technical map.

Throughout, `CONDA` is this machine's conda launcher:

```bash
CONDA="$USERPROFILE/miniconda3/condabin/conda.bat"     # Git Bash
# PowerShell:  $CONDA = "$env:USERPROFILE\miniconda3\condabin\conda.bat"
```

Everything assumes the `letizia` conda environment (Python 3.11). First-time
setup:

```bash
"$CONDA" env create -f .env/environment.yml
"$CONDA" run -n letizia pip install -e .
```

On Linux both steps (plus verification and the test suite) are one command:

```bash
bash .env/install_linux.sh              # or --mode venv if conda is unavailable
```

---

## 1. The five things you can run

There are five entry points. Everything else is a library module (imported, not
run) or a helper. Each entry point works two ways: **from the editor** (open it,
edit the `RUN_CONFIG` block at the top, press Run — no arguments needed) or **from
the terminal** (pass flags). The editor way is there so you never have to
remember flags.

### A. `run_pipeline.py` — the main pipeline

**What it does.** Takes your recordings and produces a functional-connectivity
matrix. This is the workhorse: hemodynamic correction → (optional mask + global
signal regression) → ROI averaging → correlation.

**The three choices you make** (they combine freely):

| Choice | Flag | Options |
|--------|------|---------|
| Which pipeline | `--profile` | `cerebellar_rs`, `cerebellar_stim`, `cortical_gsr` |
| How the files are stored | `--source` | `stack_file`, `frame_folder`, `interleaved_folder` |
| How much to hold in RAM | `--streaming` / `--no-streaming` | streaming = for recordings too big for memory |

**Run the cerebellar pipeline** on two multipage TIFFs:

```bash
"$CONDA" run -n letizia python run_pipeline.py \
    --profile cerebellar_rs --source stack_file --no-streaming \
    --trial gcamp.tif,emo.tif --bregma-row 121 --bregma-col 134
```

**Run the cortical pipeline** (22 ROIs, needs a brain mask), streaming a big
interleaved folder:

```bash
"$CONDA" run -n letizia python run_pipeline.py \
    --profile cortical_gsr --source interleaved_folder --streaming \
    --trial /path/to/folder --mask mask.tif \
    --bregma-row 126 --bregma-col 126
```

**What you get.** It prints the `R_mean` matrix and the stage chain it ran, and
(unless you set `--output-path` to nothing) saves an `.npz` with `temp_roi`, `R`,
`R_mean`, `averaged_traces`, `roi_labels` and the profile name. Default location:
`outputs/connectivity.npz`.

**Useful extras.**
- `--atlas my.yaml` — use your own ROI layout instead of the profile's built-in
  one (see [task: my own ROIs](#i-want-to-use-my-own-rois)).
- `--channel-order gcamp_first|emo_first` — for interleaved folders, override the
  automatic brightness-based channel detection.
- `--debug-max-frames 60` — a fast smoke run on just the first 60 frames.
- `--mode resting_state|stimulated` — **deprecated**; it still works but maps onto
  the cerebellar profiles. Prefer `--profile`.

Full flag reference and the editor `RUN_CONFIG` block: see the top of
[run_pipeline.py](run_pipeline.py) and [README.md](README.md#entrypoints).

### B. `examples/run_example.py` — the 30-second demo

**What it does.** Runs the whole cerebellar pipeline on the bundled sample data
(`data/R11_*.tif`) so you can see it work without supplying anything. The sample
is only one channel, so it synthesises a second channel deterministically — this
is a *demo*, not real science.

```bash
"$CONDA" run -n letizia python examples/run_example.py
```

**What you get.** A 4×4 `R_mean` printed, plus `examples/output/roi_overlay.png`
(the step-2 figure showing where the ROI boxes land) and an `.npz` of results.
Good first thing to run to confirm your install works.

### C. `experiments/healthy_vs_disease_day4.py` — a worked group study

**What it does.** The *group-level* analysis: take one connectivity matrix per
animal, split them into groups, average within each group, and compute the
difference `DIFF = healthy − disease` — plus the significance figures (matrix,
network, bar plots). This corresponds to MATLAB steps 5–6.

```bash
"$CONDA" run -n letizia python experiments/healthy_vs_disease_day4.py
```

**What you get.** Three PNGs in `experiments/output/`: `diff_matrix.png`,
`network.png`, `barplots.png`. By default it runs on a **synthetic** cohort (so it
works with no data); point `RUN_CONFIG["results_dir"]` at a folder of real
`run_pipeline.py` outputs to use those.

**This is a template, not a fixed tool.** It is *your* file to copy and edit — the
cohort table (who's in which group), the selections, the figure choices all live
here, not in the library. See [experiments/README.md](experiments/README.md).

### D. `benchmarks/benchmark_modalities.py` — compare the four layouts

**What it does.** Runs the same data through all four storage/memory layouts
(`stack`, `folder`, `stream`, `interleaved`), each in an isolated subprocess, and
writes a report: wall time, peak RAM, a cross-layout correctness check, and a
cross-check against the MATLAB script (if MATLAB is installed).

```bash
"$CONDA" run -n letizia python benchmarks/benchmark_modalities.py
"$CONDA" run -n letizia python benchmarks/benchmark_modalities.py --no-matlab
```

**What you get.** `outputs/modality_comparison.txt`. Use it to confirm all layouts
agree and to see the RAM/time trade-offs.

### E. `benchmarks/benchmark_scaling.py` — will my big folder fit / how long?

**What it does.** Before committing to a huge (possibly network-mounted)
recording, this measures a few short streaming runs on *cold* chunks and
extrapolates total time and peak RAM. It also asserts the constant-memory
guarantee: peak RAM must not grow with frame count.

```bash
"$CONDA" run -n letizia python benchmarks/benchmark_scaling.py \
    --folder "\\\\server\\share\\animal\\t1" --limits 200,300,400
```

**What you get.** A printed scaling report (time slope, RAM slope, a full-folder
estimate). A near-zero RAM slope means streaming is genuinely O(1) in frame count.
Needs access to a real folder; it won't run on the tiny sample.

---

## 2. What every file does

### Things you run (entry points)

| File | One-liner |
|------|-----------|
| [run_pipeline.py](run_pipeline.py) | **Main entry point.** Recordings → connectivity matrix. |
| [examples/run_example.py](examples/run_example.py) | Demo of the pipeline on the bundled sample data. |
| [experiments/healthy_vs_disease_day4.py](experiments/healthy_vs_disease_day4.py) | Worked group study (cohort → DIFF → figures). |
| [benchmarks/benchmark_modalities.py](benchmarks/benchmark_modalities.py) | Compare the 4 layouts on time/RAM/correctness. |
| [benchmarks/benchmark_scaling.py](benchmarks/benchmark_scaling.py) | Estimate time/RAM for a big folder from short runs. |
| [src/inspect_channel_intensity.py](src/inspect_channel_intensity.py) | Utility: inspect an interleaved folder's odd/even channel brightness (which is GCaMP?). Writes a CSV. |
| [roi_editor.py](roi_editor.py) | **Utility (opens a pyqtgraph/Qt window).** Shows the first image of each recording on the analysis grid; drag the ROI boxes, drag Bregma to move the whole layout at once, or drag Lambda to rescale it, then save a ROI set. See [docs/ROI_EDITOR.md](docs/ROI_EDITOR.md). |
| [run_intermingle_rs.py](run_intermingle_rs.py) | Study script: RS connectivity on interleaved folders. Point `--folder` at one recording, an animal, or a whole day — every folder holding TIFFs below it is analysed separately into `<output-dir>\<animal>\<t#>\` (`--roi-set` to use drawn ROIs). |
| [run_botox_batch.py](run_botox_batch.py) | Study script: the BOTOX manifest batch — one analysis per `t#` recording, each in its own output subfolder (`--merge-recordings` to concatenate them into one trial per animal instead; `--roi-set-dir` / `--roi-set` for drawn ROIs). |
| [scan_botox_dataset.py](scan_botox_dataset.py) | Study script: scans the dataset share and writes `manifests/botox_restani_manifest.csv`. |

### The library (`src/wfci/`) — imported, not run

These are the building blocks. You only touch them if you're extending the
package. For the full API see [LIBRARY.md](LIBRARY.md).

| Module | What it handles |
|--------|-----------------|
| [io.py](src/wfci/io.py) | Reading TIFFs; the `FrameSource` that lets any storage format stream. **The efficiency-critical file** — see the do-not-break list in LIBRARY.md. |
| [resize.py](src/wfci/resize.py) | An exact reimplementation of MATLAB's `imresize(...,'box')`. |
| [correction.py](src/wfci/correction.py) | Step 1: hemodynamic correction to ΔF/F (%). |
| [config.py](src/wfci/config.py) | `Box` (one ROI) and `ROIConfig` (per-animal Bregma + boxes). |
| [atlases.py](src/wfci/atlases.py) | The ROI layouts: `CEREBELLUM_4`, `CORTEX_22`, and load/save from files. Each `Atlas` knows the field of view it's valid for. |
| [mask.py](src/wfci/mask.py) | The brain mask stage: pixels outside the brain become NaN. |
| [gsr.py](src/wfci/gsr.py) | Global signal regression (removes the brain-wide common signal). |
| [roi.py](src/wfci/roi.py) | Step 3: averaging ΔF/F inside each ROI, then the correlation matrix. Also the bounds check that stops an ROI silently reading the wrong pixels. |
| [profiles.py](src/wfci/profiles.py) | `Profile` — the bundle that defines "which pipeline" (atlas + windows + stages). |
| [pipeline.py](src/wfci/pipeline.py) | Ties the stages together (in-memory path). |
| [streaming.py](src/wfci/streaming.py) | The constant-memory version of the whole pipeline, including GSR. |
| [visualize.py](src/wfci/visualize.py) | Step 2: draw the ROI boxes on a frame to check placement. |
| [cohort.py](src/wfci/cohort.py) | **Group layer**: stack per-animal matrices, select subsets, average, difference. Knows nothing about your specific study. |
| [significance.py](src/wfci/significance.py) | **Figure layer**: mask a matrix by a significance adjacency, draw the network and bar plots. |

### Tests (`tests/`) — how the project proves itself

Run them all with `"$CONDA" run -n letizia python -m pytest tests/ -q` (156 tests).
Each file guards one thing; the map is in [LIBRARY.md §11](LIBRARY.md#11-test-map--how-to-know-you-didnt-break-anything).
The MATLAB reference used by the parity test lives in
`tests/matlab_reference/` (regenerate `reference.mat` with
`matlab -batch "run('tests/matlab_reference/gen_reference.m')"`).

### Reference material (not code)

| Location | What it is |
|----------|-----------|
| [matlab/](matlab/) | The **original cerebellar** MATLAB scripts. The Python is validated against these to machine precision. |
| [Antea_scripts/](Antea_scripts/) | The **original cortical** MATLAB scripts. Read but not runnable here; the Python transcription is pinned to them by a test, but there is no numerical parity reference (see below). |
| [data/](data/) | Sample TIFFs used by the demo and parity test. |
| [Antea_Scripts_README.md](Antea_Scripts_README.md) | Explains the cortical scripts and how the two pipelines differ. |

---

## 3. Task cheat-sheet — "I want to…"

### …just check the install works
```bash
"$CONDA" run -n letizia python examples/run_example.py
```

### …run my cerebellar recordings
Edit the `RUN_CONFIG` block at the top of [run_pipeline.py](run_pipeline.py) (set
`trials`, `bregma_row/col`, keep `profile="cerebellar_rs"`), then run it with no
flags. Or use the terminal command in [entry point A](#a-run_pipelinepy--the-main-pipeline).

### …run the cortical (22-ROI) pipeline
Use `--profile cortical_gsr` and supply `--mask mask.tif`. The mask is a single
TIFF drawn on the *once-downsampled* field of view (e.g. 256×256 for 512×512 raw),
non-zero inside the brain. See [README: the mask flag](README.md#expected-input).

### …handle a recording too big for RAM
Add `--streaming`. Everything else is identical and the numbers match to floating-
point roundoff. Unsure if it'll fit? Run [benchmark_scaling.py](benchmarks/benchmark_scaling.py) first.

### …use my own ROIs
Export a preset to a file, edit it (one line per ROI), and pass it in:
```bash
# python -c "from wfci import CORTEX_22, save_atlas; save_atlas(CORTEX_22, 'atlas.yaml')"
"$CONDA" run -n letizia python run_pipeline.py --profile cortical_gsr \
    --atlas atlas.yaml --mask mask.tif --source interleaved_folder \
    --trial /path/folder --bregma-row 126 --bregma-col 126
```
Details and the file format: [README: bringing your own ROI atlas](README.md#bringing-your-own-roi-atlas).

### …move the ROIs / Bregma onto *this* animal's anatomy
Draw them instead of guessing coordinates:
```bash
"$CONDA" run --no-capture-output -n letizia python roi_editor.py   # window: drag, then 's'
"$CONDA" run -n letizia python run_botox_batch.py --roi-set-dir roi_sets --full
```
Two drags do most of the work: the magenta `+` (**Bregma**) translates the whole layout,
and the green `x` (**Lambda**) rescales it — its distance from Bregma is the scale the
box offsets are in, so putting both markers on the animal's real landmarks fits the
atlas to *that* brain instead of nudging 22 boxes one at a time. Press `A` then `w` to
write ONE layout used by every session instead.
Full walkthrough: [docs/ROI_EDITOR.md](docs/ROI_EDITOR.md).

### …compare groups of animals (healthy vs disease)
Copy [experiments/healthy_vs_disease_day4.py](experiments/healthy_vs_disease_day4.py),
edit its `COHORT` table and the `.select(...)` lines, and point it at your folder
of per-animal `.npz` files. The library provides the averaging and figures; the
grouping is yours. See [experiments/README.md](experiments/README.md).

### …figure out which interleaved channel is GCaMP
```bash
"$CONDA" run -n letizia python src/inspect_channel_intensity.py \
    --input-dir data --output-path examples/channel_intensity.csv
```
`channel_order="auto"` assigns the **dimmer** of the two interleaved groups to
GCaMP (the reflectance/emo channel comes back brighter). If this recording is the
exception, force it with `--channel-order gcamp_first|emo_first`.

### …confirm a change didn't break anything
```bash
"$CONDA" run -n letizia python -m pytest tests/ -q
```
If you changed the cerebellar numeric path, `test_parity.py` must still pass
exactly — that's the promise the whole port rests on.

### …understand the code well enough to extend it
Read [LIBRARY.md](LIBRARY.md) start to finish. It has the full API, the array-shape
conventions (the easiest thing to get wrong), the worked recipes, and the list of
optimisations you must *not* undo.

---

## 4. A few things that will save you a headache

- **The channel order matters.** GCaMP and `emo` are not interchangeable — swap
  them and the hemodynamic correction inverts. For interleaved folders the tool
  auto-detects by brightness; if that's ever wrong, force it with
  `--channel-order`.
- **The atlas is tied to a field of view.** ROI boxes are pixel offsets, valid
  only for the FOV they were drawn on (128×128 here). Point the wrong FOV or a
  bad Bregma at them and the pipeline now *refuses* rather than quietly averaging
  the wrong region — if you see a "box falls outside the frame" error, check
  `--bregma-row/col` and that your atlas matches your data.
- **The cortical numbers are not MATLAB-validated.** The cerebellar path matches
  MATLAB to machine precision; the cortical path is a careful, test-pinned
  transcription with no numerical reference. Trust it as *carefully read*, not
  *proven*. (See [LIBRARY.md §10](LIBRARY.md#10-what-is-and-isnt-validated).)
- **Streaming keeps no `dff_stack`.** That's deliberate (it's what makes it
  constant-memory), but it means the step-2 overlay isn't available in streaming
  mode — use `--no-streaming` if you want to inspect ROI placement.
