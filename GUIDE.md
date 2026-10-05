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
CONDA="$USERPROFILE/anaconda3/condabin/conda.bat"     # Git Bash
# PowerShell:  $CONDA = "$env:USERPROFILE\anaconda3\condabin\conda.bat"
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
| [roi_editor.py](roi_editor.py) | **Utility (opens a pyqtgraph/Qt window).** Shows the first image of each recording on the analysis grid; drag the ROI boxes, drag Bregma to move the whole layout at once, or drag Lambda to rescale it, then save a ROI set. Editing one box of a bilateral pair mirrors its twin about Bregma (`m` unlocks), so a layout cannot be saved half-nudged. See [docs/ROI_EDITOR.md](docs/ROI_EDITOR.md). |
| [run_intermingle_rs.py](run_intermingle_rs.py) | Study script: RS connectivity on interleaved folders. Point `--folder` at one recording, an animal, or a whole day — every folder holding TIFFs below it is analysed separately into `<output-dir>\<animal>\<t#>\` (`--roi-set` to use drawn ROIs). |
| [run_botox_batch.py](run_botox_batch.py) | Study script: the BOTOX manifest batch — one analysis per `t#` recording, each in its own output subfolder (`--merge-recordings` to concatenate them into one trial per animal instead; `--roi-set-dir` / `--roi-set` for drawn ROIs). `--roi-set-dir` resolves `<day>_<animal>_<t#>.yaml` first, so `roi_sets/rebuilt/` gives every recording its own geometry. Writes `roi_fluorescence_*.csv` (per-ROI ΔF/F per frame) next to each `.npz`. `--save-data` additionally dumps the per-pixel ΔF/F and both channels' raw F as `.npy` volumes (~197 MB/recording; `--save-data-root` puts them on another disk). |
| [scan_botox_dataset.py](scan_botox_dataset.py) | Study script: scans the dataset share and writes `manifests/botox_restani_manifest.csv`. |
| [test_epileptic_detection.py](test_epileptic_detection.py) | Test script: epileptiform event detection on one `roi_fluorescence_full.csv` (running-median detrend, amplitude + rise peaks, z cut-off), with trace figures. |
| [epileptic_by_area_animal_day.py](epileptic_by_area_animal_day.py) | Study script: the same detection on every recording under `outputs/botox_restani_rebuilt`, one z cut-off calibrated over all of them, then tables of event counts and rates split by area, animal and day in `outputs/epileptic_by_area_animal_day/`. Its detection functions are copies of the test script's, so keep them in sync. |
| [test_roi_mean_detection.py](test_roi_mean_detection.py) | Test script: the cohort detection on ONE `roi_fluorescence_full.csv` (edit the parameter block at the top; no CLI flags), with that recording's events split by area and hemisphere. Imports the cohort's detection functions instead of copying them — verified bit-identical to the saved cohort run on `260611_PV5/t1` (419 peaks, 220 confirmed), the cut-off being the only difference: this recording's own top 1% unless `epileptic_z_threshold` is set to the cohort's. Draws an overall-activity overview (all ROIs averaged, raw with its running median, detrended and smoothed, then the two hemispheres), an event raster per ROI ordered by area with a count bar, and zooms of the largest events with every other ROI behind. Writes to `outputs/test_roi_mean_detection/<date>_<animal>/<t#>/`. |
| [epileptic_diagnostics.py](epileptic_diagnostics.py) | Library for the two scripts above: the shared per-recording figures — detrending, detections, rise rate, and `plot_roi_overview` (every ROI overlaid over the whole recording, the counterpart of `run_botox_batch`'s `roi_traces_full.png`). |
| [plot_epileptic_per_animal_day.py](plot_epileptic_per_animal_day.py) | Test script: reads `epileptic_by_area_animal_day.csv`, sums the events over all ROIs per animal and recording day, and draws one line per animal coloured by group, every animal by default (set `only_complete_animals = True` to keep only animals recorded on every day). Writes the PNG and the plotted table to `outputs/epileptic_median_dff_analysis/epileptic_per_animal_day/`; `input_csv` at the top selects which analysis is plotted (currently the pixel/`median_dff` one). |
| [epileptic_by_area_animal_day_pixels.py](epileptic_by_area_animal_day_pixels.py) | Study script: the same cohort detector, but the ROI traces are rebuilt from the per-pixel dumps in `pixel_data/` instead of `roi_fluorescence_full.csv`. `--signal-mode median_dff` (default) corrects each pixel as `(F/Fbar)/(R/Rbar) - 1` against a centred 20 s running-median baseline per channel, then averages inside each ROI box of the recording's own atlas (`--all-rois`, 22 boxes). `--signal-mode reflectance_ratio` is the older `F * mean_t(R) / R` in raw fluorescence units. Writes the cohort tables plus, for every recording, the extracted traces, the exact geometry and the event tables into `outputs/epileptic_by_area_animal_day_pixels_median_dff_wide/`, plus two figures (`roi_traces_all.png`, `roi_traces_rise.png`) for every recording of 5 random animals per group, drawn 30x wider so a single second is readable. |
| [cache_median_dff.py](cache_median_dff.py) | Study script: computes the per-pixel `(F/Fbar)/(R/Rbar) - 1` (20 s running-median baselines) over the whole saved crop of every `pixel_data/` dump, once, and saves it next to the dump as `pixels_median_dff_20s_full.npy` (float16) plus a `_meta.json` completion marker. Resumable; `--workers` (default 8) runs recordings in parallel, ~20 min for all 545. |
| [roi_pixel_connectivity.py](roi_pixel_connectivity.py) | Study script: the connectivity of `run_botox_batch.py` extended from ROI x ROI to ROI x pixel. For every `pixel_data/` dump, the 22 seed traces are the box means of the recording's own `roi_sets/rebuilt` atlas (they reproduce the batch's `R_mean` to 3e-6), and each is Pearson-correlated with every pixel of the crop, giving 22 maps of 76 x 87. Maps are Fisher-z averaged over a unit's `t#` and then over the units of each group (the crop is Bregma-relative, so pixels line up across animals). Writes per-recording `roi_pixel_connectivity.npz`, per-unit `unit_roi_pixel_connectivity.npz` + `seed_maps.png`, and `group_roi_pixel_connectivity.npz` + `group_<G>_seed_maps.png` to `outputs/roi_pixel_connectivity/<volume stem>/`. `--volume-file pixels_median_dff_20s_full.npy` correlates the median-baseline cache instead. `--gsr` first regresses the global signal (mean of the recording's ROI-box union) out of every pixel, as in `Antea_scripts/(2)_Global_Signal_Regression_SCRIPT.txt`; the folder gets a `_gsr` suffix and the maps a -1..1 diverging scale. Resumable; ~10 min serially for 544 recordings (`260828_PV7/t2` excluded, CLAUDE.md 9.23). |
| [pixel_roi_editor.py](pixel_roi_editor.py) | Utility (pyqtgraph window, like `roi_editor.py`): choose ROIs AFTER the pixel dumps exist, while looking at the computed signal. One page per `pixel_data/` recording; background cycles through time-mean GCaMP, temporal SD, seed r of the selected box and single frames, and the selected box's trace is plotted live below. Move/resize/add/delete/rename boxes (mirror lock on by default), Draw once: `A` makes the page a template and places it on every recording by that recording's Bregma and Lambda (from `roi_editor.py`, offsets scaled by Lambda ratio, sizes fixed); `S` also writes `_template.yaml`, which `--apply-template` re-applies without the window. Writes `roi_sets/pixel_selected/<day>_<animal>_<t#>.yaml`, which `epileptic_by_area_animal_day_pixels.py`, `epileptic_by_active_pixels.py` and `roi_pixel_connectivity.py` use with `--roi-set-dir roi_sets/pixel_selected` (outputs in a `_roi_pixel_selected` folder). Never runs an analysis itself. |
| [plot_hemisphere_traces.py](plot_hemisphere_traces.py) | Study script: the hemisphere counterpart of `roi_traces_full.png`. For every `pixel_data/` dump, averages every saved pixel left and right of the Bregma midline (columns within ±`midline_half_width_px`, default 3, dropped; no ROI boxes) into one trace per hemisphere, for three signals stacked in three panels, all in %: `pixels_dff_full.npy` (mean-baseline, emo-corrected dF/F0) and the 20 s and 60 s `pixels_median_dff_*` caches. Writes `hemisphere_traces.png` and `.csv` to `outputs/hemisphere_traces/<date>_<animal>/<t#>/`, plus `hemisphere_traces_gcamp_only.png`/`.csv`: the same three baselines on raw GCaMP with no emo correction (per-pixel `F/mean(F) - 1` and `F/Fbar - 1` with 20 s / 60 s running medians, computed from `pixels_f_gcamp_full.npy`). `--figures` selects which to draw; existing PNGs are skipped per figure. Resumable, `--workers` (default 8). |
| [plot_frame_montage.py](plot_frame_montage.py) | Figure script: paper-style filmstrips. Each panel tiles `n_tiles` consecutive frames (optionally averaged over `bin_frames`, blurred by `spatial_sigma_px`) of one recording's `pixel_data/` volume (default `pixels_median_dff_20s_full.npy`) into `tile_rows` rows, labelled in the corner; all panels share one colour scale (fixed `color_limits` or shared percentiles). Panels are set in `RUN_CONFIG["panels"]` or with repeated `--panel LABEL=DAY_ANIMAL/t#@START_S` (`@peak` centres on the largest field-mean dF/F). Writes `outputs/frame_montages/<output_name>.png` and a `.json` listing every tile's frames. |
| [test_drift_vs_design.py](test_drift_vs_design.py) | Test script (flat, CLAUDE.md §6): one number per recording from `outputs/hemisphere_traces/` -- the OLS slope (% dF/F per min) of the emo-corrected and of the GCaMP-only mean-baseline trace, hemispheres averaged, and `log2(SD last third / SD first third)` of the 20 s median trace -- then asks which design columns it follows. Variance partition (random animal + session within animal, the rest is recording within session), a mixed model with `group`, `mouse_line`, `day_index` and categorical `recording` as fixed effects tested one term at a time by likelihood ratio, Kruskal-Wallis on session means per column and Friedman on t1..t5 within session. `260828_PV7_t2` excluded (§9.27). Writes `recording_drift_metrics.csv`, `variance_partition.csv`, `mixed_model_terms.csv`, `factor_tests.csv` and figures to `outputs/drift_vs_design/`. Run: `conda run --no-capture-output -n letizia python -u test_drift_vs_design.py` |
| [test_nbs_change_over_time.py](test_nbs_change_over_time.py) | Test script (flat, CLAUDE.md §6): the `nbs` package on the per-recording `R_roi` of `roi_pixel_connectivity.py`, Fisher-z averaged per session. Keeps C57 animals of groups R and T (`groups`, `mouse_lines`). Each design is animal dummies + a time slope (`time_variable`: `day_index` or `days_since_first_session`), so the change is within-animal; sessions are permuted only within an animal. Analyses: pooled, R, T and the R-T slope difference, each for increase and decrease at t > 2.5 / 3.1 / 3.5. Writes `components.csv`, `sessions.csv`, the observed t maps, one network figure per analysis, t heatmaps of every edge, the permutation nulls, the per-animal trajectory of each significant component and the per-animal mean connectivity by session to `outputs/nbs_change_over_time/<time_variable>/`. Set `NBS_CONNECTIVITY_VARIANT` (e.g. `pixels_dff_full_gsr`, `pixels_median_dff_20s_full`, `pixels_median_dff_20s_full_gsr`) to read another connectivity folder; outputs then go to `outputs/nbs_change_over_time_<variant>/`. |
| [test_nbs_pre_post_boto.py](test_nbs_pre_post_boto.py) | Test script (flat, CLAUDE.md §6): same sessions and filters as `test_nbs_change_over_time.py`, but the time slope is replaced by a period indicator, preBoto = `day_index` < `post_start_day_index` (4), postBoto = the rest. Animal dummies keep the contrast within-animal, so only animals with sessions in both periods carry it (listed in `animal_periods.csv`); sessions are permuted only within an animal. Analyses: pooled, R, T and R-T (within-animal), plus `R_vs_T_preBoto` / `R_vs_T_postBoto`: R vs T inside one period, one observation per animal (its sessions in that period averaged, `animal_period_means.csv`), group labels permuted freely and enumerated exactly. postBoto has 3 R vs 3 T animals, only 20 relabellings, so its smallest p is 0.05 and it can never be significant; the script prints a NOTE. Both directions, t > 2.5 / 3.1 / 3.5. Writes `components.csv`, `sessions.csv`, the t maps, one network figure per analysis, t heatmaps, the permutation nulls, per-group preBoto / postBoto / difference mean matrices, per-period R / T / R-T mean matrices (`r_vs_t_mean_matrices.png`) and per-animal pre-vs-post plots to `outputs/nbs_pre_post_boto/`. Set `NBS_CONNECTIVITY_VARIANT` (e.g. `pixels_dff_full_gsr`, `pixels_median_dff_20s_full`, `pixels_median_dff_20s_full_gsr`) to read another connectivity folder; outputs then go to `outputs/nbs_pre_post_boto_<variant>/`. postBoto is also later, so treatment and session order are confounded. |
| [test_nbs_pv_group.py](test_nbs_pv_group.py) | Test script (flat, CLAUDE.md §6): same session averaging as `test_nbs_change_over_time.py`, restricted to the PV-CRE line. All six PV animals are PV-CRE, so the mouse line is held fixed and PV is compared with the same-line R (R6, R7) and T (T9-T14) animals. `PV_slope`: within-animal slope over `day_index` (PV3-PV5 have days 1-4, PV6-PV8 only 1-2), sessions permuted within an animal. `PV_vs_R` / `PV_vs_T` / `PV_vs_RT`: one observation per animal, the mean of its sessions in `between_day_indices` (1-2, the only days every PV animal has, `animal_means.csv`), group labels enumerated exactly; PV vs R has 28 relabellings, smallest p 0.036. `exclude_recordings` drops `260828_PV7_t2` (§9.23/§9.27). Both directions, t > 2.5 / 3.1 / 3.5. Writes `components.csv`, `sessions.csv`, the t maps, one network figure per analysis, t heatmaps, the permutation nulls, PV / R / T mean matrices with PV - R and PV - T (`group_mean_matrices.png`) and per-animal mean z over days (`pv_by_day.png`) to `outputs/nbs_pv_group/`. Set `NBS_CONNECTIVITY_VARIANT` (e.g. `pixels_dff_full_gsr`, `pixels_median_dff_20s_full`, `pixels_median_dff_20s_full_gsr`) to read another connectivity folder; outputs then go to `outputs/nbs_pv_group_<variant>/`. Days 3-4 come from PV3-PV5 only, so the slope beyond day 2 rests on three animals. |
| [analyze_peak_tails.py](analyze_peak_tails.py) | Study script (flat, CLAUDE.md §6): for every recording of `pixel_data/experimental_design.csv` averages `pixels_median_dff_60s_full.npy` inside the 22 atlas boxes plus `Cortex_mean`/`CortexL_mean`/`CortexR_mean`, finds every peak with `scipy.signal.find_peaks` at a deliberately low prominence (0.1 % dF/F) after dropping the first 20 s (onset artifact), and keeps the top 5 % of each trace's peaks. Tail metrics are reported in % dF/F and in robust z of the trace, because noise differs 2-3x between recordings. Outputs in `outputs/peak_tails_60s/`: per recording x trace summary, every tail peak, animal-day and group x ROI aggregates, a ranking, and figures (group survival functions, per-recording strips, per-animal days, ROI x group heatmap, top recordings). Run: `conda run --no-capture-output -n letizia python -u analyze_peak_tails.py` |
| [analyze_peak_tails_emo.py](analyze_peak_tails_emo.py) | Study script (flat, CLAUDE.md §6): the counterpart of `analyze_peak_tails.py` with the signal changed and nothing else. Per saved pixel it takes `pixels_f_gcamp_full.npy / pixels_f_emo_full.npy` and averages that ratio in the same 22 atlas boxes; `signal_mode` then says what baseline it gets. `"raw"` divides by the recording's own median ratio — one constant, so the arbitrary ratio level goes but no trend does — and `"detrended"` divides by the trace's own `detrend_window_s` running median, a single high-pass where the median dF/F path applies one per pixel and the detector another on the ROI mean (§9.22). Same prominence (0.1 %), same top 5 %, same 20 s skip and same exclusion in both. `drift_ptp_pct` reports the span of each trace's own 60 s running median: the trend in `"raw"`, the residual in `"detrended"`. Each mode writes its own folder (`outputs/peak_tails_emo_ratio/`, `outputs/peak_tails_emo_ratio_detrended60s/`) with the same tables and figures as the dF/F run, plus `tail_time_course.png` and a `tail_vs_<run>.{csv,png}` against every other run listed in `reference_runs`. Measured: `"raw"` tails are mostly the bleaching trend (4.1x the uniform share of tail peaks in the first tenth of the recording), while `"detrended"` reproduces the median dF/F run (Spearman 0.99 per recording on every tail metric). Reading the raw F volumes is 158 MB per recording and the disk saturates at ~36 MB/s, so the first pass is ~40 min; both modes then share the cached ratio traces in `outputs/peak_tails_emo_ratio/roi_traces.npz` and run in about a minute. Run: `conda run --no-capture-output -n letizia python -u analyze_peak_tails_emo.py` |
| [epileptic_by_active_pixels.py](epileptic_by_active_pixels.py) | Study script: the same cohort detector, but the signal is the fraction of atlas-box pixels per hemisphere whose cached median dF/F exceeds `--pixel-z-threshold` robust SDs of that pixel's own trace (two traces, `CortexL_active`/`CortexR_active`). Amplitude and rise both come from that fraction; the rise from the smoothed fraction unless `--no-rise-on-smoothed`. At 3 SD the fraction is mostly 0 and the detector cannot scale it, so pass a lower threshold (0.5-2). Writes to `outputs/epileptic_by_active_pixels_<dff_source>_rise_<smoothed|detrended>/`. |
| [test_active_pixel_detection.py](test_active_pixel_detection.py) | Test script: the active-pixel detection on ONE `pixel_data/<date>_<animal>/<t#>/` dump (edit the parameter block at the top; no CLI flags). Imports the cohort script's `active_pixel_fraction` and checks its own traces against it, so the two cannot drift. Draws what the cohort run does not: the pixels counted per hemisphere, each pixel's active rate, the active map at the largest events, and an overall-activity overview (all counted pixels pooled into one `Cortex_active` trace -- raw with its running median, detrended and smoothed, and the two hemispheres for comparison). The overall trace is drawn and saved but not detected in. The cut-off is this recording's own top 1% of frames, so counts are not comparable across recordings. Writes to `outputs/test_active_pixel_detection/<date>_<animal>/<t#>/`. |
| [build_epileptic_groups.py](build_epileptic_groups.py) | Utility: writes `outputs/epileptic_groups/batch_summary.csv` (`day,animal,group`) from both manifests, checking that no unit has two groups, that groups agree with the connectivity summary, and that every `pixel_data/` unit is covered. Pass that folder as `--input-root` to the pixel detectors when the connectivity summary lacks a cohort. |
| [build_experimental_design.py](build_experimental_design.py) | Utility: writes `pixel_data/experimental_design.csv`, one row per recording folder under `pixel_data/`, with `day`, `animal`, `group` (the summary's `P` relabelled `PV`), `mouse_line` (`PV-CRE`/`C57`, from the hand-recorded `MOUSE_LINES` map at the top of the file -- it does not follow the group, both R and T contain each line), `day_index`/`animal_day` (that animal's sessions numbered chronologically from 1), `recording`/`recording_index`, the per-animal totals and the relative `pixel_dir`. Groups come from `outputs/epileptic_groups/batch_summary.csv`, so run [build_epileptic_groups.py](build_epileptic_groups.py) first. Raises if a unit has no group, an animal appears under two groups, or an animal is missing from `MOUSE_LINES`. |
| [plot_pixel_detection_comparisons.py](plot_pixel_detection_comparisons.py) | Test script: for a sample of the recordings of a saved pixel analysis, one row per ROI with the starting signal and its running median on the left and the detection signals on the right. Reads the traces, event flags and settings from the analysis — no redetection — and takes the trace file name, unit and titles from its `detection_config.json`, so it works on either `--signal-mode` tree. `--sample-count` (default 5) bounds the cost, each figure being ~8 MB for 22 ROIs; `--add-recording` pins extra ones. Writes to `outputs/epileptic_median_dff_analysis/sampled_plot_comparisons/`. |

`run_botox_batch.py` defaults to `workers=1`: every `t#` runs in the main process,
with no multiprocessing queue or spawned-worker DLL loading. With no arguments it
runs full streaming against `roi_sets/rebuilt` and resumes completed `.npz` jobs.
Values above 1 enable the experimental process-pool path, which is not recommended
for the full batch on this Windows machine because native worker crashes persist.
If an interruption left a completed result without its `batch_summary.csv` row,
the next run reconstructs that row from the saved metadata. Serial output is live;
Ctrl+C stops cleanly and completed results remain resumable.

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

The Network-Based Statistic lives in its own small package, [src/nbs/](src/nbs/). It follows [NBS_ALGORITHM.md](NBS_ALGORITHM.md), turns per-animal connectivity matrices into significant sub-networks, and hands each one's `adjacency` to `significance.py`. Try it with `examples/nbs_synthetic_example.py`.

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
"$CONDA" run -n letizia python run_botox_batch.py --roi-set-dir roi_sets/rebuilt --full
```
Two drags do most of the work: the magenta `+` (**Bregma**) translates the whole layout,
and the green `x` (**Lambda**) rescales it — its distance from Bregma is the scale the
box offsets are in, so putting both markers on the animal's real landmarks fits the
atlas to *that* brain instead of nudging 22 boxes one at a time. Press `A` then `w` to
write ONE layout used by every session instead.

Going through many pages: `c` (button: *Copy previous page's ROIs*) puts the previous
page's boxes, scale and Bregma on the current one, so a layout adjusted for one
recording can be carried to the next and only touched up. When you are done,
`ctrl`+`s` (button: *Save modified (N)*) writes a file for every page you changed and
leaves the untouched ones alone — `S` still rewrites all of them.
Full walkthrough: [docs/ROI_EDITOR.md](docs/ROI_EDITOR.md).

### …pick new ROIs after the pixel dumps are computed
```bash
"$CONDA" run --no-capture-output -n letizia python pixel_roi_editor.py   # edit, then A and S
"$CONDA" run --no-capture-output -n letizia python epileptic_by_area_animal_day_pixels.py --roi-set-dir roi_sets/pixel_selected
```
The window shows the saved crop of each recording. Press `b` to switch between
anatomy (mean F), activity (SD), the correlation map of the selected box (seed r)
and single frames. Double-click adds a box, and with mirror lock on its twin is
added too. Draw on ONE recording only, then `A`: every other recording gets the
same boxes placed by its own Bregma, with the offsets scaled by its Lambda distance
over the template's (both from `roi_editor.py`). Page through a few to check, then
`S`. Later, `python pixel_roi_editor.py --apply-template roi_sets/pixel_selected/_template.yaml`
redoes all of them without the window.

### …compare groups of animals (healthy vs disease)
Copy [experiments/healthy_vs_disease_day4.py](experiments/healthy_vs_disease_day4.py),
edit its `COHORT` table and the `.select(...)` lines, and point it at your folder
of per-animal `.npz` files. The library provides the averaging and figures; the
grouping is yours. See [experiments/README.md](experiments/README.md).
To test which sub-network differs, run `nbs.nbs` on the Fisher-z matrices, one per
animal or session and never one per recording. See
[examples/nbs_synthetic_example.py](examples/nbs_synthetic_example.py) and the
pitfalls in [NBS_ALGORITHM.md](NBS_ALGORITHM.md).

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

## Cortex activity relative to emo

`test_cortex_activity_emo.py` is the editor-run balanced raw-data pilot.
`analyze_cortex_activity_emo.py` processes the cohort with per-pixel
`q - running_median(q) + mean(running_median(q))`, then unique-pixel ROI-union
aggregation and candidate event intervals. `compare_cortex_activity_conditions.py`
uses saved tables for exposure-aware GEE and animal bootstrap comparisons.

```powershell
conda run --no-capture-output -n letizia python -u test_cortex_activity_emo.py
conda run --no-capture-output -n letizia python -u analyze_cortex_activity_emo.py
conda run --no-capture-output -n letizia python compare_cortex_activity_conditions.py
```

See [inputs, outputs, configuration and interpretation](docs/CORTEX_ACTIVITY_EMO_USAGE.md).
Pixel data and existing analyses are preserved. Thresholds require experimental
validation; the three-recording pilot is not a condition comparison.

Cortex/emo result gallery: `conda run --no-capture-output -n letizia python plot_cortex_activity_emo.py`.
Reads completed tables/trace archives (cohort if complete, otherwise pilot), writes
six PNG/SVG plots, plotted-data CSVs and an HTML gallery under
`outputs/cortex_activity_emo*/plots/60s_factor1/`. See
[plot usage and interpretation](docs/CORTEX_ACTIVITY_EMO_USAGE.md#cortexemo-result-plots).

Multiprocessing/resume (2026-10-01):
`conda run --no-capture-output -n letizia python analyze_cortex_activity_emo.py --workers 2`.
Workers process separate recording directories; only the parent builds aggregate
CSVs, in design order. Cache checks read metadata/file fingerprints before raw
volumes. Worker count does not invalidate results. Exact pre-parallel implementation
hash compatibility preserves completed results; scientific settings, source files,
geometry and dependency versions must still match. Two workers are the default
for the USB data drive. `processing_status.csv` records cached/computed/failed
recordings. A recording failure no longer aborts other jobs: after processing the
remaining recordings, any failures are saved to `processing_failures.json` and the
run raises without marking an incomplete cohort complete. The QC rules are unchanged.
Windows wrappers calling `run()` must use an `if __name__ == '__main__':` guard.

For plots when some recordings failed QC, explicitly select the completed subset:
`conda run --no-capture-output -n letizia python plot_cortex_activity_emo.py --input-root outputs/cortex_activity_emo --allow-incomplete`.
This verifies each recording's cache provenance, reads its saved tables, and labels
omitted recordings in the figures, gallery and plot manifest. It does not mark the
cohort complete, change exclusions, or recompute signals. Without this flag, an
incomplete run is still rejected.
