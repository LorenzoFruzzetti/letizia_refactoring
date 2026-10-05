# Cortex activity relative to emo

Implemented study scripts, separate from the `wfci` library. Read the scientific
[proposal](CORTEX_ACTIVITY_EMO_PROPOSAL.md) for interpretation and limitations.

## Run

From the repository root, in the existing `letizia` environment:

```powershell
# Balanced pilot: first eligible recording from each design group.
conda run --no-capture-output -n letizia python -u test_cortex_activity_emo.py

# Equivalent explicit, reproducible pilot selection.
conda run --no-capture-output -n letizia python -u analyze_cortex_activity_emo.py --config examples/cortex_activity_emo_pilot.json
conda run --no-capture-output -n letizia python compare_cortex_activity_conditions.py --output-root outputs/cortex_activity_emo_pilot --bootstrap 100

# Cohort, followed by comparisons from saved tables.
conda run --no-capture-output -n letizia python -u analyze_cortex_activity_emo.py
conda run --no-capture-output -n letizia python compare_cortex_activity_conditions.py

# Targeted validation.
conda run --no-capture-output -n letizia python -m pytest tests/test_cortex_activity_emo.py -q
```

Editor execution uses `RUN_CONFIG` in the cohort and comparison scripts. The
flat pilot has its parameters directly below the imports. CLI overrides editor
defaults; `--config` takes a JSON object containing any `RUN_CONFIG` keys. Unknown
keys fail. `--help` lists convenience flags. No environment changes are needed.

## Inputs

- `pixel_data/experimental_design.csv`: unique `recording_id` and `pixel_dir`,
  plus `animal`, `group`, `mouse_line`, `day`, `day_index`, `recording_index`.
- Each `pixel_dir`: float32 NumPy arrays `pixels_f_gcamp_full.npy` and
  `pixels_f_emo_full.npy`, shape `[time,y,x]`, and `pixels_meta_full.npz`.
- The ROI YAML referenced by each metadata file, with matching Bregma, grid,
  downsampling and ROI boxes. Geometry uses the existing checked helpers.
- Optional artifact CSV: `recording_id,start_s,stop_s`, half-open seconds relative
  to the saved recording. Whole frames in these intervals are invalidated.

The metadata inspected in the pilot has no timestamps, channel order or sample
rate. The configured rate is therefore explicitly recorded as an assumption:
10 Hz **per channel**, paired by saved index, without interpolation. A stored
`sampling_rate_hz` must agree with the configuration. The known acquisition
artifact `260828_PV7/t2` is excluded by default.

## Exact processing

For each pixel, in float64:

```text
q = F_gcamp / F_emo
b = centered running median(q)
mean_b = arithmetic mean of all finite b samples, including partial edge windows
residual = q - b
corrected = residual + mean_b
```

The default 60 s baseline is 601 frames; 20 and 120 s are sensitivity variants.
There is one median stage, no baseline division, dF/F, second detrend or GSR.
Every gap/artifact invalidates all median windows intersecting it. No valid
baseline for a pixel raises an error. Partial recording-edge windows are flagged.

Masks are the unique-pixel union of saved cortical ROIs, left, right and each
individual ROI. Pixels with more than 1% invalid channel samples are removed
for the entire recording; an empty region raises. Near-zero/nonpositive emo,
nonpositive GCaMP and nonfinite channels are invalid. A spatial-median raw
channel jump exceeding 5% flags that frame; this heuristic can also flag genuine
fast changes and must be inspected. Set `step_fraction=0` to disable it and use
annotations instead. Saturation in original sensor pixels cannot be determined
from these downsampled dumps.

Spatial means are formed **after** per-pixel subtraction. Trace archives also
contain explicitly separate equal-ROI means; they are not the primary area-weighted
signal. Positive activity is rectified before averaging. The high-frequency
first-difference MAD noise proxy and Gaussian positive-noise expectation are
diagnostics, not validated measurement noise or detection normalization.

## Candidate events and exposure

Defaults are exploratory common native thresholds, **not calibrated epilepsy
criteria**. They are configuration values selected before condition comparisons:

| Setting | Default |
|---|---:|
| Pixel residual threshold | 0.01 F_gcamp/F_emo ratio |
| Sensitivity multipliers of pixel threshold | 0.5, 1, 2 |
| Recruitment onset / offset | 0.10 / 0.05 of region pixels |
| Regional mean residual amplitude / prominence | 0.002 / 0.001 ratio |
| Minimum duration | 0.2 s |
| Maximum onset-to-peak rise time | 5 s |
| Merge gap | 0.3 s, only through valid time |

Detection uses recruitment hysteresis followed by duration, mean amplitude,
flank prominence and rise-time gates. The cortical event list merges the union
of accepted regional intervals, so asynchronous local events can contribute.
Left/right/ROI lists describe their own detections. **Filter events.csv by region**;
do not sum overlapping regional lists to obtain a cortical event count.
An event overlapping an analysis edge is flagged as censored. Long events are
flagged at half the baseline-window duration; masking them in a secondary slow
fit remains a follow-up if such events are common.

Primary `support=common` excludes the first/last 60 s with the default windows
(60 to 238 s for 298 s dumps). The margin is at least half the largest configured
window. Gaps/artifacts are also expanded by the longest configured window for
all primary variants, ensuring matched valid exposure. `support=method` uses each method's full-window support, still excluding
the first 20 s. A regional frame is valid only when every retained pixel in that
region has a valid baseline. Missing frames are excluded from exposure.

Thirty-second bins retain partial exposure. Event counts are assigned once to
the onset bin; occupied seconds are split across every overlapping bin. Burden
is occupied/valid seconds; rate is events/valid minutes. Native residual RMS,
MAD, positive-pixel activity, amplitudes and ratio-second AUC retain amplitude
information. Adding `mean_b` only affects display. It is not tonic activity and
does not restore amplitude loss from multiplicative bleaching.

## Outputs and resuming

Default root: `outputs/cortex_activity_emo/`; pilot:
`outputs/cortex_activity_emo_pilot/`. Inputs and existing analysis caches are not
modified. Recordings are processed by a configurable process pool (default two workers), with memory-mapped input files,
only ROI-union pixels loaded, and median working arrays chunked by 128 pixels.
Cohort tables stream to `.partial` files and replace final tables on completion;
condition comparisons read only cortical time-bin rows in chunks.

- `config.json`: formulas, units, all parameters, software versions and selection.
- `recording_qc.csv`: coverage, invalid channels, artifact flags, timing assumptions,
  raw-channel and ratio baseline slopes/relative changes and early/middle/late
  residual/noise diagnostics. Channel diagnostics use the explicitly labelled
  **median of the spatial mean**, separately from per-pixel endpoint processing.
- `window_metrics.csv`, `recording_metrics.csv`, `events.csv`: design keys,
  baseline method, support, threshold factor and region identify every row.
- Under each recording: the same tables, `traces_<window>s.npz`, channel CSVs,
  baseline/event overlays, spatial NPZ/PNG maps and exact empirical residual
  survival tables. Trace archives retain spatial summaries of q/b/residual/
  corrected/mean_b, validity, edge support, pixel membership, pixel baseline
  counts and means, noise proxies, recruitment and both aggregation schemes.
- `complete.json` per recording: parameter/source/design fingerprint for cache
  reuse. Large raw files use path, size and nanosecond modification time; small
  metadata/geometry files also use SHA256. This is not a full raw-file checksum.
  `--overwrite` forces recomputation. Changing parameters invalidates the cache.
- `run_complete.json`: written only after all selected recordings and aggregate
  tables finish. The comparison script requires this marker.
- Comparison outputs: `animal_day_metrics.csv`, `animal_metrics.csv`,
  `animal_time_metrics.csv`, `condition_contrasts.csv`, `design_overlap.csv`,
  `model_coefficients.csv`, `adjusted_trajectories.csv`, `model_diagnostics.json`,
  condition-time and threshold-sensitivity figures.

Comparison primary rows are `additive_60s`, `common`, factor 1, `cortex`.
Descriptive contrasts are animal-balanced, with whole-animal bootstrap CIs.
Adjusted count models use Poisson GEE with a log valid-minute offset and robust
animal covariance; a fixed-alpha=1 negative-binomial fit is a labelled sensitivity
model. Burden uses fractional-logit GEE weighted by relative valid exposure,
not an independent trial per frame. Models include group*time, line, ordinal
day and recording index when those covariates vary.

Adjusted comparisons restrict to mouse lines observed in all selected groups;
PV vs R vs T therefore does not extrapolate into PV/C57. Only line/day-index/
recording-index combinations observed in every selected group enter adjusted fits. Predictions average
over one shared animal-weighted covariate distribution at common times. Complete
animals are resampled within group for adjusted bootstrap intervals (independence
GEE coefficients use equivalent weighted GLM refits). Require >=3 animals/group,
>=8 total, a full-rank design, varying outcomes and successful convergence;
unstable or insufficient fits are recorded, not replaced with frame-level tests.
At least 80% and 30 successful bootstrap draws are required for adjusted CIs.
The pilot cannot establish condition differences. Acquisition-cohort overlap is
tabulated by calendar month; acquisition settings remain unavailable.

## Validation scope

Synthetic tests cover additive-slope pulse recovery, constant-offset invariance,
gap dilation, partial exposure, merging, event counts and recruitment, noise false
events, retained biological amplitude decline, residual multiplicative loss,
emo-only artifacts and absorption of sustained events. Statistical tests exercise
all three GEE families and clustered resampling. Blinded event review, acquisition
calibration and EEG/LFP validation remain experimental work, not software tests.

## Cortex/emo result plots

Run `conda run --no-capture-output -n letizia python plot_cortex_activity_emo.py`.
It selects the completed cohort if available, otherwise the completed pilot;
incomplete runs are not used. Set `--input-root outputs/cortex_activity_emo_pilot`
to select explicitly. `RUN_CONFIG` supports editor execution; `--baseline-s 60`
and `--threshold-factor 1` choose the primary rows.

Outputs under `<input-root>/plots/60s_factor1/`: an `index.html` gallery, six
PNG/SVG figures (group activity, recording-time trajectories, ROI heatmaps,
baseline/threshold sensitivity, correction examples, drift versus activity),
plotted-data CSVs and `plot_manifest.json`. No raw image volumes are reread.
Colors match the earlier peak-tail plots: PV blue, R orange, T green. Group
summaries pool exposure within animals; pooled RMS uses the mean squared RMS.
Small selections are labelled pilot/descriptive, with animal counts and no
inferential confidence intervals. Example traces use the first recording ID
in each group, not a selection based on activity.

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
