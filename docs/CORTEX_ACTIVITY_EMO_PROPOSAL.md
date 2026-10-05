# Cortex-wide activity relative to emo: implementation proposal

Status: implemented, 2026-09-28; synthetic and balanced real-data pilot validation.
See [usage and implementation choices](CORTEX_ACTIVITY_EMO_USAGE.md). Detection
thresholds remain exploratory; no validated epilepsy or whole-cohort finding is claimed.
Read alongside [README](../README.md), [CLAUDE](../CLAUDE.md), and the canonical
technical map [REFERENCE](../REFERENCE.md).

## Question and interpretation

Compare the burden and spatial spread of calcium transients between PV, R, and T,
while separately estimating the within-recording decline. Use emo as an optical
correction channel, not as a biological denominator that makes fluorescence an
absolute measure of excitability. Record the biological meaning of the groups,
awake/anesthetized state, emo wavelength, exposure/gain, and treatment timing
before interpreting effects mechanistically. The design table does not supply
these acquisition details.

Call the endpoint **candidate epileptiform calcium-event burden in the imaged
cortex**. More fluorescence alone does not establish epilepsy; EEG/LFP or blinded
expert-labelled events are needed to validate that interpretation. The stored
76 x 87 crop and 22 ROI boxes do not measure the whole brain.

## What the existing project establishes

- `outputs/drift_vs_design/recording_drift_metrics.csv` contains 544 recordings
  from 26 animals. Rechecking that table gives 535 negative corrected slopes
  (98.35%), with mean -0.552 percentage points dF/F per minute.
- CLAUDE section 9.30 reports approximately stable first-to-last fast fluctuation
  SD overall. However, that SD is computed from locally normalized 20 s dF/F:
  it does not independently prove that raw event amplitudes are stable or that
  the decline is photobleaching. The documented group effect on this metric has
  a corrected inference; the original nested-model likelihood-ratio p-values
  must not be reused.
- `analyze_peak_tails_emo.py` already computes per-pixel F_gcamp/F_emo, then ROI
  means. Its raw mode uses one constant baseline; its detrended mode divides
  each ROI trace by a 60 s running median once. Its docstring reports early-event
  selection bias when drift is left in the raw ratio.
- `Cortex_mean` in that script is the mean of ROI means. It is an equal-region
  summary, not the mean over a contiguous cortex mask.
- The top 5% of each recording's peaks is a descriptive tail, not a shared
  abnormal-event definition. Standardizing each recording by its own full-trace
  MAD can also conceal a condition-wide amplitude increase.
- Existing median-dF/F plus detector detrending applies two baseline operations
  (CLAUDE 9.22). The proposed path must apply only one.
- Preserve the exclusion of `260828_PV7/t2`: a reflectance glitch accompanies a
  persistent GCaMP step (CLAUDE 9.23/9.27). GSR must be off for this analysis,
  because the widespread component is part of the target signal.

## Related experimental evidence

1. [Valley et al., 2020, Separation of hemodynamic signals from GCaMP fluorescence
   measured with wide-field imaging](https://pubmed.ncbi.nlm.nih.gov/31747332/)
   measured substantial hemodynamic contamination, including in GFP controls,
   and showed limitations of correction near vessels. Implication: F/emo is a
   useful starting correction, not proof that vascular contamination is absent.
2. [Mesoscopic mapping of hemodynamic responses and neuronal activity during
   pharmacologically induced interictal spikes in awake and anesthetized mice,
   2024](https://pubmed.ncbi.nlm.nih.gov/38230631/) simultaneously measured calcium,
   hemodynamics, and LFP. Anesthesia affected these signals differently.
   Implication: a calcium-to-hemodynamic comparison can change because of
   neurovascular coupling or state, independently of epileptiform burden.
3. [Rahmati et al., 2016, Inferring Neuronal Dynamics from Calcium Imaging Data
   Using Biophysical Models and Bayesian Inference](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004736)
   observed downward drifts over hundreds of seconds and removed slow baselines
   before inference. This is a related calcium-imaging problem in a different
   preparation, not a validation of a particular filter for this dataset.

The following choices are proposed engineering/statistical decisions, not
parameters established by those papers for this experiment.

## Signal construction and drift

The agreed algorithm subtracts a running-median baseline and adds back the
arithmetic mean of that baseline. It does not select an early or middle reference
segment and does not divide by a baseline-derived reference.

Read raw float32 dumps, promote working chunks to float64, and compute per pixel:

    q_p(t) = F_gcamp,p(t) / F_emo,p(t)
    b_p(t) = centred running median of q_p(t), initially 60 s
    mean_b,p = arithmetic mean of b_p(t) over all valid baseline samples in the recording
    residual_p(t) = q_p(t) - b_p(t)
    corrected_p(t) = residual_p(t) + mean_b,p

Retain q, b, mean_b, residual and corrected. The baseline is estimated once on
the full recording, not independently in each analysis bin. Compute the 60 s
window in frames from the per-channel sampling rate, using an odd frame count
(601 samples at 10 Hz, as in the existing scripts). Use partial centred windows
at recording boundaries, record their support, and flag them for sensitivity
checks. Exclude invalid samples and identified artifacts; a baseline window
intersecting a flagged artifact or data gap is also invalid. Do not bridge gaps
or treat missing values as zero. The mean uses all remaining finite baseline
samples across the recording, including flagged partial-edge estimates, rather
than a selected reference segment; record the sample count and edge policy.
Raise if no valid baseline samples remain.

This replaces the estimated changing baseline with one constant per pixel and
recording. A transient 0.02 above its estimated baseline remains 0.02 above the
corrected baseline, early or late. Adding mean_b changes only the vertical offset;
it does not scale the residuals. The offset itself can depend on decline rate
and recording duration, so it must not be interpreted as corrected tonic neural
activity or compared as a measure of event strength between conditions.

Use residual for detection, amplitudes, RMS and positive event AUC. Use corrected
for plots at a representative F_gcamp/F_emo level, with mean_b drawn as the
reference line. Measurements from corrected must subtract mean_b first; do not
threshold it against absolute zero or integrate its offset as activity. Retain b
and its slope as separate outputs to study the decline.

The native unit of residual and corrected is the dimensionless F_gcamp/F_emo
ratio; event AUC has units of ratio-seconds. No percentage or dF/F conversion is
implicit. In particular, do not divide by mean_b, by a middle-interval median,
or by an early reference. Such division would rescale amplitudes according to a
recording-dependent baseline and reintroduce the normalization problem. Raw
ratio amplitudes can still depend on gain, illumination and expression between
recordings; additive detrending does not solve cross-animal optical calibration.

This method corrects additive baseline drift. If bleaching also reduces transient
amplitudes, subtraction cannot restore them. Compare q, F_gcamp, and F_emo
residual amplitudes and noise in early/middle/late intervals, and use simulations
below. The existing divisive algorithm can be retained as a labelled secondary
comparison, not as part of the agreed correction or a replacement selected for
larger group separation.

Start with one 60 s median stage, with 20 s and 120 s sensitivity variants.
Subtracting a running median is nonlinear and can remove long genuine events.
Flag events approaching the baseline-window duration; inspect their raw traces
and use an event-masked slow fit as a secondary analysis if sustained events are
common. No detrending method can recover an unknown tonic neural change that is
indistinguishable from optical drift without additional controls.

Use actual per-channel sampling metadata (existing scripts assume 10 Hz), and
check channel order and interleaving offset. Match emo to calcium times if
timestamps support interpolation; do not invent timing metadata. Log invalid or
near-zero denominators, acquisition steps, non-finite samples and clipping.
Downsampled raw dumps cannot exclude saturation in the original sensor pixels;
inspect original TIFFs where saturation is suspected (CLAUDE 9.14).

The pilot can reuse the ROI ratio cache for trace-level comparisons. Subtracting
a running median per pixel and then averaging is not equal to subtracting a
running median from the ROI average, so this pilot must be labelled separately
from the final spatial calculation.

## Spatial and event endpoints

First reuse `hemisphere_masks` and the union of the saved per-recording ROI boxes:
each pixel is counted once. Report this as the **sampled cortical ROI union**.
A separate validated brain mask is required for a contiguous imaged-cortex
analysis; never treat the whole rectangular crop as brain. Fix the quality mask
within a recording, record included area, and check coverage across conditions.

Compute these separately for bilateral cortex, left, right, and individual ROIs:

| Endpoint | Definition and purpose |
|---|---|
| Event burden, primary | Seconds inside accepted candidate event intervals / valid seconds |
| Event rate | Unique events per valid minute; merge regional detections belonging to the same global interval |
| Recruitment | Fraction of valid masked pixels active at each event peak; report bilateral co-recruitment |
| Event strength | Peak residual amplitude in F_gcamp/F_emo ratio units and positive residual event AUC in ratio-seconds |
| Continuous activity | RMS of baseline residuals, robust spread, and full amplitude survival curves |
| Drift | Slopes/relative changes of GCaMP, emo, and q baselines, reported independently |

Keep the area-weighted pixel mean and equal-ROI mean as distinct summaries.
Compute positive pixel activity before spatial averaging when measuring dispersed
activity, because a global mean can hide local asynchronous events. Positive
rectification has noise bias: report a noise/reference null and do not equate
positive area alone with epileptiform burden.

Use amplitude/prominence, rise time, duration and spatial recruitment to define
candidate events. Estimate measurement noise from event-free residuals or
controls, save the estimate, and hold its threshold fixed within each recording.
Do not re-standardize every time bin. Report native ratio amplitudes alongside
noise-standardized detections so higher activity cannot normalize itself away.
Do not assume full-trace MAD measures instrument noise.

Choose threshold and recruitment parameters using designated reference animals
and blinded annotations, then freeze them for held-out animals. Without a
biologically identified reference group, use prespecified common thresholds and
show threshold-sensitivity curves. Do not choose a separate top percentile in
each recording/group, or tune thresholds to maximize a condition difference.
Do not z-score a mostly-zero active-fraction trace: use the fraction directly.
Detection onset/offset hysteresis and temporal merging should produce event
intervals, not one event for every local maximum or ROI.

## Time and condition comparisons

Bin valid time into non-overlapping 30 s intervals and retain partial-bin
exposure. Use the same early/late support across processing variants. With
centred baselines, primary cross-window comparisons use full-window support for
the longest baseline (120 s implies excluding the first/last 60 s); report the
larger 60 s-method interval as a sensitivity analysis. Mark recording-onset
frames and filter boundaries explicitly rather than counting them as quiet time.
These intervals control event-analysis support and exposure only. They do not
define a normalization reference or change mean_b, which is computed once from
all valid baseline samples across the recording. Changing the display offset
must not change residual-based event metrics on the same analysed frames.

Join `experimental_design.csv` on recording_id. Keep within-recording seconds,
recording_index (t1..t5), day_index, and calendar date distinct. Day_index is the
ordinal visit, not elapsed days after treatment. Add actual elapsed days or
treatment-relative time if available. Audit group x mouse_line x acquisition
cohort overlap before fitting contrasts; PV exists only in PV-CRE in this table.

Fit counts with a log-link Poisson GEE clustered by animal, robust uncertainty
and log(valid minutes) offset; assess overdispersion and sparse counts and use
a suitable negative-binomial extension/sensitivity model if needed. Fixed terms:

    group + within_recording_time + group:within_recording_time
          + mouse_line + day_index + categorical(recording_index)

Use a small prespecified time spline only if a straight time trend is inadequate.
For the primary time-burden endpoint, use exposure-aware fractional-outcome GEE
and animal-cluster bootstrap intervals rather than treating frames as independent
Bernoulli trials. For amplitudes and recruitment, summarize per recording and
session, with animal-cluster bootstrap contrasts. Resample complete animals with
their visits/recordings intact; there are 26 animals, not 544 independent animals.
Report group-wise animal counts and instability when strata are small.

Report adjusted condition contrasts at common times and standardized over the
same supported covariate distribution, plus group-by-time trajectories. A true
decline in event rate/amplitude should remain visible after baseline removal.
An early-to-late ratio alone misses nonlinear changes. Do not force trajectories
to be flat, adjust away slow trends solely because they differ by condition, or
use post-treatment emo measures as causal adjustment variables without a causal
justification. Convergence/boundary problems described in CLAUDE 9.30 preclude
blindly copying the old MixedLM likelihood-ratio loop.

## Proposed implementation and outputs

1. `test_cortex_activity_emo.py`: flat, editor-runnable pilot with parameters at
   the top, following CLAUDE's test-script conventions. Read cached ROI ratios
   and a small balanced set of raw recordings. Save q, the running baseline,
   its mean, residual and corrected traces, plus early/late comparisons. Do not import procedural analysis scripts that
   execute cohort processing on import.
2. `analyze_cortex_activity_emo.py`: editor `RUN_CONFIG` plus CLI cohort runner.
   Read raw dumps, metadata, ROI sets and design; process one recording at a
   time using memory maps/chunks. Reuse geometry helpers, but keep study-specific
   endpoints/statistics outside `src/wfci`, consistent with the library boundary.
3. `compare_cortex_activity_conditions.py`: fits the exposure-aware models and
   creates animal-level plots from saved tables without rereading image volumes.
4. Add meaningful synthetic regression tests before running the whole cohort.
   Keep existing analyses and caches unchanged.

Proposed output root: `outputs/cortex_activity_emo/`:

- `recording_qc.csv`: included/excluded time, mask area, denominator and artifact
  flags, channel baselines, sampling/timing assumptions.
- `window_metrics.csv`: recording/design keys, method, region, time bounds,
  valid_seconds, event counts/occupied seconds, activity and recruitment metrics.
- `events.csv`: unique event IDs, onset/offset, amplitude, AUC, peak recruitment,
  hemispheres involved, QC flags and any expert/EEG labels.
- `recording_metrics.csv`, `animal_day_metrics.csv`, `condition_contrasts.csv`.
- `config.json`: exact formulas, units, thresholds, baseline settings, exclusions,
  source/ROI/mask fingerprints, software versions and training animal IDs.
- Per-recording trace outputs: q, b, residual, corrected, mean_b per pixel or
  reported spatial summary, and validity/window-support masks. Label pixel-level
  versus ROI-level processing explicitly.
- Diagnostic baseline/event overlays, spatial maps, condition-by-time plots and
  correction/threshold sensitivity figures. Each plot states units and mask.

Cache keys include source fingerprints, geometry, timing, method and settings;
matching recording IDs alone is insufficient. The entrypoints are now runnable;
commands and implementation details are documented in the usage guide, README,
GUIDE and REFERENCE. The pilot uses the shared per-pixel implementation on raw
data, rather than the old ROI-ratio cache. LIBRARY is unchanged because no package
API changed.

## Validation before interpreting group differences

Inject known events into realistic noise with (a) baseline-only decline, (b)
multiplicative bleaching, (c) genuinely falling event rates/amplitudes, (d)
emo-only changes, (e) channel glitches/steps, and (f) long sustained events.
Verify amplitude/duration recovery, event rate, false events, spatial recruitment
and absence of an artificial group-by-time interaction. Define acceptance
tolerances before looking at condition contrasts. Confirm the true biological
decline in case (c) is preserved, not corrected away.

Specifically verify corrected - mean_b equals residual, and that changing only
the added constant leaves detected events, peak residual amplitudes, RMS and
residual AUC unchanged. Test identical pulses on different additive decline
slopes without dividing by a baseline reference. Quantify median-window and
edge effects rather than assuming exact pulse recovery from a nonlinear filter.
Case (b) must document the remaining amplitude loss: additive detrending does
not claim to correct multiplicative bleaching.

On real data, inspect blinded events and rejected artifacts across all groups and
early/late recording periods. Assess saturation and coverage differences, compare
GCaMP-only versus emo-corrected results, and validate with EEG/LFP if available.
Until that validation, interpret differences as calcium-event burden and spread,
not a calibrated score of whole-brain epileptic severity.
