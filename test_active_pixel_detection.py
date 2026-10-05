"""Active-pixel epileptiform detection on ONE pixel dump, with the maps drawn.

The single-recording counterpart of `epileptic_by_active_pixels.py`. That script
runs the whole cohort and calibrates one cut-off over every recording's frames;
this one takes a single `pixel_data/<date>_<animal>/<t#>/` dump so the
intermediate steps can be looked at.

Signal (identical to the cohort script, whose functions are imported rather than
copied, so the two cannot drift apart):

    z(t, p)   = (dff(t, p) - median_t dff(p)) / scale_t(p)
    active    = z(t, p) > pixel_z_threshold
    signal(t) = (active pixels) / (pixels considered), per hemisphere

`dff` is a saved per-pixel volume (`pixels_median_dff_<window>s_full.npy` from
cache_median_dff.py, or run_botox_batch's `pixels_dff_full.npy`), `scale_t` is
the pixel's own robust SD, and the pixels considered are the union of the
recording's atlas boxes, split L/R. The two fractions are then detrended,
smoothed and peak-picked exactly as the cohort does.

A third trace, `Cortex_active`, pools every counted pixel of both hemispheres.
It is drawn (whole-recording overview, and beside the others in the detrending
figure) and written to the CSV, but events are NOT detected in it, so the event
tables stay the ones the cohort script would produce.

ONE DELIBERATE DIFFERENCE. The epileptiform cut-off here is the top
`calibration_frame_percent` of THIS recording's frames (or the fixed
`epileptic_z_threshold`, when set). The cohort pools the frames of all 545
recordings, so counts from this script are NOT comparable across recordings --
they answer "what did the detector do on this file", not "how many events does
this animal have".

Input:  pixel_data/<date>_<animal>/<t#>/pixels_meta_full.npz  (+ the dF/F volume
        beside it, and the roi_sets/rebuilt/*.yaml the dump names)
Output: outputs/test_active_pixel_detection/<date>_<animal>/<t#>/
          active_pixel_fraction.csv        the three traces, one row per frame
          active_pixel_events.csv          every amplitude peak with its flags
          active_pixel_epileptic_events.csv  only the epileptiform ones
          active_pixel_maps.png            which pixels count, and how often active
          active_pixel_activity_overview.png  overall activity, detrended, per hemisphere
          active_pixel_traces_detrended.png  each trace vs its running median, detrended
          active_pixel_traces_events.png   the detections on each hemisphere
          active_pixel_traces_rise.png     rise rate over the smoothed fraction
          active_pixel_event_frames.png    zoom + active map at the largest events

Run it (the env's own interpreter dies on the first BLAS call, CLAUDE.md 9.17):
    conda run --no-capture-output -n letizia python test_active_pixel_detection.py
"""

import os

# MUST run before numpy/scipy/matplotlib are imported, i.e. before MKL loads.
# MKL's mkl_intel_thread.3.dll delay-loads libiomp5md.dll by bare name; when that
# fails the process dies with native exception 0xc06d007f and no traceback
# (CLAUDE.md 9.11/9.17). The sequential layer never loads that DLL.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are only saved, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Rectangle

from epileptic_by_active_pixels import (TRACE_NAMES, active_pixel_fraction, dff_volume_path,
                                        hemisphere_masks, pixel_scales)
from epileptic_by_area_animal_day import (detect_events, remove_slow_trend, rise_by_trial,
                                          robust_z_frames, smooth_by_trial)
from epileptic_by_area_animal_day_pixels import crop_slices
from epileptic_diagnostics import plot_detections, plot_detrending, plot_rise

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent
# One pixel dump folder. 260828_PV7/t2 is the one recording to avoid: its
# reflectance channel drops field-wide for frame 1809, which reads as a
# whole-cortex event (CLAUDE.md 9.23).
dump_folder = repo_root / "pixel_data" / "260611_PV5" / "t1"

# Which saved dF/F volume the pixels are thresholded in.
#   "median_dff"   pixels_median_dff_<baseline_window_s>s_full.npy (cache_median_dff.py):
#                  (F/Fbar)/(R/Rbar) - 1 against a centred running-median baseline.
#   "pipeline_dff" pixels_dff_full.npy (run_botox_batch.py): percent dF/F against
#                  the whole-recording mean, so slow drift stays in each pixel.
dff_source = "median_dff"
baseline_window_s = 30.0  # names the median_dff cache to read; ignored by pipeline_dff

# A pixel is active above this many SDs of its OWN trace, so a noisy pixel
# (vessel, edge) is not active more often than a quiet one. 3 fails on the
# cohort: the fraction is then 0 in most frames and has no robust SD (9.23).
pixel_z_threshold = 2.0
# The SD each pixel is measured in:
#   "mad"               1.4826 * MAD of the whole trace.
#   "bottom_percentile" plain SD of the values at or below the pixel's own
#                       percentile, so the events cannot inflate the scale. It is
#                       a truncated-distribution SD, ~0.6x the MAD one, so
#                       thresholds are NOT comparable between the two scales.
pixel_scale_name = "mad"
pixel_scale_percentile = 50.0  # used only by "bottom_percentile"; 50 = bottom half

# Detection - same values as epileptic_by_active_pixels.py RUN_CONFIG["detection"].
sampling_rate_hz = 10.0      # per-channel rate, after the channels are split
median_window_s = 20.0       # running median subtracted as the slow trend
smooth_window_s = 10.0        # rolling-mean width; the result is the "amplitude"
min_event_distance_s = 0.5   # refractory period between two peaks of one trace
amplitude_prominence_sd = 3.0  # peak prominence in robust SDs of the trace itself
rise_prominence_sd = 3.0
rise_window_s = 0.5          # interval the rise is measured over
not_linear_rise = True       # True: window max - min; False: end - start
rise_on_smoothed = False      # True: rise of the smoothed fraction; False: of the detrended one
rise_lead_s = 10.0            # a rise peak this long BEFORE the amplitude peak is concurrent
rise_lag_s = 5.0             # ... and this long after it
epileptic_require_confirmed = True  # epileptiform also needs a concurrent rise

# The epileptiform cut-off in robust z. A number fixes it; None takes the top
# `calibration_frame_percent` of this recording's own frames, which is what makes
# the counts recording-specific (see the docstring).
epileptic_z_threshold = None
calibration_frame_percent = 1.0
# An epileptiform peak also needs at least this fraction of the hemisphere's
# pixels active at the peak frame, read from the RAW fraction (before detrending
# and smoothing). 0.20 = 20% of pixels; 0 disables the gate.
epileptic_min_signal = 0.20

# Column name of the whole-cortex trace: both hemispheres' pixels in one fraction.
# It is drawn and written to the CSV, but events are still detected per hemisphere,
# so the event tables stay the ones epileptic_by_active_pixels.py would produce.
OVERALL_NAME = "Cortex_active"

# Figures.
n_event_snapshots = 4        # largest events to draw a zoom and an active map for
event_zoom_window_s = 10.0   # width of each zoom panel, centred on the peak
figure_dpi = 150
trace_width_scale = 8.0      # stretch of the whole-recording rise figure
overview_width_in = 22.0     # width of the overall-activity figure
overview_row_height_in = 2.4  # height of each of its three rows

# Index-space values derived from the physical ones above.
n_median = 2 * round(median_window_s * sampling_rate_hz / 2) + 1  # odd, so the median is centred
n_smooth = max(1, round(smooth_window_s * sampling_rate_hz))      # frames per rolling mean
edge_frames = n_smooth // 2  # frames at each end averaged over a partial window
min_distance_frames = max(1, round(min_event_distance_s * sampling_rate_hz))
n_rise = max(1, round(rise_window_s * sampling_rate_hz))  # frames spanned by one rise value
rise_lead_frames = max(1, round(rise_lead_s * sampling_rate_hz))
rise_lag_frames = max(1, round(rise_lag_s * sampling_rate_hz))
n_zoom_frames = max(1, round(event_zoom_window_s * sampling_rate_hz / 2))  # half-width of a zoom

recording_label = f"{dump_folder.parent.name}/{dump_folder.name}"
output_dir = repo_root / "outputs" / "test_active_pixel_detection" / dump_folder.parent.name / dump_folder.name
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
metadata_path = dump_folder / "pixels_meta_full.npz"
with np.load(metadata_path, allow_pickle=False) as data:
    # The two mean images are large and unused here; everything else is scalar or short.
    meta = {key: data[key].tolist() for key in data.files if key not in ("mean_f", "mean_r")}
if meta["axis_order"] != "time,y,x":
    raise ValueError(f"Unsupported pixel axes in {metadata_path}")
if meta["n_written"] != meta["n_time"] or meta["n_time"] <= 0:
    raise ValueError(f"Incomplete pixel dump: {metadata_path}")

# The recording's OWN atlas, not the shared cortex22 one: each t# was drawn with
# its own Bregma, and using the wrong file is the silent failure of CLAUDE.md 9.4.
atlas_path = Path(meta["roi_set"])
if not atlas_path.is_absolute():
    atlas_path = repo_root / atlas_path
atlas = yaml.safe_load(atlas_path.read_text(encoding="utf-8"))
if list(atlas["grid"]) != meta["grid"]:
    raise ValueError(f"Atlas and pixel grid differ: {atlas_path}")
for key in ("bregma_row", "bregma_col", "downsample"):
    if atlas[key] != meta[key]:
        raise ValueError(f"Atlas {key} differs from saved metadata: {atlas_path}")

# The saved crop is Bregma-relative, so it names the same anatomy in every animal.
region = meta.get("region", [0, meta["grid"][0], 0, meta["grid"][1]])
crop_shape = (region[1] - region[0], region[3] - region[2])  # (rows, cols)
volume_path = dff_volume_path(dump_folder, dff_source, baseline_window_s)
dff = np.load(volume_path, mmap_mode="r")  # (n_frames, rows, cols), float16 or float32
if dff.shape != (meta["n_time"], *crop_shape) or list(dff.shape) != meta["shape"]:
    raise ValueError(f"dF/F volume shape disagrees with metadata: {dump_folder}")

n_frames = dff.shape[0]
time_s = np.arange(n_frames) / sampling_rate_hz  # (n_frames,)

# ---------------------------------------------------------------------------
# Which pixels are counted
# ---------------------------------------------------------------------------
# One boolean map per hemisphere: the union of that side's atlas boxes on the
# saved crop. The box boundaries are then ignored - a pixel counts once, whichever
# box it is in - but staying inside them excludes the skull the crop also holds.
masks = hemisphere_masks(atlas["boxes"], int(meta["y_1"]), int(meta["x_2"]), region, crop_shape)
cortex_mask = masks["L"] | masks["R"]  # (rows, cols)
pixel_rows, pixel_cols = np.nonzero(cortex_mask)  # (n_pixels,) each, crop-relative 0-based
pixel_is_left = masks["L"][pixel_rows, pixel_cols]  # (n_pixels,)
n_pixels = pixel_rows.size


def as_map(pixel_values: np.ndarray) -> np.ndarray:
    """Scatter one value per counted pixel back onto the crop; NaN outside the boxes."""
    image = np.full(crop_shape, np.nan)  # (rows, cols)
    image[pixel_rows, pixel_cols] = pixel_values
    return image


# ---------------------------------------------------------------------------
# Per-pixel thresholding
# ---------------------------------------------------------------------------
# Widened to float64 first: summing a float16 trace overflows (CLAUDE.md 9.16).
pixels = np.asarray(dff[:, cortex_mask], dtype=np.float64)  # (n_frames, n_pixels)
if not np.isfinite(pixels).all():
    raise ValueError("dF/F volume contains non-finite values inside the atlas boxes")
pixel_median = np.median(pixels, axis=0)  # (n_pixels,)
pixel_scale = pixel_scales(pixels, pixel_median, pixel_scale_name, pixel_scale_percentile)  # (n_pixels,)
# A flat pixel has no scale, so it is left out of both the count and the denominator.
has_scale = pixel_scale > 0
if not has_scale.any():
    raise ValueError("Every pixel inside the atlas boxes is flat; none can be thresholded")

pixel_z = np.full_like(pixels, np.nan)  # (n_frames, n_pixels)
pixel_z[:, has_scale] = (pixels[:, has_scale] - pixel_median[has_scale]) / pixel_scale[has_scale]
is_active = pixel_z > pixel_z_threshold  # (n_frames, n_pixels); NaN compares False
active_rate = np.where(has_scale, is_active.mean(axis=0), np.nan)  # (n_pixels,) frames active

# One trace per hemisphere: the fraction of that side's usable pixels active per frame.
fraction = {}
for side, trace_name in TRACE_NAMES.items():
    side_pixels = (pixel_is_left if side == "L" else ~pixel_is_left) & has_scale  # (n_pixels,)
    fraction[trace_name] = is_active[:, side_pixels].mean(axis=1)  # (n_frames,)
    # Same numbers the cohort script would produce, computed by its own function.
    reference, n_dropped = active_pixel_fraction(dff, masks[side], pixel_z_threshold,
                                                 pixel_scale_name, pixel_scale_percentile)
    if not np.array_equal(reference, fraction[trace_name]):
        raise ValueError(f"{trace_name} differs from epileptic_by_active_pixels.active_pixel_fraction")

# Overall activity: the fraction over ALL usable pixels, both hemispheres at once.
# Taken from `is_active` directly, not as the mean of the two hemisphere fractions -
# those are only the same when each side has the same number of usable pixels.
fraction[OVERALL_NAME] = is_active[:, has_scale].mean(axis=1)  # (n_frames,)

# traces: (n_frames, 5) - trial, frame, the two hemispheres and the overall
# fraction. A dump is one continuous block, hence one trial.
trace_names = list(TRACE_NAMES.values())        # the two traces events are detected in
plot_trace_names = trace_names + [OVERALL_NAME]  # + the overall one, drawn but not detected in
traces = pd.DataFrame({"trial": np.zeros(n_frames, dtype=int), "frame": np.arange(n_frames),
                       **fraction})

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
# Slow changes in the COUNT removed; everything after this works on `detrended`.
# The overall trace is detrended and smoothed with the hemispheres so the figures
# show all three treated identically; only `trace_names` feeds the detection, which
# keeps the event tables the ones the cohort script would produce.
# detrended: (n_frames, 5), trend/smoothed: (n_frames, 3)
detrended, trend = remove_slow_trend(traces, plot_trace_names, n_median)
smoothed = smooth_by_trial(detrended, plot_trace_names, n_smooth)  # the "amplitude"
# Every frame in robust z; the cut-off below is a percentile of these.
frame_z = robust_z_frames(detrended, smoothed, trace_names, edge_frames)  # (n_frames, 2)

# One row per amplitude peak, `confirmed` when a rise peak falls in its window.
events = detect_events(
    detrended, smoothed, trace_names,
    rise_traces=smoothed if rise_on_smoothed else detrended,
    sampling_rate_hz=sampling_rate_hz,
    min_distance_frames=min_distance_frames,
    amplitude_prominence_sd=amplitude_prominence_sd,
    rise_prominence_sd=rise_prominence_sd,
    n_rise=n_rise,
    rise_lead_frames=rise_lead_frames,
    rise_lag_frames=rise_lag_frames,
    not_linear_rise=not_linear_rise,
)

# The raw fraction at each peak frame, for the epileptic_min_signal gate: how
# much of the hemisphere was actually active there, before any filtering.
peak_column = events["roi"].map({name: i for i, name in enumerate(trace_names)}).to_numpy(dtype=int)
events["signal_at_peak"] = traces[trace_names].to_numpy(dtype=float)[events["frame"].to_numpy(), peak_column]

# The cut-off, then the flags. A fixed threshold is comparable across recordings;
# the percentile of this recording's own frames is not (see the docstring).
if epileptic_z_threshold is not None:
    z_threshold = float(epileptic_z_threshold)
    cutoff_label = f"z >= {z_threshold:.2f} (fixed)"
else:
    pooled_z = frame_z.to_numpy(dtype=float).ravel()
    pooled_z = pooled_z[np.isfinite(pooled_z)]
    z_threshold = float(np.percentile(pooled_z, 100.0 - calibration_frame_percent))
    cutoff_label = f"z >= {z_threshold:.2f} (top {calibration_frame_percent:g}% of this recording's frames)"

is_epileptic = events["amplitude_z"].to_numpy(dtype=float) >= z_threshold  # NaN compares False
if epileptic_require_confirmed:
    is_epileptic &= events["confirmed"].to_numpy(dtype=bool)
if epileptic_min_signal:
    is_epileptic &= events["signal_at_peak"].to_numpy(dtype=float) >= epileptic_min_signal
events["epileptic"] = is_epileptic
events["epileptic_threshold"] = z_threshold
epileptic_events = events[events["epileptic"]]

# Rise of the same signal the peaks were found in, as a table for the rise figure.
rise_per_s = rise_by_trial(detrended, trace_names, n_rise, sampling_rate_hz, not_linear_rise,
                           rise_traces=smoothed if rise_on_smoothed else detrended)  # (n_frames, 2)

# The events drawn as maps below: the largest epileptiform ones, or the largest
# peaks of any kind when the cut-off flagged nothing.
ranked = (epileptic_events if len(epileptic_events) else events).sort_values(
    "amplitude_z", ascending=False)
snapshots = ranked.head(n_event_snapshots)

# ---------------------------------------------------------------------------
# Saving tables
# ---------------------------------------------------------------------------
traces.to_csv(output_dir / "active_pixel_fraction.csv", index=False)
events.to_csv(output_dir / "active_pixel_events.csv", index=False)
epileptic_events.to_csv(output_dir / "active_pixel_epileptic_events.csv", index=False)

scale_label = ("1.4826 * MAD" if pixel_scale_name == "mad"
               else f"SD of the bottom {pixel_scale_percentile:g}%")
print(f"{dump_folder}")
print(f"  volume {volume_path.name}, {n_frames} frames at {sampling_rate_hz:g} Hz, "
      f"crop {crop_shape[0]}x{crop_shape[1]} from {atlas_path.name}")
print(f"  {n_pixels} pixels in {len(atlas['boxes'])} atlas boxes "
      f"({int(pixel_is_left.sum())} L, {int((~pixel_is_left).sum())} R), "
      f"{int((~has_scale).sum())} dropped as flat")
print(f"  active above {pixel_z_threshold:g} x {scale_label} of each pixel's own trace: "
      f"{100 * float(np.nanmean(active_rate)):.2f}% of pixel-frames")
for trace_name in plot_trace_names:
    print(f"  {trace_name}: mean {fraction[trace_name].mean():.4f}, "
          f"max {fraction[trace_name].max():.4f} of pixels active")
print(f"  {len(events)} amplitude peaks, {int(events['confirmed'].sum())} confirmed")
print(f"  {len(epileptic_events)} epileptiform ({cutoff_label}, "
      f"min signal {epileptic_min_signal:g})")
print(f"  wrote {output_dir}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
# Which pixels are counted, and how often each one was active. The box outlines
# are drawn on both so a mask that slipped off the cortex is visible.
box_rectangles = {}  # label -> (x, y, width, height) on the crop, for both panels
for label, box in atlas["boxes"].items():
    rs, cs = crop_slices(box, int(meta["y_1"]), int(meta["x_2"]), region)
    box_rectangles[label] = (cs.start - 0.5, rs.start - 0.5, cs.stop - cs.start, rs.stop - rs.start)

fig, axes = plt.subplots(1, 2, figsize=(11, 5), squeeze=False, constrained_layout=True)
side_map = as_map(np.where(pixel_is_left, 0.0, 1.0))  # (rows, cols), L = 0, R = 1
axes[0, 0].imshow(side_map, cmap="coolwarm", vmin=0, vmax=1, interpolation="nearest")
axes[0, 0].set_title(f"Pixels counted: {int(pixel_is_left.sum())} L (blue), "
                     f"{int((~pixel_is_left).sum())} R (red)")
rate_image = axes[0, 1].imshow(as_map(100 * active_rate), cmap="magma", interpolation="nearest")
fig.colorbar(rate_image, ax=axes[0, 1], label="% of frames active")
axes[0, 1].set_title(f"Active rate per pixel (> {pixel_z_threshold:g} x {scale_label})")
for ax in axes[0]:
    for x, y, width, height in box_rectangles.values():
        ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="#2b2b2b", linewidth=0.6))
    # Bregma sits at the same place inside every dump, the crop window being
    # Bregma-relative (CLAUDE.md 9.22). y_1/x_2 are Bregma on the downsampled
    # grid the crop is cut from; bregma_row/col are on the full 256 one.
    ax.plot(int(meta["x_2"]) - region[2], int(meta["y_1"]) - region[0],
            marker="+", color="#00d1b2", markersize=10)
    ax.set_xlabel("crop column")
    ax.set_ylabel("crop row")
fig.suptitle(f"Active-pixel geometry - {recording_label} ({dff_source}, {atlas_path.name})")
fig.savefig(output_dir / "active_pixel_maps.png", dpi=figure_dpi)
plt.close(fig)
print(f"  wrote {output_dir / 'active_pixel_maps.png'}")

# Overall activity over the whole recording: the pooled fraction as it is loaded,
# then the same trace after the running median is taken out, then the two
# hemispheres for comparison. The epileptiform peaks found per hemisphere are
# marked on all three, so a whole-cortex rise can be told from a one-sided one.
fig, axes = plt.subplots(3, 1, figsize=(overview_width_in, 3 * overview_row_height_in),
                         sharex=True, squeeze=False, constrained_layout=True)
overall_raw = traces[OVERALL_NAME].to_numpy()  # (n_frames,)
axes[0, 0].plot(time_s, overall_raw, color="#7a7f87", linewidth=0.8, label="active fraction")
axes[0, 0].plot(time_s, trend[OVERALL_NAME].to_numpy(), color="#e08214", linewidth=1.6,
                label=f"{median_window_s:g} s running median")
if epileptic_min_signal:
    # The gate an epileptiform peak must also clear, on this very signal.
    axes[0, 0].axhline(epileptic_min_signal, color="#2f6fd0", linewidth=1.0, linestyle="--",
                       label=f"epileptic_min_signal = {epileptic_min_signal:g}")
axes[0, 0].set_ylabel("all pixels\nfraction active")
axes[0, 0].set_title(f"Overall activity - {n_pixels} pixels of both hemispheres")

axes[1, 0].plot(time_s, detrended[OVERALL_NAME].to_numpy(), color="#c9ccd1", linewidth=0.8,
                label="detrended")
axes[1, 0].plot(time_s, smoothed[OVERALL_NAME].to_numpy(), color="#2f6fd0", linewidth=1.4,
                label=f"smoothed ({smooth_window_s:g} s)")
axes[1, 0].set_ylabel("detrended\nfraction active")
axes[1, 0].set_title("The same trace with the running median subtracted")

for trace_name in trace_names:
    axes[2, 0].plot(time_s, traces[trace_name].to_numpy(), linewidth=0.9, label=trace_name)
axes[2, 0].set_ylabel("per hemisphere\nfraction active")
axes[2, 0].set_title("Each hemisphere on its own, for comparison")
axes[2, 0].set_xlabel("Time (s)")

for ax in axes[:, 0]:
    for event_time_s in epileptic_events["time_s"]:
        ax.axvline(event_time_s, color="#c2354a", linewidth=0.8, alpha=0.5, zorder=0)
    ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Active-pixel activity over time - {recording_label} "
             f"(pixel {pixel_z_threshold:g} x {scale_label}, {dff_source}; "
             f"red lines = the {len(epileptic_events)} epileptiform peaks)")
fig.savefig(output_dir / "active_pixel_activity_overview.png", dpi=figure_dpi)
plt.close(fig)
print(f"  wrote {output_dir / 'active_pixel_activity_overview.png'}")

plot_detrending(
    traces, trend, detrended, plot_trace_names, time_s,
    output_dir / "active_pixel_traces_detrended.png",
    f"Slow-trend removal - {recording_label} ({median_window_s:g} s running median "
    f"on the active fraction)",
    dpi=figure_dpi,
)
print(f"  wrote {output_dir / 'active_pixel_traces_detrended.png'}")

plot_detections(
    detrended, smoothed, events, trace_names, time_s, cutoff_label,
    output_dir / "active_pixel_traces_events.png",
    f"Epileptiform detection on the active-pixel fraction - {recording_label} "
    f"(pixel {pixel_z_threshold:g} x {scale_label}, {median_window_s:g} s detrend, "
    f"{smooth_window_s:g} s smoothing, prominence {amplitude_prominence_sd:g} SD, {cutoff_label})",
    dpi=figure_dpi,
)
print(f"  wrote {output_dir / 'active_pixel_traces_events.png'}")

plot_rise(
    smoothed, rise_per_s, events, trace_names, time_s, n_rise / 2 / sampling_rate_hz,
    output_dir / "active_pixel_traces_rise.png",
    f"Rise rate vs smoothed amplitude - {recording_label} ({rise_window_s:g} s window, "
    f"not_linear_rise={not_linear_rise}, rise_on_smoothed={rise_on_smoothed})",
    dpi=figure_dpi, signal_unit="fraction of active pixels", width_scale=trace_width_scale,
    rise_source="smoothed amplitude" if rise_on_smoothed else "detrended trace",
)
print(f"  wrote {output_dir / 'active_pixel_traces_rise.png'}")

# The largest events seen twice: as a zoom of both fractions, and as the map of
# which pixels were active at the peak frame. A real event is a patch; a glitch
# or a threshold set too low lights the whole field or scattered single pixels.
if len(snapshots):
    fig, axes = plt.subplots(2, len(snapshots), figsize=(3.4 * len(snapshots), 6.4),
                             squeeze=False, constrained_layout=True)
    for i_event, (_, event) in enumerate(snapshots.iterrows()):
        peak_frame = int(event["frame"])
        first, last = max(0, peak_frame - n_zoom_frames), min(n_frames, peak_frame + n_zoom_frames + 1)
        ax = axes[0, i_event]
        for trace_name in trace_names:
            ax.plot(time_s[first:last], traces[trace_name].to_numpy()[first:last],
                    linewidth=1.2, label=trace_name)
        ax.axvline(time_s[peak_frame], color="#c2354a", linewidth=1.0)
        ax.set_title(f"{event['roi']} at {event['time_s']:.1f} s\n"
                     f"z {event['amplitude_z']:.1f}, {100 * event['signal_at_peak']:.0f}% active")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("fraction active")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

        ax = axes[1, i_event]
        ax.imshow(as_map(is_active[peak_frame].astype(float)), cmap="magma", vmin=0, vmax=1,
                  interpolation="nearest")
        for x, y, width, height in box_rectangles.values():
            ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="#4de0c6",
                                   linewidth=0.6))
        ax.set_title(f"frame {peak_frame}")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8)
    kind = "epileptiform" if len(epileptic_events) else "amplitude (none flagged epileptiform)"
    fig.suptitle(f"Largest {kind} events - {recording_label} "
                 f"({event_zoom_window_s:g} s zoom of the RAW fraction; active pixels bright)")
    fig.savefig(output_dir / "active_pixel_event_frames.png", dpi=figure_dpi)
    plt.close(fig)
    print(f"  wrote {output_dir / 'active_pixel_event_frames.png'}")
