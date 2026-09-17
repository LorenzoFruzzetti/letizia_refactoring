"""Shared diagnostic plots for single-recording and batch event detection."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_detrending(
    fluorescence: pd.DataFrame,
    trend: pd.DataFrame,
    detrended: pd.DataFrame,
    roi_names: list[str],
    time_s: np.ndarray,
    output_path: Path,
    title: str,
    *,
    dpi: int = 200,
) -> None:
    """One row per ROI: original with its running median (left), detrended (right)."""
    n_rois = len(roi_names)
    fig, axes = plt.subplots(n_rois, 2, figsize=(18, 1.6 * n_rois), sharex=True,
                             squeeze=False, constrained_layout=True)
    for i_roi, roi in enumerate(roi_names):
        ax_original = axes[i_roi, 0]
        ax_original.plot(time_s, fluorescence[roi].to_numpy(), color="#9aa0a6",
                         linewidth=0.8, label="original")
        ax_original.plot(time_s, trend[roi].to_numpy(), color="#d89020",
                         linewidth=1.8, label="running median")
        ax_original.set_ylabel(roi, rotation=0, ha="right", va="center")

        ax_detrended = axes[i_roi, 1]
        ax_detrended.plot(time_s, detrended[roi].to_numpy(), color="#2f6fd0",
                          linewidth=0.8, label="original - running median")
        ax_detrended.axhline(0.0, color="#444444", linewidth=0.6)

        for ax in (ax_original, ax_detrended):
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)

    axes[0, 0].legend(loc="upper right", ncol=2, frameon=False, fontsize=8)
    axes[0, 1].legend(loc="upper right", frameon=False, fontsize=8)
    axes[0, 0].set_title("Original and running median")
    axes[0, 1].set_title("Detrended")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=dpi, pil_kwargs={"compress_level": 1})
    plt.close(fig)


def plot_detections(
    traces: pd.DataFrame,
    smoothed: pd.DataFrame,
    events: pd.DataFrame,
    roi_names: list[str],
    time_s: np.ndarray,
    criterion_label: str,
    output_path: Path,
    title: str,
    *,
    dpi: int = 200,
) -> None:
    """One panel per ROI: detrended trace, smoothed trace and the detections.

    The three marker tiers are disjoint and drawn weakest claim first, so the
    epileptiform stars sit on top.
    """
    n_rois = len(roi_names)
    fig, axes = plt.subplots(n_rois, 1, figsize=(16, 1.6 * n_rois), sharex=True,
                             squeeze=False, constrained_layout=True)
    for i_roi, roi in enumerate(roi_names):
        ax = axes[i_roi, 0]
        ax.plot(time_s, traces[roi].to_numpy(), color="#c9ccd1", linewidth=0.8,
                label="detrended")
        ax.plot(time_s, smoothed[roi].to_numpy(), color="#2f6fd0", linewidth=1.6,
                label="smoothed")

        roi_events = events[events["roi"] == roi]
        roi_epileptic = roi_events[roi_events["epileptic"]]
        roi_rest = roi_events[~roi_events["epileptic"]]
        roi_unconfirmed = roi_rest[~roi_rest["confirmed"]]
        roi_confirmed = roi_rest[roi_rest["confirmed"]]
        ax.scatter(roi_unconfirmed["time_s"], roi_unconfirmed["amplitude"], s=45,
                   facecolors="none", edgecolors="#d89020", zorder=3,
                   label="amplitude peak only")
        ax.scatter(roi_confirmed["time_s"], roi_confirmed["amplitude"], s=45,
                   facecolors="none", edgecolors="#c2354a", linewidths=1.5, zorder=4,
                   label="concurrent peak")
        ax.scatter(roi_epileptic["time_s"], roi_epileptic["amplitude"], s=110,
                   marker="*", color="#c2354a", zorder=5,
                   label=f"epileptiform ({criterion_label})")

        ax.set_ylabel(roi, rotation=0, ha="right", va="center")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0, 0].legend(loc="upper right", ncol=5, frameon=False, fontsize=8)
    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=dpi, pil_kwargs={"compress_level": 1})
    plt.close(fig)


def plot_rise(
    smoothed: pd.DataFrame,
    rise_per_s: pd.DataFrame,
    events: pd.DataFrame,
    roi_names: list[str],
    time_s: np.ndarray,
    rise_time_shift_s: float,
    output_path: Path,
    title: str,
    *,
    dpi: int = 200,
    signal_unit: str = "DF/F",
    width_scale: float = 1.0,
    rise_source: str = "detrended trace",
) -> None:
    """One panel per ROI: smoothed amplitude in blue behind, rise rate in red on top.

    `rise_source` names the trace the rise was taken from, for the legend.

    `width_scale` stretches the figure horizontally (30 makes one second about
    1.6 inches wide on a 5-minute recording) and adds a tick every second.

    Both lines are half-transparent so the one underneath stays visible where
    they overlap; the markers are near-black so they stand out from the red.

    The two have different units, so the rise gets its own y-axis on the right.
    Markers sit at each confirmed event's steepest concurrent rise, i.e. the
    `rise_time_s` / `rise_per_s` of the event table, so a marker off the line
    would mean the plot and the detection disagree.
    """
    n_rois = len(roi_names)
    rise_time_s = time_s + rise_time_shift_s  # (n_frames,) window centres
    fig, axes = plt.subplots(n_rois, 1, figsize=(16 * width_scale, 1.6 * n_rois), sharex=True,
                             squeeze=False, constrained_layout=True)
    for i_roi, roi in enumerate(roi_names):
        ax_amplitude = axes[i_roi, 0]
        amplitude_line, = ax_amplitude.plot(
            time_s, smoothed[roi].to_numpy(), color="#2f6fd0", alpha=0.5, linewidth=1.4,
            label="smoothed amplitude (left axis)")
        ax_amplitude.set_ylabel(roi, rotation=0, ha="right", va="center")
        ax_amplitude.grid(alpha=0.2)
        ax_amplitude.spines[["top"]].set_visible(False)

        ax_rise = ax_amplitude.twinx()
        rise_line, = ax_rise.plot(
            rise_time_s, rise_per_s[roi].to_numpy(), color="#c2354a", alpha=0.5, linewidth=1.0,
            label=f"rise of {rise_source} (right axis, {signal_unit} per s)")
        ax_rise.axhline(0.0, color="#c2354a", linewidth=0.5, alpha=0.4)
        ax_rise.tick_params(axis="y", colors="#c2354a", labelsize=7)
        ax_rise.spines[["top"]].set_visible(False)

        roi_confirmed = events[(events["roi"] == roi) & events["confirmed"]]
        roi_epileptic = roi_confirmed[roi_confirmed["epileptic"]]
        roi_other = roi_confirmed[~roi_confirmed["epileptic"]]
        confirmed_markers = ax_rise.scatter(
            roi_other["rise_time_s"], roi_other["rise_per_s"], s=40,
            facecolors="none", edgecolors="#222222", linewidths=1.3, zorder=4,
            label="steepest rise of a concurrent peak")
        epileptic_markers = ax_rise.scatter(
            roi_epileptic["rise_time_s"], roi_epileptic["rise_per_s"], s=100,
            marker="*", color="#222222", zorder=5,
            label="steepest rise of an epileptiform peak")

        if i_roi == 0:
            handles = [amplitude_line, rise_line, confirmed_markers, epileptic_markers]
            ax_rise.legend(handles=handles, loc="upper right", ncol=4,
                           frameon=False, fontsize=8)

    axes[-1, 0].set_xlabel("Time (s)")
    _second_ticks(axes, time_s, width_scale)
    if width_scale > 1:
        # The y-label is only at the far left; repeat the ROI name inside each
        # panel so a stretched figure stays readable wherever it is scrolled to.
        for i_roi, roi in enumerate(roi_names):
            for label_time_s in np.arange(np.floor(time_s.min()) + 30, time_s.max(), 30.0):
                axes[i_roi, 0].text(label_time_s, 0.97, roi, transform=axes[i_roi, 0].get_xaxis_transform(),
                                    ha="left", va="top", fontsize=8, color="#444444")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=dpi, pil_kwargs={"compress_level": 1})
    plt.close(fig)


def _second_ticks(axes, time_s, width_scale):
    """On stretched figures, a labelled tick every 10 s and a minor one every 1 s.

    The 1 s minor grid is what makes the second readable; left at the default
    ticks a 30x figure would still only show a label every 50 s.
    """
    if width_scale <= 1:
        return
    start, stop = np.floor(time_s.min()), np.ceil(time_s.max())
    for ax in axes.ravel():
        ax.set_xticks(np.arange(start, stop + 1, 10.0))
        ax.set_xticks(np.arange(start, stop + 1, 1.0), minor=True)
        ax.grid(which="minor", axis="x", alpha=0.12, linewidth=0.5)
        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.set_xlim(start, stop)


def plot_roi_overview(
    traces: pd.DataFrame,
    roi_names: list[str],
    time_s: np.ndarray,
    output_path: Path,
    title: str,
    *,
    dpi: int = 200,
    signal_unit: str = "DF/F",
    width_scale: float = 1.0,
) -> None:
    """Every ROI of one recording overlaid on one axis, the whole recording long.

    The same "did signals come out" look as `roi_traces_full.png` written by
    run_botox_batch, but drawn from the trace table this detector actually read,
    so it shows the signal the events were found in. The legend is placed outside
    the axes because 22+ overlaid ROIs would otherwise cover the trace.
    """
    fig, axes = plt.subplots(1, 1, figsize=(16 * width_scale, 5), constrained_layout=True, squeeze=False)
    ax = axes[0, 0]
    for roi in roi_names:
        ax.plot(time_s, traces[roi].to_numpy(dtype=float), linewidth=0.5, label=roi)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(signal_unit)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    # One column per ~11 ROIs keeps the legend box from growing taller than the axes.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=7,
              ncol=max(1, (len(roi_names) + 10) // 11))
    ax.set_title(title, fontsize=10)
    _second_ticks(axes, time_s, width_scale)
    fig.savefig(output_path, dpi=dpi, pil_kwargs={"compress_level": 1})
    plt.close(fig)
