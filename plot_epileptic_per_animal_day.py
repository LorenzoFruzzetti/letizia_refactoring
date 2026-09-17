"""Total epileptiform events per animal per recording day, one line per animal.

Reads the long table written by `epileptic_by_area_animal_day.py` (one row per
group x animal x recording day x ROI), sums the events over all ROIs, and plots
each animal as a line across its recording days, coloured by group.

Every animal is plotted by default, so lines end on different days: an animal
recorded on 2 days has a 2-point line. Set `only_complete_animals = True` to keep
only animals recorded on every recording day in the table.
"""

import os

# MUST run before numpy/matplotlib are imported (MKL delay-load fault, CLAUDE.md 9.11/9.17).
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are only saved, never shown
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent
# The long table to plot. Point this at any epileptic_by_area_animal_day.csv:
#   outputs/epileptic_by_area_animal_day                     <- detection on roi_fluorescence_full.csv
#   outputs/epileptic_by_area_animal_day_pixels_median_dff   <- detection on the pixel dumps (current)
input_csv = (repo_root / "outputs" / "epileptic_by_area_animal_day_pixels_median_dff"
             / "epileptic_by_area_animal_day.csv")
count_column = "n_epileptic"  # events counted per ROI; summed over ROIs and t1..t5
only_complete_animals = False  # True: drop animals missing any recording day in the table

# Palette: fixed categorical order for groups (same slots as epileptic_by_area_animal_day.py).
group_colors = ["#2a78d6", "#eb6834", "#1baf7a"]
text_color = "#3b3b3b"   # labels wear text ink, not the series colour
surface_color = "#ffffff"  # halo behind labels, so they stay readable over lines
grid_color = "#e4e4e1"
line_width_pt = 2.0
marker_size_pt = 7.0
label_min_gap_fraction = 0.03  # minimum vertical gap between labels ending on one day, fraction of the y range

# A folder of its own, beside the other plots made from the same analysis, so the
# figure is not buried among the 545 per-recording output folders.
output_dir = repo_root / "outputs" / "epileptic_median_dff_analysis" / "epileptic_per_animal_day"
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
# long_table: (n_animals x n_recording_days x n_rois) rows
long_table = pd.read_csv(input_csv, dtype={"date": str})

# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
# Total events per animal per recording day, all ROIs summed.
# totals: one row per group x animal x recording day
totals = long_table.groupby(["group", "animal", "recording_day"], as_index=False, sort=False).agg(
    date=("date", "first"),
    n_recordings=("n_recordings", "first"),
    total_events=(count_column, "sum"),
)

# Optionally keep only animals present on every recording day that occurs in the table.
recording_days = np.sort(totals["recording_day"].unique())  # (n_recording_days,)
days_per_animal = totals.groupby("animal", sort=False)["recording_day"].nunique()
if only_complete_animals:
    kept_animals = days_per_animal.index[days_per_animal == len(recording_days)]
else:
    kept_animals = days_per_animal.index
plotted = totals[totals["animal"].isin(kept_animals)].reset_index(drop=True)
if plotted.empty:
    raise ValueError(f"no animal has all {len(recording_days)} recording days in {input_csv}")

print(f"recording days {recording_days.min()}..{recording_days.max()}: plotting "
      f"{plotted['animal'].nunique()} of {len(days_per_animal)} animals")
print("  days per plotted animal: "
      + ", ".join(f"{animal} {days_per_animal[animal]}" for animal in plotted["animal"].unique()))
# A day with fewer t# recordings has fewer minutes to count events in; flag it.
short_days = plotted[plotted["n_recordings"] < long_table["n_recordings"].max()]
if not short_days.empty:
    print("  WARNING: days with fewer recordings, their totals cover less time:")
    print(short_days.to_string(index=False))

# ---------------------------------------------------------------------------
# Saving the plotted table
# ---------------------------------------------------------------------------
plotted.to_csv(output_dir / "epileptic_total_per_animal_day.csv", index=False)

# ---------------------------------------------------------------------------
# Plot: one line per animal, colour = group
# ---------------------------------------------------------------------------
group_names = sorted(totals["group"].unique())  # every group in the table keeps its colour slot
group_color = {group: group_colors[i_group] for i_group, group in enumerate(group_names)}
plotted_groups = [group for group in group_names if group in set(plotted["group"])]

fig, axes = plt.subplots(1, 1, figsize=(10, 6.5), constrained_layout=True, squeeze=False)
ax = axes[0, 0]
end_points = []  # (last day, last total, animal) for the direct labels
for (group, animal), animal_rows in plotted.groupby(["group", "animal"], sort=False):
    animal_rows = animal_rows.sort_values("recording_day")
    ax.plot(animal_rows["recording_day"], animal_rows["total_events"],
            color=group_color[group], linewidth=line_width_pt,
            marker="o", markersize=marker_size_pt,
            markeredgecolor=surface_color, markeredgewidth=1.5)
    end_points.append((int(animal_rows["recording_day"].iloc[-1]),
                       float(animal_rows["total_events"].iloc[-1]), animal))

# Direct labels: animal name right of its last point. Labels ending on the same
# day and closer than the minimum gap are pushed apart symmetrically (each pair
# moves half the shortfall), so every label stays as near its own point as possible.
ax.set_ylim(bottom=0)
y_low, y_high = ax.get_ylim()
min_gap = label_min_gap_fraction * (y_high - y_low)
end_table = pd.DataFrame(end_points, columns=["day", "value", "animal"]).sort_values(["day", "value"])
for day, day_labels in end_table.groupby("day"):
    # A copy: to_numpy can return a read-only view of the column.
    label_positions = day_labels["value"].to_numpy(dtype=float, copy=True)  # (n_labels_on_day,)
    for _ in range(200):
        shortfall = np.clip(min_gap - np.diff(label_positions), 0, None) / 2  # (n_labels_on_day - 1,)
        if not shortfall.any():
            break
        label_positions[:-1] -= shortfall
        label_positions[1:] += shortfall
    for animal, position in zip(day_labels["animal"], label_positions):
        label = ax.annotate(animal, xy=(day, position), xytext=(8, 0),
                            textcoords="offset points", va="center", fontsize=7.5,
                            color=text_color)
        label.set_path_effects([path_effects.withStroke(linewidth=3, foreground=surface_color)])

# Group legend: one entry per group, so identity never rests on colour alone.
for group in plotted_groups:
    n_group_animals = plotted.loc[plotted["group"] == group, "animal"].nunique()
    ax.plot([], [], color=group_color[group], linewidth=line_width_pt, marker="o",
            markersize=marker_size_pt, markeredgecolor=surface_color,
            label=f"{group} ({n_group_animals} animals)")
# Above the plot area, so it can never cover a line.
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(plotted_groups),
          labelcolor=text_color)

ax.set_xticks(recording_days)
ax.set_xlim(recording_days[0] - 0.3, recording_days[-1] + 0.6)
ax.set_xlabel("recording day", color=text_color)
ax.set_ylabel("epileptiform events (all ROIs, t1..t5)", color=text_color)
ax.grid(axis="y", color=grid_color, linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(colors=text_color)
animal_selection = (f"animals with all {len(recording_days)} recording days"
                    if only_complete_animals else "all animals")
# suptitle, not ax.set_title: the legend already occupies the space above the axes.
fig.suptitle(f"Total epileptiform events per animal per recording day ({animal_selection})",
             color=text_color, fontsize=10)
fig.savefig(output_dir / "epileptic_total_per_animal_day.png", dpi=200)
plt.close(fig)

print(f"wrote {output_dir}")
