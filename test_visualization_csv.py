from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd

filename = Path("outputs/botox_restani_rebuilt/260611_PV5/t1/roi_fluorescence_full.csv")
df = pd.read_csv(filename)
# assuming 10 Hz 


# generate a new dataset as a matrix n_columns x n_dt x (n_rows-n_dt+1)
# each i_col x n_dt x i_rows are the n_dt rows of the dataframe for that column
n_dt = 10 # corresponds to 1 seconds of data at 10 Hz
starting_column = 2 # first 2 columns (trial, frame) are metadata, not data
column_names = df.columns[starting_column:]
values = df.to_numpy()[:, starting_column:]
n_rows, n_columns = values.shape
# sliding window of n_dt consecutive rows, one window per row offset i_row
windows = np.stack([values[i : i + n_dt, :].T for i in range(n_rows - n_dt + 1)], axis=-1)

# run a pca over each windows generate a 2 subplot 1. explained variance of the
# first 10 pc. 2. scatter plot of the first 2 pc for each window
n_windows = windows.shape[-1]
n_pcs = 10
# one sample per (column, window) pair, so points from the same column can be colored alike
samples = windows.transpose(0, 2, 1).reshape(n_columns * n_windows, n_dt)

# extract two features from windows: mean and diff between the first two and
# last two time points
# windows: shape (n_areas, n_dt, n_windows (frames - n_dt + 1))

windows_mean = windows.mean(axis=1)
windows_diff = windows[:, :2, :].mean(axis=1) - windows[:, -2:, :].mean(axis=1)

# Plot the two window features against the time at the center of each window.
sampling_rate_hz = 10
window_center_frame = (
    df["frame"].to_numpy()[:n_windows] + (n_dt - 1) / 2
)
time_s = window_center_frame / sampling_rate_hz

features = {
    "Window mean": windows_mean,
    "Early mean - late mean": windows_diff,
}

output_dir = filename.parent / "window_feature_plots"
output_dir.mkdir(parents=True, exist_ok=True)

# One long figure: one row per area and one column per feature.
fig, axes = plt.subplots(
    n_columns,
    2,
    figsize=(18, 3 * n_columns),
    sharex=True,
    squeeze=False,
    constrained_layout=True,
)

feature_items = list(features.items())
for area_index, area_name in enumerate(column_names):
    for feature_index, (_, feature_values) in enumerate(feature_items):
        ax = axes[area_index, feature_index]
        ax.plot(time_s, feature_values[area_index], linewidth=1)
        ax.grid(alpha=0.25)

    axes[area_index, 0].set_ylabel(str(area_name))

for feature_index, (feature_name, _) in enumerate(feature_items):
    axes[0, feature_index].set_title(feature_name)
    axes[-1, feature_index].set_xlabel("Time (s)")

fig.suptitle(f"Window features ({n_dt / sampling_rate_hz:g} s window)")
fig.savefig(output_dir / "all_areas_window_features.png", dpi=200)
plt.close(fig)


"""
samples_centered = samples - samples.mean(axis=0, keepdims=True)
# SVD-based PCA to avoid an extra dependency on scikit-learn
U, S, _ = svd(samples_centered, full_matrices=False)
explained_variance = (S ** 2) / (samples.shape[0] - 1)
explained_variance_ratio = explained_variance / explained_variance.sum()
scores = U * S

# color each point by the data column (ROI) its window was taken from
column_index = np.repeat(np.arange(n_columns), n_windows)
cmap = plt.get_cmap("nipy_spectral", n_columns)

# one explained-variance + PC1-vs-PC2 scatter plot per data column, saved to disk
output_dir = Path("outputs/pca_per_column")
output_dir.mkdir(parents=True, exist_ok=True)
for j, colname in enumerate(column_names):
    mask = column_index == j
    fig, (ax_variance, ax_scatter) = plt.subplots(1, 2, figsize=(12, 5))
    ax_variance.bar(range(1, n_pcs + 1), explained_variance_ratio[:n_pcs])
    ax_variance.set_xlabel("Principal component")
    ax_variance.set_ylabel("Explained variance ratio")

    ax_scatter.scatter(scores[mask, 0], scores[mask, 1], color=cmap(j), s=10)
    ax_scatter.set_xlabel("PC1")
    ax_scatter.set_ylabel("PC2")

    fig.suptitle(colname)
    fig.savefig(output_dir / f"{colname}.png")
    plt.close(fig)"""


