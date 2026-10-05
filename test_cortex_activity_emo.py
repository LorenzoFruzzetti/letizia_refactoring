"""Editor-run balanced raw-data pilot; outputs are separate from the cohort.

Uses per-pixel processing (not the old ROI ratio cache, whose median differs).
"""
from pathlib import Path
import pandas as pd
from analyze_cortex_activity_emo import RUN_CONFIG, run
from compare_cortex_activity_conditions import run as compare

# Parameters
repo_root = Path(__file__).resolve().parent
pixel_root = repo_root / "pixel_data"
output_root = repo_root / "outputs/cortex_activity_emo_pilot"
recordings_per_group = 1
baseline_windows_s = [60.0, 20.0, 120.0]
bootstrap_repeats = 100

# Loading: balanced by the design group, no peak-based recording selection.
design = pd.read_csv(pixel_root / "experimental_design.csv")
eligible = design[~design.pixel_dir.isin(RUN_CONFIG["exclude_recordings"])]
selected = eligible.groupby("group",sort=True).head(recordings_per_group)

# Computation, plots and saving: shared tested per-pixel implementation.
run(dict(pixel_root=str(pixel_root),output_root=str(output_root),
         recording_ids=selected.recording_id.tolist(),baseline_windows_s=baseline_windows_s))
compare(str(output_root),bootstrap=bootstrap_repeats)
print(f"Pilot saved to {output_root}. Inspect traces before interpreting candidate events.")
