"""Write a groups-only batch_summary.csv for the epileptic detectors.

`run_analysis` (epileptic_by_area_animal_day.py) reads only day/animal/group from
`<input_root>/batch_summary.csv`. The connectivity summary in
outputs/botox_restani_rebuilt covers the first 71 units only; the September cohort
(manifests/botox_restani_2026_09_manifest.csv) was batched on another machine, so
its pixel dumps are here but its groups are not. This script takes the groups from
every manifest instead and writes them to a separate folder, leaving the
connectivity summary untouched.

Output: outputs/epileptic_groups/batch_summary.csv, columns day, animal, group.
Use it with:
    python epileptic_by_active_pixels.py --input-root outputs/epileptic_groups

Edit RUN_CONFIG below to run directly from the editor without CLI arguments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent

# Edit this section to run without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "manifests": [
        REPO_ROOT / "manifests/botox_restani_manifest.csv",
        REPO_ROOT / "manifests/botox_restani_2026_09_manifest.csv",
    ],
    # Existing connectivity summary; its groups must agree with the manifests. None skips the check.
    "check_summary": REPO_ROOT / "outputs/botox_restani_rebuilt/batch_summary.csv",
    # Every unit folder here must get a group. None skips the check.
    "pixel_root": REPO_ROOT / "pixel_data",
    "output_path": REPO_ROOT / "outputs/epileptic_groups/batch_summary.csv",
    "prefer_cli_args": True,  # False: ignore CLI arguments and use this block.
}


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests", type=Path, nargs="+", default=defaults["manifests"])
    parser.add_argument("--check-summary", type=Path, default=defaults["check_summary"])
    parser.add_argument("--pixel-root", type=Path, default=defaults["pixel_root"])
    parser.add_argument("--output-path", type=Path, default=defaults["output_path"])
    return parser.parse_args()


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(
        manifests=[Path(path) for path in config["manifests"]],
        check_summary=config["check_summary"],
        pixel_root=config["pixel_root"],
        output_path=Path(config["output_path"]),
    )


def build_groups(manifests: list[Path]) -> pd.DataFrame:
    """One row per day+animal; a unit listed with two different groups is an error."""
    # day is read as text so 260611 keeps its form and matches the pixel folder names.
    rows = pd.concat([pd.read_csv(path, dtype={"day": str})[["day", "animal", "group"]]
                      for path in manifests], ignore_index=True).drop_duplicates()
    conflicting = rows[rows.duplicated(["day", "animal"], keep=False)]
    if not conflicting.empty:
        raise ValueError(f"Units with more than one group across manifests:\n{conflicting}")
    return rows.sort_values(["day", "animal"], ignore_index=True)


def main():
    args = build_runtime_args()
    groups = build_groups(args.manifests)
    keyed = groups.set_index(["day", "animal"])["group"]

    if args.check_summary is not None:
        # The connectivity summary is the source the detector used so far; a unit
        # whose group changed would silently move between groups.
        summary = pd.read_csv(args.check_summary, dtype={"day": str})
        old = summary[["day", "animal", "group"]].drop_duplicates().set_index(["day", "animal"])["group"]
        missing = old.index.difference(keyed.index)
        if len(missing):
            raise ValueError(f"Units in {args.check_summary} missing from the manifests: {list(missing)}")
        changed = old[old != keyed.loc[old.index]]
        if not changed.empty:
            raise ValueError(f"Group differs from {args.check_summary} for: {list(changed.index)}")

    if args.pixel_root is not None:
        units = {folder.name for folder in Path(args.pixel_root).iterdir() if folder.is_dir()}
        named = {f"{day}_{animal}" for day, animal in keyed.index}
        if units - named:
            raise ValueError(f"Pixel units with no group: {sorted(units - named)}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    groups.to_csv(args.output_path, index=False)
    print(f"wrote {len(groups)} units to {args.output_path}")
    print(groups.groupby("group").size().to_string())


if __name__ == "__main__":
    main()
