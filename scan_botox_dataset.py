"""Scan the BOTOX_RESTANI share and write an analysis manifest CSV.

This is a *study script* (P2 boundary: dataset policy lives here, mechanism lives
in ``wfci``). It walks the three-level folder tree of the resting-state dataset

    <root>\\<day>\\<animal>\\<recording>\\*.tif
      260611  \\   R1    \\    t1     \\  R11_00001.tif ...

and emits ONE row per (day, animal), listing that animal's recordings. The row is
the *bookkeeping* unit; the *analysis* unit is chosen by ``run_botox_batch.py``,
which by default runs each ``t#`` recording separately (one result per recording)
and can instead concatenate them into one continuous trial with
``--merge-recordings``.

Naming convention (given):
    * day     = the numeric top folder (e.g. 260611) -- one acquisition session.
    * animal  = capital ``T#`` / ``R#`` (e.g. T4, R1). The leading letter is the
                experimental group (T vs R); kept as ``group`` for later selects.
    * t1..tn  = consecutive recordings of the same animal, listed in one row.

Each recording folder holds single-page 512x512 interleaved TIFFs (odd/even
positions = the two channels), ~6000 files (=3000 frames/channel) each.

The CSV carries per-animal knobs the analysis needs but the tree does not encode
-- Bregma coordinates and channel order. Bregma defaults to the R1/t1 values that
were validated for this dataset; VERIFY and edit them per animal before trusting
the full run (ROI placement is the one thing the pipeline cannot check itself).

Run it:
    conda run -n letizia python scan_botox_dataset.py
    conda run -n letizia python scan_botox_dataset.py --root <path> --out <csv>

Editor mode: just edit ``RUN_CONFIG`` and run -- no CLI flags needed. Errors are
left to surface (no try/except) so an unreachable share fails loudly.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Edit this section to run without CLI flags.
# ---------------------------------------------------------------------------
RUN_CONFIG: dict[str, Any] = {
    # Root of the dataset tree (day\animal\recording\*.tif). Raw string so the
    # UNC backslashes are not read as escapes.
    "root": r"F:\WF_2026\starting",
    # Where the manifest CSV is written (parent dir created if missing).
    "out": r"manifests\botox_restani_manifest.csv",
    # Count the TIFFs in every recording folder. Accurate but hits the network for
    # each of ~85 folders; set False for a fast structure-only scan.
    "count_frames": True,
    # File glob used both for counting and later loading.
    "pattern": "*.tif",
    # Default per-animal Bregma (full resolution). These are the R1/t1 values
    # validated for this dataset; EDIT per animal in the CSV after verifying with
    # the ROI overlay. floor(.../2) onto the downsampled grid is applied at run time.
    "default_bregma_row": 121,
    "default_bregma_col": 134,
    # Pipeline + channel split defaults, written per row so they are editable.
    "profile": "cerebellar_rs",
    "channel_order": "auto",
    "prefer_cli_args": True,
}

# Column order of the manifest. Kept explicit so the analysis reader and the CSV
# stay in lock-step.
FIELDS = [
    "day",            # top folder = acquisition session
    "animal",         # T#/R#
    "group",          # leading letter of the animal name (T or R)
    "n_recordings",   # how many t# folders were merged
    "recordings",     # "t1;t2;..." (names)
    "total_frames",   # sum of TIFFs across recordings ("" if not counted)
    "bregma_row",     # per-animal, EDIT after verifying
    "bregma_col",
    "profile",
    "channel_order",
    "animal_path",     # <root>\<day>\<animal>
    "recording_paths",  # ";"-joined full paths of the merged t# folders
]


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    """Minimal CLI overrides for the knobs worth flipping from the terminal."""
    p = argparse.ArgumentParser(description="Scan BOTOX_RESTANI into a manifest CSV.")
    p.add_argument("--root", default=defaults["root"], help="Dataset root folder.")
    p.add_argument("--out", default=defaults["out"], help="Output CSV path.")
    p.add_argument("--pattern", default=defaults["pattern"], help="TIFF glob.")
    p.add_argument("--no-count", dest="count_frames", action="store_false",
                   default=defaults["count_frames"], help="Skip per-folder frame counts.")
    ns = p.parse_args()
    ns.default_bregma_row = defaults["default_bregma_row"]
    ns.default_bregma_col = defaults["default_bregma_col"]
    ns.profile = defaults["profile"]
    ns.channel_order = defaults["channel_order"]
    return ns


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    """Prefer CLI args when any are given, else fall back to RUN_CONFIG (editor mode)."""
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(
        root=config["root"],
        out=config["out"],
        pattern=config["pattern"],
        count_frames=bool(config["count_frames"]),
        default_bregma_row=config["default_bregma_row"],
        default_bregma_col=config["default_bregma_col"],
        profile=config["profile"],
        channel_order=config["channel_order"],
    )


def _subdirs(path: str) -> list[str]:
    """Sorted names of immediate subdirectories of ``path``."""
    return sorted(e.name for e in os.scandir(path) if e.is_dir())


def _count_tifs(folder: str, pattern: str) -> int:
    """Count files matching ``pattern`` in ``folder`` (one shallow scandir)."""
    import fnmatch
    return sum(1 for e in os.scandir(folder)
               if e.is_file() and fnmatch.fnmatch(e.name, pattern))


def scan(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Walk day -> animal -> recording and build one manifest row per animal."""
    rows: list[dict[str, Any]] = []
    for day in _subdirs(args.root):
        day_path = os.path.join(args.root, day)
        for animal in _subdirs(day_path):
            animal_path = os.path.join(day_path, animal)
            recordings = _subdirs(animal_path)
            if not recordings:
                # An animal folder with no t# recordings: record it so the gap is
                # visible in the manifest rather than silently skipped.
                print(f"  ! {day}/{animal}: no recording subfolders -- skipped")
                continue
            rec_paths = [os.path.join(animal_path, r) for r in recordings]

            total = ""
            if args.count_frames:
                counts = [_count_tifs(p, args.pattern) for p in rec_paths]
                total = sum(counts)
                # Uneven frame counts across merged recordings are worth flagging.
                if len(set(counts)) > 1:
                    print(f"  ! {day}/{animal}: uneven frame counts {counts}")

            rows.append({
                "day": day,
                "animal": animal,
                "group": animal[0],
                "n_recordings": len(recordings),
                "recordings": ";".join(recordings),
                "total_frames": total,
                "bregma_row": args.default_bregma_row,
                "bregma_col": args.default_bregma_col,
                "profile": args.profile,
                "channel_order": args.channel_order,
                "animal_path": animal_path,
                "recording_paths": ";".join(rec_paths),
            })
            print(f"  + {day}/{animal}: {len(recordings)} recordings "
                  f"[{';'.join(recordings)}]"
                  + (f", {total} frames" if total != "" else ""))
    return rows


def main() -> None:
    args = build_runtime_args()
    print(f"Root : {args.root}")
    print(f"Count frames: {args.count_frames}\n")

    rows = scan(args)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_animals = len(rows)
    n_days = len({r["day"] for r in rows})
    print(f"\nWrote {n_animals} animal-day rows across {n_days} days -> {args.out}")


if __name__ == "__main__":
    main()
