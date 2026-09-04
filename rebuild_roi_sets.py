"""Regenerate every per-recording ROI set as a pure rescale of the baseline atlas.

Why
---
The sets in ``roi_sets/`` were drawn page by page in roi_editor.py, and the editor's
per-box nudge was used on top of the global Lambda rescale.  Auditing them against the
baseline showed the atlas's structural relations survive everywhere *except* bilateral
mirror symmetry, which is broken in three regions:

    HL       294/356 files, always +1 px
    V1       289/356 files, -2 .. +6 px, drifting monotonically with acquisition date
    M2_alta  230/356 files, always +1 px

There are two distinct causes, and they need different fixes.

1. The uniform +1 px in HL and M2_alta is a ROUNDING BUG in ``scale_boxes``.  Its
   docstring claims round-half-to-even keeps a mirrored pair mirrored because it is
   symmetric about zero.  Symmetry about zero is not the property the mirror needs --
   translation invariance is.  A box of span 5 puts its edge at ``centre*f - 2.5``, so
   mirroring requires ``round(u + 2.5) == round(u - 2.5) + 5``.  Half-to-even fails
   that whenever ``u = |centre| * f`` lands on an integer: round(5.5) = 6 but
   round(0.5) = 0, a gap of 6, so the pair comes out 1 px off-centre.  Round-half-away-
   from-zero satisfies both f(-x) = -f(x) and f(x+n) = f(x)+n for integer n, so it is
   mirror-exact; this script uses it (``scale_boxes_mirrored`` below) and diverges from
   ``roi_editor.scale_boxes`` by at most 1 px, on exactly the cases that break.

2. V1's -2..+6 px error is too large for that and drifts monotonically with acquisition
   date, so it is a genuine one-hemisphere hand nudge.  Because ``c`` copies the
   previous page's layout forward, a nudge propagates to every later session and
   accumulates.  A left-vs-right V1 contrast on the current sets is confounded, and
   since the drift tracks date it is confounded with group if groups were run in date
   blocks.

Bilateral row alignment, box sizes, antero-posterior and medio-lateral ordering,
co-alignment equalities, disjointness and hemisphere sidedness are clean in all 356
files, so nothing else needs repairing.

This script throws the nudges away and rebuilds each set as the baseline constellation
scaled by that recording's own Bregma->Lambda distance, which is exactly the geometry
the editor would have produced with the rescale knob alone.  Mirror symmetry is then
exact by construction.

The reference Lambda is 55, not the 30 written in cortex22_roi_set.yaml
--------------------------------------------------------------------
``_seed_layout`` hands the editor ``args.lambda_offset`` (RUN_CONFIG["lambda_offset"],
55) when a page is seeded from the profile atlas -- it never reads the atlas file's own
``lambda_row_offset``.  Every existing set was therefore built treating CORTEX_22 as if
drawn at 55 rows.  Measured: at reference 55 the pure rescale reproduces the drawn
boxes to a median 1.5 px and reproduces 2 files exactly; at 30 it is off by 23 px.
Using 55 keeps the rebuilt ROIs the same size as the ones behind existing results, so
this is a cleanup and not a change of scale.  The `30` in the baseline file is
inconsistent with the pipeline and is flagged in CLAUDE.md section 9.

Header normalisation
--------------------
t1..t5 of one <day>_<animal> are one animal in one session, so their geometry must be
identical.  Four sessions disagreed (260611_PV5 t5 jumped 30 Bregma rows; 260611_T4 t1
missed a rescale; 260708_PV5 and 260716_PV5 t4/t5 sat 2 Bregma rows off).  Each animal
is rebuilt from the Bregma and Lambda held by the majority of its own recordings.

Run it:
    conda run --no-capture-output -n letizia python rebuild_roi_sets.py
    conda run --no-capture-output -n letizia python rebuild_roi_sets.py --out-dir roi_sets/rebuilt
    conda run --no-capture-output -n letizia python rebuild_roi_sets.py --dry-run
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from roi_editor import Box, save_roi_set
from wfci import load_atlas

# ---------------------------------------------------------------------------
# Edit this section to run the rebuild without passing CLI flags.
# ---------------------------------------------------------------------------
RUN_CONFIG: dict[str, Any] = {
    # Directory holding the hand-drawn sets to rebuild.
    "roi_set_dir": r"roi_sets",
    # The constellation every set is regenerated from.
    "baseline": r"roi_sets/cortex22_roi_set.yaml",
    # Where the regenerated sets are written. Same <key>.yaml names as the input, so
    # run_intermingle_rs.py / run_botox_batch.py can be pointed here with --roi-set-dir.
    "out_dir": r"roi_sets/rebuilt",
    # Bregma->Lambda distance the BASELINE boxes are treated as being drawn at.
    # 55 = RUN_CONFIG["lambda_offset"] in batch_roi_select.py, the value the existing
    # sets were actually built with. See the module docstring before changing this.
    "reference_lambda": 55,
    # True: give every recording of one animal the Bregma/Lambda held by the majority
    # of that animal's own files, so t1..t5 come out identical.
    "normalise_headers": True,
    # False (matches batch_roi_select.py): rescaling moves box CENTRES but keeps each
    # box's pixel area fixed, so ROI noise stays comparable across animals.
    "lambda_scales_box_size": False,
    # True: report what would be written without writing anything.
    "dry_run": False,
    "prefer_cli_args": True,
}

# t1..t5 of one <day>_<animal> are the same animal in the same session.
KEY_RE = re.compile(r"(?P<animal>\d{6}_[A-Za-z]+\d+)_t(?P<t>\d+)")

BOX_FIELDS = ("row_start", "row_end", "col_start", "col_end")


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rebuild ROI sets as a pure Lambda rescale of the baseline atlas.")
    p.add_argument("--roi-set-dir", default=defaults["roi_set_dir"])
    p.add_argument("--baseline", default=defaults["baseline"])
    p.add_argument("--out-dir", default=defaults["out_dir"])
    p.add_argument("--reference-lambda", type=int, default=defaults["reference_lambda"])
    p.add_argument("--no-normalise-headers", dest="normalise_headers",
                   action="store_false", default=defaults["normalise_headers"],
                   help="Rebuild each file from its own Bregma/Lambda instead of the "
                        "majority held by its animal.")
    p.add_argument("--dry-run", action="store_true", default=defaults["dry_run"])
    ns = p.parse_args()
    ns.lambda_scales_box_size = bool(defaults["lambda_scales_box_size"])
    return ns


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(
        roi_set_dir=config["roi_set_dir"],
        baseline=config["baseline"],
        out_dir=config["out_dir"],
        reference_lambda=config["reference_lambda"],
        normalise_headers=bool(config["normalise_headers"]),
        lambda_scales_box_size=bool(config["lambda_scales_box_size"]),
        dry_run=bool(config["dry_run"]),
    )


def round_half_away(value: float) -> int:
    """Round to nearest, ties away from zero.

    The rounding the mirror needs.  It satisfies both ``f(-x) == -f(x)`` (so the two
    hemispheres round the same way) and ``f(x + n) == f(x) + n`` for integer ``n`` (so
    the half-span shift that turns a centre into an edge cannot move one side and not
    the other).  ``round()``'s half-to-even has the first property but not the second,
    which is why it can put a mirrored pair 1 px off-centre -- see the module docstring.
    """
    return int(math.floor(value + 0.5)) if value >= 0 else -int(math.floor(-value + 0.5))


def scale_boxes_mirrored(boxes: dict[str, Box], factor: float,
                         scale_size: bool = False) -> dict[str, Box]:
    """``roi_editor.scale_boxes`` with mirror-exact rounding.

    Same contract otherwise: offsets are measured from Bregma, so multiplying them
    scales the constellation about Bregma without moving it.  ``scale_size=False``
    scales each box's centre and keeps its width and height, holding the pixel count
    behind every ROI mean fixed so noise stays comparable across animals.
    """
    scaled: dict[str, Box] = {}
    for label, box in boxes.items():
        if scale_size:
            r0, r1 = round_half_away(box.row_start * factor), round_half_away(box.row_end * factor)
            c0, c1 = round_half_away(box.col_start * factor), round_half_away(box.col_end * factor)
            # Shrinking must not invert a box: Box requires end >= start.
            scaled[label] = Box(r0, max(r0, r1), c0, max(c0, c1))
            continue
        span_r = box.row_end - box.row_start
        span_c = box.col_end - box.col_start
        r0 = round_half_away((box.row_start + box.row_end) / 2 * factor - span_r / 2)
        c0 = round_half_away((box.col_start + box.col_end) / 2 * factor - span_c / 2)
        scaled[label] = Box(r0, r0 + span_r, c0, c0 + span_c)
    return scaled


def read_header(path: Path) -> dict[str, Any]:
    """Bregma / Lambda / grid / downsample of one ROI set, as written."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        "bregma_row": int(data["bregma_row"]),
        "bregma_col": int(data["bregma_col"]),
        "lambda_row_offset": int(data["lambda_row_offset"]),
        "grid": tuple(data["grid"]),
        "downsample": float(data["downsample"]),
        "boxes": data["boxes"],
    }


def mirror_error(boxes: dict[str, Any], pairs: list[tuple[str, str, str]]) -> int:
    """Total absolute bilateral-mirror error of a layout, in pixels.

    Zero means every L box is the exact mirror of its R twin about Bregma. Used only
    to report what the rebuild removed.
    """
    total = 0
    for _, left, right in pairs:
        lc = (boxes[left]["col_start"] + boxes[left]["col_end"]) / 2
        rc = (boxes[right]["col_start"] + boxes[right]["col_end"]) / 2
        total += abs(lc + rc)
    return int(total)


def bilateral_pairs(labels: list[str]) -> list[tuple[str, str, str]]:
    """Pair each ``*L*`` label with its ``*R*`` twin: ('M2_alta', 'M2L_alta', 'M2R_alta')."""
    pairs = []
    for label in labels:
        # The side letter is the first L that has an R-twin when swapped.
        for i, ch in enumerate(label):
            if ch != "L":
                continue
            twin = label[:i] + "R" + label[i + 1:]
            if twin in labels:
                pairs.append((label[:i] + label[i + 1:], label, twin))
                break
    return pairs


def main() -> None:
    args = build_runtime_args()

    roi_dir = Path(args.roi_set_dir)
    baseline_path = Path(args.baseline)
    out_dir = Path(args.out_dir)

    # load_atlas gives the baseline the library's own validation (field names, types,
    # the grid check) instead of trusting the YAML blindly.
    base_atlas = load_atlas(baseline_path)
    base_boxes = dict(base_atlas)
    labels = list(base_boxes)
    pairs = bilateral_pairs(labels)

    base_raw = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    grid = tuple(base_raw["grid"])
    downsample = float(base_raw["downsample"])

    if mirror_error(base_raw["boxes"], pairs) != 0:
        raise ValueError(
            f"{baseline_path} is not bilaterally symmetric -- rebuilding from it "
            f"cannot restore mirror symmetry. Fix the baseline first."
        )

    # ---- collect the inputs, grouped by animal ----------------------------
    sources = sorted(p for p in roi_dir.glob("*.yaml")
                     if p.resolve() != baseline_path.resolve())
    if not sources:
        raise FileNotFoundError(f"No ROI sets to rebuild in {roi_dir}.")

    headers = {p.stem: read_header(p) for p in sources}
    animals: dict[str, list[str]] = defaultdict(list)
    ungrouped: list[str] = []
    for key in headers:
        m = KEY_RE.fullmatch(key)
        (animals[m.group("animal")].append(key) if m else ungrouped.append(key))
    for keys in animals.values():
        keys.sort(key=lambda s: int(s.rsplit("_t", 1)[1]))

    print(f"Baseline  : {baseline_path}  ({len(labels)} boxes, grid {grid[0]}x{grid[1]})")
    print(f"Reference : Bregma->Lambda = {args.reference_lambda} rows "
          f"(the scale the baseline boxes are treated as being drawn at)")
    print(f"Input     : {len(sources)} ROI sets in {roi_dir} "
          f"({len(animals)} animal-sessions"
          + (f" + {len(ungrouped)} ungrouped" if ungrouped else "") + ")")
    print(f"Output    : {out_dir}"
          + ("   [DRY RUN -- nothing will be written]" if args.dry_run else ""))
    print(f"Box size  : {'scales with Lambda' if args.lambda_scales_box_size else 'fixed'}")

    # ---- decide the Bregma/Lambda each recording is rebuilt at -------------
    # One animal is one session: its five recordings must share a geometry. Where they
    # disagree the majority wins -- a lone dissenting file is an editing slip, not a
    # measurement, and the majority is what the other four recordings agree on.
    chosen: dict[str, tuple[int, int, int]] = {}
    normalised: list[tuple[str, str, tuple, tuple]] = []
    for animal, keys in sorted(animals.items()):
        own = {k: (headers[k]["bregma_row"], headers[k]["bregma_col"],
                   headers[k]["lambda_row_offset"]) for k in keys}
        if args.normalise_headers:
            winner = Counter(own.values()).most_common(1)[0][0]
            for k in keys:
                if own[k] != winner:
                    normalised.append((animal, k, own[k], winner))
                chosen[k] = winner
        else:
            chosen.update(own)
    for key in ungrouped:
        h = headers[key]
        chosen[key] = (h["bregma_row"], h["bregma_col"], h["lambda_row_offset"])

    if normalised:
        print(f"\nHeader normalisation -- {len(normalised)} file(s) moved onto their "
              f"session's majority:")
        for animal, key, was, now in normalised:
            bits = []
            if was[:2] != now[:2]:
                bits.append(f"bregma ({was[0]},{was[1]}) -> ({now[0]},{now[1]})")
            if was[2] != now[2]:
                bits.append(f"lambda {was[2]} -> {now[2]}")
            print(f"  {key:<22} {'; '.join(bits)}")

    # ---- rebuild ----------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_mirror = 0
    total_mirror_px = 0
    changed = 0
    written: list[Path] = []
    per_animal_scale: dict[str, float] = {}

    for key in sorted(chosen):
        bregma_row, bregma_col, lam = chosen[key]
        factor = lam / args.reference_lambda
        boxes = scale_boxes_mirrored(base_boxes, factor, args.lambda_scales_box_size)

        # Everything below is reporting on what the rebuild changed.
        as_dict = {l: {f: int(getattr(b, f)) for f in BOX_FIELDS}
                   for l, b in boxes.items()}
        err = mirror_error(headers[key]["boxes"], pairs)
        if err:
            fixed_mirror += 1
            total_mirror_px += err
        if as_dict != {l: {f: int(headers[key]["boxes"][l][f]) for f in BOX_FIELDS}
                       for l in labels}:
            changed += 1
        m = KEY_RE.fullmatch(key)
        per_animal_scale[m.group("animal") if m else key] = factor

        if mirror_error(as_dict, pairs) != 0:
            raise AssertionError(f"{key}: rebuilt layout is still not mirrored.")

        if args.dry_run:
            continue
        written.append(save_roi_set(
            out_dir / f"{key}.yaml",
            boxes,
            bregma_row,
            bregma_col,
            grid,
            key,
            f"rebuilt by rebuild_roi_sets.py: {baseline_path.name} scaled by "
            f"lambda {lam}/{args.reference_lambda} = {factor:.4f}; bilaterally "
            f"symmetric by construction",
            lambda_offset=lam,
            downsample=downsample,
        ))

    scales = sorted(per_animal_scale.values())
    print(f"\nRebuilt   : {len(chosen)} ROI sets")
    print(f"  scale vs baseline : {scales[0]:.3f}x .. {scales[-1]:.3f}x "
          f"(median {scales[len(scales)//2]:.3f}x)")
    print(f"  geometry changed  : {changed}/{len(chosen)} files")
    print(f"  mirror symmetry   : repaired in {fixed_mirror} file(s), "
          f"{total_mirror_px} px of asymmetry removed in total")
    print(f"  every rebuilt set is bilaterally symmetric (verified per file)")
    if args.dry_run:
        print("\nDRY RUN -- no files written.")
    else:
        print(f"\nWrote {len(written)} files to {out_dir}")
        print(f"\nUse them:\n"
              f'  conda run -n letizia python run_botox_batch.py --roi-set-dir {out_dir} --full')


if __name__ == "__main__":
    main()
