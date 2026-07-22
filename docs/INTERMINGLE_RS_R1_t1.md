# Resting-state connectivity — BOTOX_RESTANI / 260611 / R1 / t1

What was built to analyse the interleaved ("intermingle") resting-state recording
at `\\146.48.88.209\share2\BOTOX_RESTANI\260611\R1\t1`, how to reproduce it, and
the result. Written 2026-07-21.

---

## 1. Goal

Run the **cerebellar resting-state functional-connectivity** pipeline on one
interleaved wide-field calcium-imaging folder and produce the 4×4 region-to-region
connectivity matrix `R_mean`, plus the visual checks needed to trust it. The work
was staged deliberately: **a debug smoke run on a handful of frames first**, then
the **full streaming run** over the whole recording once the setup was verified.

The analysis maths is entirely the existing `wfci` library — nothing in the
package was modified. Only a thin, study-owned run script was added (the P2
library/experiment boundary in [LIBRARY.md](../LIBRARY.md#8-the-library--experiment-boundary-p2)).

---

## 2. The dataset

| Property | Value |
|----------|-------|
| Folder | `\\146.48.88.209\share2\BOTOX_RESTANI\260611\R1\t1` |
| Files | 6000 single-page TIFFs (`R11_00001.tif …`), ~525 KB each, ~3 GB total |
| Frame size | 512 × 512, 16-bit (`uint16`) |
| Layout | **Interleaved** — odd-positioned images (1st, 3rd, …) are one channel, even the other |
| Channels | Odd = **GCaMP** (bright), even = **emo**/reflectance (dim). Auto-detected by brightness, verified: file 0 top-10% mean ≈ 34580 vs file 1 ≈ 12047 |
| Frames/channel | 3000 gcamp + 3000 emo |

Because the split alternates, one interleaved folder is **one trial**
(`gcamp`, `emo`), not a pair of paths.

---

## 3. What was built

### The script: [`run_intermingle_rs.py`](../run_intermingle_rs.py)

A self-contained, editor-first study script (repo root). Design choices:

- **Editor-first `RUN_CONFIG`** at the top (per the project convention) — set the
  folder, Bregma, profile, and debug toggle there and just run; CLI flags override
  when given (`--full`, `--debug-frames N`, `--bregma-row/col`, `--folder`).
- **A `debug` toggle that changes *both* the frame count and the memory mode:**
  - `debug = True` → load only the first `debug_max_frames` frames **per channel**
    into RAM (`run_profile`). RAM mode retains `dff_stack`, which is what lets it
    draw the **step-2 ROI-placement overlay** — the one check the script can't do
    for you (is Bregma right?).
  - `debug = False` (`--full`) → **stream** the whole folder frame-by-frame in
    constant memory (`run_streaming_profile`, two passes). This is what a ~3 GB
    recording needs; no `dff_stack` is kept, so no overlay in this mode.
- **Lazy debug loading** — the debug clip is read via `folder_frame_source(...)` +
  `itertools.islice`, so only the frames actually used are decoded off the network
  share, not all 6000.
- **No `try/except`** — a bad path or wrong shape fails loudly (project rule 5.2).
- **Library-only mechanism** — `interleaved_channel_files`, `folder_frame_source`,
  `ROIConfig.from_bregma`, `run_profile` / `run_streaming_profile`,
  `show_roi_placement`. The script holds only *policy* (which folder, which Bregma,
  which profile).

### Pipeline it runs (`cerebellar_rs` profile)

```
interleaved split → correction (ΔF/F) → ROI means (4 boxes) → connectivity (4×4 Pearson)
```

- trim 20 frames, ×0.5 box-downsample → correct → ×0.5 box-downsample,
- full-recording baseline and full-recording correlation window,
- 4 ROIs on the final 128×128 grid: `Laterale_L, Verme_L, Laterale_R, Verme_R`.

This is the MATLAB-validated cerebellar path (parity to machine precision —
[README §Validation](../README.md#validation-against-matlab)). Streaming agrees
with the in-memory path to ~1e-16.

---

## 4. How it was run (and how to reproduce)

Environment: conda env `letizia`. On this machine:

```bash
CONDA="$USERPROFILE/miniconda3/condabin/conda.bat"
```

**Step 1 — debug smoke run** (first 60 frames/channel, in-memory, writes the
overlay). This is the default `RUN_CONFIG`, so no flags are needed:

```bash
"$CONDA" run -n letizia python run_intermingle_rs.py
```

It confirmed: the interleaved split is correct (3000 + 3000, `auto`), `R_mean` is
well-formed, and the **ROI overlay** placed the 4 boxes symmetrically about the
midline in the lower-central field (caudal to Bregma = cerebellum). Bregma
`row=121, col=134` was confirmed against that overlay before the full run.

**Step 2 — full streaming run** (all 3000 frames/channel, constant memory):

```bash
"$CONDA" run -n letizia python run_intermingle_rs.py --full
```

To reproduce with a different Bregma or a longer debug clip:

```bash
"$CONDA" run -n letizia python run_intermingle_rs.py --bregma-row 121 --bregma-col 134 --full
"$CONDA" run -n letizia python run_intermingle_rs.py --debug-frames 120
```

---

## 5. Inputs

Set in `RUN_CONFIG` (or via flags):

| Key | Value used | Meaning |
|-----|-----------|---------|
| `folder` | the R1/t1 UNC path | the interleaved trial folder |
| `profile` | `cerebellar_rs` | 4 cerebellar ROIs, RS windows |
| `channel_order` | `auto` | brighter group → GCaMP (correct here) |
| `bregma_row` / `bregma_col` | `121` / `134` | per-animal Bregma (full-res; `//2` applied → `y_1=60, x_2=67`) |
| `debug` / `debug_max_frames` | `True` / `60` | smoke-run size (debug only) |
| `output_dir` | `outputs/intermingle_R1_t1` | where results land |

**Bregma is the one per-animal input the script cannot verify itself** — it was
confirmed visually via the debug overlay.

---

## 6. Outputs — [`outputs/intermingle_R1_t1/`](../outputs/intermingle_R1_t1/)

| File | From | Contents |
|------|------|----------|
| `connectivity_full.npz` | full run | `temp_roi (2980,4,1)`, `R (4,4,1)`, `R_mean (4,4)`, `averaged_traces (2980,4)`, `roi_labels`, `profile`, `bregma` |
| `roi_traces_full.png` | full run | trial-averaged ΔF/F trace per ROI |
| `connectivity_debug.npz` | debug run | same arrays on 60 frames, **plus `dff_stack`** |
| `roi_overlay_debug.png` | debug run | the 4 ROI boxes on a ΔF/F frame (Bregma check) |
| `roi_traces_debug.png` | debug run | ROI traces on the debug clip |

`temp_roi` has **2980** frames = 3000 − 20 trim. Arrays follow the library's shape
conventions (ROI outputs time-first; `R_mean` symmetric, unit diagonal, entries in
[-1, 1]).

---

## 7. Result

Full recording, `R_mean` (Pearson correlation), ROI order
`[Laterale_L, Verme_L, Laterale_R, Verme_R]`:

|            | Laterale_L | Verme_L | Laterale_R | Verme_R |
|------------|:----------:|:-------:|:----------:|:-------:|
| **Laterale_L** | 1.0000 | 0.9769 | 0.9310 | 0.9075 |
| **Verme_L**    | 0.9769 | 1.0000 | 0.9475 | 0.9519 |
| **Laterale_R** | 0.9310 | 0.9475 | 1.0000 | 0.9643 |
| **Verme_R**    | 0.9075 | 0.9519 | 0.9643 | 1.0000 |

All four cerebellar ROIs are strongly, positively correlated (0.91–0.98), as
expected for a resting-state recording; the ΔF/F traces fluctuate in a
physiological range (~−4% to +3.5%) and co-vary across regions. Highest coupling
is ipsilateral-vermis/lateral (L: 0.977; R: 0.964).

> These are the numbers for **this one animal/session**. Group contrasts
> (healthy vs treated, etc.) are a separate layer — `wfci.cohort` / `wfci.significance`
> over several `*.npz` results — see [README §Group analysis](../README.md#group-analysis-across-animals).

---

## 8. Notes & caveats

- **Bregma** was taken as the repo default and confirmed by eye on the overlay,
  not from an independent registration. If a per-animal Bregma is available from
  acquisition notes, re-run `--bregma-row/col` and re-check the overlay.
- **Channel order** `auto` is correct here (clear brightness margin). Force
  `--channel-order` only if a future dim-GCaMP recording fools the heuristic.
- **Streaming vs in-memory** agree to float roundoff; the full run used streaming
  purely for the ~3 GB memory footprint, not for any difference in the numbers.
- Nothing in `src/wfci/` was changed; only `run_intermingle_rs.py` and this doc
  were added.
