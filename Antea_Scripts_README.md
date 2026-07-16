# Antea Scripts — Cortical Wide-field GSR Connectivity (MATLAB)

The six MATLAB scripts in [Antea_scripts/](Antea_scripts/) implement a **second,
independent wide-field imaging pipeline**: dual-channel hemodynamic correction
(ΔF/F), **global signal regression** (GSR), **22 cortical ROIs** placed relative
to Bregma, a 22×22 functional-connectivity matrix per animal, and finally
**group-level statistics and figures** (healthy vs. disease, NBS network-based
statistic).

This is *not* the pipeline ported to Python. The `wfci` package documented in
[README.md](README.md) ports the **cerebellar** scripts in [matlab/](matlab/) —
4 ROIs, no GSR, resting-state *and* stimulated. The two pipelines share the same
step-1 hemodynamic maths and diverge everywhere else; see
[Differences between the two MATLAB pipelines](#differences-between-the-two-matlab-pipelines).

The scripts are shipped as **`.txt` files**, are **not functions**, and are
**not runnable end-to-end as-is**: each one is a chunk of script meant to be
pasted into the MATLAB editor and run section-by-section (`%%` cells) against a
shared base workspace. Every script depends on variables left in the workspace
by the previous one.

Reference document (Italian, includes screenshots of the intended figures):
`Antea_scripts/Analisi WF - Nuove.docx`. **Local-only** — it is git-ignored and
not distributed with this repository, so a fresh clone will not contain it.

---

## Entrypoints

There is **no single entrypoint**. The scripts run in numbered order, in one
MATLAB session, sharing the base workspace. To run them, copy each `.txt` into a
`.m` file (or paste its contents into the editor):

```bash
# From the repository root (Git Bash). Renaming is required: MATLAB will not run .txt.
mkdir -p temporary_files/antea_m
cp "Antea_scripts/(1)_correzione_emodinamica.txt"            temporary_files/antea_m/antea_step1_correzione_emodinamica.m
cp "Antea_scripts/(2)_Global_Signal_Regression_SCRIPT.txt"   temporary_files/antea_m/antea_step2_gsr.m
cp "Antea_scripts/(3)_FOV128-128_ROIposition.txt"            temporary_files/antea_m/antea_step3_roi_position.m
cp "Antea_scripts/(4)_FOV128x128_corr_SCRIPT.txt"            temporary_files/antea_m/antea_step4_corr.m
cp "Antea_scripts/(5)_matrici_e_figure.txt"                  temporary_files/antea_m/antea_step5_matrici_e_figure.m
cp "Antea_scripts/(6)_visualizzazione_connettivita_sign.txt" temporary_files/antea_m/antea_step6_nbs_figures.m
```

Then, in MATLAB, from `temporary_files/antea_m/`:

| Order | Script | What you must edit first | Runs headless? |
|-------|--------|--------------------------|----------------|
| 1 | `antea_step1_correzione_emodinamica.m` | the `dir(fullfile(...))` data path (line 2) | yes |
| 2 | `antea_step2_gsr.m` | the `uiopen(...)` mask path (line 2) | **no** — `uiopen` is a GUI dialog |
| 3 | `antea_step3_roi_position.m` | `y_1`, `x_2` (Bregma, per animal) | **no** — draws a figure |
| 4 | `antea_step4_corr.m` | nothing (reuses `y_1`, `x_2` from step 3) | yes |
| 5 | `antea_step5_matrici_e_figure.m` | the subject variable names (every `cat`/`mean` line) | **no** — draws figures |
| 6 | `antea_step6_nbs_figures.m` | the `load(...)` node-position path (line 45, line 140) | **no** — needs the NBS GUI |

```matlab
% Typical session, from temporary_files/antea_m/ in MATLAB:
antea_step1_correzione_emodinamica   % -> t_TEMP
antea_step2_gsr                      % -> t_TEMP_resized, t_TEMP_regressed   (mask dialog opens)
antea_step3_roi_position             % -> y_1, x_2 ; visual check of ROI placement
antea_step4_corr                     % -> TEMP, R, mean_R_...
% ...repeat 1-4 for every animal/day, renaming the outputs each time...
antea_step5_matrici_e_figure         % -> group means, DIFF, correlation-matrix figure
antea_step6_nbs_figures              % -> network graphs + barplots  (requires NBS toolbox running)
```

Steps 5 and 6 are **not** per-animal: they consume the per-animal results of
step 4 after you have renamed and saved them, one variable per animal per day.

Toolboxes required: Image Processing (`imresize`, `imagesc`), Statistics and
Machine Learning (`fitlm`, `corr`, `nanmean`), and — for step 6 only — the
external **NBS 1.2** toolbox (network-based statistic), which must be launched
and left open so the `global nbs` structure exists.

---

## Expected input

**Per-animal imaging data (step 1).** One folder per animal/session, e.g.
`/home/firenze/Desktop/Ts65Dn/GSR/10T2_L_WT/emo`, containing **multi-page TIFFs,
two per trial, alternating channels**. Step 1 walks the `dir` listing with
`for i = 3:2:size(matfiles,1)` — the `3` skips the `.` and `.` entries returned by
`dir` on Linux, and the stride of 2 consumes one pair per iteration:

| `dir` entry | Assigned to | Variable |
|-------------|-------------|----------|
| `matfiles(i)` | first file of the pair | `t_FILE_emo` (hemodynamic / reflectance) |
| `matfiles(i+1)` | second file of the pair | `t_FILE_gCaMP` (fluorescence) |

So **file order on disk decides which channel is which** — `emo` first, `gcamp`
second, with no intensity check. Each TIFF is read page-by-page with `imread`
into a `[y, x, time]` double array. Frames are used in full: unlike the
cerebellar pipeline, **no frames are trimmed**.

**Brain mask (step 2).** A TIFF mask at full FOV resolution, loaded
interactively via `uiopen('/home/firenze/Desktop/Ts65Dn/FOV_bl/1T_R_WT_BL.tif',1)`,
which must land in the workspace under the name `Mask`. Non-zero = brain,
zero = excluded. It is downsampled ×0.5 to match `t_TEMP_resized`, and every
pixel where the resized mask is `0` is set to `NaN` across all frames and trials.

**Bregma (step 3).** `y_1 = floor(126/2)`, `x_2 = floor(126/2)` — the Bregma row
and column at full resolution, halved to index the ×0.5-downsampled images. The
comment `% settare per ogni animale` is the point: **these are per-animal and
must be re-set for every recording**, then re-checked visually.

**Per-animal results (steps 5–6).** Step 5 expects one variable per animal per
day already in the workspace, named by hand, e.g. `mean_R_PV_F_MACCHI_DX_day4`,
`mean_R_CR1M_SX_day4`. Step 6 additionally expects `DIFF` (from step 5), a live
NBS run (`global nbs`), and a `node_positions` matrix loaded from
`nodeposition_PD.mat` (hard-coded Windows path).

---

## Expected output

Nothing is written to disk automatically. Every result is a **workspace
variable**, and figures are drawn to screen. The naming convention is set by the
comment in step 4 — *"save these files as CORR_ANIMAL_XDPL"* — i.e. you save the
workspace manually, per animal, per day.

| Step | Variable | Shape | Meaning |
|------|----------|-------|---------|
| 1 | `t_TEMP` | `[y, x, time, trial]` | ΔF/F in %, one ×0.5 downsample applied |
| 2 | `t_TEMP_resized` | `[y, x, time, trial]` | second ×0.5 downsample + mask applied (NaN outside brain) |
| 2 | `t_TEMP_regressed` | `[y, x, time, trial]` | GSR residuals — the input to every later step |
| 3 | `img_av` | `[y, x]` | one frame with the 22 ROI boxes burned in at value `1`, shown via `imagesc` |
| 4 | `TEMP` → `TEMP_regressed_ANIMAL_XDPL` | `[time, 22, trial]` | mean ΔF/F trace per ROI |
| 4 | `R` → `R_regressed_ANIMAL_XDPL` | `[22, 22, trial]` | per-trial Pearson matrix |
| 4 | `mean_R_regressed_ANIMAL_XDPL` | `[22, 22]` | trial-averaged connectivity for that animal/day |
| 5 | `mean_SANI`, `mean_PD`, `DIFF` | `[22, 22]` | group means and their difference (healthy − disease) |
| 5 | `SUBJECTS_ALL`, `SUBJECT_ALL_M/F` | `[22, 22, n]` | all subjects stacked, and the male/female splits |
| 6 | `matrix_onlysignHYPER`, `matrix_onlysign_HYPO` | `[22, 22]` | `DIFF` masked to the NBS-significant edges only |
| 6 | `connectivity_matrixHYPER/_HYPO` | `[22, 22]` | symmetrised versions used for the graph plots |

Figures produced: the `DIFF` correlation matrix with 22 region labels (step 5);
two node-and-edge network graphs over MNI-like node positions, hyper- and
hypo-connectivity, with colour-graded nodes and edge widths (step 6); and two
horizontal barplots counting significant edges per region (step 6).

**Region order** for all 22-element axes is left hemisphere then right, each in
this order:

```
MOs-a, MOs-p, MOp-a, MOp-p, SSp-bfd, SSp-tr, SSp-fl, SSp-hl, RSP, VISa, VISp
```

i.e. secondary motor (anterior/posterior), primary motor (anterior/posterior),
barrel field, trunk, forelimb, hindlimb, retrosplenial, anterior visual, primary
visual — Allen CCF naming. In step 4 the traces are built as
`regioni_L` then `regioni_R`, matching the labels in steps 5 and 6.

---

## Pipeline overview

- **Step 1 — hemodynamic correction → ΔF/F** (`(1)_correzione_emodinamica`).
  Read each channel's TIFF, `imresize(...,0.5,'box')`, take each channel's
  temporal mean over the **whole recording** (`MIf`, `MIr`), divide each channel
  by its own mean, divide the GCaMP ratio by the `emo` ratio, and express the
  result as `(ratio-1)*100` %. Accumulates one trial per loop iteration into
  `t_TEMP(:,:,:,K)`.
- **Step 2 — global signal regression** (`(2)_Global_Signal_Regression_SCRIPT`).
  Apply the second ×0.5 downsample, NaN out everything outside the brain mask,
  then per trial: compute the **global signal** as the spatial `nanmean` over all
  brain pixels, fit `fitlm(global_signal, pixel_signal)` **per pixel**, and
  subtract the fitted component (slope·global + intercept) from that pixel's
  trace. The residual is `t_TEMP_regressed`. This removes the shared,
  brain-wide fluctuation so that the correlations in step 4 reflect
  region-specific coupling rather than global drift.
- **Step 3 — ROI placement check** (`(3)_FOV128-128_ROIposition`). Burn the 22
  ROI boxes (each 6×6 px, defined as offsets from `y_1`, `x_2`) into a single
  frame and `imagesc` it. Purely a **visual check** that Bregma and the boxes sit
  on the right cortical areas for this animal — it produces no data.
- **Step 4 — ROI traces + connectivity** (`(4)_FOV128x128_corr_SCRIPT`). Extract
  the same 22 boxes from `t_TEMP_regressed`, reduce each to one trace per frame
  with `nanmean(nanmean(...,1),2)`, concatenate as `[time, 22]` per trial, then
  `corr` each trial into a 22×22 Pearson matrix and average across trials.
- **Step 5 — group matrices and figures** (`(5)_matrici_e_figure`). Concatenate
  per-animal matrices along dim 3 by group, average within group, and take
  `DIFF = mean_SANI - mean_PD`. Repeat for the male-only and female-only splits.
  Plot `DIFF` as a labelled 22×22 image.
- **Step 6 — significance visualisation** (`(6)_visualizzazione_connettivita_sign`).
  With an NBS run open, read the significant-edge adjacency from `nbs.NBS.con_mat{1}`,
  mask `DIFF` to those edges, and render the surviving network twice — **HYPER**
  (red palette, hyper-connectivity) and **HYPO** (blue palette) — as a
  node-and-edge graph plus a per-region barplot of significant-edge counts.

---

## Differences between the two MATLAB pipelines

The two pipelines answer different questions on different brain structures. They
are **not variants of each other and their outputs are not comparable**.

| | `Antea_scripts/` (this document) | `matlab/` (ported to `wfci`) |
|---|---|---|
| **Brain structure** | dorsal cortex, through-skull | cerebellum |
| **ROIs** | **22** Allen-named areas, 11 per hemisphere | **4**: `Verme_L/R`, `Laterale_L/R` |
| **Bregma used** | `floor(126/2)`, `floor(126/2)` | `floor(121/2)`, `floor(134/2)` |
| **Global signal regression** | **yes** — per-pixel `fitlm` against the brain-wide mean | **no** |
| **Brain mask** | **yes** — `uiopen` TIFF, outside-brain pixels → `NaN` | **no** mask; `nanmean` still used defensively |
| **Frame trimming** | **none** — all frames kept | first 20 dropped (`out(:,:,21:end)`) |
| **Baseline for ΔF/F** | whole-recording temporal mean, always | RS: whole recording; stimulated: pre-stimulus `1:278` |
| **Stimulated / optogenetic variant** | **none** — resting-state only | yes — separate `_stimolato` scripts, `corr` over `280:300` |
| **Where the 2nd ×0.5 downsample happens** | in step 2, before GSR | at the end of step 1 |
| **File pairing in `dir`** | `i = 3:2:end`; `matfiles(i)` → **emo**, `matfiles(i+1)` → **gcamp** | `i = 5:4:end`; `matfiles(i)` → **gcamp**, `matfiles(i+1)` → **emo** (4 files per trial) |
| **Analysis endpoint** | **group** statistics: group means, `DIFF`, NBS network stats, network + bar figures | **per-animal** `R_mean`; no group layer |
| **External toolbox** | NBS 1.2 (step 6) | none |
| **Python port** | **none** | `wfci` — see [README.md](README.md) |
| **Validated numerically** | no | yes, to machine precision — see [README.md](README.md#validation-against-matlab) |
| **Runs headless / scriptable** | no — `uiopen`, hand-renamed variables, live NBS GUI | steps 1 and 3 yes; step 2 is a figure |

**What they share.** Step 1 is the same algorithm in both, and to the frame trim
and the choice of baseline window it is the same code: read two channels,
`imresize(...,0.5,'box')`, normalise each channel by its own temporal mean,
divide GCaMP by `emo`, `(ratio-1)*100`. Both then apply a second ×0.5 box
downsample (128×128 from a 512×512 FOV), both define ROIs as fixed 6×6 offsets
from a per-animal Bregma, both reduce ROIs with `nanmean(nanmean(...,1),2)`, and
both compute connectivity as a per-trial `corr` averaged over trials. So the
`wfci` step-1 implementation is directly reusable for the Antea data; what is
missing on the Python side is the mask, the GSR regression, the 22-ROI layout,
and the group/NBS layer.

**Practical consequence.** If the Antea pipeline is ever ported, the differences
that matter most are (a) GSR is a per-pixel linear fit and is by far the most
expensive step — 128×128 `fitlm` calls per trial — and is a strong candidate for
vectorisation; (b) the channel assignment is reversed relative to `matlab/`, so
the loader cannot be reused blindly; and (c) steps 5–6 encode the experimental
design (group membership, sex) in **variable names**, which a port would have to
replace with an explicit subject table.

---

## Directory map

```
letizia/
├── Antea_Scripts_README.md                     ← this file
├── Antea_scripts/
│   ├── (1)_correzione_emodinamica.txt          ← step 1: read pairs → ΔF/F  → t_TEMP
│   ├── (2)_Global_Signal_Regression_SCRIPT.txt ← step 2: mask + per-pixel GSR → t_TEMP_regressed
│   ├── (3)_FOV128-128_ROIposition.txt          ← step 3: burn 22 ROI boxes into a frame (visual check)
│   ├── (4)_FOV128x128_corr_SCRIPT.txt          ← step 4: 22 ROI traces → 22×22 R → mean_R_...
│   ├── (5)_matrici_e_figure.txt                ← step 5: group means, DIFF, matrix figure
│   ├── (6)_visualizzazione_connettivita_sign.txt ← step 6: NBS-significant network + barplots
│   └── Analisi WF - Nuove.docx                 ← protocol notes + figures (Italian); git-ignored, local-only
├── matlab/                                     ← the OTHER pipeline (cerebellum, 4 ROIs) — see README.md
├── README.md                                   ← `wfci`, the Python port of matlab/
└── REFERENCE.md                                ← canonical technical map of the repository
```

---

## Examples

There is **no runnable example and no sample data** for this pipeline. The
sample TIFFs in [data/](data/) belong to the cerebellar pipeline (one channel,
512×512, 7 frames) and cannot exercise these scripts: step 1 needs whole folders
of paired multi-page TIFFs, step 2 needs a matching brain mask, and steps 5–6
need a full cohort of per-animal results plus a live NBS session.

To see the intended figures without running anything, open
`Antea_scripts/Analisi WF - Nuove.docx` (local-only; not in the repository).

---

## Notes & caveats

- **The scripts are `.txt`, not `.m`.** MATLAB will not execute them under that
  extension. Copy them as shown in [Entrypoints](#entrypoints).
- **Hard-coded absolute paths from three different machines.** Step 1 and 2 use
  Linux paths (`/home/firenze/...`, `/media/...`), step 6 a Windows one
  (`C:\Users\VivoBook\Documents\MATLAB Path\NBS1.2\...`). All must be edited
  before running.
- **No preallocation.** `t_TEMP`, `t_TEMP_resized`, `t_TEMP_regressed`, `TEMP`
  and `R` all grow inside loops. With `[128, 128, time, trial]` doubles this is
  slow and memory-hungry; on a long session the growth alone can dominate runtime.
- **The 4 GB TIFF limit applies.** The comment `ricorda che se pesa più di 4gb non
  lo legge!!` is in step 1 here too — MATLAB's reader mis-reads files above 4 GB.
  This pipeline has no streaming path (the Python port has one, but only for the
  cerebellar pipeline).
- **GSR drops any pixel with a single NaN.** In step 2, `if ~isnan(data(row,col,:))`
  is an `if` over an array, so it is true only when the pixel is NaN-free across
  **every** frame — one bad frame discards that pixel's whole trace. This is on
  top of the mask.
- **Channel identity rests entirely on filename sort order.** Step 1 assigns
  `emo` and `gcamp` by position in the `dir` listing with no intensity check, and
  the `i = 3:2:...` start index assumes the Linux `dir` behaviour of returning
  `.` and `..` first. A stray file in the folder shifts every pair and silently
  swaps the two channels.
- **Step 3 and step 4 must be kept in sync by hand.** The 22 box definitions are
  duplicated verbatim between them; editing the placement in step 3 without
  mirroring it into step 4 means you are validating boxes you are not measuring.
  (The cerebellar pipeline has the same duplication, and there the two copies
  have already drifted — `Laterale_L` is `x_2-37:x_2-32` in step 2 but
  `x_2-34:x_2-29` in step 3.)
- **Experimental design lives in variable names.** `mean_R_PV_F_MACCHI_DX_day4`
  encodes strain, sex, animal and day. Step 5 hard-codes group membership by
  listing these names in `cat(3, ...)`, so adding a subject means editing the
  script.
- **Step 5's female split looks inverted.** `mean_SANI_F` averages the single
  `CR1` animal while `mean_PD_F` averages the two `PV` animals — the opposite of
  the grouping used in `mean_SANI`/`mean_PD` and `mean_SANI_M`/`mean_PD_M`, where
  `PV` is healthy and `CR1` is the disease group. Worth confirming with the
  author before trusting `DIFF_F`.
- **Step 6 needs NBS open.** It reads `global nbs` from the running NBS 1.2
  toolbox; there is no saved intermediate. `degrees = connectivity_matrix...`
  followed by `degrees / max(degrees)` divides a matrix by a row vector, so
  `node_sizes` is a 22×22 matrix rather than the per-node vector the comment
  describes — `scatter` tolerates it, but the node sizes are not the node degrees.
- **`COLOR_EDGE` and `node_colors_2` persist between the HYPER and HYPO cells.**
  Neither is cleared, so if HYPO produces fewer edges than HYPER, stale rows from
  the previous plot remain and are silently reused.
