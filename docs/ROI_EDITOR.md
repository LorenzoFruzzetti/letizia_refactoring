# Changing the ROIs — `roi_editor.py` and ROI-set files

How to look at the first image of every recording, move the ROI boxes and Bregma
onto the anatomy, save the result, and feed it back into the run scripts — either
one layout per animal, or the same layout for every session. Written 2026-07-22.

---

## 1. The short version

```bash
CONDA="$USERPROFILE/miniconda3/condabin/conda.bat"

# 1. draw: opens a window, one page per animal in the manifest
"$CONDA" run --no-capture-output -n letizia python roi_editor.py

# 2a. run ONE folder with the geometry you drew
"$CONDA" run -n letizia python run_intermingle_rs.py --roi-set roi_sets/260611_R1.yaml --full

# 2b. run the whole batch, each animal with its own file
"$CONDA" run -n letizia python run_botox_batch.py --roi-set-dir roi_sets --full

# 2c. run the whole batch with ONE layout shared by every session
"$CONDA" run -n letizia python run_botox_batch.py --roi-set roi_sets/shared_roi_set.yaml --full
```

---

## 2. Where the ROIs came from before

Nothing about the boxes was ever editable from the run scripts. `run_intermingle_rs.py`
and `run_botox_batch.py` took the ROI layout straight from the profile's atlas —
[`CEREBELLUM_4`](../src/wfci/atlases.py) for `cerebellar_rs`, four fixed boxes given
as **inclusive, 1-based offsets from Bregma** — and the only per-animal knob was
Bregma itself (`bregma_row` / `bregma_col`).

That is two separate things you might want to change:

| What | Where it lived | What it does |
|------|----------------|--------------|
| **Bregma** | `RUN_CONFIG["bregma_row"/"bregma_col"]`, or the manifest column | Anchors the whole layout. Moving it translates **all** boxes together. |
| **The boxes** | `wfci.atlases.CEREBELLUM_4` (library) | The offsets of each region from that anchor, and their sizes. |

Editing the library's atlas was never the answer — that constant is MATLAB-validated
and shared by every study (P1/P2 in [LIBRARY.md](../LIBRARY.md)). The library already
supported a study bringing its own layout (`load_atlas` / `save_atlas`); what was
missing was a way to *see* where the boxes land and drag them there.

---

## 3. `roi_editor.py` — the editor

```bash
"$CONDA" run --no-capture-output -n letizia python roi_editor.py
```
or, in VS Code, the **"ROI editor (roi_editor.py)"** launch config (F5 → it opens an
external terminal plus the window, on the pinned `letizia` interpreter).

> **"I ran it and it just did everything automatically."** Then you ran a *pipeline*
> script, not the editor. `run_intermingle_rs.py` and `run_botox_batch.py` never stop
> to ask — by design, so a 15 GB batch can run unattended. Selecting ROIs happens in
> `roi_editor.py`, which computes nothing: it draws, you save, and the pipeline reads
> the file afterwards via `--roi-set`.
>
> Two smaller traps: plain `conda run` (no `--no-capture-output`) withholds every
> print until the process exits, so the terminal looks dead while the window is open;
> and the Tk window can open **behind** the terminal — check the taskbar.

**What it shows.** One page per session. For each it decodes only the first
`preview_frames` GCaMP images of that folder and applies the *same* two 0.5× box
downsamples the pipeline uses (512 → 256 → 128), so the picture is exactly the grid
the ROI offsets are resolved against: one offset unit = one pixel on screen. It is a
raw anatomy image, not ΔF/F — vasculature and the midline are visible, which is what
you actually place boxes against.

**Which sessions.** Set in `RUN_CONFIG` (or by flag):

| Source | Pages | Key (= filename stem) |
|--------|-------|-----------------------|
| `manifest` + `scope: "animal"` (default) | one per manifest row, previewing that animal's **first** recording | `260611_R1` |
| `manifest` + `scope: "recording"` | one per `t#` folder — "the first image of every file" | `260611_R1_t1` |
| `folders: [...]` (with `manifest: None`) | one per listed folder | last 3 path parts, e.g. `260611_R1_t1` |

The per-animal key is the one `run_botox_batch.py --roi-set-dir` looks up, so
`scope: "animal"` is the mode that feeds the batch run directly.

**Controls** (also printed in the window and to the terminal):

| | |
|---|---|
| left-drag inside a box | move it |
| left-drag a corner | resize it |
| right-click | put Bregma there (all boxes translate with it) |
| `ctrl`+arrows | nudge Bregma 1 px |
| arrows / `shift`+arrows | nudge the selected box 1 / 5 px |
| `+` / `-` | grow / shrink the selected box by 1 px on every side |
| `n` / `p` | next / previous session |
| `s` / `S` | save this session / save **every** session |
| `w` | write the **shared** ROI set (one file for all sessions) |
| `a` / `A` | apply this layout to all sessions — `A` also copies Bregma |
| `r` | reset this page to how it started |
| `[` / `]` | display contrast |
| `h` / `q` | print help / quit |

A box dragged off the frame turns **red** and the save is **refused** with a message
naming it. That is the same condition [`wfci.roi.box_slices_for`](../src/wfci/roi.py)
raises on — an off-frame box is not a crash in NumPy, it is a *negative index*, so a
left-hemisphere ROI would quietly average the right side of the brain and return a
perfectly ordinary number.

### The "same ROIs for every session" workflow

1. Open the editor (`scope: "animal"`), land on the first animal, get the boxes right.
2. Press `A` — copies the boxes **and** Bregma to every page.
3. Press `n` a few times to check the copy actually fits the other animals' anatomy
   (Bregma really does move between animals; that is what `a` — boxes only, each page
   keeps its own Bregma — is for).
4. Press `w` to write `roi_sets/shared_roi_set.yaml`, or `S` to write one file per session.

---

## 4. The ROI-set file

`roi_sets/<key>.yaml`, written by the editor:

```yaml
name: 260611_R1
grid: [128, 128]            # the FINAL frame these offsets were drawn for
source: drawn with roi_editor.py on the cerebellar_rs preview grid; folder \\...\R1\t1
bregma_row: 120             # the Bregma the boxes were drawn from (RUN_CONFIG units)
bregma_col: 134
boxes:                      # ORDER IS THE COLUMN ORDER OF R — do not sort
  Laterale_L: {row_start: 21, row_end: 26, col_start: -34, col_end: -29}
  Verme_L:    {row_start: 22, row_end: 27, col_start: -12, col_end: -7}
  Laterale_R: {row_start: 21, row_end: 26, col_start: 29,  col_end: 34}
  Verme_R:    {row_start: 22, row_end: 27, col_start: 0,   col_end: 5}
```

- It is a **superset of the library's atlas file**: `wfci.load_atlas` reads it
  unchanged and ignores the two `bregma_*` keys. So the same file works anywhere an
  atlas is accepted (e.g. `run_pipeline.py --atlas`).
- **Bregma travels with the boxes** because the boxes are offsets *from* it. Using one
  animal's boxes with another's Bregma moves every ROI while still producing numbers.
  `load_roi_set` returns both; the run scripts apply both.
- `bregma_row` is in `RUN_CONFIG` units — the scripts apply `// 2` to reach the
  final-grid coordinate (`y_1 = bregma_row // 2`), matching the MATLAB
  `y_1 = floor(121/2)`. The editor writes `2 × y_1`, so a Bregma of 121 comes back as
  120; both floor to the same pixel.
- `grid` is checked against the real data at run time. An atlas drawn for a different
  field of view whose boxes all still *fit* is wrong by a scale factor with nothing
  else to catch it — see [README § Why `grid` matters](../README.md#why-grid-matters).
- You can edit the file by hand; the offsets are written MATLAB-style (1-based,
  inclusive) so they diff by eye against `matlab/step3_*.m`.
- Deleting or reordering boxes is allowed and is a real change: box order **is** the
  row/column order of `R`, and `R` is sized from `len(boxes)`.

---

## 5. Loading a ROI set in the run scripts

### `run_intermingle_rs.py` (one folder)

```python
RUN_CONFIG = {
    ...
    "bregma_row": 121,
    "bregma_col": 134,
    "roi_set": r"roi_sets\260611_R1.yaml",   # None = profile atlas + the values above
}
```

or `--roi-set roi_sets/260611_R1.yaml`. When set, the file supplies the boxes **and**
the Bregma (the `bregma_row`/`bregma_col` above are then unused). The path is recorded
in the output `.npz` as `roi_set`, so a result always says which geometry produced it.

### `run_botox_batch.py` (the manifest)

```python
RUN_CONFIG = {
    ...
    "roi_set_dir": r"roi_sets",   # per-unit: roi_sets\<day>_<animal>.yaml
    "roi_set": None,              # one file for every unit
}
```

Resolution order per unit: `roi_set_dir\<day>_<animal>.yaml` → `roi_set` → the
profile's atlas with the manifest's Bregma. A unit with no per-unit file prints which
fallback it took, so a half-drawn `roi_sets/` folder never silently reverts to the
preset. Both `roi_set_dir` and `roi_set` set to `None` reproduces the previous
behaviour exactly. The chosen path is written into each unit's `.npz` and into
`batch_summary.csv`.

---

## 6. What changed in the repo

| File | Change |
|------|--------|
| `roi_editor.py` | **new** — the editor, plus `load_roi_set` / `save_roi_set` / `preview_image`, importable without opening a window |
| `run_intermingle_rs.py` | `roi_set` config key + `--roi-set`; new `apply_roi_set()` helper; `roi_set` saved in the `.npz` |
| `run_botox_batch.py` | `roi_set_dir` / `roi_set` config keys + flags; `resolve_roi_set()`; `roi_set` in the `.npz` and the summary CSV |
| `src/wfci/` | **unchanged** — the editor is study policy, the library keeps the mechanism (P2) |

Verified: the editor's geometry conversion agrees box-for-box with
`wfci.roi.box_slices_for`; a saved file round-trips through `load_roi_set` **and**
`wfci.load_atlas`; an edited 2-ROI set runs end-to-end and yields a 2×2 `R_mean`; the
full test suite (156 tests) still passes.
