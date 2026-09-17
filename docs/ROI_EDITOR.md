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

That is three separate things you might want to change:

| What | Where it lived | What it does |
|------|----------------|--------------|
| **Bregma** | `RUN_CONFIG["bregma_row"/"bregma_col"]`, or the manifest column | Anchors the whole layout. Moving it translates **all** boxes together. |
| **Lambda** | `RUN_CONFIG["lambda_offset"]` (editor only) | The distance Bregma → Lambda, i.e. the **scale** the offsets are in. Changing it stretches or shrinks the whole layout about Bregma. |
| **The boxes** | `wfci.atlases.CEREBELLUM_4` (library) | The offsets of each region from that anchor, and their sizes. |

Editing the library's atlas was never the answer — that constant is MATLAB-validated
and shared by every study (P1/P2 in [LIBRARY.md](../LIBRARY.md)). The library already
supported a study bringing its own layout (`load_atlas` / `save_atlas`); what was
missing was a way to *see* where the boxes land and drag them there.

### Current starting layout: the 22 cortical boxes

`roi_editor.py` now opens on `RUN_CONFIG["profile"] = "cortical_gsr"`, whose atlas is
[`CORTEX_22`](../src/wfci/atlases.py) — the 22 `img_av(y_1+…, x_2+…)` boxes from the
MATLAB cortical scripts. The editor reads **only** a profile's atlas and downsample
(it runs no pipeline), so that choice does not pull in `cortical_gsr`'s brain mask or
GSR. Set it back to `"cerebellar_rs"` for the four-box `CEREBELLUM_4` layout.

The same 22 boxes are checked in as [`roi_sets/cortex22_roi_set.yaml`](../roi_sets/README.md),
which both run scripts use by default — atlas only, on the unchanged `cerebellar_rs`
chain, giving a 22×22 `R`.

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
> and the Qt window can open **behind** the terminal — check the taskbar.

**Built on pyqtgraph.** Each box is a `pg.RectROI` and each landmark a `pg.TargetItem`,
so dragging, corner handles and whole-pixel snapping (`translateSnap` / `scaleSnap`) are
Qt's job rather than hand-rolled hit-testing — a box cannot acquire a fractional
offset, and rotation is disabled because `wfci` has no way to express a rotated slice.
Scroll or right-drag to zoom/pan; right-click is pyqtgraph's own menu (view range,
export image).

Qt is imported inside `ROIEditor._build_ui`, **never** at module level, so the run
scripts' `from roi_editor import load_roi_set` still works on a headless batch
machine — an unattended 15 GB run must not die because there is no display.
`tests/test_roi_editor.py` drives the real widgets under
`QT_QPA_PLATFORM=offscreen` and asserts that import stays GUI-free.

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

*Move the whole layout* — Bregma **and** every box together. This is the primary
gesture: the boxes are stored as offsets *from* Bregma, so moving the anchor
translates all of them rigidly. Nothing deforms, no box changes size.

| | |
|---|---|
| **drag the magenta `+`** | drag Bregma and every box with it |
| double-click | put Bregma there (all boxes translate with it) |
| `ctrl`+arrows | move Bregma + all boxes 1 px |
| `ctrl`+`shift`+arrows | move Bregma + all boxes 5 px |

*Scale the whole layout* — the green `x` is **Lambda**, on the midline
`lambda_offset` rows posterior to Bregma. It is a *ruler*, not a second anchor: its
distance from Bregma is the scale the offsets are in, so dragging it does not move one
landmark, it rescales the entire atlas about Bregma. This is the knob for "this brain
sits bigger in the field of view than the one the atlas came from" — line the `x` up
with the animal's real Lambda and every box moves outward in proportion.

| | |
|---|---|
| **drag the green `x`** | rescale the layout about Bregma |
| `,` / `.` | distance −1 / +1 px |
| `<` / `>` | distance −5 / +5 px |
| `l` | back to scale 1.000× (the reference layout) |

Two properties worth knowing, because both are deliberate:

- **Box sizes stay fixed** by default; only the positions scale. Holding each box's
  area constant keeps the number of pixels behind every ROI mean — and therefore its
  noise level — identical across animals, so a group comparison is not confounded by
  how big each brain happened to sit in the frame. Set
  `RUN_CONFIG["lambda_scales_box_size"] = True` for the true similarity transform,
  i.e. boxes that cover a fixed *fraction* of cortex instead of a fixed area.
- **Scaling is reversible.** Every rescale is computed from a stored reference layout,
  never from its own last output, so `30 → 44 → 30` lands back on the exact same
  integers instead of drifting by a rounding step each way. Adjusting a box by hand
  re-anchors that reference (otherwise the next Lambda nudge would throw your
  adjustment away), which is also why `l` and `r` differ: `l` removes only the stretch,
  `r` discards everything back to how the page opened.

Get the constellation onto the right anatomy this way *first* — Bregma, then Lambda;
only then adjust individual boxes:

| | |
|---|---|
| drag inside a box | move it |
| drag a corner handle | resize it (all four corners have one) |
| arrows / `shift`+arrows | nudge the selected box 1 / 5 px |
| `+` / `-` | grow / shrink the selected box by 1 px on every side |
| `m` | toggle the **mirror lock** (on at startup) |

#### The mirror lock

With the lock on, every one-box edit — drag, corner-resize, arrow nudge, `+` / `-` —
rewrites that box's bilateral twin as its exact reflection about Bregma. Nudge `V1L`
and `V1R` follows; nudge `V1R` and `V1L` follows. Rows are copied across, columns are
negated, and the twin is written as a whole box rather than as a mirrored *delta*, so
a pair that was already crooked is repaired by the next edit to either half.

This exists because every ROI set drawn before it came out asymmetric. Across the 190
sets drawn in September 2026, `V1` was off in 174 files and `M2_alta` in 165, by up to
3 px, always on one hemisphere — and because `c` carries a layout forward, one slip
propagated through the rest of the session. The GUI cannot show you a 2 px asymmetry,
and the correlation matrix that comes out of it looks entirely normal.

The twin is **not** clamped to the frame. If mirroring pushes it off the image it
turns red and blocks the save, exactly as a box dragged off the edge does. Squashing
the twin to fit would be the one outcome worse than a refusal: a silently asymmetric
pair.

Press `m` to unlock when a one-sided layout is genuinely what you want. The status
line says which way the toggle went, and the startup banner prints the state.
`--no-mirror-lock` (or `mirror_lock: False` in `RUN_CONFIG`) starts unlocked.

Everything else:

| | |
|---|---|
| `n` / `p` | next / previous session |
| `c` | copy the **previous** page's layout (boxes, scale **and** Bregma) onto this one |
| `s` / `S` | save this session / save **every** session |
| `ctrl`+`s` | save only the sessions with unsaved changes (the `*unsaved*` ones) |
| `w` | write the **shared** ROI set (one file for all sessions) |
| `a` / `A` | apply this layout to all sessions (boxes **and** scale) — `A` also copies Bregma |
| `r` | reset this page to how it started (boxes, Bregma **and** scale) |
| `[` / `]` | display contrast |
| `h` / `q` | print help / quit |

The button bar above the image duplicates the gestures that are easiest to reach for
with the mouse: `<< Previous`, `Rotate left`, `Save`, `Resume`, `Rotate right`,
`Next >>`, and on a second row `Copy previous page's ROIs` (= `c`) and
`Save modified (N)` (= `ctrl`+`s`), whose label counts the pages still unsaved and
which greys out when there are none.

### Moving between pages, and the Resume button

Paging is not stateless — with hundreds of recordings, re-fitting the atlas from
scratch on every page would be the whole job done N times:

- **The first time you open a page, it takes the layout you are looking at now**
  (boxes and Lambda scale, not Bregma). An adjustment made once follows you forward
  through a run of similar recordings.
- **Unless that page already has its own `<key>.yaml`** — seeded at startup with
  `load_existing: True`, or written earlier in this run. Those boxes were drawn for
  *that* recording, so arriving on the page keeps them and says so in the status line.
- **A page you have already visited keeps whatever you left it at**, edited or not.
- `c` overrides all of the above on demand.

**`Resume`** reloads the current page from disk, in three steps:

1. Its own `<roi_set_dir>/<key>.yaml`, if it exists — discarding only the edits made
   since that save (`r`, by contrast, goes back to how the page opened this run; `l`
   removes only the Lambda stretch). The page then matches its file, so it stops
   counting as modified and `r` is re-pointed at this layout.
2. Its own `<roi_seed_dir>/<key>.yaml`, when a read-only seed directory is configured
   (see below). Still this recording's own geometry, so the page counts as saved —
   this is the "throw my changes away and go back to the reference layout" button.
3. Otherwise **the most recently saved ROI set in `roi_set_dir`**, whichever page it
   belongs to. Early in a run almost no page has a file of its own, and the layout you
   last committed beats starting from the raw atlas. Because that geometry (Bregma
   included) was drawn for a *different* recording, the page stays **unsaved** — it
   remains in the `Save modified` batch, and `r` still returns to its own starting
   layout. Check it against this anatomy and press `s`.

If `roi_set_dir` holds no ROI set at all, Resume says so and changes nothing.

### Reviewing a reference set without overwriting it (`roi_seed_dir`)

`roi_set_dir` is where saves go. `roi_seed_dir` is an optional **read-only** directory
of starting layouts — nothing is ever written to it. Set both to open a verified set,
page through it, adjust what you want, and have the adjustments land somewhere else:

```bash
"$CONDA" run --no-capture-output -n letizia python batch_roi_select.py \
    --roi-seed-dir roi_sets/rebuilt --roi-set-dir roi_sets/reviewed
```

Starting layout per page, in order: your own saved set in `roi_set_dir` (so re-running
resumes your work), then `<roi_seed_dir>/<key>.yaml`, then `--start-from`, then the
profile atlas. Both of the first two count as the page's *own* set, so page-to-page
carry-over will not silently overwrite them.

Because each seed file carries its own `bregma_row` / `bregma_col` /
`lambda_row_offset`, a page opens exactly as it was written and at scale **1.000×** —
the rescale knob starts neutral rather than stretching an already-scaled layout again.

`roi_seed_dir: None` restores the original single-directory behaviour.

`roi_sets/rebuilt` is the set written by [`rebuild_roi_sets.py`](../rebuild_roi_sets.py):
every recording regenerated as a pure Lambda rescale of the baseline atlas, bilaterally
symmetric by construction. Keeping it as a seed directory rather than an output is what
keeps that guarantee true.

Only Bregma is clamped to the frame, not the boxes it carries — so a 22-box layout
dragged near an edge *will* push boxes out. They turn **red**, the status line names
them and counts them, and the save is **refused**. Clamping the boxes instead would
mean silently squashing the layout to fit, which is the one thing offsets exist to
prevent.

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
downsample: 0.5             # profile factor used for each of the two downsamples
lambda_row_offset: 30       # the scale these offsets are at (FINAL-grid rows)
boxes:                      # ORDER IS THE COLUMN ORDER OF R — do not sort
  Laterale_L: {row_start: 21, row_end: 26, col_start: -34, col_end: -29}
  Verme_L:    {row_start: 22, row_end: 27, col_start: -12, col_end: -7}
  Laterale_R: {row_start: 21, row_end: 26, col_start: 29,  col_end: 34}
  Verme_R:    {row_start: 22, row_end: 27, col_start: 0,   col_end: 5}
```

- It is a **superset of the library's atlas file**: `wfci.load_atlas` reads it
  unchanged and ignores the extra top-level metadata keys. So the same file works anywhere
  an atlas is accepted (e.g. `run_pipeline.py --atlas`).
- **Bregma travels with the boxes** because the boxes are offsets *from* it. Using one
  animal's boxes with another's Bregma moves every ROI while still producing numbers.
  `load_roi_set` returns both; the run scripts apply both.
- `lambda_row_offset` is a **record, not an instruction**: the offsets in the file are
  already scaled to it, so a reader that ignores it still gets the right geometry. That
  is why it is *not* part of `load_roi_set`'s tuple — making the run scripts unpack a
  value they must not act on would invite applying the scale twice. Only the editor
  reads it (`load_lambda_offset`), to resume from the same reference. Unlike
  `bregma_row` it is in **final-grid rows**, the same units as the box offsets, not
  doubled.
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
| `roi_editor.py` | **new** — the editor, plus `load_roi_set` / `load_lambda_offset` / `save_roi_set` / `scale_boxes` / `preview_image`, importable without opening a window |
| `run_intermingle_rs.py` | `roi_set` config key + `--roi-set`; new `apply_roi_set()` helper; `roi_set` saved in the `.npz` |
| `run_botox_batch.py` | `roi_set_dir` / `roi_set` config keys + flags; `resolve_roi_set()`; `roi_set` in the `.npz` and the summary CSV |
| `src/wfci/` | **unchanged** — the editor is study policy, the library keeps the mechanism (P2) |

Verified: the editor's geometry conversion agrees box-for-box with
`wfci.roi.box_slices_for`; a saved file round-trips through `load_roi_set` **and**
`wfci.load_atlas`; an edited 2-ROI set runs end-to-end and yields a 2×2 `R_mean`;
rescaling round-trips to the identical integers and keeps the 11 mirrored left/right
pairs mirrored; the full test suite (207 tests) still passes.
