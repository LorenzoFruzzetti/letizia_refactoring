# `roi_sets/` — ROI geometry drawn per animal

Files written by [`roi_editor.py`](../roi_editor.py). One YAML per session key
(`<day>_<animal>.yaml`, e.g. `260611_R1.yaml`), plus optionally one
`shared_roi_set.yaml` used by every session.

Each file holds the ROI boxes **and** the Bregma they were drawn from — they are one
measurement and must not be separated. The layout is a superset of the library's atlas
file, so `wfci.load_atlas` reads it unchanged.

## `cortex22_roi_set.yaml` — the 22-box cortical layout (checked in)

The only file here not drawn by hand. It is [`wfci.atlases.CORTEX_22`](../src/wfci/atlases.py)
verbatim — the 22 `img_av(y_1+…, x_2+…)` boxes from the MATLAB cortical scripts
(M2/M1/BFD/Tr/FL/HL/RS/V1a/V1, all 11 left then all 11 right) — at the manifest's
default Bregma (row 121, col 134) on a 128×128 grid.

Both run scripts point at it out of the box (`RUN_CONFIG["roi_set"]`). It replaces
**only the atlas**: the profile stays `cerebellar_rs`, so trim, baseline, correlation
window and the no-mask/no-GSR chain are unchanged — you get a 22×22 `R` instead of
4×4. The full `cortical_gsr` pipeline (trim 0 + brain mask + GSR) is a separate
choice; switch the profile itself for that.

Its Bregma is a **default, not a measurement for your animal**. Open the editor,
check the boxes against the anatomy, and save per-animal files — those override it.

Use them:

```bash
CONDA="$USERPROFILE/miniconda3/condabin/conda.bat"
"$CONDA" run --no-capture-output -n letizia python roi_editor.py                 # draw / edit
"$CONDA" run -n letizia python run_intermingle_rs.py --roi-set roi_sets/cortex22_roi_set.yaml --full
"$CONDA" run -n letizia python run_intermingle_rs.py --roi-set roi_sets/260611_R1.yaml --full
"$CONDA" run -n letizia python run_botox_batch.py --roi-set-dir roi_sets --full  # per animal
"$CONDA" run -n letizia python run_botox_batch.py --roi-set roi_sets/shared_roi_set.yaml --full
```

Full walkthrough: [docs/ROI_EDITOR.md](../docs/ROI_EDITOR.md).
