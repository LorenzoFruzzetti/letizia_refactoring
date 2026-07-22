# `roi_sets/` — ROI geometry drawn per animal

Files written by [`roi_editor.py`](../roi_editor.py). One YAML per session key
(`<day>_<animal>.yaml`, e.g. `260611_R1.yaml`), plus optionally one
`shared_roi_set.yaml` used by every session.

Each file holds the ROI boxes **and** the Bregma they were drawn from — they are one
measurement and must not be separated. The layout is a superset of the library's atlas
file, so `wfci.load_atlas` reads it unchanged.

Use them:

```bash
CONDA="$USERPROFILE/miniconda3/condabin/conda.bat"
"$CONDA" run --no-capture-output -n letizia python roi_editor.py                 # draw / edit
"$CONDA" run -n letizia python run_intermingle_rs.py --roi-set roi_sets/260611_R1.yaml --full
"$CONDA" run -n letizia python run_botox_batch.py --roi-set-dir roi_sets --full  # per animal
"$CONDA" run -n letizia python run_botox_batch.py --roi-set roi_sets/shared_roi_set.yaml --full
```

Full walkthrough: [docs/ROI_EDITOR.md](../docs/ROI_EDITOR.md).
