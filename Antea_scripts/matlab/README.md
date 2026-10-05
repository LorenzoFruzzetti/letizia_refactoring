# Antea's MATLAB chain, reading the raw TIFFs directly (no Fiji)

Antea's scripts `(1)`–`(5)` in the folder above are plain MATLAB. Fiji was used for
two things before them, and both are replaced here. The ROI constellation can also
come from the ROI sets drawn with the Python ROI editor instead of hard-coded offsets.

| Step | Before | Here |
|---|---|---|
| Split each recording's interleaved single-frame TIFFs (`pv41_00001.tif`, … both channels alternating) into an emo and a gCaMP stack | Fiji | `load_interleaved_folder.m`, called by `script_1_correzione_emodinamica_da_tiff.m` |
| Brain mask for script (2) | drawn in Fiji, opened with `uiopen` | `script_2a_maschera_cervello.m`: draw it, read a file, or use the union of the ROI boxes |
| ROI boxes, scripts (3)+(4) | one `y_1`/`x_2` set by hand, fixed offsets | `script_4_correlazione_roi_set.m` + `read_roi_set.m`: each recording's own `roi_sets/rebuilt/<day>_<animal>_<t#>.yaml` |

## Workflow

1. `script_1_correzione_emodinamica_da_tiff.m`: set `animal_day_folder` (the folder
   holding `t1` … `t5`) and `recording_names`, run. Leaves `t_TEMP` (256 × 256 × frames ×
   recordings, dF/F in %) and `anatomy_image`.
2. `script_2a_maschera_cervello.m`: pick `mask_source` and run. Leaves `Mask` (256 × 256).
3. Antea's script (2) **without its first `uiopen(...)` line**. Leaves `t_TEMP_regressed`.
4. `script_4_correlazione_roi_set.m`: set `roi_set_dir`, `day`, `animal`,
   `recording_names`, run. Leaves the variables of script (4)
   (`R_regressed_ANIMAL_XDPL`, `mean_R_regressed_ANIMAL_XDPL`,
   `TEMP_regressed_ANIMAL_XDPL`) and a figure of the boxes. For the matrix without GSR,
   run it after `t_TEMP_regressed = t_TEMP_resized;`.
5. Script (5) for group means and difference figures, as before.

## Sessions without raw TIFFs: start from `pixel_data`

Only some sessions have their raw TIFFs on this machine. Every session has a Python pixel
dump, `pixel_data\<day>_<animal>\<t#>\pixels_dff_full.npy`. It holds the corrected dF/F
**after both 0.5 resizes**: Antea's `t_TEMP_resized` from the start of script (2), on the
128 × 128 grid. MATLAB reads it directly (`read_npy.m` decodes the float16 values), so no
conversion or extra disk space is needed.

1. `script_1b_da_pixel_data.m`: set `day`, `animal`, `recording_names`, run. Leaves
   `t_TEMP_resized` (128 × 128 × 2980 × recordings) and `anatomy_image` (the 256 × 256 mean
   reflectance stored with the dump). Loading takes about 1 s per recording.
2. `script_2a_maschera_cervello.m`, as in the raw route.
3. `script_2_gsr_da_128.m`: Antea's script (2) without the `uiopen` and without the resize
   of `t_TEMP`, since the data are already at 128. Her code is otherwise unchanged.
4. `script_4_correlazione_roi_set.m`, as in the raw route.

How this differs from the raw route:
- **Crop:** only the 76 × 87 crop around Bregma was saved, and pixels outside it are NaN. A
  `'roi_union'` mask lies inside the crop, so its GSR is unaffected. A hand-drawn outline
  that reaches outside the crop uses only its part inside.
- **Frames:** the first 20 frames of each channel are missing (2980 frames), as in the
  Python pipeline.
- **Precision:** float16 storage moves the correlations by about 1e-5.

`pixels_median_dff_20s_full.npy` (`volume_file`) gives the 20 s running-median baseline
instead of Antea's whole-recording mean.

## All PV days at once: `make_pv_figures.m`

Runs the whole chain in MATLAB for PV3, PV4 and PV5, every session, without clicking. For
each recording it loads the session from `pixel_data`, optionally applies Antea's GSR
(`antea_gsr.m`, her script (2) loop as a function) with the recording's ROI-union mask,
takes the box means of the recording's own ROI set, and computes `corr`. Each session's
matrix is `mean_R = mean(R,3)` over its recordings. It then draws, as script (5) does,
with `imagesc`, parula, a fixed -1..1 scale and her labels:

- `outputs/pv_matlab_figures/matrices_<tag>.png`: rows PV3, PV4, PV5 and their mean;
  columns day 1..5.
- `outputs/pv_matlab_figures/difference_<tag>.png`: the same rows; columns day 2..5 minus
  day 1.
- `outputs/pv_matlab_figures/mean_R_<tag>.mat`: the matrices.

`<tag>` is `pixels_dff_full_gsr` (Antea's chain) or `pixels_dff_full_no_gsr`. Run it with
`matlab -batch "run('E:\Developing_projects\letizia\Antea_scripts\matlab\make_pv_figures.m')"`.

## The ROI sets

`read_roi_set.m` reads one YAML file written by the Python tool. Box offsets use
Antea's convention exactly: a box is
`img(y_1+row_start : y_1+row_end, x_2+col_start : x_2+col_end, :)`, with
`y_1 = floor(bregma_row/2)` and `x_2 = floor(bregma_col/2)`. The baseline
`roi_sets/cortex22_roi_set.yaml` holds her offsets unchanged (e.g. BFDR 17..22 / 37..42).
The rebuilt sets are the same constellation, rescaled per animal by its Bregma–Lambda
distance and placed at that recording's own Bregma.

## Mask choice changes the GSR result

- `'draw'` (or `'file'`): a brain outline, as Antea did.
- `'roi_union'`: only the 22 boxes, the same region our Python `_gsr` connectivity
  variants use. Antea's script (2) computes the global signal as `nanmean(nanmean(data,1),2)`:
  the mean of each column's mean, so a column holding 6 mask pixels counts as much as one
  holding 30. Since 2026-10-05 Python does the same by default (`gsr_global_mean="column"`
  in `roi_pixel_connectivity.py`, `GSRConfig(global_mean="column")` in `wfci.gsr`), so the
  `_gsr` folders match MATLAB with this mask. The earlier plain pixel mean is kept as
  `--gsr-global-mean pixel` (`_gsr_pixelmean` folders). On 260520/PV4/t1 the two differ by
  up to 0.13 in r.

## Settings that matter

- `channel_order = 'auto'` takes the dimmer group (brightest 10 % of pixels in the first
  frame) as GCaMP. Reflectance is several times brighter on this rig. Check the printed
  line for each recording: a swap would invert the correction.
- `n_trim = 0` keeps every frame, as script (1) does. The Python pipeline drops the first
  20 frames per channel; set `n_trim = 20` to match it exactly.
- Memory: each frame is downsampled as it is read, so a recording needs about 5 GB while
  loading instead of 25 GB. `t_TEMP` itself is about 1.6 GB per 3000-frame recording.

## Validation (2026-10-05, MATLAB R2024a, 260520/PV4/t1, `n_trim = 20`)

- Loading the 6000 files took 41 s. Both channel mean images equal the Python
  pipeline's (`pixel_data/260520_PV4/t1/pixels_meta_full.npz`) exactly.
- After script (2)'s 0.5 resize, the dF/F (crop 2980 × 76 × 87) matches
  `pixels_dff_full.npy` to within float16 storage rounding (max 0.008 percentage points).
- Full chain (script 1 → 2a with `'roi_union'` → Antea's script (2) → script 4):
  - Without GSR, the 22 × 22 r equals `outputs/roi_pixel_connectivity/pixels_dff_full/260520_PV4/t1`
    `R_roi` to 2.3e-6.
  - With GSR, it differs from `pixels_dff_full_gsr` by up to 0.13. The column-mean global
    signal above accounts for all of it: recomputed that way in Python, the difference
    falls to 1.6e-5.
- Antea's script (2) took 6 s on the 792 masked pixels.
- `pixel_data` route (script 1b → 2a `'roi_union'` → script_2_gsr_da_128 → 4) on the same
  recording reproduces the raw-TIFF route to 2.3e-6 without GSR and 1.6e-5 with GSR (float16
  rounding). The mean reflectance read from the `.npz` equals the raw route's exactly.
