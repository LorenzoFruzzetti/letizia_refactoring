

## 1.0 README Requirements
When creating or updating documentation, ensure README.md contains:
1. Entrypoints:
   - Exact commands for primary script/module execution from editor and terminal.
2. Expected input:
   - Required/optional formats, schema notes, and input paths.
3. Expected output:
   - Generated files/directories and runtime side effects.
4. Directory map:
   - Clear map of source code, configs, environment files, tests, scripts, and outputs.
5. Examples:
   - Pointers to runnable examples and sample data locations.

README quality rules:
- Commands should be copy/paste ready.
- Paths should be explicit and consistent with target project root.
- Update README whenever refactoring changes entrypoints, I/O contracts, or structure.
- Keep `REFERENCE.md` as the canonical technical map of the repository and reference it directly from `README.md` and from this agent file.
- Companion docs, kept in sync when the API or entrypoints change: `LIBRARY.md` (the single self-contained doc for using/editing the `wfci` package — full API, array-shape conventions, the do-not-break invariants I1–I11, the library/experiment boundary P2) and `GUIDE.md` (a human-facing tour of every script and how to run it). When editing the package, read `LIBRARY.md` first.

## 2) Operating Contract
- Work inside the target project folder unless compatibility work is explicitly required.
- Preserve current behavior unless the user requests behavior changes.
- Prefer small, verifiable edits over broad rewrites.
- Ask for confirmation before risky deletes, irreversible moves, or major archive operations.

## 3) Required Inputs to Collect First
1. Target project folder path.
2. Primary entrypoint script(s) or module(s).
3. Preferred Conda environment name (or derive from folder name).
4. Whether environment creation/maintenance should be skipped.
5. Keep/delete/archive constraints.

## 4) Project-Specific Reusable Layer

### 4.1 Proposed Reusable Project Structure
Use this as a baseline and adapt to project conventions:

```text
<target_project>/
|-- README.md
|-- .env/
|   |-- .envVariables
|   |-- ENVIRONMENT_SETUP.md
|   |-- environment.yml
|   `-- requirements.txt
|-- src/
|   |-- <classes>               <-- classes with a single responsibility, e.g. DataLoader, ModelTrainer, etc. each in its own file.
|   |-- <utils>                 <-- utility functions and helpers, e.g. data processing, evaluation metrics, etc.  
|   `-- <package_or_modules>/
|-- tests/
|-- examples/
|   |-- README.md
|   `-- data/
|-- temp_image/
|-- temporary_files/
`-- debugging_scripts/
```

### 4.2 Environment Management Rules
- This project's Conda environment is **`letizia`** (Python 3.11).
  - Location on disk: `C:\Users\utente\anaconda3\envs\letizia`
  - Interpreter: `C:\Users\utente\anaconda3\envs\letizia\python.exe`
  - Spec / setup docs: `.env/environment.yml`, `.env/requirements.txt`, `.env/ENVIRONMENT_SETUP.md`
- By default, create and maintain a Conda environment unless the user explicitly opts out.
- Use non-interactive commands.
- Do not depend on conda activate in automated flows.
- Preferred command style:
  - conda run -n <env_name> <command>
- On this machine `conda` is not on PATH directly. Use the full path to conda.bat:
  - PowerShell: `& "$env:USERPROFILE\anaconda3\condabin\conda.bat" run -n <env_name> <command>`
  - Git Bash: `"$USERPROFILE/anaconda3/condabin/conda.bat" run -n <env_name> <command>`

#### Adding a new package to the `letizia` env
1. Install it (prefer conda-forge; fall back to pip):
   - `& "$env:USERPROFILE\anaconda3\condabin\conda.bat" install -n letizia -y <package>`
   - or `& "$env:USERPROFILE\anaconda3\condabin\conda.bat" run -n letizia pip install <package>`
2. Record it in BOTH spec files so the env stays reproducible:
   - add `- <package>>=<min-version>` to `.env/environment.yml`
   - add `<package>>=<min-version>` to `.env/requirements.txt`
3. Verify the import: `conda run -n letizia python -c "import <package>; print(<package>.__version__)"`
- Keep these files aligned when environment management is enabled:
  - .env/environment.yml
  - .env/requirements.txt
  - .env/.envVariables
- When VS Code debugging is expected to work, also keep these files aligned with the environment contract:
  - .vscode/settings.json
  - .vscode/launch.json
- The debugger must use the same Python interpreter as the maintained Conda environment.
- Prefer an explicit interpreter path or an explicitly selected workspace interpreter instead of relying on auto-detection.
- Debug configurations must use the modern debugger type:
  - "type": "debugpy"
- Debug configurations should set these fields explicitly when project behavior depends on paths or environment variables:
  - "program" or "module"
  - "cwd"
  - "envFile"
  - "console"
- If the project has editor-run entrypoints, create named launch configurations for the main scripts instead of relying only on "${file}".
- For interactive UI applications such as tkinter-based tools, prefer terminal-based debugging, and use "externalTerminal" when integrated terminal behavior is unreliable on Windows.
- If debugging the current open file is part of the workflow, add a launch configuration with:
  - "purpose": ["debug-in-terminal"]
- Minimum debugger validation should confirm all of the following:
  - VS Code resolves the intended Conda interpreter for the workspace
  - launch.json parses correctly
  - the primary script starts under the pinned interpreter
  - imports required by the main entrypoints work under the debugger environment
- Minimum validation commands:
  - conda run -n <env_name> python --version
  - conda run -n <env_name> pip list
- Recommended debugger validation steps:
  - Run the main entrypoint with the environment interpreter directly
  - Run at least one VS Code debug configuration for a primary entrypoint
  - Confirm breakpoints bind in user code and stop as expected
- If conda run fails due to unavailable/misconfigured Conda, stop and ask user confirmation before proceeding with alternatives.
- If the debugger fails while terminal execution succeeds, first verify interpreter selection, launch.json settings, cwd, envFile, and console choice before changing application code.

### 4.3 README Requirements (Mandatory)
When creating or updating documentation, ensure README.md contains:
1. Entrypoints:
   - Exact commands for primary script/module execution from editor and terminal.
2. Expected input:
   - Required/optional formats, schema notes, and input paths.
3. Expected output:
   - Generated files/directories and runtime side effects.
4. Directory map:
   - Clear map of source code, configs, environment files, tests, scripts, and outputs.
5. Examples:
   - Pointers to runnable examples and sample data locations.

README quality rules:
- Commands should be copy/paste ready.
- Paths should be explicit and consistent with target project root.
- Update README whenever refactoring changes entrypoints, I/O contracts, or structure.


### 5.2 Instructions (Mandatory)
- When generating code do not use try catch blocks make all the error surface unless the user explicitly ask for error handling, in that case make sure to log the error in a way that is easy to understand and debug.
- When generating code make sure to add comments that explain the purpose of the code and any non-obvious decisions, this will help the user understand the code and make it easier for them to modify it in the future if needed.
- Make sure that the code can be run from the editor, to add argparse use a configuration file like the following example:

```python
# Edit this section to run the pipeline without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "input_path": "path/to/input", # | None
    "output_path": "path/to/output", # | None
}

# then use something like the following to parse the arguments

def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    prefer_cli_args = bool(config.get("prefer_cli_args", True))

    if prefer_cli_args and len(sys.argv) > 1:
        return parse_args(defaults=config)

    return argparse.Namespace(
        input_path=config["input_path"],
        output_path=config["output_path"],
    )
```

- If both MATLAB and Python pipelines exist, keep only the Python pipeline unless user says otherwise.
- Move unnecessary images to temp_image/.
- Move other non-output transient files to temporary_files/.
- Move code without clear product purpose to debugging_scripts/.
- Ensure all required directories exist.
- Add a minimal runnable example for each major project part in examples/, and document each example input/output in README.


## 6. Test scripts

Test scripts are small, self-contained analysis scripts that load one dataset,
compute something from it, and save plots. They are read and edited by hand much
more often than they are reused, so favour clarity over abstraction.

### Structure

Flat, top-to-bottom procedural code, in this order:

1. **Imports** — standard library first, then third-party.
2. **Parameters** — every value someone might want to change, in one block
   directly after the imports.
3. **Loading** — read the input data.
4. **Computation** — one commented block per step.
5. **Plotting and saving** — build the figure, save it, close it.

No `main()`, no `argparse`, no classes. Only add a helper function when the same
logic is needed in three or more places or for ease of understanding with a descriptive name.

### Parameters

- All parameters live at the top. Nothing configurable appears further down the
  file — if a magic number shows up in the computation section, move it up.
- Plain module-level variables, `lower_snake_case`, units in the name
  (`sampling_rate_hz`, `window_duration_s`) or in a trailing comment.
- Prefer physical quantities and derive the index-space values from them, so the
  relationship is explicit:

```python
  sampling_rate_hz = 10
  window_duration_s = 1.0
  n_dt = int(window_duration_s * sampling_rate_hz)  # rows per window
```

- Any parameter whose meaning isn't obvious from its name gets a one-line
  comment, e.g. `starting_column = 2  # first 2 columns (trial, frame) are metadata`.
- Paths are `pathlib.Path`, never strings. The output directory is derived from
  the input path and created with `mkdir(parents=True, exist_ok=True)`.

### Body

- Each computation block opens with a short comment saying what it produces in
  plain language, not a restatement of the code.
- Annotate the shape of every non-trivial array where it is created, using the
  parameter names: `# windows: (n_columns, n_dt, n_windows)`.
- Spell variable names out (`window_center_frame`, not `wcf`). Single letters
  only for loop indices, and prefer `i_row` / `i_col` over `i` / `j`.
- Keep intermediate results in named variables rather than chaining long
  expressions.

### Plots

- One figure per script section; use `constrained_layout=True` and
  `squeeze=False` when subplotting.
- Row and column labels come from the data (column names, feature names), not
  hardcoded strings.
- Save with `fig.savefig(output_dir / "<descriptive_name>.png", dpi=200)` and
  always follow with `plt.close(fig)`.
- Titles state the parameters that produced the result, e.g.
  `f"Window features ({window_duration_s:g} s window)"`.

## 7. Update
When an issuse related to this project is found update with other points the
CLAUDE.md file

## 9. Known Issues Fixed

### 9.1 `scale_boxes` breaks bilateral mirror symmetry (roi_editor.py:403-438)
**Problem.** The docstring claims round-half-to-even keeps a mirrored L/R pair mirrored
"because it is symmetric about zero". Symmetry about zero is not the property required.
A box of span 5 places its edge at `centre*factor - 2.5`, so the mirror needs
`round(u + 2.5) == round(u - 2.5) + 5` — i.e. *translation invariance*. Half-to-even
fails that whenever `u = |centre| * factor` lands on an integer: `round(5.5) == 6` but
`round(0.5) == 0`, a gap of 6, leaving the pair 1 px off-centre.

**Effect.** Silent 1-px left/right asymmetry from a pure Lambda rescale. In `roi_sets/`
it hit HL in 294/356 files and M2_alta in 230/356, always +1 px.

**Fix applied.** `rebuild_roi_sets.py` uses `round_half_away()` (ties away from zero).
`roi_editor.scale_boxes` itself is unchanged — fixing it in place would move boxes in
already-saved sets. Regenerate through `rebuild_roi_sets.py` instead.

**Correction (2026-09-10, see 9.20).** The claim above that ties-away "satisfies both
`f(-x) == -f(x)` and `f(x+n) == f(x)+n`, so it is mirror-exact" is FALSE, and so is
the reasoning that led to it. No rounding rule has both properties: oddness forces
`f(-0.5) == -f(0.5)`, translation forces `f(-0.5) == f(0.5) - 1`, so `f(0.5)` would
have to be `0.5`. Ties-away only moves the failure to pairs straddling zero
(`round_half_away(-0.5) == -1` but `round_half_away(4.5) - 5 == 0`). It happens to be
exact on every real scale factor in `roi_sets/`, which is why the per-file assertion
never fired, but it is not exact in general. `scale_boxes` now reflects explicitly
instead. Every one of the 546 rebuilt sets is byte-identical either way.

### 9.2 Baseline atlas declares `lambda_row_offset: 30` but is used at 55
**Problem.** `roi_sets/cortex22_roi_set.yaml` declares `lambda_row_offset: 30`, but
`_seed_layout` (roi_editor.py:539) passes `args.lambda_offset` when a page is seeded
from `profile.atlas` and never reads the atlas file's own value. With
`RUN_CONFIG["lambda_offset"] = 55` in `batch_roi_select.py`, every set in `roi_sets/`
was built treating CORTEX_22 as if drawn at 55 rows.

**Effect.** The two declarations disagree by 1.83×. Measured across the 356 sets: at
reference 55 a pure rescale reproduces the drawn boxes to a median 1.5 px (2 files
exactly); at 30 it is off by 23 px. Anyone trusting the `30` would generate ROIs ~1.9×
larger than every existing result.

**Fix applied.** `rebuild_roi_sets.py` takes the reference as an explicit parameter
(`RUN_CONFIG["reference_lambda"] = 55`) and documents why. The `30` in the baseline
file is left as-is because `wfci.load_atlas` ignores the key and changing it would not
alter any pipeline behaviour — but it must not be treated as the scale reference.

### 9.3 Hand-nudged V1 accumulated across sessions
**Problem.** `c` in the ROI editor copies the previous page's layout onto the current
one, so a one-hemisphere nudge propagates into every later session. V1's mirror error
drifts monotonically with acquisition date: −2 px at 260616 through +6 px at 260805.

**Effect.** Any left-vs-right V1 contrast on the original `roi_sets/` files is
confounded, and because the drift tracks date it is confounded with experimental group
if groups were run in date blocks.

**Fix applied.** `roi_sets/rebuilt/` regenerates all 356 sets as a pure rescale of the
baseline, discarding every per-box nudge. Verified: 0 violations of bilateral row
alignment, mirror symmetry, box size, AP/ML ordering, disjointness, midline sidedness;
all boxes inside the 128×128 grid; t1–t5 identical within every animal.

### 9.4 `run_botox_batch.resolve_roi_set` could not see `roi_sets/rebuilt/`
**Problem.** `roi_sets/rebuilt/` is named per RECORDING (`<day>_<animal>_<t#>.yaml`,
355 files matching the manifest exactly), but `resolve_roi_set` only looked for
`<roi_set_dir>\<day>_<animal>.yaml`. Pointing `--roi-set-dir` at `roi_sets/rebuilt`
therefore matched nothing, printed the "no ROI set" notice, and silently fell back to
the shared `cortex22_roi_set.yaml` — the wrong geometry AND the wrong Bregma (the
manifest's 121 instead of each file's own 108), with no error.

**Effect.** A full batch would have run with baseline cortex22 boxes while appearing
to honour the rebuilt sets; `roi_set` in the summary/npz would have named cortex22, so
it was detectable after the fact but not before.

**Fix applied.** `resolve_roi_set` now takes the recording name and tries
`<day>_<animal>_<t#>.yaml` → `<day>_<animal>.yaml` → shared `roi_set`. Geometry is
resolved per recording (new `_geometry_for`), not once per unit, so each `t#` gets the
boxes/Bregma drawn for it; `bregma_row`/`bregma_col`/`roi_set` moved from the per-unit
summary into the per-analysis rows accordingly.

### 9.5 Per-ROI fluorescence was only reachable through the `.npz`
**Problem.** `temp_roi` (`[time, n_rois, trial]`, the ΔF/F averaged inside each ROI box
— the intermediate every correlation is computed from) was written only inside the
`.npz`, so inspecting it meant loading numpy and knowing the array/label layout.

**Fix applied.** `_save_roi_fluorescence` also writes `roi_fluorescence_{debug,full}.csv`
next to each `.npz`: header `trial,frame,<22 ROI labels>`, one row per frame, trial-major.
The `.npz` still carries the array — the CSV is an addition, not a replacement.

### 9.6 `batch_summary.csv` was written only after the whole batch finished
**Problem.** `main()` collected every summary row in a list and wrote the CSV at the
end. A full manifest run is ~97 s per recording × 355 recordings ≈ 9.5 h; a crash or
interrupt at any point discarded the index of everything already computed (the per-
recording `.npz`/`.csv` survived on disk, but nothing listed them).

**Fix applied.** `main()` opens the CSV up front and flushes after every unit, so an
interrupted run leaves a valid partial index. Progress is printed as `[i/N]`.

### 9.7 The manifest batch was serial and took about 9.5 hours
**Problem.** The 355 independent per-recording analyses ran one after another even
though they share no writable state and each has its own input and output folder.

**Fix applied.** `run_botox_batch.py` flattens the manifest into independent jobs and
runs them with `ProcessPoolExecutor`. The default worker count is 75% of logical cores
(15 on this 20-core machine), configurable with `--workers N` or `RUN_CONFIG`; 1 keeps
the serial path for debugging. Worker output is buffered per job so logs do not
interleave. Verified on Windows with a two-worker/two-recording debug smoke test run
through the required `conda run -n letizia` environment.

### 9.8 Resume could skip results missing from `batch_summary.csv`
**Problem.** The stopped first run left three complete `.npz` files but an empty
summary. A simple `.npz`-based skip would preserve the computation while permanently
omitting those recordings from the batch index.

**Fix applied.** On resume, existing summary rows are preserved and de-duplicated by
day/animal/mode/recording. Any skipped `.npz` without a row is reconstructed from its
saved metadata and `R_mean` before new jobs start; `elapsed_s` is blank because older
result files did not store timing. The three existing PV5 rows were recovered.

### 9.9 `bake_all_images` skipped or ignored cached previews
**Problem.** The cache loop checked `session.image is not None` before checking the
alignment TIFF path. An in-memory preview with no TIFF was never written, while an
existing TIFF was ignored whenever the session already held an image.

**Fix applied.** `bake_all_images` now loads an existing alignment TIFF first. When
the TIFF is absent it reuses an in-memory preview or decodes one via `_ensure_image`,
then writes the float32 cache. Both cache tests pass, as does the full 213-test suite.

### 9.10 Ctrl+C printed a traceback but left the process pool running
**Problem.** The full batch buffers each worker's log until a recording completes,
so the first minute could look frozen. More importantly, `ProcessPoolExecutor` was
used as a context manager: after `KeyboardInterrupt`, its `__exit__` waited for the
submitted queue instead of stopping the 15 workers. The traceback appeared even
though the parent and workers remained alive.

**Fix applied.** The parent now prints a heartbeat every 15 seconds while waiting.
Pool shutdown is explicit: Ctrl+C cancels pending futures, terminates and joins active
workers, performs non-waiting executor shutdown, and exits with status 130 without a
traceback. Worker failures use the same cleanup before their original traceback is
raised. Completed `.npz` markers remain resumable; partial outputs have no marker and
are recomputed. A regression test exercises the interruption cleanup.
The full suite passes: 214 tests.

### 9.11 Spawned full workers crashed in the final correlation step
**Problem.** Full jobs completed both streaming passes, then one spawned process
terminated abruptly. The parent saw `BrokenProcessPool` plus a feeder-thread
`OSError: handle is closed`. Windows Error Reporting identified native exception
`0xc06d007f` in `KERNELBASE.dll`, with 17.3 GB RAM still free. This is a delayed DLL
load failure. The same signature had previously been captured by faulthandler at
`numpy.corrcoef` -> `wfci.roi._corrcoef_matlab`.

**Fix applied.** `_corrcoef_matlab` now computes Pearson correlation with explicit
float64 centering, elementwise products, sums, and norms. The ROI matrix is only
2980 x 22, so this remains small and avoids BLAS dispatch without adding a streaming
pass. Constant-column and NaN propagation match `np.corrcoef`/MATLAB semantics.

**Validation.** A two-process full run completed two 3000-frame recordings in 203.5 s
through the exact path that crashed. `temp_roi` and averaged traces were byte-identical
to the previous serial files; `R`/`R_mean` differed only by reduction-order roundoff
(maximum 5.6e-15). Numeric parity, streaming/GSR checks, and the full 217-test suite
pass. The heartbeat also now reports two running slots for two workers instead of the
executor's misleading three "running" futures (one was only staged in its feeder).

### 9.12 Sustained Windows process pools still fail; reliable default is serial
**Problem.** Although the two-worker full smoke test passed, the resumed study later
lost spawned workers again with the same native `0xc06d007f` exception. It completed
34 more recordings first (37 total), proving the problem is intermittent process/DLL
state rather than a deterministic recording or the final correlation alone. A second
parallel attempt failed as well. `BrokenProcessPool` cannot expose a Python traceback
because the child is terminated by the native runtime.

**Fix applied.** `RUN_CONFIG["workers"]` is 1 and serial execution is now the explicit
supported default. This path calls `execute_job` directly in the main process: no
`ProcessPoolExecutor`, spawn, multiprocessing feeder queue, or worker-side DLL state.
Logs are live rather than buffered, status says "serially in the main process", and
Ctrl+C exits cleanly with completed files resumable. Values above 1 remain available
but are labeled experimental/not recommended on this Windows machine.

**Validation.** Regression tests prove the editor default resolves to one worker and
that workers=1 never constructs a process pool. A real previously unprocessed full
recording (`260616/T7/t3`, 3000 frames/channel) completed through the serial path in
22.6 s, using its rebuilt ROI set/Bregma and updating the real summary. The real run
now contains 38 completed results.

### 9.13 The `letizia` env's editable install points at a DIFFERENT checkout
**Problem.** `site-packages\__editable__.wfci-0.1.0.pth` contains
`C:\Users\loren\Documents\letizia\src`, not `H:\Developing_projects\letizia\src`.
So `conda run -n letizia python ...` from this repo imports `wfci` from the
Documents checkout, whose HEAD is `f7e98bc` — a different history that does NOT
contain this repo's `src/wfci/` changes.

**Effect.** Edits to `src/wfci/` here have no effect on anything run through the
documented `conda run -n letizia` command, and neither do the tests that import
`wfci`. Concretely, the §9.11 fix — `_corrcoef_matlab` rewritten to avoid BLAS
dispatch, the fix for the native `0xc06d007f` worker crash — exists only in
`H:\...\src\wfci\roi.py`. The Documents copy still calls `np.corrcoef`. Every run
that reported the crash recurring (§9.12) was executing the UNFIXED function, so
§9.12's conclusion that "the fix did not hold and process pools are unusable" was
drawn from a library that never had the fix loaded. Whether pools actually work
with the real fix is untested.

**Workaround in use.** Prefix the interpreter with an explicit path:
`PYTHONPATH=H:\Developing_projects\letizia\src conda run -n letizia python ...`
(PYTHONPATH precedes `.pth` entries on `sys.path`). Verified: `wfci.__file__` then
resolves to this checkout, and the full suite passes (236 tests).

**Proper fix (NOT applied — it changes a shared env and the other checkout's
imports, so ask first).** Re-point the editable install:
`conda run -n letizia pip install -e H:\Developing_projects\letizia`
Then re-test §9.12's parallel path before trusting either conclusion about workers.

### 9.14 The manifest's `F:` paths were stale; the reflectance channel clips
**Problem.** `manifests\botox_restani_manifest.csv` had `recording_paths` under
`F:\WF_2026\starting\...`, but F: is not mounted; the dataset is now at
`D:\WF_2026\starting\...` (same layout, 6000 TIFFs per `t#`). Any batch run
against the manifest as previously committed failed to find its inputs.

**Fix applied (2026-09-04).** `scan_botox_dataset.py` RUN_CONFIG `root` and
`batch_roi_select.py` RUN_CONFIG `folder` now read `D:\WF_2026\starting`, and the
manifest was REGENERATED against D: with frame counting on. Checked field by
field against the previous file: 71 rows, identical in every column once `F:`
is replaced by `D:`, and `total_frames` is 30000 on every row -- so the D: copy
is complete, not partial. Every row's Bregma was still the 121/134 default, so
regenerating discarded no hand edits. The unreferenced root-level
`botox_restani_manifest.csv` still holds the original `\\146.48.88.209` UNC
paths and was deliberately left alone.

**Separately, observed while validating `--save-data`:** in `260611/PV5/t1` the
`emo` (reflectance) baseline image `mean_r` reaches exactly **65535** — the uint16
full-scale value — so that channel is saturating in some pixels. `mean_r` is a
divisor in the correction, so clipped pixels understate the hemodynamic term
there. Not investigated further; flagged because the dumps now make it visible.

### 9.15 `--save-data`: what was measured, and why the defaults are what they are
Settled from the real data rather than assumed, while adding the flag:
- **float16 is safe for dF/F, not for raw F.** Round-trip error on dF/F is
  <=0.004 percentage points against a signal sd of 0.66-0.97 pp. But `F_emo` was
  measured at 63194 against float16's 65504 ceiling (3.6% headroom), so raw F is
  stored float32 — which is *exactly* lossless for it, the twice-box-downsampled
  values being k/16. `PixelDump` range-checks before every cast and raises rather
  than writing `inf`.
- **There is no black background to crop away.** Otsu reports 59-63% "background"
  on the mean image, but rendering it shows the threshold cutting into real cortex
  (on `260618/PV3` it discards the whole dark upper-right quadrant, which is head),
  and the threshold moves per recording. The useful crop is geometric instead: the
  union of every ROI box across all 356 files in `roi_sets\rebuilt`, expressed
  RELATIVE TO BREGMA, is `rows -29..+47, cols -44..+43` = 76x87 = 40.4% of the
  128x128 frame, and it fits inside the grid for 356/356 recordings. Being
  Bregma-relative it names the same anatomy in every animal, so dumps stack across
  recordings — which a full-frame crop does not, the frames being aligned only to
  the camera. `tests/test_run_botox_batch.py` re-checks all 356 sets against the
  configured window, so editing those sets cannot silently outgrow it.
- **dF/F must be dumped, not derived.** The pipeline corrects at HALF resolution
  and downsamples the result; a ratio recomputed from the dumped (already
  downsampled) F volumes is close but not equal, because a box mean does not
  commute with a ratio. `tests/test_dump.py` asserts the discrepancy so nobody
  removes the `dff` volume as redundant.
- **Cost.** ~197 MB per recording (~70 GB for the 355-recording manifest) against
  ~1.9 MB for everything else a job writes. Time cost is nil: measured 300.9 s
  with the dump vs 301.0 s without, same recording, and `mean_offdiag_R` identical.

### 9.16 float16 for the pixel dumps: free for dF/F, lossy for raw F
Measured on 500 real frames of `260611/PV5/t1` at the final 128x128 grid, carried
all the way through the ROI reduction to the correlation matrix (the only place
the answer matters). Reference is the float64 pipeline.

**dF/F as float16 — effectively free.**

| stage | float16 error |
|---|---|
| pixel | 7.8e-3 pp (0.63% of one sd, sd = 1.24 pp) |
| `temp_roi` | 6.0e-4 |
| `R_mean` | **4.1e-6** |

Mean off-diagonal R: 0.737263 (float64) vs 0.737262 (float16). The connectivity
result does not move. This is why `save_data_dff_dtype` defaults to float16.

**Raw F as float16 — lossy where it counts.** The relative error looks tiny
(~0.03%), which is misleading: F is a large DC pedestal carrying a small AC
modulation, and float16 spends its precision on the pedestal.

| | F_gcamp | F_emo |
|---|---|---|
| max abs error | 8 counts | 16 counts |
| typical temporal sd of a pixel | 58 counts | 147 counts |
| **quantisation as % of the actual signal** | **13.8%** | **10.9%** |
| float32 max abs error | **0 (exact)** | **0 (exact)** |

float32 is exactly lossless because the twice-box-downsampled values are k/16 with
k well under 2^24. So float16 buys 2x on the F volumes and costs ~an eighth of the
temporal signal; float32 costs nothing. Hence `save_data_f_dtype` = float32.

**The four concrete hazards of float16 (all verified, not theoretical):**
1. **Precision is relative; the signal is absolute.** Step size is 0.001 near 1,
   but 8 near 10000 and 32 near 60000. A pixel at F=20000 modulating by +-58
   counts gets quantised into ~7 levels.
2. **The ceiling is below the camera's.** float16 max finite is 65504;
   `np.float16(65535)` is `inf`. The camera saturates at uint16 65535, and
   `mean_r` was measured at exactly 65535 on `260611/PV5/t1`, so a saturated pixel
   is literally unrepresentable. Measured peak dumped `F_emo` is 63194
   (`260616/T7`) -- 3.6% headroom. `PixelDump._check_castable` raises rather than
   writing `inf`, so this fails loudly, but it does fail.
3. **Reductions overflow.** `np.sum` of a 2980-frame float16 F trace returns
   `inf` (true value 5.96e7, float16 max 6.5e4). `np.mean` happens to survive via
   a float32 accumulator, but nothing guarantees that. **Always
   `.astype(np.float64)` before reducing a dump.** For dF/F (values ~1) this is
   not a practical risk; for raw F it is immediate.
4. **No speed benefit.** numpy has no float16 compute kernels; operations upcast.
   It is a storage format only.

**Conclusion.** The shipped defaults (dF/F float16, F float32, 197 MB/recording,
70 GB batch) are the right trade. Going all-float16 to reach 42 GB is viable ONLY
if the F volumes are never used for anything but a sanity check -- and if a
recording ever exceeds the ceiling the run stops rather than corrupting silently.
Both are now selectable per run: `--save-data-dff-dtype` / `--save-data-f-dtype`.

### 9.17 Calling `envs\letizia\python.exe` directly crashes numpy's BLAS (`0xc06d007f`)
**Problem.** On this machine (`utente`, `C:\Users\utente\anaconda3`) invoking the
environment interpreter by absolute path — without activation — kills the process
with the native Windows exception `0xc06d007f` on the very first BLAS call. It is
not project code: a bare

```python
import numpy as np; a = np.random.rand(22, 300); a @ a.T
```

is enough. faulthandler points at `numpy.lib._function_base_impl.cov` when the
call arrives via `np.corrcoef`, which is exactly the signature recorded in §9.11.

**Cause.** conda-forge's numpy loads its BLAS DLLs (MKL) from
`envs\letizia\Library\bin`, which is put on `PATH` by *activation*. Running
`python.exe` straight from the env directory never activates it, so the delay-load
fails and Windows raises `0xc06d007f` (missing dependent DLL) instead of a Python
`ImportError`.

**Rule.** Always go through `conda run` — this is what CLAUDE.md §4.2 already
mandates, and the reason is now concrete:

```bash
"$USERPROFILE/anaconda3/condabin/conda.bat" run --no-capture-output -n letizia python -m pytest -q
```

Verified: `python.exe blas_check.py` dies at `a @ a.T`; the identical script under
`conda run` prints `dot ok` / `corrcoef ok` and exits 0. The full suite run through
`conda run` is 236 passed, 3 skipped (plus one pre-existing failure, below).

**Relation to §9.11/§9.12.** Same native exception and same numpy frame, so any
future recurrence should first check how the interpreter was launched before
concluding the pipeline is at fault. This does NOT by itself prove the worker-pool
crashes were the same thing — a spawned child inherits the parent's activated
`PATH` — so §9.12's serial default stands until someone re-tests pools under a
known-activated environment.

**Use `--no-capture-output`.** Plain `conda run` buffers child stdout and, in this
Git Bash setup, frequently emitted nothing at all; every "the command produced no
output" symptom during setup was this.

### 9.18 `>=` in a package spec is eaten by `cmd.exe` when calling `conda.bat` from Git Bash
**Problem.** `conda.bat` is a batch file, so `cmd.exe` re-parses the argument list
*after* Bash has stripped the quotes. `>` is then a redirection operator, and

```bash
"$CONDA" install -n letizia -c conda-forge -y "scipy>=1.10" "pyqtgraph>=0.13"
```

silently writes conda's entire stdout into repo-root files literally named `1.10`
and `0.13`, while conda receives only the bare package names. Observed for real:
four junk files (`0.13`, `1.10`, `6.5`, `7.4` — one per pinned spec) appeared in
the project root, and the log file the command was redirected to came out 0 bytes.

**Effect.** The version pin is lost, and the install *looks* like it produced no
output. It also masks failures: the first `pyqtgraph` install appeared to succeed
with an empty log.

**Fix.** Quote the spec so `cmd` cannot see the operator, or avoid the pin:

```bash
"$CONDA" install -n letizia -c conda-forge -y 'scipy>=1.10'   # still re-parsed
"$CONDA" install -n letizia -c conda-forge -y "scipy>=1.10"   # still re-parsed
"$CONDA" install -n letizia -c conda-forge -y scipy pyqtgraph # safe
```

Only the unpinned form is reliably safe from Git Bash; use PowerShell (`& "$env:USERPROFILE\anaconda3\condabin\conda.bat" install ... "scipy>=1.10"`)
or `.env/environment.yml` when a pin actually matters. After any such install,
check `git status` for stray numeric files in the repo root.

### 9.19 Rebuilding this machine for the dumped-pixel batch (2026-09-04)
Everything needed to re-run `run_botox_batch.py` here with the per-pixel dumps.

**Dataset.** Now on `D:\WF_2026\starting` (see 9.14). `scan_botox_dataset.py`,
`batch_roi_select.py` and the regenerated manifest all point there.

**Env.** There was NO `letizia` env on this machine; it had to be built from
scratch. `conda env create -f .env/environment.yml` failed FOUR times, always
while unpacking one of the two big packages (`mkl` 109 MB, `qt6-main` 85 MB):
`[Errno 36] Resource deadlock avoided`, once `[WinError 2]` on a `.conda`
tarball that had just been downloaded. `CONDA_ALWAYS_COPY`, single-threaded
extract, and `conda clean` did not help, and the pkgs dir is writable, so it is
not permissions.

**What worked** -- build it in stages instead of one transaction:

```powershell
$C = "C:\Users\utente\anaconda3\Library\bin\conda.bat"
& $C create  -n letizia -c conda-forge python=3.11 -y
& $C install -n letizia -c conda-forge -y "numpy>=1.24" "scipy>=1.10" "pandas>=2.0" `
    "tifffile>=2023.7.10" "matplotlib>=3.7" "pyyaml>=6.0" "pytest>=7.4" pip
& $C install -n letizia -c conda-forge -y "pyqtgraph>=0.13" "pyside6>=6.5"
& $C run -n letizia pip install -e .        # from THIS repo root -- see 9.13
```

Each stage is a small transaction, so a stall does not lose the whole env. Note
the staged install also lets the headless deps land first: only `roi_editor.py`
needs Qt, so a failing `pyside6` would not block the batch.

**Verified after the rebuild.** `wfci.__file__` resolves to
`C:\Users\utente\Documents\letizia\letizia_refactoring\src\wfci\__init__.py`,
so 9.13's wrong-checkout trap is NOT present here. Suite: 238 passed, 3 skipped
through `conda run` (9.17's rule still applies -- the bare interpreter path is
fine for stdlib scripts but dies on the first BLAS call).

**Pixel dumps are now ON by default.** The batch reports the real cost up
front: **64.8 GB for 353 jobs** (the other 2 were already done), against 200 GB
free on C:. `RUN_CONFIG` has `save_data = True` and
`save_data_root = "pixel_data"`, i.e. inside this repo, gitignored along with
`batch_run.log`. `--no-save-data` turns them off; a regression test now covers
that off-switch, because with the config default ON it is the thing that can
silently cost 197 MB a recording.

**The dump was checked against the analysis, not just for existence.** Reducing
the dumped float16 dF/F over each ROI box (`y_1 + row_start - 1 : y_1 + row_end`,
minus the crop origin in `meta['region']`) reproduces `roi_fluorescence_full.csv`
-- the traces the correlations were actually computed from -- to a worst
per-frame disagreement of **9.5e-4 pp across all 22 ROIs, 0.045% of one signal
sd**. That is the float16 storage cost 9.16 predicts and nothing else. Measured
on `260611/PV5/t1`: shapes `(2980, 76, 87)`, dtypes float16/float32/float32,
`f_emo` peak 58607 (11% under float16's 65504 ceiling -- still why raw F is
float32).

**Do not pipe the batch through `Tee-Object`.** PowerShell buffers the whole
pipeline, so a 10-hour run produces a 0-byte log and no visible progress. Use
cmd's OS-level redirection with unbuffered Python:

```powershell
& cmd /c '"%USERPROFILE%\anaconda3\Library\bin\conda.bat" run --no-capture-output -n letizia python -u run_botox_batch.py > batch_run.log 2>&1'
```

Interrupting is safe either way: the resume marker is the analysis `.npz`, and a
killed recording leaves an EMPTY dump folder (verified), not a truncated volume.

**The run completed: 355/355 recordings, 611.6 min (10.2 h), no errors.**
`batch_summary.csv` has 355 unique day/animal/recording rows, all three groups
(P/R/T), and `mean_offdiag_R` on every one. `pixel_data` is 65.5 GB in 1420
files -- exactly 4 per recording -- leaving 138 GB free on C:.

Audited afterwards, not just counted:
- 355/355 have all four dump files AND an analysis `.npz`.
- Every volume is `(2980, 76, 87)` float16, one uniform window.
- Every dump's `meta['bregma_row']/['bregma_col']` equals its OWN
  `roi_sets/rebuilt/<day>_<animal>_<t#>.yaml`. **53 distinct Bregmas** appear
  across the 355 -- if 9.4's shared-fallback bug had come back there would be
  exactly 1, and nothing in the summary would have said so.
- On 8 recordings sampled across days, animals and groups, reducing the dumped
  pixels over the ROI boxes reproduces each one's `roi_fluorescence_full.csv`
  to a worst 1.13e-3 pp (0.069% of one sd) -- the float16 cost, nothing more.

### 9.20 The new cohort repeated 9.1 and 9.3 exactly; both are now fixed at source
**What happened (2026-09-10).** 190 ROI sets for 38 new animal-sessions (dates
260807-260909) were drawn through `batch_roi_select.py` into `roi_sets/reviewed/`.
They came out with the same two defects as the original 356, in the same regions:

| region | files off | error |
|---|---|---|
| V1 | 174/190 | +1, +2, +3 px |
| M2_alta | 165/190 | +1, +2 px |
| M2_bassa, HL, RS_alta | 10/190 each | +1 px |

The 1 px entries are 9.1's rounding tie. The 2-3 px entries are 9.3's one-hemisphere
hand nudge, identical across all five `t#` of a session because `c` carries a layout
forward. Row alignment was clean everywhere. Six files also disagreed with their own
session's Bregma; `260828_PV7_t1` still held the RUN_CONFIG fallback 120/134 while its
four siblings sat at 86/132, i.e. it was saved before Bregma was ever placed.

**Rebuilt.** `rebuild_roi_sets.py --roi-set-dir roi_sets/reviewed --out-dir
roi_sets/rebuilt` wrote all 190 alongside the existing 356 (no key collides; the old
files were checksummed before and after and are untouched). `roi_sets/rebuilt` now
holds 546 sets, scale 0.782x-0.964x for the new cohort, 589 px of asymmetry removed,
and all 546 pass the mirror, row-alignment, box-size, ordering and grid checks with
t1-t5 identical within every animal. `run_botox_batch.py` picks the new ones up with
no flag change.

**Fixed at source, so the next cohort does not need this.**
1. `roi_editor.scale_boxes` no longer relies on a rounding rule to carry the mirror.
   A box left of Bregma is computed as the negated reflection of the same box on the
   right, so a pair is symmetric by construction at any factor and any span. See the
   correction in 9.1 for why no rounding rule can do this. `round_half_away` moved into
   `roi_editor.py` and `rebuild_roi_sets.scale_boxes_mirrored` now just forwards, so
   there is one implementation. Verified: all 546 rebuilt sets regenerate byte-identical.
2. **Mirror lock** (`ROIEditor._mirror_to_twin`, `RUN_CONFIG["mirror_lock"] = True`,
   `m` toggles, `--no-mirror-lock` opts out). Every one-box edit — mouse drag, corner
   resize, arrow nudge, `+`/`-` — rewrites the bilateral twin as the exact reflection.
   It writes the mirrored BOX, not a mirrored delta, so one edit also repairs a pair
   that was already crooked. The twin is not clamped: if mirroring pushes it off the
   frame it turns red and blocks the save, the same visible refusal an off-frame drag
   gets, because silently squashing the twin is the one outcome worse than a refusal.

`mirror_twin` reads the side letter as the first `L`/`R` whose swap names another box
in the same atlas, so `RSL_alta`'s leading `R` is not mistaken for the side.

**Not done, offered and declined for now:** a save-time refusal on a broken mirror,
and save-time warnings for a still-default Bregma or geometry disagreeing with an
already-saved sibling `t#`. Those would have caught the six header slips above, which
right now only `rebuild_roi_sets.py`'s majority normalisation repairs.

**Watch out.** `rebuild_roi_sets.py` RUN_CONFIG still defaults to `roi_set_dir:
roi_sets` and `out_dir: roi_sets/rebuilt`. Run it with no flags and it rebuilds the
ORIGINAL hand-drawn cohort, not whatever `batch_roi_select.py` last wrote. Pass
`--roi-set-dir` explicitly.

Tests: 248 passed, 3 skipped. New coverage in `tests/test_roi_editor.py` for the
impossibility of the rounding rule, mirror-exactness of `scale_boxes` across factors,
mirror symmetry over the editor's whole Lambda range, `mirror_twin`'s side-letter
rule, and the lock on the nudge, resize and mouse-drag paths plus its repair
behaviour and its `m` toggle.

### 9.21 `--all-rois` silently replaced four atlas boxes with non-mirrored ones
**Problem.** `epileptic_by_area_animal_day_pixels.py` built its ROI set as
`boxes = dict(atlas["boxes"]); boxes.update(CUSTOM_BOXES)`. The four custom boxes
are named `M2R_alta`, `M2R_bassa`, `M1R_alta`, `M1R_bassa` -- the SAME names the
cortex22 atlas uses -- so with `--all-rois` they did not add anything, they
overwrote the atlas's own versions. The custom offsets differ from the atlas ones
(`M2R_alta` -23..-18 / 8..13 vs the atlas's -25..-20 / 9..14), and they are fixed
Bregma-relative constants rather than reflections of `M2L_alta`, so 4 of the 22
ROIs came out NOT mirror-symmetric with their left twins. Exactly the class of
defect 9.1/9.3/9.20 are about, reintroduced downstream of the fixed editor.

**Effect.** Any left-vs-right contrast on an `--all-rois` run was confounded for
M1R/M2R, with nothing in the output saying so -- the column names are the atlas's.

**Fix applied (2026-09-16).** The two box sources are now independent:
`--all-rois` is the atlas as drawn, `--custom-boxes` is the four hand boxes, and
`use_custom_boxes` defaults the latter to ON only when `--all-rois` is OFF (the
old single-purpose behaviour). `resolve_boxes` raises if both are off. The
provenance JSON records `custom_boxes: null` when they were not used.

### 9.22 Median-baseline dF/F for the pixel detector, and what it costs
**Added (2026-09-16).** `--signal-mode median_dff` (now the default) computes
`(F/Fbar)/(R/Rbar) - 1` per saved pixel and then the spatial ROI mean, where
Fbar/Rbar are each pixel's CENTRED RUNNING MEDIAN over
`--baseline-median-window-s` (20 s = 201 frames at 10 Hz), not the
whole-recording mean `wfci.correction.hemodynamic_correction` uses. The old
`F * mean_t(R) / R` is `--signal-mode reflectance_ratio`, unchanged and verified
bit-identical. Result is a RATIO, not a percent; x100 gives run_botox_batch's %.
The scale is irrelevant to detection -- every threshold downstream is in robust
SDs of the trace itself.

**Verified numerically, not just by eye.** Against a brute-force per-pixel median
(`np.median` over each window, partial windows at the ends) on four real ROIs of
`260611/PV5/t1`: max abs error **9.99e-17**. `pandas.rolling(center=True,
min_periods=1).median()` is the implementation, matching `remove_slow_trend`'s
edge convention. Cost is ~1.8 s per recording for 22 boxes x 2 channels, i.e.
not the bottleneck.

**THE SIGNAL IS NOW HIGH-PASSED TWICE.** The per-pixel 20 s median baseline
removes slow drift BEFORE the ROI average; the detector then subtracts its own
20 s running median (`--median-window-s`) AFTER it. The two act on different
quantities so it is not literally redundant, but anyone comparing amplitudes with
a `reflectance_ratio` run must know it. Raise `--median-window-s` or switch modes
to get one stage only.

**Output tree is named for the mode** (`outputs/epileptic_by_area_animal_day_
pixels_median_dff/`). The per-recording file names are shared between modes, so a
single tree would have had one mode overwrite the other's tables and figures. The
trace CSV is also named per mode (`roi_median_dff.csv` vs
`roi_reflectance_corrected_fluorescence.csv`), which is what lets
`plot_pixel_detection_comparisons.py` and `plot_pixel_peak_zooms.py` keep reading
the older `outputs/epileptic_by_area_animal_day_pixels/` untouched.

**Per-recording figures are now drawn for EVERY recording** (`debug_plot_count:
None` / `--debug-plot-count all`), and there is a new one:
`epileptic_diagnostics.plot_roi_overview` -> `roi_traces_all.png`, all 22 ROIs
overlaid over the whole recording, the counterpart of run_botox_batch's
`roi_traces_full.png`. Cost 6.3 MB per recording, 3.4 GB for the cohort.

**Full run (545 recordings, 109 day+animal units, all 22 atlas ROIs).** 18.5 min
detection + ~16 min figures. Cut-off `z >= 3.37` over 35,610,300 pooled frames
(achieved 1.004%); 18,633 epileptiform events of 220,188 peaks (110,055
confirmed). Audited afterwards, not just counted: 545/545 have all six expected
files; one single ROI column set (22) across both cohorts; every
`roi_geometry.json` names its OWN `roi_sets/rebuilt/<day>_<animal>_<t#>.yaml`
with **72 distinct absolute Bregmas**, so 9.4's shared-fallback bug is not back.

**Watch out.** `bregma_crop_zero_based` is IDENTICAL on all 545 and that is
correct, not a fallback symptom: the saved crop window is itself Bregma-relative
(9.15), so Bregma sits at the same place inside every crop by construction. Check
`y_1`/`x_2` (absolute, on the full grid) when auditing for 9.4.

### 9.23 Median dF/F cache: one reflectance glitch frame, and the active-pixel scale trap
**Cache built (2026-09-16).** `cache_median_dff.py` wrote
`pixel_data/*/*/pixels_median_dff_20s_full.npy` for all 545 dumps: 20.1 min on 8
workers, 21.48 GB, all `(2980, 76, 87)` float16, 545 meta markers, no `.partial`
leftovers. Averaging the cached volume over each of the 22 boxes reproduces
`median_dff_roi_trace` with difference 0. Parallel scaling measured here: 1 worker
107 min, 4 -> 29, 8 -> 21, 12 -> 17, 20 -> 13, ~1.1 GB RAM per worker. The pool is
`multiprocessing.Pool` (context exit terminates workers, so 9.10's trap does not
apply) and the job does no BLAS work, so 9.11/9.12 did not recur.

**Glitch frame.** `260828_PV7/t2` frame 1809: the reflectance channel drops
field-wide from ~45,500 to ~3,900 (about the GCaMP level) for ONE frame, so
dF/F there reaches 10.8 and the field median is 1.4. It is the only such frame in
all 545 recordings (next-worst recording's field median max 0.08 in the 99th
percentile). It is also why the cache's worst float16 error is 1.95e-3 instead of
the typical 6e-5 (float16 error scales with the value). The cache is NOT
corrected: any detector run on this recording sees a fake whole-cortex event at
180.9 s. `epileptic_by_active_pixels.py` therefore excludes the whole recording by
default (`exclude_recordings` / `--exclude-recordings`); the ROI-mean scripts
(`epileptic_by_area_animal_day*.py`) still include it.

**Active-pixel scale trap.** `epileptic_by_active_pixels.py` counts pixels above a
per-pixel robust-z threshold. Active pixels are spatially correlated, so at high
thresholds the fraction is exactly 0 in most frames, its MAD is 0 and
`run_analysis` has no scale: at 3 SD the run stops with "No finite, non-flat ROI
frames available for calibration". Measured on 8 recordings of the median dF/F:
3 SD -> 0 in 77-92% of frames, robust SD 0; 2.5 SD -> robust SD 0.001-0.004,
below one pixel of ~396 (0.0025); 2 SD -> 0.008-0.020; 1.5 SD -> 0.034-0.058.
On `pixels_dff_full.npy` (mean baseline) the usable range is lower still (<= 1 SD).
The default is therefore 1.5 SD.

**Bottom-percentile scale (added 2026-09-17).** `--pixel-scale bottom_percentile`
uses the plain SD of each pixel's values at or below its
`--pixel-scale-percentile` (default 50). That is a truncated-distribution SD:
measured 0.58-0.61x the MAD SD on 8 recordings (Gaussian: 0.60x), so thresholds
are NOT comparable between the two scales -- 2 bottom-50% SDs ~ 1.2 robust SDs.
Same trap as above, shifted: at 2.0 the fraction's robust SD is 0.007-0.075, at
2.5 some recordings already give 0. Its outputs go to a `_bottom<N>sd` folder.

**Watch out.** `median_dff_volume` divides in place; it must take `np.array`
copies, not `np.asarray` — with float64 input `asarray` returns the caller's own
array and the division overwrote it (caught by `tests/test_active_pixels.py`;
the float32 dumps always forced a copy, so the built cache is unaffected).

### 9.24 The pooled cut-off silently saturated at the top of `z_histogram_range`
**Problem (2026-09-17).** `run_analysis` clips every frame z into the histogram
range, so frames above the top edge pile into the last bin. When more than
`calibration_frame_percent` of frames sit there, the "top 1%" cut-off comes out
as the range's edge instead. The active-pixel fraction has a tiny robust SD, so
its z values are huge: at a 2 SD pixel threshold, 16 recordings gave
`cut-off z >= 29.99 ... achieved 4.036%` -- four times too many events flagged,
with only the achieved-percent in the log to show it. The true cut-off is 63.46.

**Fix applied.** `run_analysis` raises "widen z_histogram_range" when the cut-off
falls in the last bin. `epileptic_by_active_pixels.py` uses (-10, 1000). The
earlier full runs were not affected (ROI means 3.37, active pixels at 1.5 SD 13.14).

**Also added with it.** `run_analysis(workers=N)` runs detection and figures on a
`multiprocessing.Pool` (module-level `detect_recording` / `write_recording_outputs`;
the trace loader must be picklable). Tables are identical to the serial run
(`tests/test_run_analysis_workers.py`, and 16 real recordings: 66 s -> 15 s on 8).
`epileptic_min_signal` gates epileptiform peaks on the RAW loaded value at the
peak frame (`signal_at_peak`, a new column in every `all_peaks.csv`); the
active-pixel script sets it to 0.20. On the 16 recordings at the true cut-off it
removed 2 of 76 events: past the cut-off, most peaks already have >20% active.

### 9.25 September cohort had no group on this machine; the full pixel run could not start
**Problem.** `pixel_data/` holds 109 units (545 recordings), but
`outputs/botox_restani_rebuilt/batch_summary.csv` lists only the 71 units of
`botox_restani_manifest.csv`. The 38 units of `botox_restani_2026_09_manifest.csv`
were batched on another machine and only their pixel dumps were copied here.
`run_analysis` takes groups from `<input_root>/batch_summary.csv` and raises
`no group in batch_summary.csv for: [...]` before detection. Under
`conda run -n letizia` (without `--no-capture-output`) that traceback was not
shown, so the run looked like it exited silently after printing `Output:`,
leaving an empty output folder.

**Fix applied.** `build_epileptic_groups.py` writes
`outputs/epileptic_groups/batch_summary.csv` (`day,animal,group`, 109 units:
P 18, R 33, T 58) from both manifests, raising on a unit with two groups, a group
that disagrees with the connectivity summary, or a pixel unit with no group.
Run the pixel detectors with `--input-root outputs/epileptic_groups`; the
connectivity summary is untouched. Use `--no-capture-output` (or the env's
`python.exe` directly) to see errors.

### 9.26 This machine's conda and checkout paths differ from §4.2/§9.19 (2026-09-21)
**Observed on the `loren` machine.** §4.2 and §9.19 document
`C:\Users\utente\anaconda3\...\conda.bat`; here there is no `utente` profile and no
Anaconda. Conda is **Miniconda**:

- `C:\Users\loren\miniconda3\condabin\conda.bat`
- env `letizia` at `C:\Users\loren\miniconda3\envs\letizia`

```powershell
& "$env:USERPROFILE\miniconda3\condabin\conda.bat" run --no-capture-output -n letizia python -u <script>.py
```

**Git Bash produced NO output at all** from `"$USERPROFILE/anaconda3/condabin/conda.bat"
run --no-capture-output ...` (the path does not exist there, and the miniconda path
also returned nothing). Use the PowerShell form above; it works.

**§9.13's wrong-checkout trap IS present here.** Under `conda run -n letizia`,
`wfci.__file__` resolves to `C:\Users\loren\Documents\letizia\src\wfci\__init__.py`,
NOT `H:\Developing_projects\letizia\src`. Anything importing `wfci` from this repo
needs `PYTHONPATH=H:\Developing_projects\letizia\src` (or the editable install
re-pointed, which changes the other checkout — ask first). The root-level epileptic
scripts import each other by module name, so they resolve to this repo and are not
affected; the `wfci` package and its tests are.

### 9.27 `260828_PV7/t2`: the glitch frame also carries a persistent GCaMP step
**Found (2026-09-22) with `plot_hemisphere_traces.py`'s GCaMP-only figure.** 9.23
describes frame 1809 as a one-frame reflectance glitch. It is more than that. The
field-mean raw counts:

| | frames 1700-1799 | frames 1820-1919 | step |
|---|---|---|---|
| `F_gcamp` | 8429 | 8103 | **-3.9 %, persistent** |
| `F_emo` | 20812 | 20695 | -0.6 % |

Reflectance drops to ~8000 for frame 1809 alone and recovers by 1811; GCaMP drops by
~330 counts at 1810 and stays there for the rest of the recording (illumination or
acquisition change, not hemodynamics -- the emo channel does not follow). Effects:
a mean-baseline dF/F (`pixels_dff_full.npy`, run_botox_batch's traces) has a -4 %
offset for the last 40% of the recording; running-median baselines absorb it
after a transient of one window (20 s / 60 s). The emo correction cannot remove it.
Excluding the recording (9.23's default in `epileptic_by_active_pixels.py`) is still
the right call.

### 9.28 Editing a CRLF doc from Python writes `\r\r\n`, and the obvious check misses it
**Problem (2026-09-23).** The repo's `.md` files are CRLF. A helper that inserts a
line the natural way --

```python
text = path.read_text(encoding="utf-8")          # universal newlines: CRLF -> \n
...                                              # insert the line
path.write_text(text.replace("\n", "\r\n"))       # text mode translates AGAIN
```

writes `\r\r\n` on every line: the explicit `\r\n` still contains a `\n`, and Windows
text mode turns that one into `\r\n` as well.

**Why it survived a check.** Counting `b"\r\n"` and bare `b"\n"` reports *CRLF 948,
bare LF 0* on such a file -- exactly what a clean CRLF file reports, because every
`\r\r\n` contains one `\r\n` and no bare LF. That is the check that pronounced
README.md "clean, nothing was actually damaged" after an earlier edit. It was not.
Count `b"\r\r\n"`, or `b"\r"` minus `b"\r\n"`, instead.

**Second edit doubles the file.** Reading `\r\r\n` back with universal newlines
yields `\n\n`, i.e. a blank line after every line, so the next edit turns 948 lines
into 1899 and writes them out doubled again. That is visible damage, unlike the
first pass. Caught here on README.md (`git diff --stat`: 1899 insertions, 878
deletions on a two-line edit) and repaired: the artifact blank after each line is
removed, the endings rewritten as plain CRLF, leaving a pure-insertion diff.
GUIDE.md and REFERENCE.md had only the first-pass form and were rewritten.

**Rule.** Write repo files as bytes (`path.write_bytes(text.replace("\n", "\r\n")
.encode("utf-8"))`) or open with `newline=""`, and verify with `git diff --stat`:
any deletion count on an insert-only edit means the line endings moved.

### 9.29 pytest cannot create its default temp/cache folders on the `loren` machine
**Problem (2026-09-27).** Every test that takes `tmp_path` errored with
`PermissionError: [WinError 5] Accesso negato: C:\Users\loren\AppData\Local\Temp\pytest-of-loren`,
and `.pytest_cache` in the repo is not writable either. These are environment
failures, not test failures: 12 errors, all in tests using `tmp_path`.

**Workaround.** Point pytest at a writable base temp and skip the cache:

```powershell
$env:PYTHONPATH = "E:\Developing_projects\letizia\src;E:\Developing_projects\letizia"
& "$env:USERPROFILE\miniconda3\condabin\conda.bat" run --no-capture-output -n letizia python -m pytest -q -p no:cacheprovider --basetemp <writable dir> tests
```

With that, the full suite is 286 passed (including the new
`tests/test_pixel_roi_editor.py`). `PYTHONPATH` is still needed for 9.26's
wrong-checkout trap.
### 9.30 statsmodels `MixedLM` with a nested session component: the default optimizer fails silently
**Found (2026-09-27) in `test_drift_vs_design.py`.** The model is a random intercept
per animal plus `vc_formula={"session": "0 + C(session)"}`. With the default
`fit()` (bfgs) the full model stopped at log-likelihood -69.80 while lbfgs, powell
and nm all agree on -61.80; the only symptom was a `ConvergenceWarning` and a
likelihood-ratio statistic of **-1.02**, i.e. the reduced model "fitting better"
than the full one, which is impossible at a real optimum. lbfgs alone then failed on
a different model (the GCaMP-drift null model), so no single optimizer is enough.

**Fix applied.** `fit_nested_model` fits with `method=["lbfgs", "powell", "nm"]`
(statsmodels moves to the next on failure) and raises if `converged` is still
False. Any LR statistic below 0 means a fit did not converge: do not report it.

**Also:** `statsmodels` was not in the env; it is now (`statsmodels>=0.14`, installed
0.15.0 from conda-forge) and recorded in `.env/environment.yml` and
`.env/requirements.txt`.

**Measured on the 544 recordings (for whoever looks at the drift next).** The
mean-baseline dF/F falls in 98% of recordings, about -0.55 % dF/F per minute
(sd 0.28), with or without the emo correction. Variance of that slope: animal 7%,
session within animal 12%, recording within session 81% (GCaMP-only: 2 / 19 / 80).
So the decline is universal, and what varies around it is mostly
recording-to-recording, NOT a property of the session. Group and mouse_line have no
effect in the mixed model. GCaMP-only drift grows with `day_index` (-0.023 %/min
per session, p = 0.001) and depends on recording order (t1 declines least, t2 most,
then recovering toward t5; LR p < 1e-5). The fast-fluctuation SD does not shrink
(median log2 ratio 0.00), so it is a baseline decline, not less activity; PV-group
sessions are the exception, their fluctuations shrink slightly (see correction below).

**Correction (2026-09-27): the fluctuation-ratio LR tests are unreliable.** For
`fluct_log2_ratio` the animal variance is ~0 (on the boundary), and full and reduced
models land on different optima even when both report `converged`: one LR came out
-11.2, others gave p = 0.005 for a coefficient of 0.014 +- 0.017. The script's
`group p = 0.0004` for this metric is one of those. Refitted two ways that do not
depend on that optimum -- random intercept per SESSION only (REML, Wald) and OLS
with standard errors clustered by animal -- they agree: PV-group shrinkage is
about -0.06 log2 (~4 %) vs R/T, joint group Wald p = 0.020 / 0.026. **mouse_line
has no effect** on the fluctuation ratio: +0.014 +- 0.016, p = 0.39 / 0.24 adjusted
for group; within R p = 0.89 / 0.75, within T p = 0.20 / 0.13. Among PV-CRE animals
only, PV group vs R/T is -0.054, p = 0.011 / 0.043, so the PV shrinkage belongs to
the group, not the line. For this metric, trust Wald/cluster-robust tests over LR.
Six recordings rise instead (`260708_PV5` and `260716_PV5`, t1-t3, about +1 %/min):
their GCaMP-only drift is about 0, so the rise is the reflectance channel falling
and the emo correction dividing it in.

### 9.31 Interpreting baseline decline and cortex-wide event burden
**Analysis caveats identified (2026-09-28); proposal, not a detector fix.**
Section 9.30's stable fast-fluctuation SD is measured after local 20 s dF/F
normalization. It supports separating baseline and transient activity, but does
not establish stable raw event amplitudes or prove photobleaching as the cause.
Local division can increase apparent late amplitudes when only the baseline
falls; compare additive and multiplicative corrections without flattening genuine
event-rate changes.

`analyze_peak_tails_emo.py` selects the top 5% of peaks separately per recording
and defines `Cortex_mean` as the mean of ROI means. These are descriptive tail
and sampled-region summaries, not a common abnormal-event threshold or a
whole-brain measurement. The proposal in
[docs/CORTEX_ACTIVITY_EMO_PROPOSAL.md](docs/CORTEX_ACTIVITY_EMO_PROPOSAL.md)
specifies shared event criteria, spatial recruitment, exposure-aware time-bin
comparisons and animal-level inference. No existing pipeline behavior changes.

### 9.32 Cortex/emo additive activity implementation (2026-09-28)

`analyze_cortex_activity_emo.py`, `compare_cortex_activity_conditions.py` and the
flat pilot `test_cortex_activity_emo.py` implement the agreed additive algorithm.
Per pixel: q=F/emo; b=centered median; residual=q-b; corrected=residual+mean_valid(b).
No reference-interval normalization, GSR or second detrend. Event metrics use the
residual; mean_b is display-only. Masks use each recording's saved cortical ROI
union, not the rectangular crop. Defaults are exploratory native thresholds with
sensitivity variants, not validated epileptiform labels. Whole-animal comparisons
use valid exposure and require shared mouse-line support; tiny pilots skip GEE.
The inspected dump metadata lacks channel timing, so 10 Hz/index alignment is an
explicit assumption. The known 260828_PV7/t2 exclusion remains.
Read [usage](docs/CORTEX_ACTIVITY_EMO_USAGE.md) for commands, output schemas,
cache fingerprints, artifact annotations and validation scope. No package API
changes. The environment on this workstation is under USERPROFILE/miniconda3.

### 9.33 `pixel_roi_editor.py` showed the first page as one giant pixel (2026-10-01)
**Problem.** `_show_page` called `image_item.setRect(QRectF(col_0, row_0, w, h))`
before `_refresh_background` assigned the image. pyqtgraph 0.14's `setRect` scales
by `rect / current image size`, and with no image yet it uses size 1. So each crop
pixel was stretched to the whole crop (87x76 -> 7569x5776 plot units), and the view
showed a single pixel. Display only: traces, seed-r maps and saved boxes were correct.

**Fix applied.** The crop is 1:1 with the full 128 grid, so the image is placed with
`resetTransform()` + `setPos(col_0, row_0)`, which does not depend on whether an image
is set. Verified headlessly: the item's bounds are now `(col_0, row_0, 87, 76)`.
Rule: never call `ImageItem.setRect` before `setImage`.

### 9.33 NBS connectivity had neither GSR nor drift removal; corrected variants added (2026-10-01)
**Problem.** `R_roi` in `outputs/roi_pixel_connectivity/pixels_dff_full/` (the input of
the three `test_nbs_*.py` scripts) is a plain Pearson r of box means of the
mean-baseline dF/F. No GSR was ever applied anywhere in the batch chain (all 109
manifest rows use `cerebellar_rs`; `cortical_gsr` is unused; the streaming pixel dump
refuses GSR). So every edge carries the cortex-wide component, and the -0.55 %/min
mean-baseline drift (9.30) adds a shared slow trend that inflates every r.

**Added.** `roi_pixel_connectivity.py --gsr`: the global signal is the per-frame mean of
the recording's ROI-box union (stand-in for Antea's brain mask), regressed out of every
pixel by closed-form OLS (`regress_global_signal`, einsum, no BLAS), folder suffix
`_gsr`. The NBS scripts read `NBS_CONNECTIVITY_VARIANT` (default `pixels_dff_full`,
original outputs unchanged) and write to `outputs/nbs_<name>_<variant>/`. Built:
`pixels_dff_full_gsr`, `pixels_median_dff_20s_full`, `pixels_median_dff_20s_full_gsr`
(544 recordings each, ~10.5 min on 8 workers); all NBS reruns ~4 min total.

**What survived (t > 2.5 unless noted; edge overlap between variants in brackets).**
- `PV_slope` increase: 50 edges originally, 9 (mean+GSR) / 12 (median+GSR); 8-10 of
  those are in the original. Core: V1L/V1aL with M2L_bassa, M1L_bassa, RSL_alta, and
  V1aL/TrL with M2R. The big original component was mostly global signal.
- `T` decrease over `day_index`: 16 orig, 44 median, 7 median+GSR; the GSR remnant is
  RS-centred (RSL/RSR_alta with BFDL, TrL, TrR).
- pooled decrease over `day_index`, only after GSR: M1/M2-V1 edges, 9 of 10 shared by
  mean+GSR and median+GSR.
- `PV_vs_R` increase, only after GSR (M1L_bassa/M2 with HLR/TrR, 8 shared): p = 0.036 is
  the smallest attainable (28 relabellings, 2 R animals) -- fragile.
- pre/post Boto: nothing originally; GSR variants give 2-6 edge components that do not
  agree with each other. Treat as null.
- `T` increase and `R_minus_T` increase appear in median+GSR only.

**Caveats.** Analyses x directions x thresholds x variants are not corrected against
each other; trust components that recur across variants. After GSR an edge change is
relative to the global mean, and negative r are produced by construction.

### 9.34 New session 260608 PV5: one download was unsynchronised, the other is clean (2026-10-02)
**Two downloads, only one usable.** `C:\Users\loren\Downloads\PV5\PV5\t1-t5` (6000
TIFFs each) is NOT two interleaved channels: odd and even frames have equal medians,
and frame brightness follows a ~4-frame wave that drifts against the camera clock
(~1 frame per 1200), so every frame position sweeps through both illuminations. LED
controller and camera were not synchronised. Running the pipeline on it would split
two streams that each swing between both lights. Ignored.
`C:\Users\loren\Downloads\PV5B\PV5B\t1-t5` is a normal recording: alternating frames at
~6000 (GCaMP) and ~15200 (reflectance), stable over the whole recording; t3 starts on
reflectance, t5 on GCaMP (`channel_order: auto` handles it). It is not one of the four
known PV5 sessions (closest is 260611: image r = 0.92, but fast fluctuations r < 0.1).
The user dated it **260608**.

**Set up.** A directory junction `C:\Users\loren\Downloads\WF_2026_new\260608\PV5` ->
`PV5B\PV5B` gives the `<day>\<animal>\<t#>` layout `roi_editor.folder_key` needs (keys
are the last 3 path parts; the raw path would give `PV5B_PV5B_t1`). The files were not
moved. `batch_roi_select.py` RUN_CONFIG `folder` points at `WF_2026_new` and
`start_from` at `roi_sets/rebuilt/260611_PV5_t1.yaml` (its Bregma/Lambda as the first
guess). `manifests/botox_restani_manifest.csv` has a 72nd row, 260608 PV5, group P,
manifest Bregma 121/134 (the saved ROI set's own Bregma wins, as for every row).
Restore `folder` to `D:\WF_2026\starting` and `start_from` to None for the full cohort.

**Watch out.** 260608 precedes PV5's first known session (260611), so once it enters
`pixel_data/experimental_design.csv` every PV5 `day_index` shifts by one, and the
PV-group NBS (`between_day_indices = [1, 2]`, `PV_slope`) changes for PV5.

The user confirmed 260608 is PV5's day 1 (later sessions shift to 2-5). The first rerun
(`run_botox_batch` -> caches -> design -> connectivity -> PV NBS) stopped when the Claude
session closed: the 260608 pipeline and 20 s cache finished, `cache_median_dff.py
--window-s 30` died with exit 0x40010004 (console closed, not a code error), nothing
after it ran. Everything is resumable, so the next run picks up from there.

**Second new session: 260520 PV4 (same setup, 2026-10-02).** `C:\Users\loren\Downloads\pv4\pv4\t1-t5`,
6000 TIFFs each, cleanly interleaved (~7,500 GCaMP vs ~52,000 reflectance, stable over the
recording; t1/t4/t5 start on GCaMP, t2/t3 on reflectance). **t1's files are named `pv31_#####`**,
the rest `pv41_#####`: a filename slip, not a different animal -- t1's image matches t2-t5
at r = 1.00. Junction `WF_2026_new\260520\PV4` -> `pv4\pv4`; manifest row 73 (260520 PV4, P).
`batch_roi_select.py` `folder` is now `WF_2026_new\260520` (the day folder, so the editor
shows only PV4, not 260608 PV5) and `start_from` is `roi_sets/rebuilt/260618_PV4_t1.yaml`
(Bregma 122/132, Lambda 47). 260520 is PV4's day 1: it precedes 260618, so PV4's later
sessions shift to day_index 2-5 too, and both PV5 and PV4 change the PV-group NBS.


**Rerun with both new sessions (2026-10-02, `scratchpad/run_new_sessions.ps1`, 14 min, no
errors).** 260520 PV4 ROI sets saved at Bregma 124/132, Lambda 47 (rebuild: 0.855x, 0 px
repaired). Design: 555 recordings / 111 units, PV group 100 recordings / 6 animals; PV4 and
PV5 now day_index 1-5 with 260520 / 260608 as day 1 (verified). Connectivity: 554 recordings
(260828_PV7_t2 excluded), 111 units per variant. PV-group NBS, largest component, before ->
after (t > 2.5 unless noted):
- `PV_slope` increase grows and sharpens in every variant: mean 50 -> 100 edges (p 0.010 ->
  0.002; 48 of the old 50 kept), mean+GSR 9 -> 20 (p 0.040 -> 0.013; 8 kept), median
  15 -> 45 (p 0.056 -> 0.001), median+GSR 12 -> 31 (p 0.015 -> 0.0008; 11 kept). Adding
  an earlier day 1 to two animals lengthens their slope, so this is expected to move.
- `PV_vs_R` / `PV_vs_T` (day_index 1-2 means): edges mostly the same, p values move by one
  relabelling step (28 relabellings -> p floor 0.036); still fragile, as 9.33 says.
- New small `PV_slope` decrease components in the GSR variants (9 edges p = 0.098 mean+GSR;
  3 edges at t > 3.5 p = 0.029 median+GSR), not shared between variants. Treat as noise.
The C57-only NBS scripts (`test_nbs_pre_post_boto.py`, `test_nbs_change_over_time.py`) are
unaffected and were not rerun. Pre-change component tables: `scratchpad/before_260608/`.


**Third new session: 260520 PV3 (2026-10-02).** `E:\pv3\t1-t5`, 6000 TIFFs each, all named
`pv31_#####`, file times 2026-05-20 09:07. Cleanly interleaved (~6,900 GCaMP vs ~44,400
reflectance, stable; t1/t3 start on reflectance). Junction `WF_2026_new\260520\PV3` -> `E:\pv3`
(a junction may cross local volumes); manifest row 74 (260520 PV3, P). `batch_roi_select.py`
`folder` is `WF_2026_new\260520\PV3` (the animal folder, so 260520 PV4 is not shown) and
`start_from` is `roi_sets/rebuilt/260618_PV3_t1.yaml` (Bregma 100/132, Lambda 50).
260520 is PV3's day 1; its later sessions shift to day_index 2-5.
**PV4's `pv31`-named t1 is PV4, not PV3 (checked).** A band-passed (sigma 1 - sigma 8)
reflectance image separates animals cleanly: within PV3 or within PV4 r >= 0.99 across
t1-t5, PV3 vs PV4 r = -0.03 for every pair, including PV4 t1. The earlier r = 1.00 for PV4
was on a smooth 4x-downsampled image, which does not discriminate animals -- use the
band-passed comparison. The files also differ (md5).


**Rerun with 260520 PV3 (2026-10-02, `scratchpad/run_pv3.ps1`, 20 min, no errors).** ROI sets
saved at Bregma 100/132, Lambda 49 (rebuild 0.891x, 0 px repaired). Design: 560 recordings /
112 units, PV group 105 recordings / 6 animals; PV3, PV4 and PV5 each now day_index 1-5
(verified). Connectivity: 559 recordings, 112 units per variant. PV-group NBS, largest
component, before (with PV4/PV5) -> after, t > 2.5:
- `PV_slope` increase grows again without GSR: mean 100 -> 133 edges (p 0.0018 -> 0.0010,
  99 of 100 kept), median 45 -> 68 (p 0.001 -> 0.0004, 44 kept). With GSR it is stable:
  mean+GSR 20 -> 21 (p 0.003, 18 kept), median+GSR 31 -> 26 (p 0.001, 22 kept).
- `PV_slope` decrease now recurs across both GSR variants with the SAME core edges
  V1aL-FLR, V1aL-HLR, V1L-FLR (t > 3.5: p = 0.027 mean+GSR, 0.021 median+GSR; median+GSR
  t > 3.1 adds V1L-HLR, p = 0.034). Left visual vs right forelimb/hindlimb weakening over
  sessions in PV. Small (3-4 edges) and only after GSR, so relative to the global mean.
- `PV_vs_R` / `PV_vs_T`: same edges, p moves by one relabelling step (fragile, as before).
Pre-PV3 component tables: `scratchpad/before_pv3/`. E: has 8.3 GB free after this run
(~1.5 GB used per new session including caches).

### 9.35 PV3/PV4/PV5 late session vs day 1 (`test_pv_session_vs_day1.py`, 2026-10-03)
**Day indices moved.** After 260520 became day 1 for PV3/PV4 (9.34), 260716 PV3/PV4 is
**day 4** and 260805 PV5 is **day 5**. Older notes calling them day 3/4 are stale.
Each session is compared with its animal's day-1 session, using its 5 recordings and all 4 connectivity variants.
Tests are within-animal: recordings are the replicates, exact 5-vs-5 permutation, p floor 0.008.
Outputs are in `outputs/pv_session_vs_day1/`.

**Findings.**
- Without GSR, mean z over all edges rises in all three: dff +0.56 / +0.13 / +0.22 (PV3/PV4/PV5).
  50 edges rise in all three animals with |t| > 2.5. With GSR the global level change is about 0
  for PV3/PV4, so their increase is the global component (as 9.33 found for the NBS).
- PV5 day 5 is the strongest change and it survives GSR. The left hemisphere becomes one strongly
  coupled block (intra-left +0.53 z, every variant, p = 0.008) and decouples from the right
  (homotopic -0.49/-0.55 z with GSR). Edge-pattern r between days is 0.68-0.79, against
  0.96-0.98 within a session, so the pattern itself changed. That is not true of PV3/PV4
  (0.84-0.94).
- It is not an amplitude effect: PV5's left ROI traces have ~0.7x the right's SD on BOTH
  days, and both sides fall ~30% on day 5. The ratio is unchanged.
- The GSR-variant PV_slope decrease core (V1L/V1aL with FLR/HLR) falls in all three animals,
  but PV5 dominates it (-0.5 to -0.6 z vs -0.1 to -0.2). PV5 also dominates the GSR
  PV_slope increase. The PV_slope result after GSR therefore leans heavily on PV5's
  lateralised change. This check is circular: these sessions are part of the NBS input.

### 9.36 PV3/PV4/PV5 every day vs day 1 (`test_pv_days_vs_day1.py`, 2026-10-05)
Same input and within-animal tests as 9.35, over days 2-5. Outputs: `outputs/pv_days_vs_day1/`.
- **PV5 changes in one step between day 2 (260611) and day 3 (260708)**, then stays.
  GSR intra-left z 0.34 / 0.24 / 0.95 / 0.91 / 0.88 (mean+GSR); homotopic 0.41 / 0.47 /
  -0.12 / -0.06 / -0.07. Edge-pattern r between days 1-2 and days 3-5 is 0.65-0.82, and
  0.92-0.98 within either block. Note day 1 -> 2 is only 3 days apart, day 2 -> 3 four weeks.
- PV3/PV4 have no such step. Pattern r stays 0.84-0.95 across all days.
- The V1L/V1aL-limbR core (9.34) falls gradually in all three with GSR, PV5 most.
- Caveat: `difference_similarity.csv` correlates (day k - day 1) with (day 5 - day 1), and
  both share the day-1 mean, so r is inflated. PV5 day 2's r ~ 0.05 shows that term is small.

### 9.37 Antea's MATLAB chain can read the raw TIFFs directly; Fiji is not needed (2026-10-05)
Antea's scripts (1)-(5) are plain MATLAB. Fiji only did two things: it split each
recording's interleaved single-frame TIFFs into the emo/gCaMP stack pairs that script (1)
reads, and it drew the brain `Mask` that script (2) opens with `uiopen`.
`Antea_scripts/matlab/` replaces both:
- `load_interleaved_folder.m` sorts the files by name, splits odd/even files, and calls the
  dimmer group GCaMP, the same rule as `wfci.io.interleaved_channel_files`. It downsamples
  each frame with a 0.5 box resize as it is read.
- `script_1_correzione_emodinamica_da_tiff.m` builds `t_TEMP`.
- `script_2a_maschera_cervello.m` draws or reads `Mask`.

**Validated** on 260520/PV4/t1 with `n_trim = 20`. Both mean images equal the dump's
`mean_f`/`mean_r` exactly. The dF/F after script (2)'s resize matches
`pixels_dff_full.npy` to within float16 rounding (max 0.008 pp).

Notes:
- MATLAB R2024a is installed on the `loren` machine:
  `"C:\Program Files\MATLAB\R2024a\bin\matlab.exe" -batch "run('<file>.m')"`.
- Antea's `y_1`/`x_2` are `floor(bregma_row/2)`/`floor(bregma_col/2)` from
  `roi_sets/rebuilt/*.yaml`; they equal the dump metadata's `y_1`/`x_2`.
- Script (1) does not trim frames, but the Python pipeline drops 20 per channel.

**ROI sets in MATLAB, and the GSR difference (2026-10-05).** `read_roi_set.m` reads
`roi_sets/rebuilt/*.yaml`, and `script_4_correlazione_roi_set.m` replaces (3)+(4) with
each recording's own boxes and Bregma. The box convention is Antea's exactly: her
offsets are `cortex22_roi_set.yaml`. `script_2a` adds a `'roi_union'` mask. Full chain on
260520/PV4/t1: **without GSR** the r matrix equals `pixels_dff_full` `R_roi` to 2.3e-6.
**With GSR it differs by up to 0.13.** Antea's script (2) builds the global signal as
`nanmean(nanmean(data,1),2)`: the mean of the column means, which is not the pixel mean
for a non-rectangular mask (here 64 columns with 6-30 pixels each). `wfci.gsr.global_signal`
and `roi_pixel_connectivity.regress_global_signal` both take the pixel mean. The
`wfci.gsr` docstring says the two "agree exactly" on clean data, which is false unless
every mask column has the same pixel count. Recomputing the column-mean version in Python
reproduces MATLAB to 1.6e-5. The Python `_gsr` variants are therefore close to Antea's GSR
but not equal to it. Not changed (it would move every `_gsr` result); ask before switching.

### 9.38 GSR global signal is now Antea's column mean everywhere; pixel_data in MATLAB (2026-10-05)
**Default changed.** The plain pixel mean in Python was not Antea's script (2) (see 9.37).
The global signal is now `nanmean(nanmean(data,1),2)` by default in three places:
- `roi_pixel_connectivity.py` (`gsr_global_mean` / `--gsr-global-mean`, default `column`);
- `wfci.gsr` (`GSRConfig(global_mean="column")`, `global_signal(..., global_mean)`, and the
  shared `spatial_mean`);
- the streaming GSR in `wfci.streaming`.

The old version is `pixel`; its connectivity goes to `_gsr_pixelmean` folders. The tests
had missed the difference because their masks removed whole columns, and on such a mask
the two means are equal. New tests use irregular masks. Suite: 322 passed.

**Recomputed.** The two `roi_pixel_connectivity/*_gsr` variants were rebuilt (559
recordings each, 10.6 min on 8 workers). The old ones were renamed `*_gsr_pixelmean`, not
deleted, and so were the six NBS `_gsr` output folders. Then all three NBS scripts (both
time axes for change_over_time; set `NBS_TIME_VARIABLE=days_since_first_session` for the
second) and the four `test_pv_*` scripts were rerun. New `_gsr` R_roi on 260520/PV4/t1
equals Antea's MATLAB chain to 1.6e-5 (float16).

**What moved (largest components, t > 2.5 unless noted).**
- `PV_slope` increase grew: mean+GSR 21 -> 31 edges (p 0.003 -> 0.0002, 20 kept),
  median+GSR 26 -> 39 (p 0.001 -> 0.0002, all 26 kept).
- `PV_slope` decrease and `PV_vs_R` barely moved. `PV_vs_R` is still at its 28-relabelling
  p floor.
- `change_over_time` pooled decrease is unchanged (9 of 10 edges, p 0.007).
- Several small pre/post-Boto and T components that were near p 0.05 lost it (e.g. T
  increase t > 3.5 p 0.045 -> 0.20). A median+GSR `R_vs_T_postBoto` increase appeared
  (14 edges, p 0.05). These were already flagged as fragile in 9.33.

**PV metrics.**
- The non-GSR variants are unchanged.
- PV5's day 2 -> 3 step is unchanged: intra-left 0.29 / 0.22 / 0.92 / 0.89 / 0.83
  (mean+GSR); homotopic about 0.45 -> -0.1.
- The V1L/V1aL-limbR core is 0.1-0.2 z higher everywhere but still declines over sessions
  in all three animals, most in PV5 (-0.32 -> -0.79).

**pixel_data in MATLAB.**
- `Antea_scripts/matlab/read_npy.m` reads `.npy` and decodes float16 itself.
- `load_pixel_dump.m` puts the 76x87 crop back on the 128 grid (NaN outside) and reads the
  `.npz` metadata through `unzip`.
- `script_1b_da_pixel_data.m` builds `t_TEMP_resized`.
- `script_2_gsr_da_128.m` is Antea's (2) without the uiopen and without the first resize.

On 260520/PV4/t1 this route equals the raw-TIFF route to 2.3e-6 (no GSR) and 1.6e-5 (GSR).
A drawn mask that reaches past the crop only uses its part inside the crop.

**MATLAB figures (2026-10-05).** `Antea_scripts/matlab/make_pv_figures.m` runs the whole
chain in MATLAB for PV3/PV4/PV5, days 1-5:
pixel_data -> `antea_gsr.m` (her script (2) loop, ROI-union mask) -> each recording's ROI
set -> `corr` -> `mean(R,3)`.
It writes `outputs/pv_matlab_figures/{matrices,difference}_pixels_dff_full_{gsr,no_gsr}.png`
and `mean_R_*.mat`. `mean_R` equals the Python connectivity (plain mean of `R_roi`) to
3e-15 in all 15 sessions, with and without GSR. Runtime: 7.1 min with GSR, 0.9 min without.
`recompute = false` reuses the saved .mat and only redraws.
Batch-MATLAB gotcha: a figure is clipped to the screen size, so `exportgraphics` gave small
panels. Set PaperPosition/PaperSize and use `print(..., '-r200')` instead.

### 9.39 Frame montages for every animal; what GSR does to the slow decline (2026-10-05)
**Montages.** `plot_frame_montages_by_animal.py` draws, for each of the 26 animals, 10
figures, one per peak rank, with one panel per session (t1). Each is drawn twice from the
same frames: mean-baseline dF/F without GSR and with GSR (ROI-union column mean).
Output: `outputs/frame_montages/by_animal/{no_gsr,gsr}/<animal>/`, 520 PNG, 557 MB, plus
`peaks.csv` (1120 peaks, 10 per session). About 2 min on 8 workers.

Peaks are ranked on the field mean minus its 20 s running median. The raw mean-baseline
maximum always falls in the first seconds (9.30). Peaks are at least 3 s apart, and the
first and last 10 s are skipped.

`plot_frame_montage.py` gained `field_mean_trace`, `ranked_peak_frames` and
`draw_montage_figure` (the drawing moved out of `main()`). Its own output is unchanged:
same frames and colour limits. The saved `T9_t1_peak.png` differs only in panel spacing,
from an older version of the script. `plot_frame_montage.py` is untracked, so there is no
git baseline for it.

**GSR and the slow decline, measured on 8 recordings.** GSR removes 80-90 % of a pixel's
linear trend (median |slope| 0.27-0.97 -> 0.08-0.17 % dF/F/min, inside and outside the ROI
union alike) but not all of it, because each pixel's slope against the global signal is fitted over
the whole recording, so only the part of its drift proportional to the global drift goes.
Where the shared drift is small (260709_T9/t1, field slope -0.16) GSR barely changes the
pixel trends (0.20 -> 0.15): what is left there is pixel-specific. GSR also removes genuine
cortex-wide activity: a whole-field flash becomes a near-flat or inverted frame.

**Git here** needs `git -c safe.directory=E:/Developing_projects/letizia ...` (dubious
ownership: the repo was created under another user SID). This is a per-command flag; the
global config was not changed.
