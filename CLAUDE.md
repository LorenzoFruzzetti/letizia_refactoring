

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
  - Location on disk: `C:\Users\loren\miniconda3\envs\letizia`
  - Interpreter: `C:\Users\loren\miniconda3\envs\letizia\python.exe`
  - Spec / setup docs: `.env/environment.yml`, `.env/requirements.txt`, `.env/ENVIRONMENT_SETUP.md`
- By default, create and maintain a Conda environment unless the user explicitly opts out.
- Use non-interactive commands.
- Do not depend on conda activate in automated flows.
- Preferred command style:
  - conda run -n <env_name> <command>
- On this machine `conda` is not on PATH directly. Use the full path to conda.bat:
  - PowerShell: `& "$env:USERPROFILE\miniconda3\condabin\conda.bat" run -n <env_name> <command>`
  - Git Bash: `"$USERPROFILE/miniconda3/condabin/conda.bat" run -n <env_name> <command>`

#### Adding a new package to the `letizia` env
1. Install it (prefer conda-forge; fall back to pip):
   - `& "$env:USERPROFILE\miniconda3\condabin\conda.bat" install -n letizia -y <package>`
   - or `& "$env:USERPROFILE\miniconda3\condabin\conda.bat" run -n letizia pip install <package>`
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

## 6. Update
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

**Fix applied.** `rebuild_roi_sets.py` uses `round_half_away()` (ties away from zero),
which satisfies both `f(-x) == -f(x)` and `f(x+n) == f(x)+n`, so it is mirror-exact.
`roi_editor.scale_boxes` itself is unchanged — fixing it in place would move boxes in
already-saved sets. Regenerate through `rebuild_roi_sets.py` instead.

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

### 9.14 The manifest's `F:` paths are stale; the reflectance channel clips
**Problem.** `manifests\botox_restani_manifest.csv` has `recording_paths` under
`F:\WF_2026\starting\...`, but F: is not mounted; the dataset is now at
`D:\WF_2026\starting\...` (same layout, 6000 TIFFs per `t#`). Any batch run
against the manifest as committed fails to find its inputs. Regenerate with
`scan_botox_dataset.py` against the current drive letter before the real run.

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
