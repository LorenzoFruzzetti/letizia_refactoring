# Environment setup

Conda environment: **`letizia`** (Python 3.11).

## Linux: one-command install

`.env/install_linux.sh` builds the whole environment (deps + editable `wfci`)
and verifies it. Run it from the repo root:

```bash
# conda mode (default): creates/updates the `letizia` env from environment.yml.
# Uses conda, mamba or micromamba, whichever is found first.
bash .env/install_linux.sh

# venv mode: no conda needed; creates .venv/ and pip-installs requirements.txt
bash .env/install_linux.sh --mode venv
```

| Flag | Effect |
| --- | --- |
| `--mode conda\|venv` | Installer backend (default `conda`) |
| `--name <env>` | Conda env name (default `letizia`) |
| `--venv-path <dir>` | Venv location (default `<repo>/.venv`) |
| `--python <bin>` | Interpreter used to bootstrap venv mode (default `python3`) |
| `--force` | Delete and recreate the env from scratch |
| `--no-test` | Skip the post-install `pytest` run |
| `--help` | Usage summary |

`roi_editor.py` is a **pyqtgraph (Qt)** window, so it needs `pyqtgraph` plus a Qt
binding — `pyside6` is what this project is tested against, though the code reaches
Qt only through `pyqtgraph.Qt`, so PyQt6 / PyQt5 / PySide2 work too. Both are in
`environment.yml` / `requirements.txt`, and also available as the `gui` extra
(`pip install -e ".[gui]"`).

On a headless Linux box a Qt wheel may still need system X/GL libraries
(`sudo apt install libgl1 libegl1 libxkbcommon-x11-0`); `QT_QPA_PLATFORM=offscreen`
runs it without a display, which is how `tests/test_roi_editor.py` drives it.

Every other script is headless and does not need Qt at all — and neither does
`from roi_editor import load_roi_set`, which the run scripts use to read ROI sets:
the editor imports Qt inside `ROIEditor._build_ui`, never at module level. The
installer reports whether `pyqtgraph` imports.

## Windows

On this machine `conda` is not on PATH; use the full path to `conda.bat`:

- Git Bash: `"$USERPROFILE/miniconda3/condabin/conda.bat"`
- PowerShell: `& "$env:USERPROFILE\miniconda3\condabin\conda.bat"`

## Create / update

```bash
CONDA="$USERPROFILE/miniconda3/condabin/conda.bat"

# Create the environment
"$CONDA" env create -f .env/environment.yml

# Install the wfci package itself (editable) from the repo root
"$CONDA" run -n letizia pip install -e .

# Update after editing environment.yml
"$CONDA" env update -n letizia -f .env/environment.yml --prune
```

## Add a new package

1. Install it into the env (conda-forge preferred; fall back to pip):

   ```bash
   CONDA="$USERPROFILE/miniconda3/condabin/conda.bat"
   "$CONDA" install -n letizia -y <package>      # e.g. pandas
   # or, if not on conda-forge:
   "$CONDA" run -n letizia pip install <package>
   ```

2. Record it in **both** spec files so the env stays reproducible:
   - add `- <package>>=<min-version>` to `.env/environment.yml`
   - add `<package>>=<min-version>` to `.env/requirements.txt`

3. Verify: `"$CONDA" run -n letizia python -c "import <package>; print(<package>.__version__)"`

## Validate

```bash
"$CONDA" run -n letizia python --version
"$CONDA" run -n letizia pip list
"$CONDA" run -n letizia python -c "import wfci; print(wfci.__version__)"
"$CONDA" run -n letizia python -m pytest tests/ -s -v
```

## Files kept aligned

- `environment.yml` — conda spec (source of truth for the env)
- `requirements.txt` — pip deps mirror
- `.envVariables` — environment variables (loaded by VS Code launch configs)

## Interpreter for VS Code

`.vscode/settings.json` pins:
`C:\Users\loren\miniconda3\envs\letizia\python.exe`

The debugger (`.vscode/launch.json`, `type: debugpy`) uses the same interpreter.
If the interpreter is not auto-selected, run **Python: Select Interpreter** and
choose the `letizia` env.

## Notes

- MATLAB (R2024a) is only needed to (re)generate the parity reference
  (`tests/matlab_reference/gen_reference.m`); the package and its tests otherwise
  run on Python alone once `reference.mat` exists.
