# Environment setup

Conda environment: **`letizia`** (Python 3.11).

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
