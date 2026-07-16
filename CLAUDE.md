

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