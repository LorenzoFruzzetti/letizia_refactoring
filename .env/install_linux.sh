#!/usr/bin/env bash
#
# Linux installer for the `letizia` / wfci environment.
#
# Two modes:
#   conda  (default) -- build the env from .env/environment.yml with conda,
#                       mamba or micromamba, whichever is found first.
#   venv             -- build a plain python -m venv and install
#                       .env/requirements.txt with pip.
#
# In both modes the wfci package itself is installed in editable mode
# (`pip install -e .`) from the repo root, so `import wfci` works from
# anywhere and edits under src/wfci/ take effect immediately.
#
# Usage:
#   bash .env/install_linux.sh                    # conda mode, env name "letizia"
#   bash .env/install_linux.sh --mode venv        # venv mode -> .venv/ in repo root
#   bash .env/install_linux.sh --name myenv       # custom conda env name
#   bash .env/install_linux.sh --venv-path /opt/e # custom venv location
#   bash .env/install_linux.sh --force            # recreate the env from scratch
#   bash .env/install_linux.sh --no-test          # skip the post-install pytest run
#
# -e: stop at the first failing command, -u: error on unset variables,
# -o pipefail: a failure anywhere in a pipe fails the whole pipe.
# Errors are deliberately left to surface rather than being swallowed.
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (edit here to run the script without passing any flags)
# ---------------------------------------------------------------------------
MODE="conda"          # conda | venv
ENV_NAME="letizia"    # conda env name
VENV_PATH=""          # resolved to <repo_root>/.venv when empty
PYTHON_BIN="python3"  # interpreter used to bootstrap venv mode
FORCE=0               # 1 -> delete an existing env before creating it
RUN_TESTS=1           # 1 -> run pytest at the end

# Repo root = parent of the directory holding this script (.env/ -> repo root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)      MODE="$2"; shift 2 ;;
        --name)      ENV_NAME="$2"; shift 2 ;;
        --venv-path) VENV_PATH="$2"; shift 2 ;;
        --python)    PYTHON_BIN="$2"; shift 2 ;;
        --force)     FORCE=1; shift ;;
        --no-test)   RUN_TESTS=0; shift ;;
        -h|--help)
            # Print the comment header above as the help text.
            sed -n '3,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown option '$1' (try --help)" >&2
            exit 2
            ;;
    esac
done

if [[ "${MODE}" != "conda" && "${MODE}" != "venv" ]]; then
    echo "ERROR: --mode must be 'conda' or 'venv', got '${MODE}'" >&2
    exit 2
fi

: "${VENV_PATH:=${REPO_ROOT}/.venv}"

ENV_YML="${REPO_ROOT}/.env/environment.yml"
REQ_TXT="${REPO_ROOT}/.env/requirements.txt"

echo "=============================================================="
echo " wfci / letizia -- Linux environment install"
echo "   repo root : ${REPO_ROOT}"
echo "   mode      : ${MODE}"
echo "=============================================================="

# ---------------------------------------------------------------------------
# System-level note: the ROI editor (roi_editor.py) selects matplotlib's
# TkAgg backend, which needs Tk bindings. conda-forge's python ships them;
# a distro python usually does not, so venv mode checks and tells the user
# the exact package to install.
# ---------------------------------------------------------------------------
report_tk() {
    local py="$1"
    if "${py}" -c "import tkinter" >/dev/null 2>&1; then
        echo "  tkinter        : OK (roi_editor.py GUI will work)"
    else
        echo "  tkinter        : MISSING -- roi_editor.py needs it. Install with:"
        echo "                     Debian/Ubuntu : sudo apt install python3-tk"
        echo "                     Fedora/RHEL   : sudo dnf install python3-tkinter"
        echo "                     Arch          : sudo pacman -S tk"
        echo "                   (all headless scripts still work without it)"
    fi
}

# ===========================================================================
# conda mode
# ===========================================================================
if [[ "${MODE}" == "conda" ]]; then
    # Pick the first available solver. mamba/micromamba are drop-in and faster.
    CONDA_BIN=""
    for candidate in micromamba mamba conda; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            CONDA_BIN="${candidate}"
            break
        fi
    done

    # Not on PATH: probe the usual install locations before giving up.
    if [[ -z "${CONDA_BIN}" ]]; then
        for prefix in "${HOME}/miniconda3" "${HOME}/anaconda3" "${HOME}/miniforge3" \
                      "${HOME}/mambaforge" "/opt/conda" "/usr/local/miniconda3"; do
            if [[ -x "${prefix}/bin/conda" ]]; then
                CONDA_BIN="${prefix}/bin/conda"
                break
            fi
        done
    fi

    if [[ -z "${CONDA_BIN}" ]]; then
        echo "ERROR: no conda/mamba/micromamba found on PATH or in the usual" >&2
        echo "       install prefixes. Either install Miniforge:" >&2
        echo "         https://github.com/conda-forge/miniforge#install" >&2
        echo "       or rerun this script in venv mode:" >&2
        echo "         bash .env/install_linux.sh --mode venv" >&2
        exit 1
    fi

    echo "[1/4] solver: ${CONDA_BIN}"

    # micromamba needs an explicit root prefix and does not read a base config.
    if [[ "$(basename "${CONDA_BIN}")" == "micromamba" ]]; then
        export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"
    fi

    # `env list` output is one env per line; match the name in column 1.
    env_exists() {
        "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"
    }

    if env_exists && [[ "${FORCE}" -eq 1 ]]; then
        echo "[2/4] --force: removing existing env '${ENV_NAME}'"
        "${CONDA_BIN}" env remove -n "${ENV_NAME}" -y
    fi

    if env_exists; then
        echo "[2/4] env '${ENV_NAME}' exists -- updating from environment.yml"
        "${CONDA_BIN}" env update -n "${ENV_NAME}" -f "${ENV_YML}" --prune
    else
        echo "[2/4] creating env '${ENV_NAME}' from ${ENV_YML}"
        "${CONDA_BIN}" env create -n "${ENV_NAME}" -f "${ENV_YML}"
    fi

    # conda/mamba buffer subprocess output and only flush it at the end;
    # --no-capture-output streams it live. micromamba streams already and
    # rejects the flag, so it is only added for the other two.
    RUN_PREFIX=("${CONDA_BIN}" run -n "${ENV_NAME}" --cwd "${REPO_ROOT}")
    if [[ "$(basename "${CONDA_BIN}")" != "micromamba" ]]; then
        RUN_PREFIX+=(--no-capture-output)
    fi

    echo "[3/4] installing wfci in editable mode"
    # `-e .` must run with the repo root as cwd so setuptools finds pyproject.toml.
    "${RUN_PREFIX[@]}" pip install -e .

    RUN_PY=("${RUN_PREFIX[@]}" python)
    ACTIVATE_HINT="conda activate ${ENV_NAME}"

# ===========================================================================
# venv mode
# ===========================================================================
else
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "ERROR: '${PYTHON_BIN}' not found. Pass a different interpreter with --python." >&2
        exit 1
    fi

    # The package declares requires-python >= 3.10; refuse to build on older.
    "${PYTHON_BIN}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
        echo "ERROR: ${PYTHON_BIN} is $("${PYTHON_BIN}" -V 2>&1); wfci needs Python >= 3.10." >&2
        exit 1
    }

    echo "[1/4] interpreter: $(command -v "${PYTHON_BIN}") ($("${PYTHON_BIN}" -V 2>&1))"

    if [[ -d "${VENV_PATH}" && "${FORCE}" -eq 1 ]]; then
        echo "[2/4] --force: removing existing venv ${VENV_PATH}"
        rm -rf "${VENV_PATH}"
    fi

    if [[ -d "${VENV_PATH}" ]]; then
        echo "[2/4] reusing venv ${VENV_PATH}"
    else
        echo "[2/4] creating venv ${VENV_PATH}"
        "${PYTHON_BIN}" -m venv "${VENV_PATH}"
    fi

    VENV_PY="${VENV_PATH}/bin/python"

    echo "[3/4] installing dependencies + wfci (editable)"
    "${VENV_PY}" -m pip install --upgrade pip setuptools wheel
    "${VENV_PY}" -m pip install -r "${REQ_TXT}"
    "${VENV_PY}" -m pip install -e "${REPO_ROOT}"

    RUN_PY=("${VENV_PY}")
    ACTIVATE_HINT="source ${VENV_PATH}/bin/activate"
fi

# ===========================================================================
# Verification -- import every third-party dependency the pipeline uses.
# ===========================================================================
echo "[4/4] verifying installation"
"${RUN_PY[@]}" - <<'PY'
import importlib
import sys

print(f"  python         : {sys.version.split()[0]}  ({sys.executable})")

# Every third-party import that appears anywhere in the repo.
for module in ("numpy", "scipy", "pandas", "tifffile", "matplotlib", "yaml", "pytest"):
    mod = importlib.import_module(module)
    version = getattr(mod, "__version__", "n/a")
    print(f"  {module:<15}: {version}")

import wfci
print(f"  wfci           : {wfci.__version__}  ({wfci.__file__})")
PY

# tkinter is optional (GUI-only), so it is reported separately and never fatal.
if [[ "${MODE}" == "conda" ]]; then
    TK_PY="$("${RUN_PY[@]}" -c 'import sys; print(sys.executable)')"
else
    TK_PY="${VENV_PY}"
fi
report_tk "${TK_PY}"

# ---------------------------------------------------------------------------
# Optional test run. `|| true` is intentional: a failing test suite should be
# reported but must not make the installer look like the install itself broke.
# ---------------------------------------------------------------------------
if [[ "${RUN_TESTS}" -eq 1 ]]; then
    echo
    echo "--- running test suite (skip with --no-test) ---"
    "${RUN_PY[@]}" -m pytest "${REPO_ROOT}/tests" -q || {
        echo
        echo "NOTE: some tests failed. The parity test needs" >&2
        echo "      tests/matlab_reference/reference.mat, which is not tracked;" >&2
        echo "      other failures are worth investigating." >&2
    }
fi

cat <<EOF

==============================================================
 Done. Activate the environment with:

   ${ACTIVATE_HINT}

 Then, from ${REPO_ROOT}:

   python run_pipeline.py --help
   python run_intermingle_rs.py --help
   python roi_editor.py --help        # needs tkinter
   python -m pytest tests/ -v
==============================================================
EOF
