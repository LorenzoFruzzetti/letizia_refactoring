#!/usr/bin/env bash
#
# Start the letizia dev container on the Ubuntu 16.04 host (IN-CONFERENCE).
#
# Usage:
#   bash docker/run.sh                  # interactive shell in the container
#   bash docker/run.sh --gui            # also forward X11 (for roi_editor.py)
#   bash docker/run.sh -- python run_pipeline.py --help    # run one command
#   bash docker/run.sh --name letizia2  # a second, independent container
#
# Everything the container writes to /work lands in the bind-mounted repo on the
# INTERNAL disk. The USB stick only ever holds image layers -- see
# CONTAINER_STICK_README.md section 6 for why that split matters at 40 MB/s.
#
# -e: stop at the first failure, -u: error on unset variables, -o pipefail.
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults -- edit here to run the script without passing any flags.
# ---------------------------------------------------------------------------
IMAGE="letizia:dev"
NAME="letizia"
REPO_DIR="${HOME}/projects/letizia"        # the git clone, on the internal disk
HOME_DIR="${HOME}/.letizia-container"      # persistent container home: ~/.claude,
                                           # the VS Code tunnel token, shell history
DATA_DIR="${HOME}/data"                    # recordings; mounted read-only at /data
GUI=0                                      # 1 -> forward X11 so Qt windows can open
STICK_MOUNT="/srv/containers"              # checked, not used directly

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gui)   GUI=1; shift ;;
        --name)  NAME="$2"; shift 2 ;;
        --repo)  REPO_DIR="$2"; shift 2 ;;
        --data)  DATA_DIR="$2"; shift 2 ;;
        --home)  HOME_DIR="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        -h|--help) sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        --)      shift; break ;;
        *)       break ;;
    esac
done

# ---------------------------------------------------------------------------
# Preflight. The number-one failure mode on this host is Docker silently
# writing to the root disk because the stick is not mounted -- the root disk was
# 87% full, so this is worth one loud check before every run.
# ---------------------------------------------------------------------------
if ! mountpoint -q "${STICK_MOUNT}"; then
    echo "WARNING: ${STICK_MOUNT} is not a mount point -- the container stick is absent."
    echo "         Docker would write image layers to the ROOT disk."
    echo "         Fix: sudo mount /dev/disk/by-label/CONTAINERS ${STICK_MOUNT}"
    echo "              sudo systemctl restart docker"
    echo
    read -r -p "Continue anyway? [y/N] " reply
    [[ "${reply}" == "y" || "${reply}" == "Y" ]] || exit 1
fi

if [[ ! -d "${REPO_DIR}/src/wfci" ]]; then
    echo "ERROR: ${REPO_DIR} does not look like the letizia repo (no src/wfci)." >&2
    echo "       Clone it first, or pass --repo /path/to/clone." >&2
    exit 1
fi

# The container home must exist on the host before the bind mount, otherwise
# Docker creates it root-owned and the non-root container user cannot write
# ~/.claude into it.
mkdir -p "${HOME_DIR}"

# ---------------------------------------------------------------------------
# Assemble the run command.
# ---------------------------------------------------------------------------
ARGS=(
    --rm -it
    --name "${NAME}"
    --hostname letizia-dev
    -v "${REPO_DIR}:/work"
    -v "${HOME_DIR}:/home/dev"
    -w /work
    -e "HOME=/home/dev"
    # Host networking: the VS Code tunnel and Claude Code both need outbound
    # access, and neither needs an inbound published port. It also means any
    # port a script opens is reachable at localhost on the host.
    --network host
)

if [[ -d "${DATA_DIR}" ]]; then
    # Read-only on purpose: an agent iterating inside the container must not be
    # able to modify or delete the recordings.
    ARGS+=(-v "${DATA_DIR}:/data:ro")
fi

if [[ "${GUI}" -eq 1 ]]; then
    if [[ -z "${DISPLAY:-}" ]]; then
        echo "ERROR: --gui was requested but DISPLAY is unset -- there is no X server" >&2
        echo "       to forward. Run this from a graphical session on the host." >&2
        exit 1
    fi
    xhost +local: >/dev/null
    ARGS+=(-e "DISPLAY=${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix)
fi

echo "repo   : ${REPO_DIR}  -> /work"
echo "home   : ${HOME_DIR}  -> /home/dev"
# Written as an `if`, not `[[ … ]] && echo`: under `set -e` a false test as the
# whole command would abort the script before the container ever starts.
if [[ -d "${DATA_DIR}" ]]; then
    echo "data   : ${DATA_DIR}  -> /data (read-only)"
fi
echo "image  : ${IMAGE}"
echo

exec docker run "${ARGS[@]}" "${IMAGE}" "${@:-bash}"
