# Ubuntu 22.04 (glibc 2.35) development container for the wfci / letizia project.
#
# Why this exists: the target host (IN-CONFERENCE) runs Ubuntu 16.04 with glibc
# 2.23. Modern VS Code, its remote server, and Node.js 18+ all require
# glibc >= 2.28, so none of them can run on the host at all. They run in here
# instead, alongside the conda env and the project itself -- i.e. this container
# is the development machine, not just the runtime.
#
# Build, from the repo root on the host:
#   docker build -f docker/Dockerfile -t letizia:dev \
#       --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
#
# Only .env/environment.yml is baked in; the source tree is bind-mounted at run
# time (see docker/run.sh). That follows CONTAINER_STICK_README.md section 6:
# image layers live on the USB stick (write-once, read-often), while source and
# caches stay on the internal disk.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# 1. System packages
#    - ca-certificates/curl/git : fetching the repo and the toolchains below
#    - build-essential          : any package that needs to compile a wheel
#    - tmux                     : a detached agent session must survive a
#                                 dropped SSH link or a closed terminal
#    - ripgrep                  : Claude Code's file search backend
#    - lib{gl,egl,xkbcommon,xcb-*} : Qt runtime for roi_editor.py's pyqtgraph
#                                 window (only used when X11 is forwarded in)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git openssh-client \
        build-essential tmux less nano ripgrep \
        libgl1 libegl1 libglib2.0-0 libdbus-1-3 libfontconfig1 \
        libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 \
        libxcb-shape0 libxcb-xinerama0 libxcb-randr0 libxcb-render-util0 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Node.js 20 + Claude Code
#    Node is here purely for Claude Code. The autoupdater is disabled further
#    down because the global npm prefix is root-owned and the container runs as
#    a non-root user -- update by rebuilding this image instead.
# ---------------------------------------------------------------------------
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code \
    && npm cache clean --force

# ---------------------------------------------------------------------------
# 3. VS Code CLI
#    Provides `code tunnel`, which makes an OUTBOUND connection to vscode.dev
#    and lets a VS Code desktop elsewhere attach to this container. No inbound
#    port, no sshd, nothing for the conference firewall to block.
# ---------------------------------------------------------------------------
RUN curl -fsSL "https://code.visualstudio.com/sha/download?build=stable&os=cli-linux-x64" \
        -o /tmp/vscode-cli.tar.gz \
    && tar -xzf /tmp/vscode-cli.tar.gz -C /usr/local/bin \
    && rm /tmp/vscode-cli.tar.gz \
    && code --version

# ---------------------------------------------------------------------------
# 4. Miniforge + the `letizia` conda env (python 3.11)
#    Built into the image so container start-up needs no solving and the whole
#    env lands in a read-only layer on the stick.
# ---------------------------------------------------------------------------
RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:${PATH}

COPY .env/environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy \
    && rm /tmp/environment.yml

# Put the env's bin FIRST, so plain `python` / `pytest` are the letizia ones and
# no `conda activate` or `conda run` wrapper is needed anywhere.
ENV PATH=/opt/conda/envs/letizia/bin:${PATH}

# ---------------------------------------------------------------------------
# 5. Non-root user matching the host's lorenzo (UID/GID 1000 by default)
#    Without this, every file the agent writes into the bind-mounted repo comes
#    out root-owned and the host user needs sudo to touch its own source tree.
#    A real passwd entry (rather than `docker run --user`) matters because git,
#    ssh and node all call getpwuid() and fail on an unknown UID.
# ---------------------------------------------------------------------------
ARG UID=1000
ARG GID=1000
RUN groupadd -f -g ${GID} dev \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash dev

# PYTHONPATH replaces `pip install -e .`: `import wfci` resolves against the
# bind-mounted source, so edits take effect immediately and nothing is written
# into the image layer at run time.
ENV PYTHONPATH=/work/src \
    DISABLE_AUTOUPDATER=1 \
    QT_X11_NO_MITSHM=1

USER dev
WORKDIR /work
CMD ["bash"]
