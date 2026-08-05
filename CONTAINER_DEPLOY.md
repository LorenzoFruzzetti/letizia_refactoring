# Deploying `letizia` onto the container host — step by step

How to get this repository running on **IN-CONFERENCE** (Ubuntu 16.04 + the
Lexar container stick), with **Claude Code** and optionally **VS Code** attached
so an agent can iterate there.

Companion doc: [CONTAINER_STICK_README.md](CONTAINER_STICK_README.md) — what the
stick is, how it mounts, and how to look after it. Read section 3 of that file
first if `df -h /srv/containers` looks wrong.

---

## 0. The constraint that shapes everything

**VS Code and Claude Code cannot run on the host.** Ubuntu 16.04 ships
**glibc 2.23**, and:

| Tool | Needs | Host has | Verdict |
|---|---|---|---|
| VS Code desktop ≥ 1.86 | glibc ≥ 2.28 | 2.23 | ✗ |
| VS Code **Remote-SSH server** ≥ 1.86 | glibc ≥ 2.28 | 2.23 | ✗ |
| Node.js ≥ 18 (Claude Code) | glibc ≥ 2.28 | 2.23 | ✗ |
| All of the above, in the container | glibc ≥ 2.28 | **2.35** | ✓ |

So "SSH from my laptop into the machine and open the folder in VS Code" fails at
*Installing VS Code Server* — that is a hard floor, not a configuration problem.
Pinning VS Code to 1.85 (January 2024) would work but leaves you on a frozen
editor that current extensions no longer support.

**The fix: the container is the development machine.** The editor server, the
agent, the conda env and the project all live inside Ubuntu 22.04. The host only
runs the Docker daemon and holds the files.

```
  your laptop                  IN-CONFERENCE (Ubuntu 16.04, glibc 2.23)
  ┌──────────────┐             ┌──────────────────────────────────────────┐
  │  VS Code     │             │  dockerd  →  /srv/containers (USB stick)  │
  │  desktop     │             │                                          │
  └──────┬───────┘             │   container: ubuntu 22.04, glibc 2.35    │
         │                     │   ┌────────────────────────────────────┐ │
         │  outbound tunnel    │   │ code tunnel · claude · conda env    │ │
         └────── vscode.dev ───┼──►│ /work ── bind-mount ──┐             │ │
                               │   └───────────────────────┼─────────────┘ │
                               │   ~/projects/letizia ◄────┘ internal disk  │
                               └──────────────────────────────────────────┘
```

---

## 1. Preflight on the host

Open a terminal **on the host** (physically, or plain `ssh` — plain SSH is fine,
it is only the VS Code *server* that cannot be installed there).

```bash
mountpoint -q /srv/containers && echo "stick OK" || echo "STICK NOT MOUNTED"
docker info --format '{{.DockerRootDir}} | {{.Driver}} | {{.ServerVersion}}'
```

Expected: `stick OK`, then `/srv/containers/docker | overlay2 | 19.03.12`.

If the stick is not mounted, stop and fix it — otherwise the image build fills
the root disk, which was already 87% full:

```bash
sudo mount /dev/disk/by-label/CONTAINERS /srv/containers
sudo systemctl restart docker
```

Also confirm you are in the `docker` group (`docker ps` without `sudo`). If it
says *permission denied … docker.sock*, run `newgrp docker` or log in again.

---

## 2. Clone the repository — onto the internal disk

**Not onto the stick.** Source is write-heavy (git, editors, agents); the stick
is USB 2.0 at 40 MB/s and flash handles many small writes badly.

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/LorenzoFruzzetti/letizia_refactoring.git letizia
cd letizia
```

The clone is ~21 MB. If the repository is private, use a fine-grained personal
access token as the password at the prompt. Don't store it — `git config
--global credential.helper store` writes it to `~/.git-credentials` in plain
text, which on a shared conference machine is a real exposure.

> **If the clone fails with a TLS or certificate error**, 16.04's CA bundle is
> too old. Either `sudo apt-get install --reinstall ca-certificates` (the host
> repos were already repointed to `old-releases.ubuntu.com`), or do the clone
> from a throwaway modern container, which writes to the same host directory:
>
> ```bash
> docker run --rm -v ~/projects:/p -w /p ubuntu:22.04 bash -c \
>   'apt-get update -qq && apt-get install -y -qq git ca-certificates && \
>    git clone https://github.com/LorenzoFruzzetti/letizia_refactoring.git letizia'
> sudo chown -R "$USER:$USER" ~/projects/letizia
> ```

---

## 3. Build the image

```bash
cd ~/projects/letizia
docker build -f docker/Dockerfile -t letizia:dev \
    --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
```

[`docker/Dockerfile`](docker/Dockerfile) installs, on Ubuntu 22.04: the Qt system
libraries, Node 20 + Claude Code, the VS Code CLI, Miniforge, and the `letizia`
conda env built from [`.env/environment.yml`](.env/environment.yml).

`--build-arg UID/GID` matters: it creates a container user with **your** UID, so
files the agent writes into the bind-mounted repo are owned by you and not by
root. Skip it and you will need `sudo` to edit your own source tree.

Expect **15–30 minutes** — most of it is conda solving and writing PySide6 to a
40 MB/s stick. It is a one-time cost; the layers are cached.

Check the space it took:

```bash
docker system df
```

---

## 4. Start the container

```bash
bash docker/run.sh
```

[`docker/run.sh`](docker/run.sh) checks the stick is mounted, then bind-mounts:

| Host | Container | Why |
|---|---|---|
| `~/projects/letizia` | `/work` | the source, read-write |
| `~/.letizia-container` | `/home/dev` | persistent home — Claude login, tunnel token, history survive `--rm` |
| `~/data` (if present) | `/data` (read-only) | recordings the agent must not be able to delete |

Edit the defaults at the top of the script, or pass `--repo` / `--data` /
`--home`. `--gui` additionally forwards X11 so `roi_editor.py` can open a window.

Verify the environment inside the container:

```bash
python -V                      # 3.11.x, from /opt/conda/envs/letizia
python -c "import wfci; print(wfci.__version__, wfci.__file__)"
claude --version
code --version
```

`import wfci` resolves via `PYTHONPATH=/work/src`, so no `pip install -e .` is
needed and edits under `src/wfci/` take effect immediately.

---

## 5. Connect Claude Code

Inside the container:

```bash
claude
```

On first run it needs authentication. With no browser available it prints a URL —
open that on **any** machine (your laptop, your phone), approve, and paste the
code back into the terminal. Alternatively set `ANTHROPIC_API_KEY` in the
environment to bill the API instead of a subscription.

Credentials land in `/home/dev/.claude`, which is the bind-mounted
`~/.letizia-container` on the host — so you log in **once**, not once per
container.

**Run long agent sessions under tmux**, so a dropped SSH connection doesn't kill
the agent mid-run:

```bash
tmux new -s agent      # then: claude
# detach: Ctrl-b d      reattach: tmux attach -t agent
```

### About permissions

The container is a reasonable place to loosen Claude Code's permission prompts —
but note the bind mounts: `/work` is your real source tree and `/data` is real
recordings. `--dangerously-skip-permissions` inside this container is *not*
sandboxed from those. `/data` is mounted read-only precisely so that the worst
case there is a wasted run, not lost recordings.

---

## 6. Connect VS Code (optional)

### Recommended: a tunnel out of the container

Inside the container:

```bash
code tunnel --accept-server-license-terms --name letizia-conf
```

It prints a device-code URL — authenticate with a GitHub or Microsoft account,
and it gives you a `https://vscode.dev/tunnel/letizia-conf` link. Then either:

- open that link in a browser anywhere, or
- in VS Code desktop on your laptop, install the **Remote - Tunnels** extension
  and pick *Connect to Tunnel → letizia-conf*.

The VS Code server runs inside the container, on glibc 2.35, so it just works.
The connection is **outbound only** — no inbound port, nothing for a conference
firewall or NAT to block. The token persists in the mounted home, so subsequent
runs skip the authentication.

Run it under tmux too, or as a background service (`code tunnel service install`).

### Alternative: no VS Code at all

Plain terminal + `claude` under tmux is fully sufficient, and one fewer moving
part. The only thing you lose is the graphical diff view.

### Alternative: Remote-SSH into the container

Only worth it if your laptop is on the same LAN. Remote-SSH must target the
**container**, never the host. That means adding `openssh-server` to the image,
generating host keys, publishing a port (e.g. `-p 2222:22` instead of
`--network host`), and managing a key for the `dev` user. More setup than the
tunnel, for the same result.

---

## 7. Verify the project actually runs

Inside the container, from `/work`:

```bash
python -m pytest tests/ -q
python examples/run_example.py
python run_pipeline.py --help
```

`tests/test_parity.py` will **fail or error** on a fresh clone: it needs
`tests/matlab_reference/reference.mat`, which is git-ignored (~23 MB) and can
only be regenerated by MATLAB. That single failure is expected. Everything else
should pass — if other tests fail, that is worth investigating.

`examples/run_example.py` runs on the sample TIFFs in `data/`, which **are**
tracked, so the example works with nothing else copied over.

---

## 8. Getting the real data there

The recordings are not in git. Options, in order of preference:

1. **Mount the lab share on the host** and bind-mount it read-only. The README's
   batch commands point at `\\146.48.88.209\share2\BOTOX_RESTANI\...`; on Linux:

   ```bash
   sudo mkdir -p /mnt/botox
   sudo mount -t cifs //146.48.88.209/share2 /mnt/botox -o ro,username=<user>,vers=2.0
   bash docker/run.sh --data /mnt/botox
   ```

   (16.04's default SMB dialect is old; `vers=2.0` or `vers=3.0` may be needed.)

2. **`rsync` or copy a subset** to `~/data` on the internal disk — enough for one
   animal is usually enough to iterate on.

3. **Not the stick.** Recordings on a 40 MB/s USB 2.0 device will dominate every
   runtime measurement you take.

Inside the container the data is at `/data`, so:

```bash
python run_intermingle_rs.py --folder /data/BOTOX_RESTANI/260611 \
    --output-dir outputs/intermingle_260611
```

---

## 9. The agent-iteration loop

The container definition lives **in the repo**, which is bind-mounted, so an
agent inside the container can edit `docker/Dockerfile` and `docker/run.sh` — it
just cannot apply them to itself. Split it:

| Who | Does what |
|---|---|
| agent, inside the container | edits source, runs the pipeline and tests, edits `docker/Dockerfile` when a new dependency is needed |
| you, in a host terminal | `docker build …` and restart via `docker/run.sh` |

Keep a second terminal on the host for exactly this. Adding a package usually
means: agent adds it to `.env/environment.yml` **and** `.env/requirements.txt`
(per [CLAUDE.md](CLAUDE.md) §4.2), you rebuild, and the layer cache means only
the conda step re-runs.

**Do not mount `/var/run/docker.sock` into the container** to let the agent
rebuild itself. That grants effective root on the host — on a machine that also
carries someone else's 3.7 TB drive (`/dev/sdb`), that is not a trade worth
making.

---

## 10. Shutting down cleanly

**Never unplug the stick with containers running** — it corrupts the overlay2
store.

```bash
exit                            # leave the container (--rm removes it)
docker stop $(docker ps -q)     # anything still running
sudo systemctl stop docker docker.socket
sudo umount /srv/containers
```

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Remote-SSH: *Installing VS Code Server…* then fails | You targeted the host. glibc 2.23 — impossible. Use `code tunnel` from inside the container (§6). |
| `claude: command not found` on the host | Expected. Claude Code only exists inside the container. |
| Build fills the root disk | Stick not mounted when dockerd started. `mount`, then `sudo systemctl restart docker`, then rebuild. |
| Files in the repo owned by `root` after an agent run | Image built without `--build-arg UID=$(id -u)`. `sudo chown -R "$USER:$USER" ~/projects/letizia` and rebuild. |
| `~/.claude` empty / login asked again every start | `~/.letizia-container` was created root-owned. `sudo chown -R "$USER:$USER" ~/.letizia-container`. |
| `roi_editor.py`: *could not connect to display* | Start with `bash docker/run.sh --gui`, and check `xhost +local:` succeeded on the host. |
| `qt.qpa.plugin: could not load the Qt platform plugin "xcb"` | Missing Qt system library. Diagnose with `QT_DEBUG_PLUGINS=1 python roi_editor.py` and add the library to the Dockerfile's apt list. |
| Container exits instantly, *Operation not permitted* | seccomp. Diagnose with `--security-opt seccomp=unconfined`; the host's backported `libseccomp2 2.5.1` normally makes this a non-issue. |
| `errors pretty printing info` from `docker info` | Cosmetic client/daemon version mismatch. Use `docker info --format '…'`. |
| `test_parity.py` fails | Expected on a fresh clone — needs the git-ignored `reference.mat`. See §7. |

---

## Quick reference

```bash
# --- host ---
mountpoint -q /srv/containers && echo OK
git clone https://github.com/LorenzoFruzzetti/letizia_refactoring.git ~/projects/letizia
cd ~/projects/letizia
docker build -f docker/Dockerfile -t letizia:dev \
    --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
bash docker/run.sh

# --- inside the container ---
tmux new -s agent
claude                                   # first run: paste the login URL into any browser
code tunnel --accept-server-license-terms --name letizia-conf    # optional VS Code
python -m pytest tests/ -q
python examples/run_example.py
```
