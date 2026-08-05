# Container Stick — Reference

Working notes for the Lexar 128 GB stick used as a Docker store on `IN-CONFERENCE`.
Written 2026-08-05. Keep a copy on the stick itself (`/srv/containers/README.md`).

---

## 1. What this is

The host machine runs **Ubuntu 16.04.7 (xenial)** with **glibc 2.23**, which is far
too old for most current toolchains. Rather than upgrade or break the host, the
project runs inside a **Docker container based on Ubuntu 22.04 (glibc 2.35)**.

The stick holds the Docker **image store** — the layers that make up those
container filesystems. It does *not* hold the project source (see §6).

Host and container run **at the same time**. A container is just a set of processes
on the host with isolated namespaces; there is no VM, no boot, nothing to switch
between.

---

## 2. Hardware and layout

| Item | Value |
|---|---|
| Device | Lexar USB Flash Drive, 128 GB nominal / **116.1 GiB** actual |
| Serial | `A083053000000F40` |
| Partition table | GPT, single partition |
| Filesystem | ext4, label `CONTAINERS`, reserved blocks set to 0 (`-m 0`) |
| Mount point | `/srv/containers` |
| Docker data-root | `/srv/containers/docker` |
| Measured write speed | **40 MB/s on USB 2.0** (port-limited, see §7) |

Device letter is usually `/dev/sdc1` but **is not stable** — it depends on what else
is plugged in. Always identify by label or UUID, never by letter.

> **Fill this in:** run `sudo blkid /dev/sdc1` and record the UUID here →
> `UUID=________________________________________`

### Other disks on the host — do not confuse them

| Device | What it is |
|---|---|
| `/dev/sda` | 447 GB internal SATA SSD (ADATA SU630), the root filesystem |
| `/dev/sdb` | **3.7 TB external NTFS drive, label `EXTERNAL_USB`** — someone else's data, leave alone |
| `/dev/sdc` | This stick |

---

## 3. How it mounts

An entry in `/etc/fstab` mounts it automatically by UUID:

```
UUID=<uuid>  /srv/containers  ext4  defaults,noatime,nofail,x-systemd.device-timeout=10  0  2
```

- `nofail` — the machine still boots if the stick is absent. Without this you get
  dropped to an emergency shell.
- `noatime` — no access-time writes, which matters on flash.
- Mounting by UUID means the device letter can change freely.

Owned by `lorenzo:lorenzo`, so no `sudo` is needed to write to it.

### Checking it is present

```bash
df -h /srv/containers
```

Correct output shows `/dev/sdcN` and ~115G. **If it shows `/dev/sda1` mounted on `/`,
the stick is not mounted** — that path is then just an empty directory on the
internal disk, and Docker will silently fill the root disk instead.

### Mounting manually if needed

```bash
sudo mount /dev/disk/by-label/CONTAINERS /srv/containers
```

---

## 4. Accessing Docker

`lorenzo` is in the `docker` group, so no `sudo` is required. (After a fresh
`usermod -aG`, you need a new login session or `newgrp docker`.)

```bash
docker ps                # running containers
docker images            # local images
docker system df         # space used, broken down
```

Daemon config lives in `/etc/docker/daemon.json`:

```json
{
  "data-root": "/srv/containers/docker",
  "storage-driver": "overlay2"
}
```

After editing it: `sudo systemctl restart docker`.

Verify where the daemon is actually writing:

```bash
docker info --format '{{.DockerRootDir}} | {{.Driver}} | {{.ServerVersion}}'
```

Expected: `/srv/containers/docker | overlay2 | 19.03.12`

> Note: plain `docker info` prints `errors pretty printing info`. That is a cosmetic
> bug from the client (20.10.7) being newer than the daemon (19.03.12). Use
> `--format` as above to get the values.

---

## 5. What you get in the container that the host cannot do

Verified working:

```
$ docker run --rm ubuntu:22.04 sh -c 'ldd --version | head -1'
ldd (Ubuntu GLIBC 2.35-0ubuntu3.14) 2.35
```

| | Host | Container |
|---|---|---|
| Distro | Ubuntu 16.04 (EOL 2021) | Ubuntu 22.04 |
| glibc | 2.23 | **2.35** |
| GCC | 5.4 | 11.x (12/13 available) |
| Python | 3.5 | 3.10 |
| apt repos | `old-releases.ubuntu.com` only | fully live |

So: modern compilers, modern C++ standard library, current Python, working package
management, and anything requiring glibc ≥ 2.24 (which is most of what has been
released in the last decade).

The seccomp concern that normally breaks glibc ≥ 2.34 on old hosts **does not apply
here** — xenial received a backported `libseccomp2 2.5.1`, which understands the
`clone3` and `faccessat2` syscalls. No `--security-opt seccomp=unconfined` needed.

### What it still cannot do

Containers share the **host kernel**. Nothing above changes that.

- Kernel stays **4.4.0-210**. No io_uring, no modern BPF, no recent nftables.
- **cgroup v1 only.** For this reason do not go past 22.04 — Ubuntu 24.04's
  systemd 255 treats cgroup v1 as deprecated. 22.04 is the sensible ceiling.
- No kernel modules, no host driver changes from inside a container.
- GPU/CUDA would need extra work (nvidia-container-toolkit) and is not set up.

If the project genuinely needs a newer *kernel*, a container is the wrong tool — the
machine has 32 cores and 125 GB RAM and `/dev/kvm` is worth checking for a VM instead.

---

## 6. Porting a project onto this — the intended pattern

**Keep the source and all caches on the internal disk. Bind-mount them in.**

The stick is USB 2 at 40 MB/s, and flash handles many small writes badly. Image
layers are written once and read often, which is fine. Build output and package
caches are write-heavy and must not live there.

```bash
docker run -it --name proj \
  -v /home/lorenzo/projects/myproject:/work \
  -v /home/lorenzo/.cache/pip:/root/.cache/pip \
  -w /work \
  ubuntu:22.04 bash
```

Edit files on the host with your normal editor; compile and run inside the container.
Both sides see the same files instantly.

Useful additions as needed:

| Flag | Effect |
|---|---|
| `--user $(id -u):$(id -g)` | files created in `/work` are owned by you, not root |
| `-p 8080:8080` | expose a port to the host |
| `--network host` | container shares the host network stack |
| `--rm` | delete the container on exit (image is kept) |

Once the dependency list is known, replace ad-hoc `apt install` with a Dockerfile so
the environment is reproducible:

```dockerfile
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
```

```bash
docker build -t myproject:dev .
docker run -it --rm -v /home/lorenzo/projects/myproject:/work myproject:dev bash
```

---

## 7. Operational notes

**Do not unplug while containers are running.** Stop everything first, then unmount.
Yanking it mid-write corrupts the overlay2 store.

```bash
docker stop $(docker ps -q)
sudo systemctl stop docker docker.socket
sudo umount /srv/containers
```

**Space.** `-m 0` removed the usual 5% root reserve, so there is no accidental buffer
at the top end. Prune periodically:

```bash
docker system df           # check first
docker system prune -af    # removes dangling images and build cache
docker builder prune -af   # build cache only
```

**USB 3 upgrade — worth doing.** The stick is currently on a USB 2.0 port (`dmesg`
reports `high-speed`, not `SuperSpeed`), capping it at 40 MB/s. Bus 4 on this machine
is a 6-port SuperSpeed hub. Moving to a blue port should give roughly 3×. **No
reconfiguration is required** — fstab mounts by UUID. Confirm with:

```bash
dmesg | tail -5     # want "new SuperSpeed USB device"
```

**Reclaiming root disk space.** The old store at `/var/lib/docker` (16 GB) is still
present but unused; the daemon now writes to the stick. The root disk was 87% full.
Once this setup has proven itself over a few days:

```bash
sudo rm -rf /var/lib/docker
```

**Host apt repos** were repointed from `it.archive.ubuntu.com` to
`old-releases.ubuntu.com`, since 16.04 went EOL in April 2021. Backup at
`/etc/apt/sources.list.bak`.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `df -h /srv/containers` shows `/dev/sda1 on /` | Stick not mounted. `sudo mount -a`, then restart Docker. |
| Root disk filling up unexpectedly | Docker writing to `/var/lib/docker` — `daemon.json` not picked up. `sudo systemctl restart docker`, then check `docker info --format '{{.DockerRootDir}}'`. |
| `errors pretty printing info` | Cosmetic client/daemon version mismatch. Use `docker info --format '...'`. |
| `permission denied ... /var/run/docker.sock` | Not in the `docker` group in this session. `newgrp docker` or re-login. |
| Container exits instantly, `Operation not permitted` | seccomp. Diagnose with `--security-opt seccomp=unconfined`; if that fixes it, either upgrade the daemon or drop to `ubuntu:20.04` (glibc 2.31). |
| Boot hangs / emergency shell | Missing `nofail` in the fstab line, or a bad UUID. |
| Device letter changed | Expected. Irrelevant — everything goes by UUID/label. |

---

## 9. Rebuilding from scratch

If the stick dies, nothing of value is lost — the store is fully reproducible.
On a replacement, with `/dev/sdX` **carefully verified** by label, size and `dmesg`:

```bash
sudo wipefs -a /dev/sdX
sudo parted /dev/sdX -- mklabel gpt mkpart primary ext4 1MiB 100%
sudo partprobe /dev/sdX && sleep 2
sudo mkfs.ext4 -L CONTAINERS -m 0 /dev/sdX1
sudo blkid /dev/sdX1                      # copy UUID into /etc/fstab
sudo mkdir -p /srv/containers && sudo mount -a
sudo chown lorenzo:lorenzo /srv/containers
sudo systemctl restart docker
docker pull ubuntu:22.04
```

Then rebuild images from the Dockerfile — which is the reason to keep one.
