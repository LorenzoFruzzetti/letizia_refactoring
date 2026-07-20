# Merging Plan — one Python framework for both wide-field pipelines

**Status: Phases 0–7 implemented.** The cortical pipeline runs end to end, in
memory and streaming, from the CLI and the API; the generic cohort/figure layer
and a worked study script are in place; and the agent-facing `LIBRARY.md` (plus a
human-facing `GUIDE.md`) is written. Phase 8 (cortical MATLAB parity) remains
deferred by design (P4).

| Phase | State | Where |
|-------|-------|-------|
| 0 — freeze the invariants | done | `tests/test_efficiency_invariants.py` (15 tests, I1–I11); RAM-slope assert in `benchmarks/benchmark_scaling.py` |
| 1 — generalise geometry | done (+ extended, see deviation 0) | `src/wfci/atlases.py`; `tests/test_atlas_transcription.py`, `test_atlas_generality.py`, `test_atlas_files.py`, `test_roi_bounds.py` |
| 2 — mask stage | done | `src/wfci/mask.py`; `tests/test_mask.py` |
| 3 — GSR, in-memory | done | `src/wfci/gsr.py`; `tests/test_gsr.py` |
| 4 — profiles + wiring | done | `src/wfci/profiles.py`, `pipeline.py`, `run_pipeline.py`; `tests/test_cli.py` |
| 5 — GSR under streaming | done | `src/wfci/streaming.py`; `tests/test_streaming_gsr.py` |
| 6 — cohort layer + study script | done | `src/wfci/cohort.py`, `significance.py`; `experiments/healthy_vs_disease_day4.py`; `tests/test_cohort.py`, `test_significance.py` |
| 7 — `LIBRARY.md` | done | `LIBRARY.md` (agent-facing, API generated from real signatures) + `GUIDE.md` (human-facing script tour); cross-linked from README/REFERENCE/CLAUDE.md |
| 8 — cortical MATLAB parity | deferred by design (P4) | see §7 below |

**Acceptance results.**

- **P1 held, measured not asserted:** the README's documented commands produce
  **byte-identical** output to the pre-merge code (`max|new − old| = 0` for
  `temp_roi`, `R`, `R_mean`, `averaged_traces`, both in-memory and streaming), and
  the MATLAB parity test is unchanged (`dff_stack` diff `0`, exact).
- **Phase 3:** vectorised OLS ≡ per-pixel `lstsq` at **8e-15**.
- **Phase 5:** streaming+GSR ≡ in-memory+GSR at **9e-15** (4 ROIs) / **1.8e-14**
  (22 ROIs); decode count still exactly 2N per channel; peak memory grew 24 KB
  over a 4× longer recording (budget 983 KB).
- **Phase 0 guard proven to bite:** each of five plausible regressions (cached
  `open()`, decoding in `tiff_frame_count`, reading every image in the split,
  dropping path coercion, an extra pass) fails the guard — while the pre-existing
  test suite stays green on all of them, which is exactly the blind spot §3
  predicted.
- **Phase 6:** `cohort.mean()`/`DIFF` match a plain-numpy hand computation; the
  worked study renders its three figures headless; and **P2 is a test** — a grep
  for `sani|pd_|macchi|day4|sex` over `src/wfci/` returns nothing, so no study
  knowledge can leak into the library without a red suite. The step-5 female-split
  discrepancy is a one-line selection in the study script, not a library concern
  (as designed).

**Deviations from this plan, and why** (all deliberate; see the phase notes):

0. **An atlas is a self-describing `Atlas`, not a bare dict, and can live in a
   file.** §4.1 has `atlases.py` hold "box dicts + labels". It holds
   :class:`Atlas` objects instead — a read-only `Mapping` (so a drop-in for the
   dicts) that also carries `grid` (the FOV the offsets were drawn for) and
   `source` (provenance) — plus `load_atlas`/`save_atlas` for YAML/JSON, and a
   `--atlas PATH` flag. **Why:** ROI boxes are not anatomy, they are anatomy
   projected through one optical setup, so they silently stop being valid on a
   different rig. Two failures were unguarded and both produced normal-looking
   numbers: a box running off the left edge became a *negative* NumPy index and
   averaged the **opposite hemisphere** (MATLAB raises here — the port was more
   permissive than its source); and an atlas drawn for another FOV whose boxes all
   still fit was wrong by a scale factor with nothing to detect it. `grid` +
   `wfci.roi.box_slices_for` close both. The file support answers the same concern
   from the other side: geometry is experimental design, so a study can own it
   without editing the library. See `tests/test_roi_bounds.py`,
   `tests/test_atlas_files.py`.

1. **`labels` is derived, not stored.** §4.2 gives `Profile` a `labels: list[str]`
   field and §5 Phase 1 gives one to `ROIConfig`. Both instead expose `labels` as
   a **property** over the atlas dict's key order. A second list is free to
   disagree with the boxes it names (wrong length, stale order), and a
   mislabelled-but-valid correlation matrix is the one error nothing downstream
   can detect. One source of truth instead.
2. **`io.py` was touched** — a non-goal in §8. §4.3 requires an explicit
   `channel_order`, and the only place the odd/even split exists is
   `interleaved_channel_files`. Implementing the flag anywhere else would
   duplicate that split and invite exactly the drift I6 exists to prevent. The
   change is one optional parameter defaulting to today's behaviour; every
   invariant still holds (and an explicit order now reads **zero** images instead
   of two).
3. **`global_signal` ignores `inf`, not just `NaN`.** MATLAB's `nanmean` +
   `~isnan` both propagate `inf`, and ΔF/F divides by the emo channel, so a zero
   there yields one. The MATLAB would feed that straight to `fitlm`; we exclude
   non-finite values instead. Identical on clean data, no silent garbage on dirty
   data. (P4: the general option over the MATLAB's accident.)
4. **Channel order defaults to `auto` on every profile**, not to the MATLAB's
   positional convention. Brightness-based identification is what this package
   already did (P1) and is more robust than position — which is the very
   channel-swap risk §4.3 wants gone. `gcamp_first`/`emo_first` are available to
   force it.

---

## Original plan (as proposed)

**Goal.** Extend the existing `wfci` package so that it runs **both** MATLAB
pipelines — the cerebellar one in [matlab/](matlab/) (already ported, 4 ROIs, no
GSR) and the cortical one in [Antea_scripts/](Antea_scripts/) (22 ROIs, brain
mask, global signal regression, group/NBS layer) — without regressing the
current pipeline's numerics **or its import efficiency**.

**Verdict: yes, and the gap is smaller than it looks.** See
[Why this is feasible](#2-why-this-is-feasible). The two genuinely missing
capabilities are the **mask + GSR stage** and a **generic group-statistics
layer**; almost everything else is already generic or is a default value.

Background reading: [README.md](README.md) (the `wfci` port),
[Antea_Scripts_README.md](Antea_Scripts_README.md) (the cortical scripts and how
the two pipelines differ), [REFERENCE.md](REFERENCE.md).

---

## 1. Design principles

These four principles decide every open question below. They are the plan's
constitution — if a later phase conflicts with one, the phase is wrong.

### P1 — The existing Python implementation is the reference

`wfci` as it stands today is the ground truth, not the MATLAB it came from. Its
outputs must not change, at any phase, for any reason. The cerebellar path is
*already* MATLAB-validated to machine precision, so this costs nothing there.
Where MATLAB fidelity and the current Python behaviour ever conflict, **the
Python wins** and the divergence gets documented.

Practical consequence: every phase ends with the existing test suite green and
[README.md](README.md#entrypoints)'s commands producing byte-identical output.

### P2 — The library is general; experiments live in scripts

Nothing inside `src/wfci/` may know about a cohort, a group name, an animal, a
sex, a day, or a disease model. The library provides **mechanism**; a per-study
script in `experiments/` provides **policy**.

This is what resolves the step-5 female-split question and its whole family:
`SANI`/`PD`, the male/female splits, which animal belongs to which group, and the
colour thresholds in the NBS figures are all **experiment-specific**. They become
arguments to generic primitives, written in a study script the user owns and
edits. The library never encodes them, so it never has to be forked or patched
when the next experiment has different groups.

The test for whether something belongs in the library: *would the next study with
different animals still want this, unchanged?* If no, it goes in `experiments/`.

### P3 — The efficiency invariants are a contract

The `FrameSource` design and the constant-memory guarantees ([§3](#3-invariants--the-import-efficiency-work-to-preserve))
are load-bearing, enforced by tests from Phase 0 onward, and are not negotiable
against convenience later.

### P4 — MATLAB compatibility is an option, never a constraint

For the cortical pipeline there is no MATLAB reference and **we are not going to
block on producing one**. The transcription of the box coordinates and the GSR
maths is **assumed correct** for now (it is a direct reading of scripts (1)–(4)).

Where a defensible choice differs from what the MATLAB happens to do, the
library takes the general option and exposes the MATLAB behaviour behind an
explicit flag. A future, optional phase ([Phase 8](#phase-8--deferred-optional-cortical-matlab-parity))
can add the reference and prove the match — designed for now, not built now.

---

## 2. Why this is feasible

Two findings drive the whole plan.

### 2.1 `wfci` is already anatomy-agnostic

The cerebellum lives in **defaults and docstrings, not in structure**:

- [config.py:44-51](src/wfci/config.py#L44-L51) — the four `Verme`/`Laterale`
  boxes are a `default_factory` on a plain field, i.e. an **argument**, not a
  constant.
- [roi.py:41-42](src/wfci/roi.py#L41-L42) and
  [streaming.py:147-151](src/wfci/streaming.py#L147-L151) size their outputs
  from `list(cfg.boxes.keys())` / `len(names)`; `functional_connectivity` reads
  `n_reg` from `temp_roi.shape[1]`. **Pass 22 boxes and they produce
  `[time, 22, trial]` and a 22×22 `R` with no code change.** Every `4` in those
  files is in a comment.
- [correction.py](src/wfci/correction.py), [resize.py](src/wfci/resize.py) and
  [io.py](src/wfci/io.py) contain no anatomical assumptions at all.

So "generalise to 22 ROIs" is a **config-and-labels task**, not a refactor.

### 2.2 GSR does not break constant-memory streaming — verified

This was the real architectural risk. GSR looks fundamentally anti-streaming: it
regresses **each pixel's whole time-series** against the global signal, which
naively means holding `[y, x, time]` resident — destroying the property that
makes [streaming.py](src/wfci/streaming.py) worth having.

It doesn't, for two reasons:

1. **Per-pixel OLS needs only sufficient statistics**, all accumulable one frame
   at a time: `N`, `Σg`, `Σg²` (scalars) and `Σp`, `Σgp` (two `[y, x]` images).
   The global signal `g(t)` is a *spatial* mean of frame `t`, so it is known at
   frame `t` — no lookahead. Then
   `a = (N·Σgp − Σg·Σp) / (N·Σg² − Σg²)`, `b = (Σp − a·Σg) / N`.
2. **The ROI mean is linear and `g(t)` is a scalar per frame**, so the regressed
   ROI trace never needs the regressed *stack*:

   ```
   T_B(t) = mean_B( p(t) − a·g(t) − b )
          = mean_B(p(t)) − g(t)·mean_B(a) − mean_B(b)
   ```

   Every term is either accumulated per frame (`mean_B(p(t))`, `g(t)`) or
   computed once at the end from the two stat images (`mean_B(a)`, `mean_B(b)`).

**Verified numerically** against a dense per-pixel OLS reference in the MATLAB
order (mask → per-pixel fit → residual stack → ROI `nanmean`):

```
max |reference − sufficient-statistics| = 3.33e-16
```

i.e. float roundoff. So GSR fits the **existing two-pass structure** with a
constant-memory footprint of two extra `[y, x]` images and two `[time]` vectors.

**The one precondition** (see [§6.1](#61-gsr-the-static-nan-precondition)): the
set of invalid pixels must be **static over time**. Under mask-only NaNs it is —
confirmed in the same check.

---

## 3. Invariants — the import efficiency work to preserve

the list below was
reconstructed by reading [io.py](src/wfci/io.py),
[streaming.py](src/wfci/streaming.py) and
[run_pipeline.py](run_pipeline.py). **Please confirm nothing is missing before
Phase 0 starts** — everything here becomes a locked invariant with a test.

| # | Invariant | Where | Why it matters |
|---|-----------|-------|----------------|
| I1 | **Frame count without decoding pixels** — IFD headers only | [io.py:27-35](src/wfci/io.py#L27-L35) `tiff_frame_count` | Lets the baseline window be sized before any frame is read; cheap on multi-GB files |
| I2 | **Lazy per-frame decode** | [io.py:38-48](src/wfci/io.py#L38-L48) `iter_tiff_frames`, [io.py:87-95](src/wfci/io.py#L87-L95) `iter_folder_frames` | Never more than one frame resident |
| I3 | **`FrameSource` = count + re-iterable `open()`** | [io.py:51-74](src/wfci/io.py#L51-L74) | The abstraction that decouples **storage format** from **memory strategy**; two passes require a *fresh* iterator per pass |
| I4 | **Storage × memory axes stay orthogonal** — any `--source` runs streaming or not | [io.py:77-119](src/wfci/io.py#L77-L119), [run_pipeline.py:145-167](run_pipeline.py#L145-L167) | 4 working layouts; a merged framework must not collapse this back into a matrix of special cases |
| I5 | **Interleaved split without decoding the recording** — file-list split, only **2 images** read for the brightness decision | [io.py:159-209](src/wfci/io.py#L159-L209) `interleaved_channel_files` | A folder too large for RAM can still be split and streamed |
| I6 | **Same split feeds both** full-load and streaming | [io.py:212-234](src/wfci/io.py#L212-L234) + [run_pipeline.py:157-159](run_pipeline.py#L157-L159) | One code path, no divergence |
| I7 | **Path coercion / back-compat** — a bare path still works where a `FrameSource` is expected | [streaming.py:60-69](src/wfci/streaming.py#L60-L69) `_as_frame_source` | Older call style keeps working |
| I8 | **Debug limit stays lazy and re-iterable** — `islice` over `open()`, not a truncated load | [run_pipeline.py:104-113](run_pipeline.py#L104-L113) `_limited_source` | `--debug-max-frames N` reads only N frames off disk |
| I9 | **Debug full-load routes through the lazy sources** | [run_pipeline.py:121-129](run_pipeline.py#L121-L129) `_load_trials` | A debug in-memory run doesn't read the whole folder either |
| I10 | **Two passes, running sums, no stack** | [streaming.py:128-163](src/wfci/streaming.py#L128-L163) | The constant-memory guarantee itself |
| I11 | **Per-frame resize ≡ whole-stack resize** | [streaming.py:72-85](src/wfci/streaming.py#L72-L85) `_half_res_frames` | Why streaming can match the in-memory path exactly |

### Regression guard (Phase 0, before any change)

Invariants I1–I11 are currently protected by **convention only** — a plausible
refactor could quietly turn `FrameSource.open()` into a cached list and every
existing test would still pass, only slower and fatter. So Phase 0 adds:

- **`tests/test_efficiency_invariants.py`** — a `CountingFrameSource` wrapping a
  real source and counting `decode` calls and `open()` calls. Asserts:
  - a streaming run over N frames decodes **exactly 2N** frames (two passes) and
    calls `open()` twice per channel — not 1 (cached) and not 3;
  - `--debug-max-frames k` decodes **exactly 2k**, not 2N;
  - `interleaved_channel_files` on a folder of N images decodes **exactly 2**;
  - `tiff_frame_count` decodes **0**.
- **a peak-RSS assertion** in the existing benchmark: streaming peak RAM must stay
  **independent of frame count** (already measured per-layout in
  `benchmarks/benchmark_modalities.py` — assert the slope, don't just print it).

**These tests are written and green *before* Phase 1 touches anything.** They are
the contract that the merge cannot silently cost you the import work (P3).

---

## 4. Target architecture

Keep **one package**, `wfci`, extended along the axis it already has (config +
orthogonal knobs). Do **not** fork a second package: the two pipelines share
step 1, the resize, the whole I/O layer, and the correlation step — a fork
duplicates all of it and guarantees drift.

The organising idea: today `run_pipeline` is **correction → ROI → connectivity**,
a fixed chain. Make it **correction → [mask] → [GSR] → ROI → connectivity**,
where the bracketed stages are **optional and configured**, and a `Profile`
bundles the per-pipeline defaults.

### 4.1 The library / experiment boundary (P2)

```
src/wfci/                 LIBRARY — general mechanism, no study knowledge
├── io.py             UNCHANGED  (I1-I9 live here; the merge must not touch it)
├── resize.py         UNCHANGED
├── correction.py     UNCHANGED  (step 1 is already common to both pipelines)
├── roi.py            unchanged logic; docstrings de-cerebellumed
├── config.py         + labels on ROIConfig; + Box presets
├── atlases.py        NEW  CEREBELLUM_4 / CORTEX_22 box dicts + labels (presets, overridable)
├── mask.py           NEW  load / resize / apply brain mask -> NaN
├── gsr.py            NEW  vectorised per-pixel OLS; in-memory + streaming stats
├── profiles.py       NEW  Profile: trim, baseline, corr window, atlas, stages
├── pipeline.py       + optional mask/GSR stages; run_profile(...)
├── streaming.py      + optional GSR via sufficient statistics (still 2 passes)
├── visualize.py      + label-aware overlay; frame_index off the profile
├── cohort.py         NEW  GENERIC: stack results, mean over a selection, difference
└── significance.py   NEW  GENERIC: mask a matrix by an adjacency, network + bar plots

experiments/              POLICY — one script per study, user-owned, freely edited
├── README.md
└── ts65dn_pd_day4.py     e.g. cohort table, group/sex splits, DIFF, figure calls
```

**What this buys.** `cohort.py` never hears the words `SANI`, `PD`, `day4` or
`sex`. It offers primitives like:

```python
results = load_results("outputs/*.npz")          # -> list of per-animal R_mean + metadata
table   = CohortTable(results)                    # any columns the study script defines
a = table.select(group="healthy").mean()          # generic selection + mean
b = table.select(group="disease").mean()
diff = a - b
```

The study script decides that `group` exists, who is in it, and whether to split
by `sex`. The MATLAB's `cat(3, mean_R_PV_F_MACCHI_DX_day4, ...)` — experimental
design encoded in **variable names** — becomes a table the script fills in.

The step-5 female-split discrepancy dissolves under this: there is no library
behaviour to port correctly or incorrectly. The study script does
`table.select(group="healthy", sex="F").mean()`, and if the original grouping was
wrong, that is a one-line fix in a script the user owns, not a library bug. (Still
worth raising with the original author — but it no longer blocks anything.)

Likewise `significance.py` takes an adjacency matrix and a value matrix and draws
the network; the colour thresholds (`> 0.6` → dark red, etc.) and the node
positions are **arguments**, defaulting to something sensible, set per study.

### 4.2 `Profile` — where the *pipeline* differences live

```python
@dataclass(frozen=True)
class Profile:
    name: str
    atlas: dict[str, Box]        # CEREBELLUM_4 | CORTEX_22 | your own
    labels: list[str]            # display/plot order
    trim: int                    # 20 cerebellar | 0 cortical
    baseline: slice              # slice(None) | slice(0, 278)
    corr_window: slice           # slice(None) | slice(279, 300)
    use_mask: bool               # False | True
    gsr: GSRConfig | None        # None = off
    overlay_frame: int           # step-2 default frame

CEREBELLAR_RS   = Profile("cerebellar_rs",   CEREBELLUM_4, ..., trim=20, baseline=slice(None),  corr_window=slice(None),    use_mask=False, gsr=None,        overlay_frame=302)
CEREBELLAR_STIM = Profile("cerebellar_stim", CEREBELLUM_4, ..., trim=20, baseline=slice(0,278), corr_window=slice(279,300), use_mask=False, gsr=None,        overlay_frame=302)
CORTICAL_GSR    = Profile("cortical_gsr",    CORTEX_22,    ..., trim=0,  baseline=slice(None),  corr_window=slice(None),    use_mask=True,  gsr=GSRConfig(), overlay_frame=10)
```

Profiles are **presets, not a closed set** — a study script constructs its own
`Profile` (or `dataclasses.replace`s a preset) without touching the library.

Under P1, every existing call is exactly a profile call:
`run_resting_state(trials, cfg)` ≡ `run_profile(trials, cfg, CEREBELLAR_RS)`,
with the old signature kept as a wrapper.

### 4.3 What stays out

- **NBS itself.** It is an external MATLAB toolbox with a GUI. `significance.py`
  **ingests** an adjacency matrix (exported from NBS, or from a future Python
  permutation test) and does the masking + figures. Re-implementing the
  network-based statistic is a separate project — explicitly a non-goal.
- **Trial discovery from `dir` order.** The Python side deliberately makes the
  caller list trials explicitly, which sidesteps the fragile
  `i = 3:2:end` / `i = 5:4:end` stride logic *and* the silent channel-swap risk
  described in [Antea_Scripts_README.md](Antea_Scripts_README.md#notes--caveats).
  Keep it that way. The **channel order differs between pipelines** (cortical:
  emo first; cerebellar: gcamp first), so `_load_trials` gains an explicit
  `channel_order` argument rather than a guess.

---

## 5. Phased plan

Each phase is independently shippable and ends green. **No phase may make the
previous phase's tests fail** — that is the whole point of the ordering, and the
mechanism behind P1.

### Phase 0 — freeze the invariants
- Write `tests/test_efficiency_invariants.py` (decode-count + open-count asserts, §3).
- Add the peak-RAM-vs-frame-count slope assert to the benchmark.
- **Acceptance:** new tests green on today's code; `pytest tests/ -s -v` still green.
- **Risk if skipped:** the import efficiency work becomes unenforced and will rot.

### Phase 1 — generalise geometry (no behaviour change)
- `atlases.py`: `CEREBELLUM_4` (copy of today's default) + `CORTEX_22`
  (transcribed from `(3)_FOV128-128_ROIposition.txt`, cross-checked box-for-box
  against `(4)_FOV128x128_corr_SCRIPT.txt` — they are duplicated in the MATLAB
  and **must be verified identical** during transcription; the cerebellar pair
  has already drifted, so assume nothing).
- `ROIConfig` gains `labels`; default `boxes` becomes `CEREBELLUM_4`.
- De-cerebellum the docstrings in `roi.py` / `__init__.py`.
- **Acceptance (P1):** existing parity + streaming tests **byte-identical**;
  a new test builds a 22-box `ROIConfig` and gets `[time, 22, trial]` out of both
  the in-memory and streaming paths.

### Phase 2 — mask stage
- `mask.py`: `load_mask`, `resize_mask` (reuse `imresize_box`), `apply_mask`
  (mask == 0 → NaN across time and trials).
- Loading is **explicit path in, array out** — no `uiopen` equivalent.
- **Acceptance:** unit test on a synthetic mask; masked pixels are NaN, ROI
  `nanmean` ignores them.

### Phase 3 — GSR, in-memory, vectorised
- `gsr.py`: `global_signal(dff)` and `regress_global(dff, cfg)` using the
  **closed-form vectorised OLS** (identical to `fitlm` with an intercept, which
  is plain OLS) instead of 16 384 `fitlm` calls per trial. Correctness path and a
  large speedup at once.
- `GSRConfig(nan_policy="drop_pixel")` — P4 in action. `"drop_pixel"` reproduces
  MATLAB's `if ~isnan(data(row,col,:))` (any NaN in time ⇒ pixel dropped
  entirely) and is the default because it is deterministic and matches the source
  script; `"per_frame"` is reserved as the future general alternative. The choice
  is an **option from day one**, so no future phase has to change behaviour under
  anyone's feet.
- **Acceptance:** vectorised OLS ≡ per-pixel `lstsq` reference to ~1e-16 (this is
  the check already run to validate §2.2). **No MATLAB required** (P4).

### Phase 4 — profiles and stage wiring
- `profiles.py` + `run_profile(trials, cfg, profile)`.
- `run_pipeline` grows optional `mask=` / `gsr=` stages between
  `build_dff_stack` and `extract_roi_timeseries` — **the natural insertion point,
  already the right order in [pipeline.py:41-45](src/wfci/pipeline.py#L41-L45)**.
- `run_resting_state` / `run_stimulated` / `run_streaming_*` become thin wrappers
  over profiles, keeping **exactly** their current signatures and defaults (P1).
- `run_pipeline.py` CLI: add `--profile {cerebellar_rs,cerebellar_stim,cortical_gsr}`,
  `--mask PATH`, `--channel-order {gcamp_first,emo_first}`. `--mode` stays as a
  deprecated alias mapping onto the cerebellar profiles.
- **Acceptance (P1):** every command in [README.md](README.md#entrypoints)
  produces identical output to before; Phase 0 tests still green.

### Phase 5 — GSR under streaming (the sufficient-statistics path)
- Extend `stream_trial_roi` pass 2 to accumulate `Σp`, `Σgp` (two `[y, x]`
  images), `Σg`, `Σg²`, `N`, plus the `[time]` vectors `g(t)` and the raw ROI
  means; then apply `T_B(t) = raw_B(t) − g(t)·mean_B(a) − mean_B(b)`.
- **Still two passes. Still constant memory.** Phase 0's decode-count test is the
  proof — it must still read exactly 2N.
- Implement the static-NaN precondition check of [§6.1](#61-gsr-the-static-nan-precondition),
  with an explicit three-pass fallback.
- **Acceptance:** streaming+GSR ≡ in-memory+GSR (Phase 3) to ~1e-13; decode count
  unchanged at 2N; peak RAM still flat in frame count.

### Phase 6 — generic cohort layer + one example study script
- `cohort.py` and `significance.py` as described in [§4.1](#41-the-library--experiment-boundary-p2)
  — **primitives only**, no study knowledge.
- `experiments/ts65dn_pd_day4.py`: a worked study script that defines the cohort
  table, the group/sex selections, `DIFF`, and the figure calls. This is the file
  the MATLAB steps 5–6 actually correspond to, and the place their
  experiment-specific choices belong.
- The known step-6 MATLAB quirks are not ported into the library: node sizes are
  a per-node **vector** (the MATLAB divides a matrix by a row vector), and edge
  colours are **local**, not stale globals reused between the HYPER and HYPO
  cells.
- **Acceptance:** on a synthetic cohort, group means and `DIFF` match a hand
  computation; figures render headless (Agg); `grep -riE "sani|pd_|macchi|day4|sex"
  src/wfci/` returns **nothing** (P2, enforceable as a test).

### Phase 7 — the agent-facing library document
- **`LIBRARY.md`** — the deliverable the user asked for: a single, detailed,
  self-contained explanation of the library that can be handed to an agent (or a
  new person) as its entire working context. Contents:
  - the mental model: two orthogonal knobs (source × streaming) + a profile;
  - the **full public API**, every exported name with signature, shapes, dtypes,
    and units — generated from the real signatures, not hand-copied;
  - **array-shape conventions** (`[y, x, time, trial]`, MATLAB order) stated once,
    prominently, because it is the single easiest thing to get wrong;
  - **worked recipes**: run the cerebellar RS pipeline; run the cortical GSR
    pipeline; define a custom atlas; define a custom profile; write a study script;
  - the **do-not-break list** (I1–I11) with the reasoning, so an agent editing the
    package knows which "simplifications" are actually regressions;
  - the library/experiment boundary (P2) stated as a rule an agent can apply.
- **Naming.** Proposed as `LIBRARY.md`, not `README.md`: per [CLAUDE.md](CLAUDE.md),
  `README.md` is the project entrypoint doc (commands, I/O, directory map) and
  `REFERENCE.md` is the canonical technical map — a third file called README would
  collide with both. `LIBRARY.md` is cross-linked from `README.md`, `REFERENCE.md`
  and `CLAUDE.md`. **Cheap to rename if you'd rather it be the README** — say so.
- Also in this phase: update `README.md` (profiles, mask/GSR knobs, new flags),
  `REFERENCE.md` (new modules), and fix the "Practical consequence" paragraph in
  [Antea_Scripts_README.md](Antea_Scripts_README.md) — it currently overstates the
  porting work, since the 22-ROI layout is a config argument, not a code change.
- Extend `benchmarks/benchmark_modalities.py` with the cortical profile so the GSR
  path is covered by the RAM/time report.

### Phase 8 — deferred, optional: cortical MATLAB parity
**Not scheduled. Designed for, not built** (P4). If the cortical numbers are ever
questioned, this is the escape hatch:

- `tests/matlab_reference/gen_reference_antea.m` — transcribe the *maths* of
  scripts (1)–(4) into a headless `.m` (deterministic synthetic two-channel data +
  synthetic mask; MATLAB's own `imresize`, `fitlm`, `nanmean`, `corr`), saving
  inputs and outputs to `reference_antea.mat`, exactly as the cerebellar
  `gen_reference.m` already does.
- `tests/test_parity_antea.py` asserts the Python matches.
- `GSRConfig(nan_policy=...)` is the knob that would absorb any discrepancy found.

**What we accept by deferring:** the `CORTEX_22` box coordinates and the GSR
transcription are **assumed correct** from a careful reading of the scripts, not
proven. Phase 3's `lstsq` check validates the *vectorisation*, not the
*transcription*. `LIBRARY.md` must say so plainly rather than imply the cortical
path carries the same machine-precision pedigree as the cerebellar one.

---

## 6. Known hard points

### 6.1 GSR: the static-NaN precondition

The two-pass identity in [§2.2](#22-gsr-does-not-break-constant-memory-streaming--verified)
computes each frame's ROI mean **during** pass 2, but `nan_policy="drop_pixel"`
decides which pixels are valid using the **whole** time-series. If a pixel is
valid in some frames and NaN in others, streaming would include it in the frames
where it happens to be valid, while the in-memory path excludes it from **all**
frames. The results would then differ — subtly, and only on real data.

This only bites if NaNs can appear **mid-recording** (e.g. a zero in `emo(t)`
making the ratio non-finite). Under mask-only NaNs the invalid set is static and
the identity is exact — confirmed in the verification check.

**Plan:** in pass 1, flag zero/non-finite pixels in the baseline mean images; in
pass 2, accumulate a per-pixel NaN counter (one more `[y, x]` image). If the
final counter shows any pixel that is *partially* NaN, **raise** and tell the
caller to re-run with `--no-streaming` (or an opt-in third pass). Never silently
return numbers that differ from the in-memory path — P1 means the in-memory
result is the one that is right. Both branches get a test with a synthetic
partial-NaN pixel.

### 6.2 Cost of GSR

Per-pixel `fitlm` in MATLAB is 128×128 = 16 384 model fits per trial and is by
far the slowest step in the Antea pipeline. The vectorised closed form removes it
entirely (a handful of array ops). Expect the Python cortical run to be
**dramatically** faster than the MATLAB original — which is a reason to
double-check parity carefully rather than to celebrate early.

---

## 7. Open decisions

Most of the earlier open questions are now settled by [§1](#1-design-principles):
the female split and friends move to `experiments/` (P2); MATLAB parity is
deferred behind an option (P4); defaults never change (P1). What remains:

1. ~~**Is the invariant list in [§3](#3-invariants--the-import-efficiency-work-to-preserve) complete?**~~
   **Settled:** confirmed complete before Phase 0. I1–I11 are now enforced by
   `tests/test_efficiency_invariants.py`.
2. ~~**`LIBRARY.md` vs `README.md`**~~ **Settled:** `LIBRARY.md`, cross-linked —
   but Phase 7 is not implemented, so the file does not exist yet.
3. **Should the cerebellar pipeline gain optional GSR?** The framework makes it a
   one-flag change. A *scientific* decision, not a technical one; defaults stay
   off, preserving today's behaviour exactly (P1).
4. **Package name.** `wfci`'s docstring says "cerebellar". Recommendation: keep
   the name (no import churn), fix the docstring.

---

## 8. Non-goals

- Re-implementing the NBS network-based statistic in Python.
- Producing a cortical MATLAB reference now (deferred — Phase 8, P4).
- Porting the MATLAB `dir`-order trial discovery (stride + positional channel
  assignment) — explicitly replaced by explicit trial lists.
- Encoding any study's cohort, groups, or sex splits in the library (P2).
- Reproducing the MATLAB figures pixel-for-pixel; the plots are re-implemented,
  not cloned.
- Touching [io.py](src/wfci/io.py). The merge needs nothing from it that is not
  already there — which is the strongest evidence that the `FrameSource` design
  was the right abstraction.
