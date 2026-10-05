# Examples

## `run_example.py`

End-to-end resting-state run on the bundled sample data.

```bash
conda run -n letizia python examples/run_example.py
```

**Input:** `data/R11_00001.tif … R11_00007.tif` — 7 single-page 512×512 16-bit
TIFFs = 7 time frames of **one** channel.

**Output** (written to `examples/output/`):
- `example_results.npz` — `R_mean`, `temp_roi`, `averaged_traces`
- `roi_overlay.png` — step-2 style ROI-placement overlay on a ΔF/F frame

> **Note on the sample data.** The real pipeline needs *two* channels (GCaMP +
> hemodynamic `emo`) per trial. The sample is a single channel, so the example
> **synthesises** a second channel and a second trial deterministically, purely
> to exercise the full code path. Because both channels are linear functions of
> the same frames, the ROI traces are perfectly correlated and `R_mean` comes
> out as all ones — that is expected for this demo. Real dual-channel
> acquisitions produce a meaningful connectivity matrix.

## `nbs_synthetic_example.py`

Network-Based Statistic (`src/nbs/`, see [../NBS_ALGORITHM.md](../NBS_ALGORITHM.md))
on synthetic data. No input files are needed.

```bash
conda run --no-capture-output -n letizia python examples/nbs_synthetic_example.py
```

**Input:** generated in the script. Two groups of 12 animals, one 22x22 CORTEX_22
matrix each (r = 0.4 plus noise), and +0.25 Fisher-z on 5 connected M1L/M2L
edges in group A. All parameters are at the top of the file.

**Output:** printed components (size, FWER p, edges) at t > 2.5 / 3.1 / 3.5. The
three thresholds share one set of 5000 permutations through `rethreshold`. The
figure is written to `examples/output/nbs/nbs_synthetic_components.png`: the t map
masked to the significant component, one panel per threshold. The planted
component is found at every threshold (p ~ 0.0002-0.0014). At t > 2.5 it also
takes in one spurious edge, which is why NBS supports claims about the component,
never about a single edge. Runs in about 3 s.

## Numerical validation against MATLAB

The authoritative check that the Python port reproduces the MATLAB pipeline is
the parity test in [`../tests/test_parity.py`](../tests/test_parity.py), which
compares against MATLAB reference outputs generated from this same sample data.
See the repository [README](../README.md#validation-against-matlab).

## Cortex/emo additive pilot

From the repository root:

```powershell
conda run --no-capture-output -n letizia python analyze_cortex_activity_emo.py --config examples/cortex_activity_emo_pilot.json
conda run --no-capture-output -n letizia python compare_cortex_activity_conditions.py --output-root outputs/cortex_activity_emo_pilot --bootstrap 100
```

Requires the three selected raw recordings in `pixel_data` and their saved ROI
YAMLs. Writes only `outputs/cortex_activity_emo_pilot/`. See the
[run guide](../docs/CORTEX_ACTIVITY_EMO_USAGE.md) for all schemas and settings.
