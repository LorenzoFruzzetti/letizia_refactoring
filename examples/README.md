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

## Numerical validation against MATLAB

The authoritative check that the Python port reproduces the MATLAB pipeline is
the parity test in [`../tests/test_parity.py`](../tests/test_parity.py), which
compares against MATLAB reference outputs generated from this same sample data.
See the repository [README](../README.md#validation-against-matlab).
