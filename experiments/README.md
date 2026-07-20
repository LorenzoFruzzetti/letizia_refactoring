# experiments/ — study scripts (the policy layer)

This folder is **owned by the study, not the library**. It is where all the
experiment-specific knowledge lives that `wfci` deliberately refuses to hold
(MERGING_PLAN.md **P2**): which animals exist, which group and sex each belongs
to, how the cohort splits, which edges are significant, and how figures are
laid out and coloured.

The rule (MERGING_PLAN.md P2): *would the next study with different animals still
want this, unchanged?* If **no**, it belongs here. If **yes**, it belongs in
`src/wfci/`. The library provides mechanism (`wfci.cohort`, `wfci.significance`);
these scripts provide the experiment.

## Scripts

| Script | What it is | MATLAB origin |
|--------|-----------|---------------|
| [`healthy_vs_disease_day4.py`](healthy_vs_disease_day4.py) | A worked group-contrast study: cohort table → group means → `DIFF` → significance figures. | `Antea_scripts/(5)_matrici_e_figure.txt` + `(6)_visualizzazione_connettivita_sign.txt` |

## Running

```bash
conda run -n letizia python experiments/healthy_vs_disease_day4.py
```

**Expected input.** By default, none — the script builds a **synthetic** cohort
with a known injected group difference, so it runs and is testable without real
data. To run on real recordings, first produce one `.npz` per animal with
`run_pipeline.py` (each carries `R_mean` and `roi_labels`), then set
`RUN_CONFIG["results_dir"]` to that folder. The script maps each file to its
group/sex by matching the animal id in `COHORT` against the filename.

**Expected output.** Three PNGs in `experiments/output/`:

- `diff_matrix.png` — the `DIFF = healthy − disease` correlation matrix (step 5).
- `network.png` — the significant-connection network (step 6). The demo marks the
  strongest `|DIFF|` edges as "significant"; a real study loads this adjacency
  from **NBS** (or another permutation test). `wfci` does **not** compute the
  network statistic — that is an explicit non-goal.
- `barplots.png` — per-node counts of significant increases / decreases (step 6).

## What this demonstrates

- **The library never learns the study.** `grep -riE "sani|pd_|macchi|day4|sex"
  src/wfci/` returns nothing; every one of those terms lives only in this folder.
  A test (`tests/test_cohort.py`) enforces it.
- **The female-split discrepancy dissolves.** MATLAB step (5) selected the female
  groups inconsistently with its own main grouping. Here there is no library
  behaviour to port wrongly: the grouping is one `.select(...)` line in
  `compute_contrasts`, which the study owns and can fix in place. See the comment
  there.
- **Writing your own study** = copy this file, edit `COHORT` and the selections,
  point it at your `.npz` folder. You never touch `src/wfci/`.
