# Analysis pipeline

Compares an EpiHiper agent-based transmission forest (ground truth: who infected
whom, and when) against a Nextstrain/augur phylogeny built from sequences sampled
out of that same simulated epidemic.

Run everything with defaults:

```bash
./run_analyses.sh
```

Run a subset, against a different tree:

```bash
./run_analyses.sh --gen-graph data/other_tree.gpickle --run-label expt_b --analyses 2,3
./run_analyses.sh --list        # what's available
./run_analyses.sh --help        # full option list
./run_analyses.sh --dry-run     # show resolved settings without running
```

`run_analyses.sh` is the documented entry point; `run_analyses.py` is the driver
it calls. All selected analyses run in **one** process because loading the
~15M-node transmission forest takes several minutes and every analysis needs it
— prefer `--analyses 1,2,3` over three separate invocations.

## The analyses

| # | Name | What it measures |
|---|------|------------------|
| 1 | `ascertainment_over_time` | Per-day infections, sequenced infections, and the ascertainment rate — how much of the epidemic surveillance actually saw, and when. |
| 2 | `generation_dating_bias` | Error in the phylogeny's estimate of the time between two infections, binned by how many transmission generations really separate them. Days and percent-of-truth, chain-clustered. |
| 3 | `cross_chain_linkage` | Where the tree groups infections from transmission chains sharing *no* transmission edge, and how those spurious pairings differ from genuine within-chain pairs in timing, age, and county. |

## Outputs

Written to `--out-dir` (default `data/results/<run-label>/`):

```
00_run_manifest.json                      inputs, parameters, what ran
01_ascertainment_over_time.csv|.png
01_ascertainment_over_time_summary.csv
02_generation_dating_bias.csv|.png        per generation bin
02_generation_dating_bias_pairs.csv       pair-level detail
02_generation_dating_bias_summary.csv
03_cross_chain_linkage_nodes.csv          one row per merge node
03_cross_chain_linkage_by_chain.csv       per-chain exposure
03_cross_chain_linkage_pairs.csv          cross-chain tip pairs + timing/demography
03_cross_chain_linkage_depth.png
03_cross_chain_linkage_vs_within.png
03_cross_chain_linkage_summary.csv
run_analyses.log
```

Naming is `<NN>_<semantic_name>.<ext>` so results sort by analysis and are
self-describing once many runs sit side by side.

## Cross-experiment summarization

Every CSV carries a `run_label` column. The `*_summary.csv` files all share one
tidy schema:

```
run_label, analysis_index, analysis, metric, value
```

so they concatenate across runs and analyses without reshaping:

```bash
python collect_summaries.py --results-dir ../data/results --out all_runs_summary.csv
```

or directly:

```python
import glob, pandas as pd
df = pd.concat(pd.read_csv(p) for p in glob.glob("data/results/*/*_summary.csv"))
df.pivot_table(index="run_label", columns="metric", values="value")
```

Selected metrics that are meant to be compared across experiments:

- **1** `ascertainment_rate_overall`, `ascertainment_rate_in_window`, `seq_window_days`
- **2** `bias_days_chain_clustered` (± `_sem`), `bias_days_per_generation_slope`,
  `median_abs_pct_error`, `corr_generations_vs_actual_duration`
- **3** `n_merge_nodes`, `pct_merges_with_single_tip_side`, `n_pure_cherries`,
  `cross_same_county_rate` vs `within_same_county_rate` vs `random_pair_same_county_rate`,
  `within_bias_days_chain_clustered`, `cross_min_error_days_node_clustered`,
  `p_cross_looks_closer_than_within`

## Adding an analysis

Create `analysis_NN_<name>.py` exposing `INDEX`, `NAME`, `DESCRIPTION`, and
`run(ctx)`, then add the module to `ANALYSES` in `run_analyses.py`. It becomes
selectable by index or name immediately. Use `ctx.write_table()` for detail
tables, `ctx.add_metric()` for scalars, and `ctx.write_summary()` at the end;
those enforce the naming and the `run_label` stamping.

The computation library is `../notebooks/chain_capture_data.py` (kept in one
place, shared with `chain_capture_analysis.ipynb`, which documents how each
statistic was developed and validated). `pipeline_common` puts it on the path
and re-exports it as `cc`.

## Notes and caveats

- **Analysis 3's demographic comparison needs `--metadata`** (the augur input
  metadata TSV, which supplies tip age band and county FIPS). If the file is
  missing or carries no usable values, that part is skipped with a warning and
  the rest of analysis 3 still runs. County names are resolved from FIPS via a
  Virginia lookup table in `chain_capture_data.VA_FIPS_TO_COUNTY`; a non-Virginia
  population would need that extended.
- **A failing analysis does not abort the others.** The driver logs the
  traceback, continues, and exits non-zero so a wrapper can still detect it.
- **Pair sampling is capped** at `--max-pairs` per chain (default 500) and
  seeded (`--seed`, default 42), so pairs are not independent draws. That's why
  analyses 2 and 3 report chain-clustered statistics alongside pooled ones.
