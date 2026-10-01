"""Pool the per-run tidy summaries into one table for cross-experiment work.

Every analysis writes `<NN>_<name>_summary.csv` with the same columns
(run_label, analysis_index, analysis, metric, value), so pooling across runs is
just a concatenation. This emits both shapes:

    long  -- one row per (run_label, analysis, metric), the concatenation
    wide  -- one row per run_label, one column per metric (with --wide)

Usage
    python collect_summaries.py                                  # long, to stdout path
    python collect_summaries.py --wide --out all_runs_wide.csv
    python collect_summaries.py --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "results"


def collect(results_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(results_dir / "*" / "*_summary.csv")))
    if not paths:
        raise SystemExit(f"no *_summary.csv files found under {results_dir}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        missing = {"run_label", "analysis", "metric", "value"} - set(df.columns)
        if missing:
            print(f"skipping {p}: missing columns {sorted(missing)}")
            continue
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    print(f"pooled {len(frames)} summary files -> {len(out):,} metric rows, "
          f"{out['run_label'].nunique()} run(s)")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                   help="directory holding one subdirectory per run (default: %(default)s)")
    p.add_argument("--out", type=Path, default=None,
                   help="output CSV (default: <results-dir>/all_runs_summary[_wide].csv)")
    p.add_argument("--wide", action="store_true",
                   help="pivot to one row per run_label, one column per metric")
    args = p.parse_args(argv)

    df = collect(args.results_dir)
    if args.wide:
        df = df.pivot_table(index="run_label", columns="metric",
                            values="value", aggfunc="first").reset_index()
    out = args.out or (args.results_dir /
                       ("all_runs_summary_wide.csv" if args.wide else "all_runs_summary.csv"))
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
