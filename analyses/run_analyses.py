"""Driver for the genomic-vs-simulation analysis pipeline.

Normally invoked through `run_analyses.sh`, which documents the options and
supplies defaults; this module is the thing that actually loads the inputs and
dispatches the selected analyses.

All selected analyses run inside this one process on purpose: reading the
~15M-node transmission forest and building its indices takes several minutes
and every analysis needs it, so running `--analyses 1,2,3` costs that once
rather than three times.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# pipeline_common must come first: it puts the computation library on sys.path.
from pipeline_common import (DEFAULT_EPI_GRAPH, DEFAULT_GEN_GRAPH, DEFAULT_METADATA,
                             AnalysisContext, log)
import analysis_01_ascertainment as a01
import analysis_02_generation_bias as a02
import analysis_03_cross_chain as a03

# The registry the CLI exposes. Index -> module; each module supplies NAME,
# DESCRIPTION and run(ctx). Add new analyses here and they become selectable
# by index or by name with no other changes.
ANALYSES = {a.INDEX: a for a in (a01, a02, a03)}


def parse_selection(raw: str) -> list:
    """'all' | '1,3' | 'ascertainment_over_time,3' -> sorted list of indices."""
    if raw.strip().lower() in ("all", ""):
        return sorted(ANALYSES)
    by_name = {a.NAME: i for i, a in ANALYSES.items()}
    chosen = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            idx = int(token)
            if idx not in ANALYSES:
                raise SystemExit(f"unknown analysis index {idx}; choose from {sorted(ANALYSES)}")
            chosen.add(idx)
        elif token in by_name:
            chosen.add(by_name[token])
        else:
            raise SystemExit(
                f"unknown analysis '{token}'; use an index {sorted(ANALYSES)} "
                f"or a name {sorted(by_name)}"
            )
    return sorted(chosen)


def build_parser():
    p = argparse.ArgumentParser(
        prog="run_analyses.py",
        description="Compare an EpiHiper transmission forest against a Nextstrain tree.",
    )
    p.add_argument("--epi-graph", type=Path, default=DEFAULT_EPI_GRAPH,
                   help="pickled EpiHiper transmission forest (default: %(default)s)")
    p.add_argument("--gen-graph", type=Path, default=DEFAULT_GEN_GRAPH,
                   help="pickled Nextstrain/augur tree (default: %(default)s)")
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA,
                   help="augur input metadata TSV supplying tip age/county; "
                        "analysis 3 skips its demographic comparison if absent")
    p.add_argument("--run-label", default=None,
                   help="identifier stamped into every output row "
                        "(default: the gen-graph filename stem)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output directory (default: data/results/<run-label>)")
    p.add_argument("--analyses", default="all",
                   help="which analyses to run: 'all', or a comma-separated list of "
                        "indices/names (default: all)")
    p.add_argument("--max-pairs-per-chain", type=int, default=500,
                   help="cap on sampled pairs per chain (default: %(default)s)")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for pair sampling (default: %(default)s)")
    p.add_argument("--list", action="store_true", help="list available analyses and exit")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list:
        print("Available analyses:\n")
        for idx, mod in sorted(ANALYSES.items()):
            print(f"  {idx}  {mod.NAME}\n     {mod.DESCRIPTION}")
        return 0

    selected = parse_selection(args.analyses)
    run_label = args.run_label or args.gen_graph.stem
    out_dir = args.out_dir or (Path(__file__).resolve().parent.parent
                               / "data" / "results" / run_label)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path, what in ((args.epi_graph, "epi graph"), (args.gen_graph, "gen graph")):
        if not path.exists():
            raise SystemExit(f"error: {what} not found: {path}")
    metadata = args.metadata if args.metadata and args.metadata.exists() else None
    if args.metadata and metadata is None:
        log(f"note: metadata not found at {args.metadata}; "
            "analysis 3 will skip its demographic comparison")

    log(f"run label : {run_label}")
    log(f"output dir: {out_dir}")
    log(f"analyses  : {', '.join(f'{i} ({ANALYSES[i].NAME})' for i in selected)}")

    ctx = AnalysisContext(
        run_label=run_label, out_dir=out_dir,
        epi_graph_path=args.epi_graph, gen_graph_path=args.gen_graph,
        metadata_path=metadata, max_pairs_per_chain=args.max_pairs_per_chain,
        seed=args.seed,
    ).load()

    t0 = time.time()
    completed, failed = [], []
    for idx in selected:
        mod = ANALYSES[idx]
        try:
            mod.run(ctx)
            completed.append(idx)
        except Exception as exc:  # keep going: one bad analysis shouldn't sink the rest
            failed.append((idx, mod.NAME, repr(exc)))
            log(f"ERROR in analysis {idx} ({mod.NAME}): {exc!r}")
            import traceback
            traceback.print_exc()

    ctx.write_manifest([{"index": i, "name": ANALYSES[i].NAME} for i in completed])
    log(f"done in {time.time()-t0:.0f}s -- completed {completed}"
        + (f", FAILED {[f'{i} ({n})' for i, n, _ in failed]}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
