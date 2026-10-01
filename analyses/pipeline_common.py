"""Shared plumbing for the analysis pipeline: input loading, output naming,
and the tidy per-run summary rows that make cross-experiment comparison work.

The expensive part of every analysis is the same: reading the ~15M-node
EpiHiper transmission forest and building its tick/parent indices (several
minutes). `AnalysisContext.load()` does that exactly once per invocation and
hands the result to every selected analysis, which is why `run_analyses.py`
dispatches multiple analyses inside one process rather than shelling out per
analysis.

Output naming convention, so results stay sortable and self-describing when
many runs are pooled later:

    <NN>_<semantic_name>.csv / .png     e.g. 01_ascertainment_over_time.csv

Every CSV carries a `run_label` column, and every analysis also emits a tidy
`<NN>_<name>_summary.csv` of (run_label, analysis, metric, value) rows. Those
summary files are the intended input to cross-experiment summarization: they
concatenate cleanly across runs regardless of which analyses each run used.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# `chain_capture_data` is the computation library and lives with the notebook
# that developed it; keep one copy rather than forking a pipeline duplicate.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "notebooks"))
import chain_capture_data as cc  # noqa: E402

DEFAULT_EPI_GRAPH = REPO_ROOT / "data" / "epi_graph.gpickle"
DEFAULT_GEN_GRAPH = REPO_ROOT / "data" / "nextstrain_tree_graph_300day.gpickle"
DEFAULT_METADATA = Path(cc.TIP_METADATA_TSV)

# Shared figure styling, matching chain_capture_analysis.ipynb so pipeline
# figures and notebook figures are visually interchangeable.
PLOT_RC = {
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
}
COL = {"captured": "#12866E", "total": "#4C6EF5", "zero": "#C2255C",
       "bias": "#F08C00", "mae": "#4C6EF5", "rmse": "#C2255C", "window": "#EDF2FF"}


def apply_plot_style():
    import matplotlib
    matplotlib.use("Agg")  # headless: the pipeline only ever writes files
    import matplotlib.pyplot as plt
    plt.rcParams.update(PLOT_RC)
    return plt


@dataclass
class AnalysisContext:
    """Everything the analyses share, loaded once."""

    run_label: str
    out_dir: Path
    epi_graph_path: Path
    gen_graph_path: Path
    metadata_path: Path | None
    max_pairs_per_chain: int = 500
    seed: int = 42

    # populated by load()
    epi_tick: dict = field(default_factory=dict, repr=False)
    epi_parent: dict = field(default_factory=dict, repr=False)
    gen_G: object = field(default=None, repr=False)
    gen_parent: dict = field(default_factory=dict, repr=False)
    gen_num_date: dict = field(default_factory=dict, repr=False)
    gen_to_epi: dict = field(default_factory=dict, repr=False)
    epi_to_gen: dict = field(default_factory=dict, repr=False)
    captured_epi_nodes: set = field(default_factory=set, repr=False)
    component_df: object = field(default=None, repr=False)
    node_to_component: dict = field(default_factory=dict, repr=False)
    tip_chain: dict = field(default_factory=dict, repr=False)
    tick_of_gen_leaf: dict = field(default_factory=dict, repr=False)
    seq_window: tuple = (0, 0)

    _summary_rows: list = field(default_factory=list, repr=False)

    def load(self):
        """Read both graphs and build every shared index. Minutes, not seconds."""
        t0 = time.time()
        log(f"loading transmission forest: {self.epi_graph_path}")
        epi_G = cc.load_epi_graph(str(self.epi_graph_path))
        log(f"  {epi_G.number_of_nodes():,} infections, {epi_G.number_of_edges():,} edges "
            f"[{time.time()-t0:.0f}s]")

        self.epi_tick, self.epi_parent = cc.build_epi_indices(epi_G)
        epi_pid_tick_index = cc.build_epi_pid_tick_index(epi_G)
        log(f"  built tick/parent indices [{time.time()-t0:.0f}s]")

        log(f"loading genomic tree: {self.gen_graph_path}")
        self.gen_G = cc.load_gen_graph(str(self.gen_graph_path))
        self.seq_window = cc.compute_sequencing_window(self.gen_G)
        self.gen_to_epi = cc.match_gen_tips_to_epi_nodes(self.gen_G, epi_pid_tick_index)
        self.epi_to_gen = {v: k for k, v in self.gen_to_epi.items()}
        self.captured_epi_nodes = set(self.gen_to_epi.values())
        self.gen_parent = {v: u for u, v in self.gen_G.edges()}
        self.gen_num_date = {n: d.get("num_date") for n, d in self.gen_G.nodes(data=True)}
        log(f"  {self.gen_G.number_of_nodes():,} tree nodes, {len(self.gen_to_epi):,} sequenced "
            f"tips matched to infections; sequencing window {self.seq_window[0]}-{self.seq_window[1]}")

        self.component_df, self.node_to_component = cc.build_component_stats(
            epi_G, self.epi_tick, self.captured_epi_nodes, window=self.seq_window
        )
        del epi_G  # everything downstream uses the indices, not the graph

        self.tip_chain = {g: self.node_to_component[e] for g, e in self.gen_to_epi.items()}
        self.tick_of_gen_leaf = {g: self.epi_tick[e] for g, e in self.gen_to_epi.items()}
        log(f"  {len(self.component_df):,} transmission chains "
            f"({(self.component_df['n_captured'] > 0).sum()} with >=1 sequenced tip) "
            f"[{time.time()-t0:.0f}s total]")
        return self

    # ---- output helpers -------------------------------------------------

    def path_for(self, index: int, name: str, ext: str) -> Path:
        return self.out_dir / f"{index:02d}_{name}.{ext}"

    def write_table(self, df: pd.DataFrame, index: int, name: str) -> Path:
        """Write a detail table, stamped with the run label for later pooling."""
        out = df.copy()
        out.insert(0, "run_label", self.run_label)
        path = self.path_for(index, name, "csv")
        out.to_csv(path, index=False)
        log(f"  wrote {path.name}  ({len(out):,} rows)")
        return path

    def add_metric(self, index: int, analysis: str, metric: str, value):
        """Record one scalar for the tidy cross-experiment summary."""
        self._summary_rows.append({
            "run_label": self.run_label, "analysis_index": index,
            "analysis": analysis, "metric": metric, "value": value,
        })

    def write_summary(self, index: int, analysis: str, name: str) -> Path:
        """Flush this analysis's tidy metrics to <NN>_<name>_summary.csv."""
        rows = [r for r in self._summary_rows if r["analysis_index"] == index]
        df = pd.DataFrame(rows)
        path = self.path_for(index, f"{name}_summary", "csv")
        df.to_csv(path, index=False)
        log(f"  wrote {path.name}  ({len(df)} metrics)")
        return path

    def write_manifest(self, analyses_run: list) -> Path:
        """Record inputs and parameters so a pooled result set stays traceable."""
        manifest = {
            "run_label": self.run_label,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epi_graph": str(self.epi_graph_path),
            "gen_graph": str(self.gen_graph_path),
            "metadata": str(self.metadata_path) if self.metadata_path else None,
            "sequencing_window": list(self.seq_window),
            "max_pairs_per_chain": self.max_pairs_per_chain,
            "seed": self.seed,
            "analyses_run": analyses_run,
            "n_chains": int(len(self.component_df)),
            "n_sequenced_tips": int(len(self.gen_to_epi)),
        }
        path = self.out_dir / "00_run_manifest.json"
        path.write_text(json.dumps(manifest, indent=2))
        log(f"wrote {path.name}")
        return path


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
