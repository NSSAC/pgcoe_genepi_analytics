"""Analysis 2 -- generation_dating_bias.

How wrong is the phylogeny's estimate of the time separating two infections,
as a function of how many transmission generations actually separate them?

For each sampled pair of sequenced infections within a chain, the true
separation is exact (both tips' distance to their real shared infector in the
transmission forest) and the genomic estimate uses the tree's own inferred
MRCA date. Error is reported in absolute days and as a percent of the true
duration -- the percent view controls for the mechanical fact that pairs
separated by more time have more days available to be wrong by.

Errors are reported chain-clustered (mean of per-chain means, with SEM across
chains) as well as pair-pooled: pair sampling is capped per chain, so pairs
are not independent draws and a pooled mean lets a few heavily-sampled chains
dominate.

Outputs
    02_generation_dating_bias.csv           error stats per generation bin
    02_generation_dating_bias_pairs.csv     pair-level detail (re-analysable)
    02_generation_dating_bias.png           bias and error vs. generations
    02_generation_dating_bias_summary.csv   tidy scalars for cross-run pooling
"""

import numpy as np

from pipeline_common import cc, COL, apply_plot_style, log

INDEX = 2
NAME = "generation_dating_bias"
DESCRIPTION = "Genomic dating bias vs. transmission generations between a pair"

# Generation bins: 5 generations wide, out to 95. Bins with no pairs are dropped.
GEN_BIN_EDGES = list(range(0, 96, 5))


def run(ctx):
    log(f"[{INDEX}] {NAME}: sampling within-chain pairs "
        f"(max {ctx.max_pairs_per_chain}/chain, seed {ctx.seed})")
    pairs = cc.sample_pairs_per_component(
        ctx.component_df, ctx.node_to_component, ctx.captured_epi_nodes,
        max_pairs_per_component=ctx.max_pairs_per_chain, seed=ctx.seed,
    )
    pair_df = cc.compute_pair_duration_errors(
        pairs, ctx.epi_tick, ctx.epi_parent, ctx.epi_to_gen,
        ctx.gen_num_date, ctx.gen_parent, ctx.component_df,
    )
    log(f"  {len(pair_df):,} pairs across {pair_df['component_id'].nunique()} chains")

    binned = cc.error_by_count_bin(pair_df, GEN_BIN_EDGES, col="n_transmission_events")
    ctx.write_table(binned, INDEX, NAME)
    ctx.write_table(
        pair_df[["component_id", "epi_a", "epi_b", "actual_duration_days",
                 "genomic_duration_days", "error_days", "abs_error_days",
                 "n_transmission_events"]],
        INDEX, f"{NAME}_pairs",
    )

    per_chain = pair_df.groupby("component_id")["error_days"].mean()
    pct_error = (pair_df["error_days"] / pair_df["actual_duration_days"].replace(0, np.nan)) * 100

    add = lambda m, v: ctx.add_metric(INDEX, NAME, m, v)
    add("n_pairs", int(len(pair_df)))
    add("n_chains_contributing_pairs", int(pair_df["component_id"].nunique()))
    add("bias_days_chain_clustered", float(per_chain.mean()))
    add("bias_days_chain_clustered_sem",
        float(per_chain.std(ddof=1) / np.sqrt(len(per_chain))) if len(per_chain) > 1 else np.nan)
    add("bias_days_pair_pooled", float(pair_df["error_days"].mean()))
    add("mae_days", float(pair_df["abs_error_days"].mean()))
    add("rmse_days", float(np.sqrt((pair_df["error_days"] ** 2).mean())))
    add("median_abs_pct_error", float(pct_error.abs().median()))
    add("mape_pct", float(pct_error.abs().mean()))
    add("generations_median", float(pair_df["n_transmission_events"].median()))
    add("generations_max", int(pair_df["n_transmission_events"].max()))
    add("corr_generations_vs_actual_duration",
        float(pair_df["n_transmission_events"].corr(pair_df["actual_duration_days"])))

    # One comparable number for "how fast does bias grow per generation":
    # slope of chain-clustered bias against bin midpoint, over populated bins.
    populated = binned.dropna(subset=["chain_mean_bias_days"])
    if len(populated) >= 2:
        mids = populated["count_bin"].astype(str).map(
            lambda s: np.mean([float(x) for x in s.split("-")])
        )
        slope = float(np.polyfit(mids, populated["chain_mean_bias_days"], 1)[0])
        add("bias_days_per_generation_slope", slope)
        add("n_generation_bins_populated", int(len(populated)))

    _plot(ctx, binned)
    ctx.write_summary(INDEX, NAME, NAME)


def _plot(ctx, binned):
    plt = apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = np.arange(len(binned))
    labels = binned["count_bin"].astype(str)

    def _xaxis(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)

    ax = axes[0, 0]
    ax.bar(x, binned.chain_mean_bias_days, color=COL["bias"], alpha=.85,
           yerr=binned.chain_bias_sem_days, capsize=3,
           error_kw={"ecolor": "0.3", "elinewidth": .8})
    ax.axhline(0, color="0.5", lw=.8)
    _xaxis(ax)
    ax.set_ylabel("chain-clustered mean bias (days)")
    ax.set_title("Absolute bias by transmission generations")

    ax = axes[0, 1]
    ax.plot(x, binned.mae_days, "s-", color=COL["mae"], label="MAE")
    ax.plot(x, binned.rmse_days, "^-", color=COL["rmse"], label="RMSE")
    _xaxis(ax)
    ax.set_ylabel("days")
    ax.legend(fontsize=9)
    ax.set_title("Absolute error by transmission generations")

    ax = axes[1, 0]
    ax.bar(x, binned.chain_mean_bias_pct, color=COL["bias"], alpha=.85,
           yerr=binned.chain_bias_pct_sem, capsize=3,
           error_kw={"ecolor": "0.3", "elinewidth": .8})
    ax.axhline(0, color="0.5", lw=.8)
    _xaxis(ax)
    ax.set_xlabel("transmission generations between the pair")
    ax.set_ylabel("chain-clustered mean bias (% of true duration)")
    ax.set_title("Percent bias -- controls for duration")

    ax = axes[1, 1]
    ax.plot(x, binned.mape_pct, "s-", color=COL["mae"], label="mean abs % error")
    ax.plot(x, binned.median_abs_pct_error, "^-", color=COL["rmse"], label="median abs % error")
    _xaxis(ax)
    ax.set_xlabel("transmission generations between the pair")
    ax.set_ylabel("% of true duration")
    ax.legend(fontsize=9)
    ax.set_title("Percent error by transmission generations")

    fig.suptitle(f"Genomic dating bias by transmission generations -- {ctx.run_label}", y=1.01)
    fig.tight_layout()
    path = ctx.path_for(INDEX, NAME, "png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {path.name}")
