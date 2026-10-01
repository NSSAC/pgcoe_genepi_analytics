"""Analysis 3 -- cross_chain_linkage.

Where does the phylogeny group together infections from transmission chains
that share no transmission edge at all (different weakly-connected components
of the epi forest), and how do those spurious pairings differ from genuine
within-chain pairs?

Three parts:
  a) structure  -- how many merge nodes, how shallow, how many per chain
  b) demography -- age band and county agreement, cross-chain vs. within-chain
                   vs. a random-pair baseline (skipped if metadata is absent)
  c) timing     -- the separation the tree asserts, cross-chain vs. within-chain

On (c): a cross-chain pair has no shared infector, so the within-chain error
(estimate minus truth) is undefined -- the truth is not a larger number, it is
"no common ancestor exists". Two things are still measurable: what the tree
claims (directly comparable between groups), and a provable lower bound on the
truth, since any common ancestry must predate both chains' seeding events.

Outputs
    03_cross_chain_linkage_nodes.csv        one row per merge node
    03_cross_chain_linkage_by_chain.csv     per-chain exposure
    03_cross_chain_linkage_pairs.csv        cross-chain tip pairs + timing/demography
    03_cross_chain_linkage_depth.png        how shallow merges are
    03_cross_chain_linkage_vs_within.png    demography + timing comparison
    03_cross_chain_linkage_summary.csv      tidy scalars for cross-run pooling
"""

import numpy as np
import pandas as pd

from pipeline_common import cc, COL, apply_plot_style, log

INDEX = 3
NAME = "cross_chain_linkage"
DESCRIPTION = "Cross-chain phylogenetic linkages and how they differ from within-chain pairs"

# "Shallow" = the smaller side of the merge holds at most this many tips.
# Beyond this the merge is a deep combination of established clades, which is
# expected from shared outbreak origin rather than a surprising local grouping.
SHALLOW_MAX_TIPS = 20
DEPTH_BIN_EDGES = [0, 1, 5, 20, 100, 1000, 10_000_000]
DEPTH_BIN_LABELS = ["=1", "2-5", "6-20", "21-100", "101-1000", "1000+"]


def run(ctx):
    add = lambda m, v: ctx.add_metric(INDEX, NAME, m, v)

    # ---- (a) structure --------------------------------------------------
    log(f"[{INDEX}] {NAME}: finding cross-chain merges in the phylogeny")
    _, gen_children_of, gen_root = cc.build_gen_tree_indices(ctx.gen_G)
    gen_order = cc.gen_postorder(gen_children_of, gen_root)
    merges_df, _ = cc.compute_chain_merges(
        gen_children_of, gen_order, ctx.tip_chain, ctx.tick_of_gen_leaf, ctx.gen_num_date
    )
    merges_df = cc.annotate_merge_earliest(merges_df, ctx.tip_chain, ctx.tick_of_gen_leaf)

    node_summary = cc.summarize_merge_nodes(merges_df)
    node_summary["size_bin"] = pd.cut(node_summary["smallest_side_tips"],
                                       bins=DEPTH_BIN_EDGES, labels=DEPTH_BIN_LABELS)
    chain_summary = cc.summarize_chain_linkages(merges_df, ctx.tip_chain)
    events = cc.classify_merge_anchors(merges_df, ctx.tip_chain)

    n_internal = sum(1 for n in ctx.gen_G.nodes if ctx.gen_G.out_degree(n) > 0)
    chain_tip_counts = pd.Series(ctx.tip_chain).value_counts()
    log(f"  {len(node_summary):,} merge nodes across {len(chain_tip_counts)} sequenced chains")

    ctx.write_table(node_summary, INDEX, f"{NAME}_nodes")
    ctx.write_table(chain_summary, INDEX, f"{NAME}_by_chain")

    add("n_chains_sequenced", int(len(chain_tip_counts)))
    add("n_chains_singleton", int((chain_tip_counts == 1).sum()))
    add("n_merge_nodes", int(len(node_summary)))
    add("n_internal_tree_nodes", int(n_internal))
    add("pct_internal_nodes_that_are_merges", 100 * len(node_summary) / n_internal if n_internal else np.nan)
    add("n_pure_cherries", int(node_summary["is_pure_cherry"].sum()))
    add("pct_merges_with_single_tip_side",
        100 * float((node_summary["smallest_side_tips"] == 1).mean()))
    add("median_smallest_side_tips", float(node_summary["smallest_side_tips"].median()))
    add("n_lone_tip_events", int(len(events)))
    add("n_lone_tip_events_singleton_chain", int(events["is_singleton_chain"].sum()))

    multi = events[~events["is_singleton_chain"]]
    determinable = multi["is_earliest"].dropna()
    add("n_multitip_outlier_events", int(len(multi)))
    add("n_chains_with_multitip_outliers", int(multi["chain"].nunique()) if len(multi) else 0)
    add("pct_multitip_outliers_that_are_chains_earliest",
        100 * float(determinable.mean()) if len(determinable) else np.nan)

    # Shallow-restricted per-chain exposure: unrestricted "other chains met"
    # saturates at n_chains-1 for every chain (a single tree always unites
    # everything eventually), so it is only informative once restricted.
    shallow_nodes = set(node_summary.loc[node_summary["smallest_side_tips"] <= SHALLOW_MAX_TIPS, "node"])
    shallow = cc.summarize_chain_linkages(
        merges_df[merges_df["node"].isin(shallow_nodes)], ctx.tip_chain
    )
    add("shallow_max_tips_threshold", SHALLOW_MAX_TIPS)
    add("n_shallow_merge_nodes", int(len(shallow_nodes)))
    if len(shallow):
        add("median_other_chains_shallow", float(shallow["n_other_chains"].median()))
        add("max_other_chains_shallow", int(shallow["n_other_chains"].max()))
        add("corr_chain_size_vs_shallow_other_chains",
            float(np.corrcoef(np.log(shallow["n_sequenced_tips"]), shallow["n_other_chains"])[0, 1])
            if len(shallow) > 2 else np.nan)

    # ---- cross-chain tip pairs (the unit for (b) and (c)) ---------------
    cross_pairs = cc.build_cross_chain_tip_pairs(merges_df, ctx.tip_chain)
    n_shared = sum(
        cc.lowest_common_ancestor(ctx.gen_to_epi[r.tip_a], ctx.gen_to_epi[r.tip_b], ctx.epi_parent) is not None
        for r in cross_pairs.itertuples(index=False)
    )
    if n_shared:
        raise AssertionError(
            f"{n_shared} 'cross-chain' pairs share a transmission ancestor -- "
            "these are not cross-chain pairs and would corrupt the comparison"
        )
    log(f"  {len(cross_pairs):,} cross-chain tip pairs (verified: none share a transmission ancestor)")
    add("n_cross_chain_pairs", int(len(cross_pairs)))

    # Within-chain comparison set, same construction as analysis 2.
    within_pairs = cc.sample_pairs_per_component(
        ctx.component_df, ctx.node_to_component, ctx.captured_epi_nodes,
        max_pairs_per_component=ctx.max_pairs_per_chain, seed=ctx.seed,
    )
    within_df = cc.compute_pair_duration_errors(
        within_pairs, ctx.epi_tick, ctx.epi_parent, ctx.epi_to_gen,
        ctx.gen_num_date, ctx.gen_parent, ctx.component_df,
    )
    within_df["tip_a"] = within_df["epi_a"].map(ctx.epi_to_gen)
    within_df["tip_b"] = within_df["epi_b"].map(ctx.epi_to_gen)
    add("n_within_chain_pairs", int(len(within_df)))

    # ---- (c) timing -----------------------------------------------------
    chain_seed_tick = dict(zip(ctx.component_df["component_id"], ctx.component_df["min_tick"]))
    cross_err = cc.compute_cross_chain_duration_errors(
        cross_pairs, ctx.gen_num_date, ctx.tick_of_gen_leaf, ctx.tip_chain, chain_seed_tick
    )
    per_node = cross_err.groupby("node")["min_error_days"].mean()
    per_chain_within = within_df.groupby("component_id")["error_days"].mean()

    add("within_genomic_duration_median_days", float(within_df["genomic_duration_days"].median()))
    add("within_true_duration_median_days", float(within_df["actual_duration_days"].median()))
    add("within_bias_days_chain_clustered", float(per_chain_within.mean()))
    add("within_bias_days_chain_clustered_sem",
        float(per_chain_within.std(ddof=1) / np.sqrt(len(per_chain_within))))
    add("cross_genomic_duration_median_days", float(cross_err["genomic_duration_days"].median()))
    add("cross_lower_bound_duration_median_days", float(cross_err["min_true_duration_days"].median()))
    add("cross_min_error_days_node_clustered", float(per_node.mean()))
    add("cross_min_error_days_node_clustered_sem",
        float(per_node.std(ddof=1) / np.sqrt(len(per_node))) if len(per_node) > 1 else np.nan)
    add("cross_pct_provably_too_short", 100 * float((cross_err["min_error_days"] < 0).mean()))

    # Separability: can the tree's own asserted distance tell a real link from
    # a fabricated one? 0.5 would mean completely indistinguishable.
    a = cross_err["genomic_duration_days"].values
    b = np.sort(within_df["genomic_duration_days"].values)
    add("p_cross_looks_closer_than_within",
        float((len(b) - np.searchsorted(b, a, side="right")).sum() / (len(a) * len(b))))
    add("pct_cross_below_within_median_separation",
        100 * float((cross_err["genomic_duration_days"] < within_df["genomic_duration_days"].median()).mean()))

    # ---- (b) demography (optional) --------------------------------------
    cross_demo = within_demo = None
    random_same_county = np.nan
    tip_demo = _load_demographics(ctx)
    if tip_demo:
        cross_demo = cc.annotate_pair_demographics(cross_pairs, "tip_a", "tip_b", tip_demo)
        within_demo = cc.annotate_pair_demographics(within_df, "tip_a", "tip_b", tip_demo)
        counties = pd.Series([v["county"] for v in tip_demo.values() if v["county"]])
        random_same_county = float((counties.value_counts(normalize=True) ** 2).sum())

        add("random_pair_same_county_rate", random_same_county)
        add("cross_same_county_rate", float(cross_demo["same_county"].mean()))
        add("within_same_county_rate", float(within_demo["same_county"].mean()))
        add("cross_mean_age_band_gap", float(cross_demo["age_band_gap"].mean()))
        add("within_mean_age_band_gap", float(within_demo["age_band_gap"].mean()))
        cross_err = cross_err.merge(
            cross_demo[["tip_a", "tip_b", "age_band_gap", "same_county"]],
            on=["tip_a", "tip_b"], how="left",
        )
    else:
        log("  demographics unavailable -- skipping the age/county comparison")

    ctx.write_table(cross_err, INDEX, f"{NAME}_pairs")

    _plot_depth(ctx, node_summary)
    _plot_comparison(ctx, cross_err, within_df, cross_demo, within_demo, random_same_county)
    ctx.write_summary(INDEX, NAME, NAME)


def _load_demographics(ctx):
    """Tip age band + county, or None if this run has no usable metadata."""
    if ctx.metadata_path is None:
        return None
    try:
        tip_demo = cc.load_tip_demographics(str(ctx.metadata_path))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        log(f"  could not read tip metadata ({exc.__class__.__name__}: {exc})")
        return None
    usable = sum(1 for v in tip_demo.values() if v["age_band"] is not None or v["county"])
    if not usable:
        log("  tip metadata present but carries no usable age/county values")
        return None
    log(f"  tip demographics: {usable:,} tips with age and/or county")
    return tip_demo


def _plot_depth(ctx, node_summary):
    plt = apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = node_summary["size_bin"].value_counts().reindex(DEPTH_BIN_LABELS)
    ax.bar(DEPTH_BIN_LABELS, counts.values, color=COL["bias"], alpha=.85)
    for i, v in enumerate(counts.values):
        if not np.isnan(v):
            ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("sequenced tips on the smaller side of the merge (shallower = fewer)")
    ax.set_ylabel("number of merge nodes")
    ax.set_title(f"How shallow are cross-chain merges? -- {ctx.run_label}")
    fig.tight_layout()
    path = ctx.path_for(INDEX, f"{NAME}_depth", "png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {path.name}")


def _plot_comparison(ctx, cross_err, within_df, cross_demo, within_demo, random_same_county):
    """Timing on the top row, demography on the bottom (when available)."""
    plt = apply_plot_style()
    has_demo = cross_demo is not None
    nrows = 2 if has_demo else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(13.5, 5 * nrows), squeeze=False)

    ax = axes[0, 0]
    bins = np.linspace(0, max(within_df["genomic_duration_days"].max(),
                              cross_err["genomic_duration_days"].max()), 40)
    ax.hist(within_df["genomic_duration_days"], bins=bins, density=True, alpha=.6,
            color=COL["total"], label=f"within-chain, real link (n={len(within_df):,})")
    ax.hist(cross_err["genomic_duration_days"], bins=bins, density=True, alpha=.6,
            color=COL["zero"], label=f"cross-chain, no link (n={len(cross_err):,})")
    ax.axvline(within_df["genomic_duration_days"].median(), color=COL["total"], ls="--", lw=1.2)
    ax.axvline(cross_err["genomic_duration_days"].median(), color=COL["zero"], ls="--", lw=1.2)
    ax.set_xlabel("separation the tree asserts (days)")
    ax.set_ylabel("density")
    ax.set_title("What the tree claims")
    ax.legend(fontsize=8.5)

    ax = axes[0, 1]
    bins2 = np.linspace(-350, 350, 45)
    ax.hist(within_df["error_days"], bins=bins2, density=True, alpha=.6, color=COL["total"],
            label="within-chain: exact error")
    ax.hist(cross_err["min_error_days"], bins=bins2, density=True, alpha=.6, color=COL["zero"],
            label="cross-chain: error vs. lower bound")
    ax.axvline(0, color="0.3", lw=1)
    ax.set_xlabel("genomic estimate minus truth (days); negative = tree too short")
    ax.set_ylabel("density")
    ax.set_title("Error, where a truth exists to compare against")
    ax.legend(fontsize=8.5)

    if has_demo:
        ax = axes[1, 0]
        gaps = [0, 1, 2, 3, 4]
        cf = cross_demo["age_band_gap"].value_counts(normalize=True).reindex(gaps, fill_value=0)
        wf = within_demo["age_band_gap"].value_counts(normalize=True).reindex(gaps, fill_value=0)
        x = np.arange(len(gaps))
        ax.bar(x - .175, cf.values, .35, color=COL["zero"], alpha=.85, label="cross-chain")
        ax.bar(x + .175, wf.values, .35, color=COL["total"], alpha=.85, label="within-chain")
        ax.set_xticks(x); ax.set_xticklabels(gaps)
        ax.set_xlabel("age-band gap (0 = same band)")
        ax.set_ylabel("fraction of pairs")
        ax.set_title("Age similarity")
        ax.legend(fontsize=9)

        ax = axes[1, 1]
        groups = ["random pair\n(baseline)", "cross-chain\nmerge", "within-chain\n(real link)"]
        rates = [random_same_county, cross_demo["same_county"].mean(),
                 within_demo["same_county"].mean()]
        ax.bar(groups, rates, color=["0.6", COL["zero"], COL["total"]], alpha=.85)
        for i, r in enumerate(rates):
            ax.text(i, r, f"{r:.1%}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("same-county rate")
        ax.set_title("Geography")

    fig.suptitle(f"Cross-chain merges vs. genuine within-chain links -- {ctx.run_label}", y=1.005)
    fig.tight_layout()
    path = ctx.path_for(INDEX, f"{NAME}_vs_within", "png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {path.name}")
