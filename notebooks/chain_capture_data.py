"""Compare EpiHiper ground-truth transmission chains to what the genomic
(Nextstrain) tree recovers: what fraction of infections in a long-running
chain were ever sequenced, and how well does the phylogeny's own inferred
divergence timing recover the true time between infections.

Two ground truths come from `data/epi_graph.gpickle` (`tree_reading_tools.ipynb`
+ `linelist_eda.ipynb`): every infection is a node named `{raw_pid}_{n}` (the
`_n` suffix disambiguates reinfections), and the single in-edge on that node
carries the `tick` (day) it happened on -- `contact_pid_updated -> pid_updated`.
Exogenous seeds are self-loops. This makes the graph a forest: one parent per
node, so per-pair common-ancestor lookups are a cheap walk up parent pointers
rather than a generic shortest-path search (the thing that made the old
sampling loops in `linelist_eda.ipynb` take ~2h for 1000 pairs).

The genomic side comes from `data/nextstrain_tree_graph_300day.gpickle`, the
augur/TreeTime output for the `run_03_vadelta_2026_03_22_128to428_SURS`
PhyloGAS run. Tip names carry the same `pid.tick` identity
(`USA/VA-EHip-{pid}.{tick:03d}/{year}`), which is how a sequenced tip is
matched back to its epi node. Every one of its 14,791 simulated tips matches
an epi node exactly one-to-one, confirming the two files describe the same
underlying epidemic.

Tip `num_date` is fixed input (not estimated -- `num_date__inferred` is False
for every leaf), so a tip-to-tip date difference would just reproduce the
known tick difference. The genuine test of the phylogeny is its *internal*
node dates, which TreeTime estimates from the molecular clock plus topology.
So "genomic-estimated duration between infections" is built the same way the
existing notebooks already do it (`linelist_eda.ipynb` distances_df): each
tip's days-from-common-ancestor, summed over the pair, using the tree's own
inferred MRCA date -- compared against the same quantity computed on the real
transmission forest, where the MRCA and its tick are exactly known.
"""

import pickle
import random
import re
from collections import deque

import networkx as nx
import numpy as np
import pandas as pd

WORKING_DIR = "/sfs/gpfs/tardis/project/bii_nssac/people/bl4zc/pgcoe_genepi_analytics"
EPI_GPICKLE = f"{WORKING_DIR}/data/epi_graph.gpickle"
GEN_GPICKLE = f"{WORKING_DIR}/data/nextstrain_tree_graph_300day.gpickle"
RESULTS_DIR = f"{WORKING_DIR}/data/results/chain_capture"

# EHip identifiers are always "9-digit pid . 3-digit zero-padded tick".
EHIP_PATTERN = re.compile(r"EHip-(?P<sim_pid>\d{9})\.(?P<tick>\d{3})/")

# Duration thresholds requested: chains lasting 4 through 26 weeks, weekly steps.
THRESHOLD_WEEKS = list(range(4, 27))

# Empirical span of the sequenced tips (simulation days), i.e. the only period
# for which we have any genomic observation at all. Chains routinely run long
# before and after this -- see chain_capture_analysis.ipynb section 2 -- so an
# analysis that wants to reason only about "what genomic surveillance could
# have seen" needs to measure chain duration *within* this window, not over a
# chain's full lifetime. 256 days (36.6 weeks), comfortably wider than the
# requested 4-26 week sweep, so thresholds in that range aren't vacuous here
# the way they are against full-chain duration.
SEQ_WINDOW = (169, 425)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_epi_graph(path=EPI_GPICKLE):
    """Ground-truth EpiHiper transmission forest (~15.4M nodes)."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_gen_graph(path=GEN_GPICKLE):
    """Nextstrain/augur phylogeny (~29k nodes, a single rooted tree)."""
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------
# Epi forest indices: tick + parent per node, one pass over the edges.
# --------------------------------------------------------------------------

def build_epi_indices(epi_G):
    """Return (tick, parent) dicts, one entry per node.

    Every non-root node has exactly one in-edge (its infector); a self-loop
    marks an exogenous seed, whose parent is None and whose own tick comes
    from the loop's `tick` attribute.

    `process_seed_nodes` (in `tree_reading_tools.ipynb`/`linelist_eda.ipynb`)
    adds each seed's self-loop keyed on the *original* `pid`, not the
    `pid_updated` label used for every other node -- so the properly-labeled
    seed node (e.g. `"366151724_1"`) never actually appears as an edge target
    and is missing from `tick`/`parent` even though it is the root of a real,
    non-trivial chain (~3.8k of 15.4M nodes). Left unpatched this makes any
    lowest-common-ancestor walk that bottoms out at one of these roots raise
    a KeyError. Patched here by taking the earliest known tick among the
    node's direct children as a lower-bound proxy for its own -- a parent's
    tick is always <= every child's tick, so this is the best available
    estimate of when a seed was introduced.
    """
    tick = {}
    parent = {}
    children_of = {}
    for u, v, d in epi_G.edges(data=True):
        tick[v] = int(d["tick"])
        parent[v] = None if u == v else u
        if u != v:
            children_of.setdefault(u, []).append(v)

    orphan_roots = [n for n in children_of if n not in tick]
    for orphan in orphan_roots:
        kid_ticks = [tick[k] for k in children_of[orphan] if k in tick]
        if kid_ticks:
            tick[orphan] = min(kid_ticks)
            parent[orphan] = None
    still_missing = [n for n in orphan_roots if n not in tick]
    if still_missing:
        print(f"Warning: {len(still_missing)} chain-root nodes have no recoverable tick "
              "(their whole subtree is missing tick data) and will be skipped by LCA lookups")
    return tick, parent


def build_epi_pid_tick_index(epi_G):
    """(raw_pid, tick) -> epi node label, for matching genomic tip ids."""
    index = {}
    for u, v, d in epi_G.edges(data=True):
        raw_pid = int(str(v).split("_")[0])
        index[(raw_pid, int(d["tick"]))] = v
    return index


def parse_gen_sim_tips(gen_G):
    """{gen leaf name -> (sim_pid, tick)} for every EHip-labeled leaf."""
    out = {}
    for n in gen_G.nodes:
        if gen_G.out_degree(n) == 0:
            m = EHIP_PATTERN.search(str(n))
            if m:
                out[n] = (int(m.group("sim_pid")), int(m.group("tick")))
    return out


def compute_sequencing_window(gen_G):
    """Empirical (min_tick, max_tick) actually covered by sequenced tips.

    Distinct from the PhyloGAS run's configured range (128-428 for `run_03`):
    that's the config, this is what tips actually exist.
    """
    sim_ticks = [t for (_, t) in parse_gen_sim_tips(gen_G).values()]
    return (min(sim_ticks), max(sim_ticks))


def match_gen_tips_to_epi_nodes(gen_G, epi_pid_tick_index):
    """{gen leaf name -> epi node label} for every genomic tip found in the epi forest."""
    gen_sim_tips = parse_gen_sim_tips(gen_G)
    matched = {}
    unmatched = []
    for gen_name, key in gen_sim_tips.items():
        epi_node = epi_pid_tick_index.get(key)
        if epi_node is None:
            unmatched.append(gen_name)
        else:
            matched[gen_name] = epi_node
    if unmatched:
        print(f"Warning: {len(unmatched)}/{len(gen_sim_tips)} genomic tips had no matching epi node")
    return matched


# --------------------------------------------------------------------------
# Chain (weakly-connected-component) stats
# --------------------------------------------------------------------------

def build_component_stats(epi_G, tick, captured_epi_nodes, window=None):
    """One row per transmission chain: size, [min_tick, max_tick], and how
    many of its infections were ever sequenced.

    If `window` (lo, hi) is given, also compute the same size/span restricted
    to infections whose tick falls inside it -- `windowed_size`,
    `windowed_min_tick`, `windowed_max_tick`, `windowed_duration_days`,
    `windowed_frac_captured`. A chain's `n_captured` is unchanged by
    windowing: every captured (sequenced) node already has a tick inside the
    sequencing window by construction, so it is also the windowed numerator.
    Chains with no infections in the window get `windowed_size` 0 and
    `windowed_frac_captured` NaN.

    Returns (component_df, node_to_component) where node_to_component maps
    every epi node to its integer component id, for later pair sampling.
    """
    records = []
    node_to_component = {}
    for cid, nodes in enumerate(nx.weakly_connected_components(epi_G)):
        ticks = [tick[n] for n in nodes if n in tick]
        n_captured = len(captured_epi_nodes.intersection(nodes))
        record = {
            "component_id": cid,
            "size": len(nodes),
            "min_tick": min(ticks) if ticks else None,
            "max_tick": max(ticks) if ticks else None,
            "duration_days": (max(ticks) - min(ticks)) if ticks else 0,
            "n_captured": n_captured,
            "frac_captured": n_captured / len(nodes),
        }
        if window is not None:
            lo, hi = window
            w_ticks = [t for t in ticks if lo <= t <= hi]
            record["windowed_size"] = len(w_ticks)
            record["windowed_min_tick"] = min(w_ticks) if w_ticks else None
            record["windowed_max_tick"] = max(w_ticks) if w_ticks else None
            record["windowed_duration_days"] = (max(w_ticks) - min(w_ticks)) if w_ticks else 0
            record["windowed_frac_captured"] = (n_captured / len(w_ticks)) if w_ticks else np.nan
        records.append(record)
        for n in nodes:
            node_to_component[n] = cid
    return pd.DataFrame.from_records(records), node_to_component


def capture_by_duration_threshold(component_df, threshold_weeks=THRESHOLD_WEEKS,
                                   duration_col="duration_days", size_col="size",
                                   captured_col="n_captured", frac_col="frac_captured"):
    """Pooled + per-chain capture fraction among chains lasting >= each threshold.

    Pooled fraction sums infections and captures across all qualifying chains
    first, then divides -- so it isn't dominated by the handful of enormous
    chains the way a plain per-chain average would be. `mean_chain_fraction`
    is reported alongside for exactly that comparison. Pass
    `duration_col="windowed_duration_days"`, `size_col="windowed_size"`,
    `frac_col="windowed_frac_captured"` to categorize by in-window persistence
    instead of full chain lifetime.
    """
    rows = []
    for weeks in threshold_weeks:
        threshold_days = weeks * 7
        qualifying = component_df[component_df[duration_col] >= threshold_days]
        total = qualifying[size_col].sum()
        captured = qualifying[captured_col].sum()
        rows.append({
            "threshold_weeks": weeks,
            "threshold_days": threshold_days,
            "n_chains": len(qualifying),
            "total_infections": total,
            "total_captured": captured,
            "pooled_frac_captured": captured / total if total else np.nan,
            "mean_chain_frac_captured": qualifying[frac_col].mean(),
            "median_chain_frac_captured": qualifying[frac_col].median(),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Lowest-common-ancestor via parent-pointer walk (forest, so no cycles).
# --------------------------------------------------------------------------

def ancestor_chain(node, parent):
    """List of (node, ...) from `node` up to its root, inclusive, via `parent`."""
    chain = [node]
    while parent.get(chain[-1]) is not None:
        chain.append(parent[chain[-1]])
    return chain


def lowest_common_ancestor(a, b, parent):
    """First shared node walking up from `a` and `b` via `parent`.

    Cheap because forest depth here is generation count (tens), not
    population size -- unlike a generic shortest-path search over the
    15.4M-node graph, which is what made the old per-pair sampling slow.
    Returns None if `a` and `b` share no ancestor (different chains).
    """
    ancestors_a = set(ancestor_chain(a, parent))
    node = b
    while node is not None:
        if node in ancestors_a:
            return node
        node = parent.get(node)
    return None


def lca_with_distances(a, b, parent):
    """Like `lowest_common_ancestor`, but also returns how many edges
    (generations of transmission, on the epi side) separate each endpoint
    from it.

    Returns (mrca, dist_a, dist_b); `dist_a + dist_b` is the number of
    transmission events on the tree path between `a` and `b` -- i.e. how
    many person-to-person infections separate the pair, one more than the
    number of intervening nodes. (None, None, None) if they share no
    ancestor. One pass per endpoint, same cost as `lowest_common_ancestor`.
    """
    ancestors_a = {}
    node, d = a, 0
    while True:
        ancestors_a[node] = d
        nxt = parent.get(node)
        if nxt is None:
            break
        node, d = nxt, d + 1

    node, d = b, 0
    while True:
        if node in ancestors_a:
            return node, ancestors_a[node], d
        nxt = parent.get(node)
        if nxt is None:
            return None, None, None
        node, d = nxt, d + 1


# --------------------------------------------------------------------------
# Pairwise duration-between-infections error
# --------------------------------------------------------------------------

def sample_pairs_per_component(component_df, node_to_component, captured_epi_nodes,
                                max_pairs_per_component=500, seed=42):
    """Sample up to `max_pairs_per_component` sequenced-tip pairs within each
    chain that has at least 2 captured infections.

    A component's own tips are grouped first so a chain with only a handful
    of sequenced tips isn't asked for more pairs than it has combinations of.
    """
    rng = random.Random(seed)
    tips_by_component = {}
    for node in sorted(captured_epi_nodes):
        cid = node_to_component.get(node)
        if cid is not None:
            tips_by_component.setdefault(cid, []).append(node)

    pairs = []
    for cid, tips in sorted(tips_by_component.items()):
        if len(tips) < 2:
            continue
        max_possible = len(tips) * (len(tips) - 1) // 2
        n_pairs = min(max_pairs_per_component, max_possible)
        if max_possible <= max_pairs_per_component:
            for i in range(len(tips)):
                for j in range(i + 1, len(tips)):
                    pairs.append((cid, tips[i], tips[j]))
        else:
            seen = set()
            while len(seen) < n_pairs:
                a, b = rng.sample(tips, 2)
                key = (a, b) if a < b else (b, a)
                if key not in seen:
                    seen.add(key)
                    pairs.append((cid, key[0], key[1]))
    return pairs


def compute_pair_duration_errors(pairs, epi_tick, epi_parent, gen_tick_of_epi_node,
                                  gen_num_date, gen_parent, component_df):
    """For each sampled pair, the real vs. genomically-inferred time separating
    the two infections through their common ancestor.

    Real side: exact, from the transmission forest -- (tick_a - tick_mrca) +
    (tick_b - tick_mrca), where tick_mrca is the actual infector shared by
    both lineages. Genomic side: the same construction on the phylogeny,
    where the "ancestor" is an internal NODE_* whose date TreeTime inferred
    from genetic divergence, not observed directly -- this is the piece that
    carries real estimation error. `gen_tick_of_epi_node` maps an epi node to
    its corresponding genomic tip name.

    Also records `n_transmission_events`: the number of real person-to-person
    infections separating the pair on the epi tree (dist to MRCA summed over
    both endpoints) -- a longer chain can persist a long time in the window
    while any given sampled pair is only a few generations apart, or vice
    versa, so this is a distinct axis from calendar duration.
    """
    day_scale = 365.25  # num_date is decimal-year
    records = []
    for cid, epi_a, epi_b in pairs:
        gen_a = gen_tick_of_epi_node.get(epi_a)
        gen_b = gen_tick_of_epi_node.get(epi_b)
        if gen_a is None or gen_b is None:
            continue

        epi_mrca, dist_a, dist_b = lca_with_distances(epi_a, epi_b, epi_parent)
        gen_mrca = lowest_common_ancestor(gen_a, gen_b, gen_parent)
        if epi_mrca is None or gen_mrca is None:
            continue
        if epi_mrca not in epi_tick or epi_a not in epi_tick or epi_b not in epi_tick:
            continue

        actual_a = epi_tick[epi_a] - epi_tick[epi_mrca]
        actual_b = epi_tick[epi_b] - epi_tick[epi_mrca]
        genomic_a = (gen_num_date[gen_a] - gen_num_date[gen_mrca]) * day_scale
        genomic_b = (gen_num_date[gen_b] - gen_num_date[gen_mrca]) * day_scale

        records.append({
            "component_id": cid,
            "epi_a": epi_a,
            "epi_b": epi_b,
            "actual_duration_days": actual_a + actual_b,
            "genomic_duration_days": genomic_a + genomic_b,
            "error_days": (genomic_a + genomic_b) - (actual_a + actual_b),
            "n_transmission_events": dist_a + dist_b,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["abs_error_days"] = df["error_days"].abs()
        merge_cols = [c for c in ("duration_days", "size", "windowed_duration_days",
                                   "windowed_size", "windowed_frac_captured")
                      if c in component_df.columns]
        df = df.merge(
            component_df[["component_id", *merge_cols]],
            on="component_id", how="left",
        ).rename(columns={"duration_days": "chain_duration_days", "size": "chain_size"})
    return df


def _error_summary(df):
    """Pair-pooled point estimates, plus a chain-clustered bias + its SEM,
    in both absolute (days) and percent-of-actual-duration terms.

    Pairs are not independent draws -- up to 500 can come from a single
    chain (`sample_pairs_per_component`), so a bin dominated by one or two
    heavily-sampled chains would otherwise look far more precise than it is.
    `chain_mean_bias_days`/`chain_bias_sem_days` (and the `_pct` versions)
    instead average per-chain mean error first, treating each *chain* as the
    independent unit; NaN when fewer than 2 chains contribute.

    Percent error (`error_days / actual_duration_days`) controls for the
    mechanical fact that pairs separated by more time have more days to be
    wrong by -- but it's a ratio with real infections in the denominator
    (min observed: 4 days), so a handful of pairs with a small actual
    duration and a mistimed genomic estimate can produce large percent
    swings. Report `median_abs_pct_error` alongside `mape_pct` for that
    reason -- the median is far less moved by those outliers.
    """
    n = len(df)
    per_chain_bias = df.groupby("component_id")["error_days"].mean() if n else pd.Series(dtype=float)
    n_chains = len(per_chain_bias)

    pct_error = (df["error_days"] / df["actual_duration_days"].replace(0, np.nan)) * 100
    abs_pct_error = pct_error.abs()
    per_chain_pct_bias = pct_error.groupby(df["component_id"]).mean() if n else pd.Series(dtype=float)

    return {
        "n_pairs": n,
        "n_chains": n_chains,
        "bias_days": df["error_days"].mean(),
        "mae_days": df["abs_error_days"].mean(),
        "rmse_days": np.sqrt((df["error_days"] ** 2).mean()) if n else np.nan,
        "median_abs_error_days": df["abs_error_days"].median(),
        "chain_mean_bias_days": per_chain_bias.mean() if n_chains else np.nan,
        "chain_bias_sem_days": per_chain_bias.std(ddof=1) / np.sqrt(n_chains) if n_chains > 1 else np.nan,
        "bias_pct": pct_error.mean(),
        "mape_pct": abs_pct_error.mean(),
        "median_abs_pct_error": abs_pct_error.median(),
        "chain_mean_bias_pct": per_chain_pct_bias.mean() if n_chains else np.nan,
        "chain_bias_pct_sem": per_chain_pct_bias.std(ddof=1) / np.sqrt(n_chains) if n_chains > 1 else np.nan,
    }


def error_by_duration_threshold(pair_error_df, threshold_weeks=THRESHOLD_WEEKS,
                                 duration_col="chain_duration_days"):
    """Bias/MAE/RMSE of the genomic duration estimate, restricted to pairs
    whose chain's duration meets or exceeds each threshold (cumulative, so
    thresholds overlap heavily whenever a handful of chains dominate).

    Pass `duration_col="windowed_duration_days"` to categorize by how long a
    chain persisted *within the sequencing window* instead of its full
    lifetime.
    """
    rows = []
    for weeks in threshold_weeks:
        threshold_days = weeks * 7
        sub = pair_error_df[pair_error_df[duration_col] >= threshold_days]
        rows.append({"threshold_weeks": weeks, **_error_summary(sub)})
    return pd.DataFrame(rows)


def error_by_duration_bin(pair_error_df, bin_edges_weeks, duration_col="windowed_duration_days"):
    """Bias/MAE/RMSE within *non-overlapping* chain-duration bins.

    Unlike the cumulative threshold sweep (where every chain long enough to
    clear the lowest threshold keeps reappearing at every higher one), each
    pair here counts toward exactly one bin -- so this is the more direct
    "does accuracy vary by how long the chain persisted" categorization.
    """
    edges_days = [w * 7 for w in bin_edges_weeks]
    labels = [f"{lo}-{hi}wk" for lo, hi in zip(bin_edges_weeks[:-1], bin_edges_weeks[1:])]
    df = pair_error_df.copy()
    df["duration_bin"] = pd.cut(df[duration_col], bins=edges_days, labels=labels, include_lowest=True)
    rows = []
    for label, sub in df.groupby("duration_bin", observed=True):
        rows.append({"duration_bin": label, **_error_summary(sub)})
    out = pd.DataFrame(rows)
    out["duration_bin"] = pd.Categorical(out["duration_bin"], categories=labels, ordered=True)
    return out.sort_values("duration_bin").reset_index(drop=True)


def error_by_count_bin(pair_error_df, bin_edges, col="n_transmission_events"):
    """Bias/MAE/RMSE/percent-error within non-overlapping bins of a plain
    count column (e.g. transmission generations between the pair).

    Same non-overlapping-bin logic as `error_by_duration_bin`, but works
    directly in `col`'s own units -- no week-to-day conversion, since a
    generation count isn't a time unit. Bins with no pairs are silently
    dropped rather than appearing as empty rows.
    """
    labels = [f"{lo}-{hi}" for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
    df = pair_error_df.copy()
    df["count_bin"] = pd.cut(df[col], bins=bin_edges, labels=labels, include_lowest=True)
    rows = []
    for label, sub in df.groupby("count_bin", observed=True):
        rows.append({"count_bin": label, **_error_summary(sub)})
    out = pd.DataFrame(rows)
    out["count_bin"] = pd.Categorical(out["count_bin"], categories=labels, ordered=True)
    return out.sort_values("count_bin").reset_index(drop=True)


# --------------------------------------------------------------------------
# Epidemiological context: incidence, realized R, serial interval by day.
# --------------------------------------------------------------------------

MONTH_DAYS = 30  # calendar-month proxy, for widening the reporting window

def build_daily_epi_stats(tick, parent, seq_window=SEQ_WINDOW,
                           months_before=3, months_after=3):
    """Global (all-chains) per-day incidence, realized R, and serial interval.

    All three are computed from the *entire* tick/parent dicts -- restricting
    to the reporting window only happens at the end, when building the
    returned rows -- so a case infected near the window's edge still gets
    full credit for secondary infections and serial intervals that land
    outside it (no right-censoring within the reporting window itself).

    - `n_infected`: incidence, count of infections with that tick.
    - `realized_r`: mean number of secondary infections caused by the cases
      infected *on* that day (their out-degree in the transmission forest,
      regardless of when those secondary infections happened).
    - `serial_interval_days`: mean (tick of secondary infection - tick of
      infector), over the same day-t cohort's outgoing infections -- i.e.
      paired with `realized_r` by the same cohort definition, so the two can
      be read together ("cases infected on day t went on to cause R more
      infections, an average of S days later").

    Returns a DataFrame indexed by `tick` over
    [seq_window[0] - months_before*30, seq_window[1] + months_after*30].
    """
    out_degree = {}
    day_serial_sum = {}
    day_serial_count = {}
    for child, p in parent.items():
        if p is None or p not in tick or child not in tick:
            continue
        out_degree[p] = out_degree.get(p, 0) + 1
        t_p = tick[p]
        day_serial_sum[t_p] = day_serial_sum.get(t_p, 0) + (tick[child] - tick[p])
        day_serial_count[t_p] = day_serial_count.get(t_p, 0) + 1

    day_n_infected = {}
    day_out_degree_sum = {}
    for n, t in tick.items():
        day_n_infected[t] = day_n_infected.get(t, 0) + 1
        day_out_degree_sum[t] = day_out_degree_sum.get(t, 0) + out_degree.get(n, 0)

    lo = seq_window[0] - months_before * MONTH_DAYS
    hi = seq_window[1] + months_after * MONTH_DAYS

    records = []
    for d in range(lo, hi + 1):
        n_inf = day_n_infected.get(d, 0)
        n_ser = day_serial_count.get(d, 0)
        records.append({
            "tick": d,
            "n_infected": n_inf,
            "realized_r": (day_out_degree_sum.get(d, 0) / n_inf) if n_inf else np.nan,
            "n_secondary": day_serial_count.get(d, 0),
            "serial_interval_days": (day_serial_sum.get(d, 0) / n_ser) if n_ser else np.nan,
        })
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Cross-chain genomic linkages: where the phylogeny groups together tips
# from transmission chains that share no edge in the real simulation.
# --------------------------------------------------------------------------

def build_gen_tree_indices(gen_G):
    """Parent/children pointers and the single root of the genomic tree.

    `gen_G` is a single rooted tree (confirmed: 28,980 nodes, one node with
    in-degree 0). Distinct from the flat `gen_parent` built ad hoc elsewhere
    (child -> parent only, for LCA walks) -- this also keeps `children_of`,
    needed to walk the tree top-down/bottom-up for the merge analysis below.
    """
    parent = {}
    children_of = {}
    for u, v in gen_G.edges():
        parent[v] = u
        children_of.setdefault(u, []).append(v)
    roots = [n for n in gen_G.nodes if n not in parent]
    if len(roots) != 1:
        raise ValueError(f"expected a single-rooted tree, found {len(roots)} roots")
    return parent, children_of, roots[0]


def gen_postorder(children_of, root):
    """Node order with every descendant appearing before its ancestor.

    Non-recursive (tree depth isn't bounded a priori) and works for any
    branching factor, not just binary -- this tree has a few large polytomies
    (one node has 86 children). Standard two-stack trick: a preorder DFS
    (stack1/stack2) visits a node before any of its descendants are even
    discovered, so reversing stack2 guarantees descendants precede ancestors.
    """
    stack1 = [root]
    stack2 = []
    while stack1:
        node = stack1.pop()
        stack2.append(node)
        stack1.extend(children_of.get(node, []))
    return stack2[::-1]


def compute_chain_merges(children_of, order, tip_chain, tick_of_gen_leaf,
                          gen_num_date, small_clade_cap=8):
    """Find every point in the genomic tree where two *distinct* transmission
    chains (real weakly-connected components of `epi_G`, sharing no
    transmission edge whatsoever) sit as descendants of a common ancestor --
    i.e. everywhere the phylogeny "links" a chain to a sample that isn't
    actually a member of it.

    For each internal node, every pair of its own children whose descendant
    chain-sets are disjoint contributes one row per (chain_a, chain_b) pair
    newly brought together there. This is exactly each chain pair's MRCA in
    the phylogeny (or one of several, if a chain's samples aren't
    monophyletic -- itself worth knowing, so instances aren't deduplicated).
    `tip_chain` deliberately omits the two non-EHip reference/outgroup
    leaves (`Wuhan/Hu-1/2019`, `21L`), which contribute no chain identity.

    `tip_list` (exact descendant tip names) is retained per node only while
    the subtree has <= `small_clade_cap` sequenced tips -- large clades are
    the *expected*, deep/early merges this analysis isn't scrutinizing for
    tip-level precision; small ones are the interesting, possibly-misleading
    case, and stay cheap to store precisely because they're small.

    Returns (merges_df, node_stats_df). `node_stats_df` is indexed by every
    node with a resolved chain identity (n_chains, n_tips, min_tick,
    num_date) for characterizing overall tree structure (e.g. is chain
    diversity concentrated near the root).
    """
    desc_chains, n_tips, min_tick, tip_list = {}, {}, {}, {}
    rows = []

    for node in order:
        kids = children_of.get(node, [])
        if not kids:
            cid = tip_chain.get(node)
            t = tick_of_gen_leaf.get(node)
            desc_chains[node] = frozenset({cid}) if cid is not None else frozenset()
            n_tips[node] = 1 if t is not None else 0
            min_tick[node] = t
            tip_list[node] = [node] if t is not None else []
            continue

        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                ci, cj = kids[i], kids[j]
                for a in desc_chains[ci]:
                    for b in desc_chains[cj]:
                        if a == b:
                            continue
                        if a < b:
                            lo, hi, side_lo, side_hi = a, b, ci, cj
                        else:
                            lo, hi, side_lo, side_hi = b, a, cj, ci
                        rows.append({
                            "node": node, "chain_a": lo, "chain_b": hi,
                            "num_date": gen_num_date.get(node),
                            "n_tips_a": n_tips[side_lo], "n_tips_b": n_tips[side_hi],
                            "tips_a": tip_list.get(side_lo),
                            "tips_b": tip_list.get(side_hi),
                        })

        seen = set()
        total_tips = 0
        mn = None
        combined_tips = []
        capped = False
        for child in kids:
            seen |= desc_chains[child]
            total_tips += n_tips[child]
            if min_tick[child] is not None:
                mn = min_tick[child] if mn is None else min(mn, min_tick[child])
            if not capped:
                cl = tip_list.get(child)
                if cl is None:
                    capped = True
                else:
                    combined_tips.extend(cl)
                    if len(combined_tips) > small_clade_cap:
                        capped = True
        desc_chains[node] = frozenset(seen)
        n_tips[node] = total_tips
        min_tick[node] = mn
        tip_list[node] = None if capped else combined_tips

    merges_df = pd.DataFrame(rows)
    node_stats_df = pd.DataFrame({
        "node": list(desc_chains.keys()),
        "n_chains": [len(desc_chains[n]) for n in desc_chains],
        "n_tips": [n_tips[n] for n in desc_chains],
        "min_tick": [min_tick[n] for n in desc_chains],
        "num_date": [gen_num_date.get(n) for n in desc_chains],
    })
    return merges_df, node_stats_df


def annotate_merge_earliest(merges_df, tip_chain, tick_of_gen_leaf):
    """For each merge row, was the specific tip anchoring each side actually
    that chain's own earliest sequenced sample?

    Only answerable precisely when that side's subtree is small enough to
    have kept an exact `tips_a`/`tips_b` list (see `compute_chain_merges`);
    otherwise the flag is NaN rather than a guess. Chain-wide earliest tick
    is computed once from every sequenced tip of that chain, not just the
    ones on this side of this particular merge.
    """
    chain_earliest_tick = {}
    for gname, cid in tip_chain.items():
        t = tick_of_gen_leaf[gname]
        if cid not in chain_earliest_tick or t < chain_earliest_tick[cid]:
            chain_earliest_tick[cid] = t

    def side_min_tick(tips, cid):
        if tips is None:
            return None
        own = [tick_of_gen_leaf[t] for t in tips if tip_chain.get(t) == cid]
        return min(own) if own else None

    df = merges_df.copy()
    df["min_tick_a"] = [side_min_tick(t, c) for t, c in zip(df["tips_a"], df["chain_a"])]
    df["min_tick_b"] = [side_min_tick(t, c) for t, c in zip(df["tips_b"], df["chain_b"])]
    df["chain_a_earliest_tick"] = df["chain_a"].map(chain_earliest_tick)
    df["chain_b_earliest_tick"] = df["chain_b"].map(chain_earliest_tick)
    df["a_is_earliest"] = np.where(df["min_tick_a"].notna(),
                                    df["min_tick_a"] <= df["chain_a_earliest_tick"], np.nan)
    df["b_is_earliest"] = np.where(df["min_tick_b"].notna(),
                                    df["min_tick_b"] <= df["chain_b_earliest_tick"], np.nan)
    return df


def summarize_merge_nodes(merges_df):
    """Collapse `merges_df`'s (node, chain_a, chain_b) rows down to one row
    per distinct merge *node*.

    The row count in `merges_df` is a combinatorial artifact -- a hub node
    with 40 chains on one side and 3 on the other produces 120 pair-rows for
    a single structural event -- so it isn't a meaningful count of "how many
    places the phylogeny crosses chains". This is: `smallest_side_tips` (how
    shallow the merge is -- the smallest sequenced-tip count among any
    contributing sibling) and `is_pure_cherry` (the node's entire content is
    two individual tips from two different chains -- the most literal
    reading of "the tree links this sample to a chain it isn't in") are the
    two fields that matter for characterizing severity.
    """
    def _agg(g):
        chains = set(g["chain_a"]) | set(g["chain_b"])
        smallest = min(g["n_tips_a"].min(), g["n_tips_b"].min())
        is_cherry = bool(((g["n_tips_a"] == 1) & (g["n_tips_b"] == 1)).any() and len(chains) == 2)
        return pd.Series({
            "n_chains_involved": len(chains),
            "n_pair_rows": len(g),
            "smallest_side_tips": smallest,
            "num_date": g["num_date"].iloc[0],
            "is_pure_cherry": is_cherry,
        })
    return merges_df.groupby("node").apply(_agg, include_groups=False).reset_index()


def classify_merge_anchors(merges_df, tip_chain):
    """One row per distinct (node, chain) lone-tip event: every place a
    chain contributes exactly *one* sequenced tip to one side of a
    cross-chain merge, deduplicated across the combinatorial explosion of
    whatever sits on the other side (an anomalous tip at a hub node with 40
    chains opposite it is one event, not 40 -- see `summarize_merge_nodes`).

    Splits into two very different situations:
    - a genuine singleton chain (only ever sequenced once) -- "is this the
      earliest sample" is tautological here, since it's the only sample;
      this is about sampling sparsity, not the phylogeny doing anything odd.
    - a multi-tip chain caught contributing a lone outlier tip outside its
      own main clade -- a meaningful test of whether that outlier is
      genuinely the chain's earliest (least-diverged) sample, or a later one
      that landed elsewhere for other reasons (e.g. placement uncertainty
      within a large, long-running, densely-sampled chain).
    """
    chain_total_tips = pd.Series(tip_chain).value_counts()
    singleton_chains = set(chain_total_tips[chain_total_tips == 1].index)

    def _side(chain_col, tips_col, tick_col, earliest_col, is_earliest_col):
        sub = merges_df[merges_df[tips_col] == 1].copy()
        sub["is_singleton_chain"] = sub[chain_col].isin(singleton_chains)
        sub = sub.drop_duplicates(subset=["node", chain_col])
        return sub.rename(columns={
            chain_col: "chain", tick_col: "min_tick",
            earliest_col: "chain_earliest_tick", is_earliest_col: "is_earliest",
        })[["node", "chain", "num_date", "min_tick", "chain_earliest_tick",
            "is_earliest", "is_singleton_chain"]]

    a = _side("chain_a", "n_tips_a", "min_tick_a", "chain_a_earliest_tick", "a_is_earliest")
    b = _side("chain_b", "n_tips_b", "min_tick_b", "chain_b_earliest_tick", "b_is_earliest")
    return pd.concat([a, b]).drop_duplicates(subset=["node", "chain"]).reset_index(drop=True)


def summarize_chain_linkages(merges_df, tip_chain):
    """One row per transmission chain that appears in >=1 cross-chain merge:
    how many distinct merge nodes it shows up in, and how many distinct
    *other* chains it ever gets grouped with there.

    Deliberately not deduplicated the way `summarize_merge_nodes` /
    `classify_merge_anchors` are -- a chain that meets 40 other chains at one
    big hub node legitimately has 40 distinct linkages from its own point of
    view, even though that's one structural event from the node's point of
    view. Double-counting across nodes is expected and left in: the point
    here is comparing chains' overall cross-chain exposure against each
    other (does chain X show up everywhere, or just once), not counting
    independent events (that's what `summarize_merge_nodes` is for).
    """
    long = pd.concat([
        merges_df[["node", "chain_a", "chain_b"]].rename(columns={"chain_a": "chain", "chain_b": "other"}),
        merges_df[["node", "chain_a", "chain_b"]].rename(columns={"chain_b": "chain", "chain_a": "other"}),
    ], ignore_index=True)
    chain_total_tips = pd.Series(tip_chain).value_counts()
    out = long.groupby("chain").agg(
        n_merge_nodes=("node", "nunique"),
        n_pair_rows=("other", "size"),
        n_other_chains=("other", "nunique"),
    ).reset_index()
    out["n_sequenced_tips"] = out["chain"].map(chain_total_tips)
    return out.sort_values("n_other_chains", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Visualizing one cross-chain merge in its local tree context.
# --------------------------------------------------------------------------

def extract_context_subtree(node, children_of, parent, n_tips, max_tips=30, max_ancestor_steps=6):
    """Climb from a chosen merge `node` toward the root, one ancestor at a
    time, while the ancestor's own subtree stays under `max_tips` (or until
    `max_ancestor_steps` ancestors have been climbed) -- giving a bare
    cross-chain merge some surrounding same-chain context to be read
    against, rather than showing it in isolation.

    Returns the resulting ancestor node (or `node` itself if its parent's
    subtree already exceeds `max_tips`). `n_tips` is the per-node descendant
    tip count from `compute_chain_merges`'s `node_stats_df`.
    """
    anchor = node
    steps = 0
    while steps < max_ancestor_steps:
        p = parent.get(anchor)
        if p is None:
            break
        if n_tips.get(p, 0) > max_tips:
            break
        anchor = p
        steps += 1
    return anchor


def subtree_nodes_edges(anchor, children_of):
    """All nodes and (parent, child) edges in the subtree rooted at `anchor`."""
    nodes = []
    edges = []
    stack = [anchor]
    while stack:
        n = stack.pop()
        nodes.append(n)
        for c in children_of.get(n, []):
            edges.append((n, c))
            stack.append(c)
    return nodes, edges


def compute_tree_layout(anchor, children_of, gen_num_date):
    """x/y layout for drawing the subtree rooted at `anchor` as a
    rectangular phylogram: x is each node's own `num_date` (so horizontal
    position reflects inferred time), y is leaf order (tips keep their
    left-to-right traversal order) with each internal node at the mean y of
    its children -- the standard recursive tree-drawing layout.

    Returns (pos, edges, leaves_in_order); `pos` is {node: (x, y)}. Assumes
    the subtree is small (as produced by `extract_context_subtree`) --
    recursive, not iterative.
    """
    leaves_in_order = []

    def _order(n):
        kids = children_of.get(n, [])
        if not kids:
            leaves_in_order.append(n)
            return
        for c in kids:
            _order(c)

    _order(anchor)
    y_of = {leaf: i for i, leaf in enumerate(leaves_in_order)}

    pos = {}
    edges = []

    def _layout(n):
        kids = children_of.get(n, [])
        if not kids:
            pos[n] = (gen_num_date.get(n), y_of[n])
            return y_of[n]
        ys = [_layout(c) for c in kids]
        for c in kids:
            edges.append((n, c))
        y = sum(ys) / len(ys)
        pos[n] = (gen_num_date.get(n), y)
        return y

    _layout(anchor)
    return pos, edges, leaves_in_order


# --------------------------------------------------------------------------
# Demographic/geographic context for sequenced tips: age band and county.
# --------------------------------------------------------------------------

TIP_METADATA_TSV = (
    f"{WORKING_DIR}/data/nextstrain_tree_sample/"
    "run_03_vadelta_2026_03_22_128to428_SURS/metadata_with_index.tsv"
)

# Ordinal order over this run's five age_group buckets (also present
# directly on gen_G's tip node attributes) -- lets "how far apart in age"
# be a plain integer difference instead of a category mismatch.
AGE_GROUP_ORDER = {
    "Preschool (0-4)": 0, "Student (5-17)": 1, "Adult (18-49)": 2,
    "Older adult (50-64)": 3, "Senior (65+)": 4,
}

# Virginia county FIPS -> name, sourced once from the population linelist
# (`data/initial_linelist.csv.xz`) purely as a lookup table: that file's own
# (sim_pid, sim_tick) identifiers are from an unrelated simulation replicate
# (0 of this run's 14,791 sequenced tips match it directly), but the FIPS
# codes themselves are standard, stable Census codes independent of replicate.
VA_FIPS_TO_COUNTY = {
    51001: 'Accomack VA', 51003: 'Albemarle VA', 51005: 'Alleghany VA', 51007: 'Amelia VA',
    51009: 'Amherst VA', 51011: 'Appomattox VA', 51013: 'Arlington VA', 51015: 'Augusta VA',
    51017: 'Bath VA', 51019: 'Bedford VA', 51021: 'Bland VA', 51023: 'Botetourt VA',
    51025: 'Brunswick VA', 51027: 'Buchanan VA', 51029: 'Buckingham VA', 51031: 'Campbell VA',
    51033: 'Caroline VA', 51035: 'Carroll VA', 51036: 'Charles City VA', 51037: 'Charlotte VA',
    51041: 'Chesterfield VA', 51043: 'Clarke VA', 51045: 'Craig VA', 51047: 'Culpeper VA',
    51049: 'Cumberland VA', 51051: 'Dickenson VA', 51053: 'Dinwiddie VA', 51057: 'Essex VA',
    51059: 'Fairfax VA', 51061: 'Fauquier VA', 51063: 'Floyd VA', 51065: 'Fluvanna VA',
    51067: 'Franklin VA', 51069: 'Frederick VA', 51071: 'Giles VA', 51073: 'Gloucester VA',
    51075: 'Goochland VA', 51077: 'Grayson VA', 51079: 'Greene VA', 51081: 'Greensville VA',
    51083: 'Halifax VA', 51085: 'Hanover VA', 51087: 'Henrico VA', 51089: 'Henry VA',
    51091: 'Highland VA', 51093: 'Isle of Wight VA', 51095: 'James City VA',
    51097: 'King and Queen VA', 51099: 'King George VA', 51101: 'King William VA',
    51103: 'Lancaster VA', 51105: 'Lee VA', 51107: 'Loudoun VA', 51109: 'Louisa VA',
    51111: 'Lunenburg VA', 51113: 'Madison VA', 51115: 'Mathews VA', 51117: 'Mecklenburg VA',
    51119: 'Middlesex VA', 51121: 'Montgomery VA', 51125: 'Nelson VA', 51127: 'New Kent VA',
    51131: 'Northampton VA', 51133: 'Northumberland VA', 51135: 'Nottoway VA',
    51137: 'Orange VA', 51139: 'Page VA', 51141: 'Patrick VA', 51143: 'Pittsylvania VA',
    51145: 'Powhatan VA', 51147: 'Prince Edward VA', 51149: 'Prince George VA',
    51153: 'Prince William VA', 51155: 'Pulaski VA', 51157: 'Rappahannock VA',
    51159: 'Richmond VA', 51161: 'Roanoke VA', 51163: 'Rockbridge VA', 51165: 'Rockingham VA',
    51167: 'Russell VA', 51169: 'Scott VA', 51171: 'Shenandoah VA', 51173: 'Smyth VA',
    51175: 'Southampton VA', 51177: 'Spotsylvania VA', 51179: 'Stafford VA', 51181: 'Surry VA',
    51183: 'Sussex VA', 51185: 'Tazewell VA', 51187: 'Warren VA', 51191: 'Washington VA',
    51193: 'Westmoreland VA', 51195: 'Wise VA', 51197: 'Wythe VA', 51199: 'York VA',
    51510: 'Alexandria City VA', 51520: 'Bristol City VA', 51530: 'Buena Vista City VA',
    51540: 'Charlottesville City VA', 51550: 'Chesapeake City VA',
    51570: 'Colonial Heights City VA', 51580: 'Covington City VA', 51590: 'Danville City VA',
    51595: 'Emporia City VA', 51600: 'Fairfax City VA', 51610: 'Falls Church City VA',
    51620: 'Franklin City VA', 51630: 'Fredericksburg City VA', 51640: 'Galax City VA',
    51650: 'Hampton City VA', 51660: 'Harrisonburg City VA', 51670: 'Hopewell City VA',
    51678: 'Lexington City VA', 51680: 'Lynchburg City VA', 51683: 'Manassas City VA',
    51685: 'Manassas Park City VA', 51690: 'Martinsville City VA', 51700: 'Newport News City VA',
    51710: 'Norfolk City VA', 51720: 'Norton City VA', 51730: 'Petersburg City VA',
    51735: 'Poquoson City VA', 51740: 'Portsmouth City VA', 51750: 'Radford City VA',
    51760: 'Richmond City VA', 51770: 'Roanoke City VA', 51775: 'Salem City VA',
    51790: 'Staunton City VA', 51800: 'Suffolk City VA', 51810: 'Virginia Beach City VA',
    51820: 'Waynesboro City VA', 51830: 'Williamsburg City VA', 51840: 'Winchester City VA',
}


def load_tip_demographics(path=TIP_METADATA_TSV):
    """{gen tip name -> {"age_group", "age_band", "county"}} for every
    sequenced tip (and the two reference/outgroup leaves, which get all-None).

    Sourced from the augur run's own input metadata rather than `gen_G`
    directly: `age_group` agrees with what's already on the tip node
    attributes, but county name is not on the graph at all -- this file's
    own `county` column is entirely empty (0 of 14,793 rows), so county name
    is resolved from `county_fips` via `VA_FIPS_TO_COUNTY` instead.

    A handful of metadata rows have ragged column counts (free-text fields
    occasionally confuse the C parser), so this reads with the slower
    python engine and warns on, rather than fails on, bad lines.
    """
    df = pd.read_csv(path, sep="\t", engine="python", on_bad_lines="warn",
                      usecols=["strain", "age_group", "county_fips"])
    out = {}
    for row in df.itertuples(index=False):
        age_group = row.age_group if isinstance(row.age_group, str) else None
        fips = int(row.county_fips) if pd.notna(row.county_fips) else None
        out[row.strain] = {
            "age_group": age_group,
            "age_band": AGE_GROUP_ORDER.get(age_group),
            "county": VA_FIPS_TO_COUNTY.get(fips),
        }
    return out


def annotate_pair_demographics(pairs_df, tip_a_col, tip_b_col, tip_demo):
    """Add `age_band_gap` (absolute difference in ordinal age band, NaN if
    either side's age is unknown) and `same_county` (True/False, NaN if
    either side's county is unknown) for a DataFrame of genomic tip-name
    pairs -- works the same way regardless of whether the pairs came from a
    cross-chain merge or a within-chain sample, so the two are directly
    comparable.
    """
    def _lookup(col, key):
        return pairs_df[col].map(lambda t: tip_demo.get(t, {}).get(key))

    age_a = _lookup(tip_a_col, "age_band")
    age_b = _lookup(tip_b_col, "age_band")
    county_a = _lookup(tip_a_col, "county")
    county_b = _lookup(tip_b_col, "county")

    out = pairs_df.copy()
    out["age_band_gap"] = (age_a - age_b).abs()
    out["same_county"] = np.where(county_a.notna() & county_b.notna(), county_a == county_b, np.nan)
    return out


def compute_cross_chain_duration_errors(cross_pairs_df, gen_num_date, tick_of_gen_leaf,
                                         tip_chain, chain_seed_tick):
    """Genomic-asserted separation vs. a provable lower bound on the true
    separation, for tip pairs sitting on opposite sides of a cross-chain merge.

    These pairs have *no* common ancestor in the transmission forest at all
    (they're in different weakly-connected components), so the Section 5
    error -- genomic estimate minus true time through a shared infector -- is
    undefined here: there is no shared infector, and the honest "truth" is
    not a larger number but "never related within this simulation".

    What *is* well defined is a lower bound. Any common ancestry between two
    separately-seeded chains has to predate both seeds, so the true
    separation is at least `(tick_a - seed_a) + (tick_b - seed_b)`, walking
    each tip back to its own chain's introduction. `min_error_days` is the
    genomic estimate minus that bound -- a conservative, signed statement of
    how far the phylogeny falls short of even the most generous reading of
    the truth, directly comparable in days to the within-chain bias.

    The pair's genomic MRCA is the merge node itself by construction (the two
    sides are disjoint child subtrees of it), so no LCA search is needed.
    `chain_seed_tick` maps component_id -> that chain's root tick
    (`component_df.min_tick`; a parent's tick never exceeds its children's,
    so the component minimum is the seed).
    """
    day_scale = 365.25  # num_date is decimal-year
    records = []
    for row in cross_pairs_df.itertuples(index=False):
        a, b, node = row.tip_a, row.tip_b, row.node
        tick_a, tick_b = tick_of_gen_leaf.get(a), tick_of_gen_leaf.get(b)
        chain_a, chain_b = tip_chain.get(a), tip_chain.get(b)
        if tick_a is None or tick_b is None or chain_a is None or chain_b is None:
            continue
        seed_a, seed_b = chain_seed_tick.get(chain_a), chain_seed_tick.get(chain_b)
        node_date = gen_num_date.get(node)
        if seed_a is None or seed_b is None or node_date is None:
            continue

        genomic = ((gen_num_date[a] - node_date) + (gen_num_date[b] - node_date)) * day_scale
        min_true = (tick_a - seed_a) + (tick_b - seed_b)
        records.append({
            "node": node, "tip_a": a, "tip_b": b,
            "chain_a": chain_a, "chain_b": chain_b,
            "genomic_duration_days": genomic,
            "min_true_duration_days": min_true,
            "min_error_days": genomic - min_true,
            "calendar_gap_days": abs(tick_a - tick_b),
        })
    return pd.DataFrame(records)


def build_cross_chain_tip_pairs(merges_df, tip_chain):
    """Explicit (tip_a, tip_b) pairs sitting on opposite sides of a
    cross-chain merge, for the merges where both sides kept an exact tip list.

    A merge row's `tips_a`/`tips_b` are the *whole* tip lists of the two
    sibling subtrees, and a subtree can carry more than one chain -- so the
    raw cartesian product of the two sides also contains **same-chain** pairs:
    both tips from one chain that happens to appear on both sides of the node
    (a chain whose samples aren't monophyletic there). Those are ordinary
    within-chain relatives that do share a real transmission ancestor, not
    cross-chain linkages, and including them silently contaminates any
    cross-vs-within comparison -- in this build they are 263 of the 927 raw
    products, concentrated in the big non-monophyletic chains (164, 240, 562,
    136, 187). Only pairs whose tips genuinely belong to different chains are
    kept, so every returned pair is guaranteed to have no common ancestor in
    the transmission forest at all. The pair's genomic MRCA is the merge node.
    """
    rows = []
    both = merges_df[merges_df["tips_a"].notna() & merges_df["tips_b"].notna()]
    for row in both.itertuples(index=False):
        for ta in row.tips_a:
            for tb in row.tips_b:
                ca, cb = tip_chain.get(ta), tip_chain.get(tb)
                if ca is None or cb is None or ca == cb:
                    continue
                rows.append({"node": row.node, "tip_a": ta, "tip_b": tb,
                             "chain_a": ca, "chain_b": cb})
    return pd.DataFrame(rows).drop_duplicates(subset=["tip_a", "tip_b"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Ascertainment over time: what share of infections were ever sequenced.
# --------------------------------------------------------------------------

def build_ascertainment_timeseries(tick, captured_epi_nodes, window=None):
    """Per-day infection count, sequenced count, and the ascertainment rate.

    `tick` covers every infection in the forest; `captured_epi_nodes` is the
    subset that was sequenced, so the ratio per day is the empirical
    ascertainment (detection) rate that day -- the simplest descriptive
    summary of how much of the epidemic genomic surveillance actually saw,
    and how that changed over the run.

    Rows span every day from the first to the last infection, so days with
    zero sequencing still appear (rate 0) rather than being silently dropped.
    `in_seq_window` flags days inside `window` (lo, hi) when given.
    """
    total, seq = {}, {}
    for n, t in tick.items():
        total[t] = total.get(t, 0) + 1
    for n in captured_epi_nodes:
        t = tick.get(n)
        if t is not None:
            seq[t] = seq.get(t, 0) + 1

    if not total:
        return pd.DataFrame(columns=["tick", "n_infections", "n_sequenced",
                                      "ascertainment_rate", "in_seq_window"])

    rows = []
    for d in range(min(total), max(total) + 1):
        n_inf = total.get(d, 0)
        n_seq = seq.get(d, 0)
        rows.append({
            "tick": d,
            "n_infections": n_inf,
            "n_sequenced": n_seq,
            "ascertainment_rate": (n_seq / n_inf) if n_inf else np.nan,
            "in_seq_window": (window is not None and window[0] <= d <= window[1]),
        })
    return pd.DataFrame(rows)
