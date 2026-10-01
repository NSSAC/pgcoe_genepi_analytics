"""Loaders and statistics for the cross-variant, cross-state wave-correlation
analysis: did states that ran unusually high on one variant wave later run
unusually high (or low) on a different variant 6-12 months on, versus their
peers?  An ecological design over `data/variants/outbreak_info_variants_states_long.csv`.

REQUIRES a local copy of pango-designation's `alias_key.json` at
`data/pango_alias_key.json`.  `pango_aliasor.Aliasor()` fetches that file from
GitHub when given no path, and this cluster's compute nodes have no outbound
internet -- fetch it once from a machine that does (e.g.
https://raw.githubusercontent.com/cov-lineages/pango-designation/master/pango_designation/alias_key.json)
and place it at the path above.  `load_aliasor()` below raises loudly instead
of ever attempting the network call.

The source CSV mixes two row shapes in one file: national rows (`division`
empty, `fips == "US"`) carry pre-computed `proportion`/CI/rolling columns;
state rows (`division` = full state name, `fips` = state FIPS) carry only
`lineage_count` -- every proportion used here is recomputed from state-level
counts post-clustering, never read from the file's own `proportion` column.
"""

import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, false_discovery_control
from scipy.stats import t as _student_t

from ascertainment_data import STATE_FIPS, FIPS_STATE, _STATE_NAMES

WORKING_DIR = "/sfs/gpfs/tardis/project/bii_nssac/people/bl4zc/pgcoe_genepi_analytics"
VARIANTS_LONG_FILE = f"{WORKING_DIR}/data/variants/outbreak_info_variants_states_long.csv"
PANGO_ALIAS_KEY_PATH = f"{WORKING_DIR}/data/pango_alias_key.json"

# fips codes present in the source file that aren't in ascertainment_data.STATE_FIPS
TERRITORY_FIPS = {"60": "AS", "66": "GU", "69": "MP", "72": "PR", "78": "VI"}

# Standard 9 Census Bureau divisions, used to pool low-volume states rather
# than discard them outright.  DC folded into South Atlantic (its Census
# division).  No `us`/`pycountry` package is installed in the analysis env,
# so this is just hardcoded -- it never changes.
CENSUS_DIVISIONS = {
    "CT": "New England", "ME": "New England", "MA": "New England",
    "NH": "New England", "RI": "New England", "VT": "New England",
    "NJ": "Mid-Atlantic", "NY": "Mid-Atlantic", "PA": "Mid-Atlantic",
    "IL": "East North Central", "IN": "East North Central", "MI": "East North Central",
    "OH": "East North Central", "WI": "East North Central",
    "IA": "West North Central", "KS": "West North Central", "MN": "West North Central",
    "MO": "West North Central", "NE": "West North Central", "ND": "West North Central",
    "SD": "West North Central",
    "DE": "South Atlantic", "DC": "South Atlantic", "FL": "South Atlantic",
    "GA": "South Atlantic", "MD": "South Atlantic", "NC": "South Atlantic",
    "SC": "South Atlantic", "VA": "South Atlantic", "WV": "South Atlantic",
    "AL": "East South Central", "KY": "East South Central", "MS": "East South Central",
    "TN": "East South Central",
    "AR": "West South Central", "LA": "West South Central", "OK": "West South Central",
    "TX": "West South Central",
    "AZ": "Mountain", "CO": "Mountain", "ID": "Mountain", "MT": "Mountain",
    "NV": "Mountain", "NM": "Mountain", "UT": "Mountain", "WY": "Mountain",
    "AK": "Pacific", "CA": "Pacific", "HI": "Pacific", "OR": "Pacific", "WA": "Pacific",
}

# Prevalence-gated collapsing: a lineage only merges into its parent if its
# *current* cluster is still below this share of national prevalence at its
# peak week, and it may climb at most MAX_CLIMBS times.  This is a ceiling,
# not a mandate -- a lineage that clears the threshold on its own (e.g. BA.5,
# BQ.1) never merges, regardless of depth.  Sweep SIGNIFICANCE_THRESHOLD_GRID
# in the notebook to see how sensitive the correlation results are to this
# choice, per the caveat that every collapse step dilutes the immune signal.
MAX_CLIMBS = 2
# 0.01 keeps known-important-but-diluted raw labels (e.g. bare "BA.5", whose
# own literal share is ~1.8% even though its numbered descendants collectively
# dominated a whole wave) from merging away at the default setting; the sweep
# grid's higher values (0.02, 0.05) deliberately cross that point so the
# notebook can show the trade-off explicitly rather than hide it.
DEFAULT_SIGNIFICANCE_THRESHOLD = 0.01
SIGNIFICANCE_THRESHOLD_GRID = [0.005, 0.01, 0.02, 0.05]
DEFAULT_OTHER_THRESHOLD = 0.005  # terminal floor after max climbs -> "Other"

DEFAULT_VOLUME_THRESHOLD = 20_000  # total sequences/state over the full window
WAVE_HALF_MAX_FRACTION = 0.5
MIN_WAVE_PEAK_SHARE = DEFAULT_SIGNIFICANCE_THRESHOLD  # below this, don't treat it as a "wave" at all
MIN_WAVE_WINDOW_WEEKS = 2  # a 1-week window is usually one state's reporting blip, not a wave
LAG_MONTHS_RANGE = (6, 12)
FDR_ALPHA = 0.05
WEEK_CONVENTION = "W-SAT"  # week-ending Saturday, matching NHSN elsewhere in the repo


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_variant_states_long(path=VARIANTS_LONG_FILE):
    """Raw outbreak.info long file, both national and state rows, dates parsed."""
    df = pd.read_csv(path, dtype={"fips": str, "division": str}, low_memory=False)
    n_before = len(df)
    df = df.dropna(subset=["lineage"])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"load_variant_states_long: dropped {n_dropped} rows with lineage=NaN")
    df["date"] = pd.to_datetime(df["date"])
    return df


def split_national_and_state_rows(df, include_territories=False):
    """Split into (national_df, state_df).

    State identity is joined via `fips` -> FIPS_STATE, never by string-matching
    `division` -- the file spells DC as "Washington DC" while this repo's own
    `_STATE_NAMES["DC"]` is "District of Columbia", so a name join silently
    drops DC.
    """
    national_df = df[df["fips"] == "US"].copy()
    state_df = df[df["fips"] != "US"].copy()

    if not include_territories:
        territory_mask = state_df["fips"].isin(TERRITORY_FIPS)
        n_territory = territory_mask.sum()
        if n_territory:
            print(f"split_national_and_state_rows: dropping {n_territory} territory rows "
                  f"({sorted(state_df.loc[territory_mask, 'fips'].map(TERRITORY_FIPS).unique())})")
        state_df = state_df[~territory_mask]

    state_df["state"] = state_df["fips"].map(FIPS_STATE)
    unmapped = state_df["state"].isna()
    if unmapped.any():
        raise ValueError(f"unmapped fips codes: {sorted(state_df.loc[unmapped, 'fips'].unique())}")
    return national_df, state_df


# --------------------------------------------------------------------------
# Weekly aggregation
# --------------------------------------------------------------------------

def assign_week_ending(dates, convention=WEEK_CONVENTION):
    """Bucket dates to their week-ending label (Saturday, by default)."""
    dates = pd.to_datetime(dates)
    return dates.dt.to_period(convention).dt.end_time.dt.normalize()


def weekly_state_lineage_counts(state_df):
    """(state, lineage, week_ending) -> summed lineage_count."""
    df = state_df.copy()
    df["week_ending"] = assign_week_ending(df["date"])
    return (df.groupby(["state", "lineage", "week_ending"])["lineage_count"]
              .sum().reset_index())


def weekly_state_totals(weekly_counts):
    """(state, week_ending) -> total sequenced genomes that week (all lineages)."""
    return (weekly_counts.groupby(["state", "week_ending"])["lineage_count"]
              .sum().rename("total_count").reset_index())


def weekly_state_shares(weekly_counts, weekly_totals):
    """Merge counts + totals, share = lineage_count / total_count."""
    out = weekly_counts.merge(weekly_totals, on=["state", "week_ending"], how="left")
    out["share"] = out["lineage_count"] / out["total_count"]
    return out


# --------------------------------------------------------------------------
# Clustering (prevalence-gated collapsing via pango_aliasor)
# --------------------------------------------------------------------------

def load_aliasor(alias_path=PANGO_ALIAS_KEY_PATH):
    """Load a local pango-designation alias_key.json.  Never falls back to a
    network fetch (this cluster has no outbound internet)."""
    if not os.path.exists(alias_path):
        raise FileNotFoundError(
            f"{alias_path} not found. Fetch it from a machine with internet access: "
            "https://raw.githubusercontent.com/cov-lineages/pango-designation/master/"
            "pango_designation/alias_key.json"
        )
    from pango_aliasor.aliasor import Aliasor
    return Aliasor(alias_file=alias_path)


def national_weekly_cluster_shares(weekly_counts, weekly_totals, group_col="lineage"):
    """National weekly share per value of `group_col` (raw lineage, or a
    cluster column after `assign_clusters`).  Recomputed from the state
    panel, not the file's own `fips=="US"` rows, so it reflects whatever
    clustering has (or hasn't) been applied."""
    nat_counts = (weekly_counts.groupby([group_col, "week_ending"])["lineage_count"]
                  .sum().rename("count").reset_index())
    nat_totals = (weekly_totals.groupby("week_ending")["total_count"]
                  .sum().rename("total").reset_index())
    out = nat_counts.merge(nat_totals, on="week_ending", how="left")
    out["share"] = out["count"] / out["total"]
    return out


def raw_lineage_prevalence(weekly_counts, weekly_totals):
    """dict[lineage] -> national peak weekly share, at raw-lineage resolution."""
    nat = national_weekly_cluster_shares(weekly_counts, weekly_totals, group_col="lineage")
    return nat.groupby("lineage")["share"].max().to_dict()


def climb_step(aliasor, cluster_of, lineage_prevalence, significance_threshold):
    """One round: any lineage whose *current* cluster prevalence is still
    below `significance_threshold` climbs to its cluster's parent; lineages
    whose current cluster already clears the bar are left untouched."""
    cluster_prevalence = defaultdict(float)
    for lineage, cluster in cluster_of.items():
        cluster_prevalence[cluster] += lineage_prevalence.get(lineage, 0.0)

    updated = dict(cluster_of)
    for lineage, cluster in cluster_of.items():
        if cluster_prevalence[cluster] >= significance_threshold:
            continue
        parent = aliasor.parent(cluster)
        if parent == "":  # already at the root, nothing higher to climb to
            continue
        updated[lineage] = parent
    return updated


def build_prevalence_aware_clusters(aliasor, lineage_prevalence, max_climbs=MAX_CLIMBS,
                                     significance_threshold=DEFAULT_SIGNIFICANCE_THRESHOLD,
                                     other_threshold=DEFAULT_OTHER_THRESHOLD,
                                     other_label="Other"):
    """dict[lineage] -> cluster_label, dict[lineage] -> climb_count.

    Bottom-up merge, capped at `max_climbs` rounds: a lineage only climbs
    while its current cluster's total prevalence is below
    `significance_threshold`, so a lineage that's already prevalent on its
    own (e.g. BA.5, BQ.1) is frozen at climbs=0 and never merges away, even
    if less-prevalent siblings nearby get merged into their shared parent.
    Anything still below `other_threshold` after the climb budget is spent
    is folded into `other_label`.
    """
    cluster_of = {lineage: lineage for lineage in lineage_prevalence}
    climb_count = {lineage: 0 for lineage in lineage_prevalence}

    for _ in range(max_climbs):
        updated = climb_step(aliasor, cluster_of, lineage_prevalence, significance_threshold)
        for lineage in cluster_of:
            if updated[lineage] != cluster_of[lineage]:
                climb_count[lineage] += 1
        cluster_of = updated

    cluster_prevalence = defaultdict(float)
    for lineage, cluster in cluster_of.items():
        cluster_prevalence[cluster] += lineage_prevalence.get(lineage, 0.0)

    known_lineages = set(lineage_prevalence)

    def _prettify(label):
        # Aliasor.compress() only re-abbreviates once a name has a *full*
        # extra 3-segment indirection beyond its alias root, so a climb that
        # lands exactly on a root alias (e.g. dealiased "B.1.1.529") comes
        # back unabbreviated. realias_dict has the exact reverse mapping for
        # those root cases -- but some roots (e.g. "B.1.1.7") are *also*
        # directly-observed raw lineages with their own, more recognizable
        # name (Alpha), and a second alias (here "Q") for their own deep
        # sublineages; prefer the directly-observed name when the climbed-to
        # label is itself one of the dataset's raw lineages, and only fall
        # back to realias_dict's shorter synonym otherwise. Purely a label
        # prettifier -- doesn't change which lineages end up grouped together.
        if label in known_lineages:
            return label
        return aliasor.realias_dict.get(label, label)

    final_map = {
        lineage: (other_label if cluster_prevalence[cluster] < other_threshold
                  else _prettify(cluster))
        for lineage, cluster in cluster_of.items()
    }
    return final_map, climb_count


def assign_clusters(df, cluster_map, lineage_col="lineage", out_col="cluster"):
    out = df.copy()
    out[out_col] = out[lineage_col].map(cluster_map)
    return out


def clustered_weekly_counts(weekly_counts, cluster_map):
    """Re-sum weekly counts by (state, cluster, week_ending) -- multiple raw
    lineages (including everything folded into "Other") land on the same
    cluster and must be summed, not just relabeled."""
    df = assign_clusters(weekly_counts, cluster_map)
    return (df.groupby(["state", "cluster", "week_ending"])["lineage_count"]
              .sum().reset_index())


def build_clustered_panel(aliasor, weekly_counts, weekly_totals,
                           max_climbs=MAX_CLIMBS,
                           significance_threshold=DEFAULT_SIGNIFICANCE_THRESHOLD,
                           other_threshold=DEFAULT_OTHER_THRESHOLD):
    """Orchestrates raw prevalence -> cluster map -> re-summed weekly panel
    for one parameter setting. Returns (weekly_counts_clustered, cluster_map,
    climb_count)."""
    lineage_prevalence = raw_lineage_prevalence(weekly_counts, weekly_totals)
    cluster_map, climb_count = build_prevalence_aware_clusters(
        aliasor, lineage_prevalence, max_climbs=max_climbs,
        significance_threshold=significance_threshold, other_threshold=other_threshold,
    )
    weekly_counts_clustered = clustered_weekly_counts(weekly_counts, cluster_map)
    return weekly_counts_clustered, cluster_map, climb_count


# --------------------------------------------------------------------------
# Lineage lookup (search by variant name/fragment)
# --------------------------------------------------------------------------

def matches_lineage(aliasor, label, query):
    """True if `label` IS `query`, or a true phylogenetic descendant of it.

    Compared in fully dealiased Pango-tree space, not by string prefix --
    "JN.1" should also catch a descendant like KP.3.1.1, which picked up its
    own alias letter along the way and doesn't share "JN.1" as a string
    prefix at all (KP.3.1.1 dealiases to B.1.1.529.2.86.1.1.11.1.3.1.1, which
    *does* start with JN.1's dealiased B.1.1.529.2.86.1.1). A raw string
    match would silently miss cases like this.
    """
    if label == "Other":
        return False
    dealiased_query = aliasor.uncompress(query)
    dealiased_label = aliasor.uncompress(label)
    return dealiased_label == dealiased_query or dealiased_label.startswith(dealiased_query + ".")


def find_matching_clusters(aliasor, clusters, query):
    """Subset of `clusters` (iterable of cluster labels) matching `query`,
    per `matches_lineage`."""
    return [c for c in clusters if matches_lineage(aliasor, c, query)]


# --------------------------------------------------------------------------
# Weak-state handling
# --------------------------------------------------------------------------

def state_signal_strength(weekly_totals, start=None, end=None):
    """Total sequenced genomes per state over the window -- the "how much
    signal does this state actually carry" metric.  No external population
    data needed; sequencing volume is the more relevant denominator here."""
    df = weekly_totals
    if start is not None:
        df = df[df["week_ending"] >= start]
    if end is not None:
        df = df[df["week_ending"] <= end]
    return df.groupby("state")["total_count"].sum().sort_values()


def classify_states_by_volume(strength, threshold):
    strong = set(strength[strength >= threshold].index)
    weak = set(strength[strength < threshold].index)
    return strong, weak


def _pool_by_region(df, weak_states, division_map, value_col):
    df = df.copy()
    def remap(state):
        if state in weak_states:
            region = division_map.get(state)
            return f"REGION:{region}" if region else state
        return state
    df["state"] = df["state"].map(remap)
    group_cols = [c for c in df.columns if c != value_col]
    return df.groupby(group_cols, as_index=False)[value_col].sum()


def build_state_panel(weekly_counts_clustered, weekly_totals, mode="pool",
                       volume_threshold=DEFAULT_VOLUME_THRESHOLD,
                       division_map=CENSUS_DIVISIONS):
    """Classify states by sequencing volume, drop or pool the weak ones, then
    recompute shares -- shares are always computed *after* thinning/pooling,
    never before, so a pooled region's share reflects its pooled total.
    Returns (panel, report) where panel has columns geo, cluster, week_ending,
    count, total_count, share.
    """
    strength = state_signal_strength(weekly_totals)
    strong, weak = classify_states_by_volume(strength, volume_threshold)

    if mode == "drop":
        counts = weekly_counts_clustered[weekly_counts_clustered["state"].isin(strong)]
        totals = weekly_totals[weekly_totals["state"].isin(strong)]
        report = {"mode": "drop", "n_dropped": len(weak), "dropped_states": sorted(weak)}
    elif mode == "pool":
        counts = _pool_by_region(weekly_counts_clustered, weak, division_map, "lineage_count")
        totals = _pool_by_region(weekly_totals, weak, division_map, "total_count")
        report = {"mode": "pool", "n_pooled": len(weak), "pooled_states": sorted(weak)}
    else:
        raise ValueError(f"unknown mode {mode!r}")

    counts = counts.rename(columns={"state": "geo", "lineage_count": "count"})
    totals = totals.rename(columns={"state": "geo"})
    panel = counts.merge(totals, on=["geo", "week_ending"], how="left")
    panel["share"] = panel["count"] / panel["total_count"]
    return panel, report


# --------------------------------------------------------------------------
# Wave detection
# --------------------------------------------------------------------------

def find_wave_window(national_series, half_max_fraction=WAVE_HALF_MAX_FRACTION):
    """national_series: Series indexed by week_ending, national share values.
    Returns dict(peak_week, peak_share, window_start, window_end) for the
    contiguous run of weeks around the peak where share >= half_max_fraction
    of the peak, or None if the series is empty/all-zero."""
    s = national_series.dropna().sort_index()
    if s.empty or s.max() <= 0:
        return None
    peak_week = s.idxmax()
    peak_share = s.loc[peak_week]
    threshold = peak_share * half_max_fraction
    above = (s >= threshold).values
    weeks = s.index
    peak_pos = weeks.get_loc(peak_week)
    lo, hi = peak_pos, peak_pos
    while lo > 0 and above[lo - 1]:
        lo -= 1
    while hi < len(weeks) - 1 and above[hi + 1]:
        hi += 1
    return {"peak_week": peak_week, "peak_share": peak_share,
            "window_start": weeks[lo], "window_end": weeks[hi]}


def build_wave_windows(national_cluster_shares, half_max_fraction=WAVE_HALF_MAX_FRACTION,
                        exclude=("Other",), min_peak_share=MIN_WAVE_PEAK_SHARE,
                        min_window_weeks=MIN_WAVE_WINDOW_WEEKS):
    """national_cluster_shares: long df with columns cluster, week_ending, share
    (i.e. `national_weekly_cluster_shares(..., group_col='cluster')`).
    One row per cluster with a detected wave; clusters with no real peak, in
    `exclude`, below `min_peak_share`, or whose half-max window spans fewer
    than `min_window_weeks` are dropped. The width/share floors matter in
    practice: a cluster that just emerged in the file's final weeks can show
    up in a single state for a single week purely from reporting lag, which
    produces a "wave" and "relative level" that are pure noise -- and,
    downstream, spuriously perfect correlations against equally sparse
    partners (confirmed empirically: single-week windows detected for
    clusters with peak_share ~1%, nonzero in exactly one state)."""
    rows = []
    for cluster, sub in national_cluster_shares.groupby("cluster"):
        if cluster in exclude:
            continue
        s = sub.set_index("week_ending")["share"]
        window = find_wave_window(s, half_max_fraction)
        if window is None:
            continue
        if window["peak_share"] < min_peak_share:
            continue
        window_weeks = (window["window_end"] - window["window_start"]).days / 7 + 1
        if window_weeks < min_window_weeks:
            continue
        rows.append({"cluster": cluster, **window})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Relative-level (z-score) computation
# --------------------------------------------------------------------------

def state_wave_level(state_panel, cluster, window):
    """Each geo's mean share of `cluster` over `window`'s weeks, zero-filled
    for geo/week combos absent from the panel (no detected sequences that
    week) rather than excluded -- excluding would bias sparse states upward."""
    all_geos = state_panel["geo"].unique()
    weeks = pd.date_range(window["window_start"], window["window_end"], freq=WEEK_CONVENTION)
    sub = state_panel[state_panel["cluster"] == cluster]
    pivot = sub.pivot_table(index="geo", columns="week_ending", values="share", aggfunc="sum")
    pivot = pivot.reindex(index=all_geos, columns=weeks, fill_value=0.0).fillna(0.0)
    return pivot.mean(axis=1)


def _zscore(level):
    std = level.std(ddof=0)
    if std == 0 or np.isnan(std):
        return level * 0.0
    return (level - level.mean()) / std


def relative_level_zscore(state_panel, wave_windows):
    """Long df: geo, cluster, level (mean share over its wave window),
    zscore (cross-geo standardized level) -- the value everything downstream
    correlates on."""
    rows = []
    for _, w in wave_windows.iterrows():
        level = state_wave_level(state_panel, w["cluster"], w)
        z = _zscore(level)
        for geo in level.index:
            rows.append({"geo": geo, "cluster": w["cluster"],
                         "level": level[geo], "zscore": z[geo]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Correlation + FDR
# --------------------------------------------------------------------------

def candidate_pairs_by_peak_gap(wave_windows, lag_months=LAG_MONTHS_RANGE):
    """Ordered cluster pairs (x, y) whose national peaks are naturally
    `lag_months` apart -- the primary, data-driven pairing."""
    lo_days, hi_days = lag_months[0] * 30.44, lag_months[1] * 30.44
    wv = wave_windows.set_index("cluster")["peak_week"]
    rows = []
    for x in wv.index:
        for y in wv.index:
            if x == y:
                continue
            gap_days = (wv[y] - wv[x]).days
            if lo_days <= gap_days <= hi_days:
                rows.append({"cluster_x": x, "cluster_y": y,
                             "lag_days": gap_days, "lag_months": gap_days / 30.44})
    return pd.DataFrame(rows)


def pairwise_spearman(relative_level, cluster_x, cluster_y):
    x = relative_level[relative_level["cluster"] == cluster_x].set_index("geo")["zscore"]
    y = relative_level[relative_level["cluster"] == cluster_y].set_index("geo")["zscore"]
    common = x.index.intersection(y.index)
    if len(common) < 3:
        return np.nan, np.nan, len(common)
    r, p = spearmanr(x.loc[common], y.loc[common])
    return r, p, len(common)


def correlation_table(relative_level, pairs):
    rows = []
    for _, row in pairs.iterrows():
        r, p, n = pairwise_spearman(relative_level, row["cluster_x"], row["cluster_y"])
        rows.append({**row.to_dict(), "n_states": n, "spearman_r": r, "p_value": p})
    return pd.DataFrame(rows)


def add_fdr_correction(table, p_col="p_value", alpha=FDR_ALPHA):
    out = table.copy()
    out["q_value"] = np.nan
    valid = out[p_col].notna()
    if valid.any():
        out.loc[valid, "q_value"] = false_discovery_control(out.loc[valid, p_col], method="bh")
    out["fdr_significant"] = out["q_value"] < alpha
    return out


def shifted_window(window, lag_months):
    """Slide a wave window earlier by `lag_months` (~30.44 days/month),
    keeping its width -- used to force a comparison window rather than rely
    on a cluster's own detected peak."""
    delta = pd.Timedelta(days=round(lag_months * 30.44))
    return {"peak_week": window["peak_week"] - delta,
            "window_start": window["window_start"] - delta,
            "window_end": window["window_end"] - delta}


def fixed_lag_correlation_table(state_panel, wave_windows, relative_level,
                                 lag_months_grid=range(6, 13)):
    """Secondary/robustness sweep: for every ordered pair (x, y) and lag in
    the grid, force x's comparison window to sit `lag` months before y's own
    detected peak (rather than using x's own detected window), and correlate
    against y's already-computed relative level. Generalizes the primary
    peak-pairing to pairs that don't have a natural 6-12 month gap."""
    wv = wave_windows.set_index("cluster")
    rows = []
    for y in wv.index:
        y_window = wv.loc[y]
        y_z = relative_level[relative_level["cluster"] == y].set_index("geo")["zscore"]
        for x in wv.index:
            if x == y:
                continue
            for lag in lag_months_grid:
                forced_window = shifted_window(y_window, lag)
                level_x = state_wave_level(state_panel, x, forced_window)
                if level_x.sum() == 0:
                    # forced window falls entirely before x existed (or after
                    # it died out) -- not a meaningful comparison, skip rather
                    # than emit a spurious all-zero-vs-real correlation
                    continue
                z_x = _zscore(level_x)
                common = z_x.index.intersection(y_z.index)
                if len(common) < 3:
                    continue
                r, p = spearmanr(z_x.loc[common], y_z.loc[common])
                rows.append({"cluster_x": x, "cluster_y": y, "lag_months": lag,
                             "n_states": len(common), "spearman_r": r, "p_value": p})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Confound control
# --------------------------------------------------------------------------

def state_local_peak_week(state_panel, cluster, window, pad_weeks=4):
    """Each geo's own local peak week for `cluster`, searched in a window
    padded around the national wave window (a state can lead or lag the
    national peak)."""
    all_geos = state_panel["geo"].unique()
    weeks = pd.date_range(window["window_start"] - pd.Timedelta(weeks=pad_weeks),
                           window["window_end"] + pd.Timedelta(weeks=pad_weeks),
                           freq=WEEK_CONVENTION)
    sub = state_panel[state_panel["cluster"] == cluster]
    pivot = sub.pivot_table(index="geo", columns="week_ending", values="share", aggfunc="sum")
    pivot = pivot.reindex(index=all_geos, columns=weeks, fill_value=0.0).fillna(0.0)
    return pivot.idxmax(axis=1)


def state_timing_rank(state_panel, wave_windows):
    """Each state's average early/late rank of its own local peak, averaged
    across every retained cluster -- a general "always first/always last to
    every wave" tendency that could otherwise masquerade as a real X->Y
    signal in every pairwise correlation."""
    ranks = []
    for _, w in wave_windows.iterrows():
        peak_weeks = state_local_peak_week(state_panel, w["cluster"], w)
        ranks.append(peak_weeks.rank(method="average").rename(w["cluster"]))
    rank_df = pd.concat(ranks, axis=1)
    return rank_df.mean(axis=1).rename("timing_rank")


def partial_spearman(relative_level, cluster_x, cluster_y, timing_rank):
    """Spearman correlation of X and Y after partialling out each state's
    general timing_rank -- computed as the Pearson partial-correlation
    formula applied to rank-transformed X, Y, and the covariate, which is
    exactly a partial Spearman correlation."""
    x = relative_level[relative_level["cluster"] == cluster_x].set_index("geo")["zscore"]
    y = relative_level[relative_level["cluster"] == cluster_y].set_index("geo")["zscore"]
    common = x.index.intersection(y.index).intersection(timing_rank.index)
    if len(common) < 5:
        return np.nan, np.nan
    xr = rankdata(x.loc[common])
    yr = rankdata(y.loc[common])
    zr = rankdata(timing_rank.loc[common])
    r_xy = np.corrcoef(xr, yr)[0, 1]
    r_xz = np.corrcoef(xr, zr)[0, 1]
    r_yz = np.corrcoef(yr, zr)[0, 1]
    denom = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    if denom == 0:
        return np.nan, np.nan
    partial_r = (r_xy - r_xz * r_yz) / denom
    n = len(common)
    df = n - 3
    if df <= 0 or abs(partial_r) >= 1:
        p = np.nan
    else:
        t_stat = partial_r * np.sqrt(df / (1 - partial_r ** 2))
        p = 2 * (1 - _student_t.cdf(abs(t_stat), df))
    return partial_r, p


def add_partial_correlation(table, relative_level, timing_rank):
    out = table.copy()
    partial_r, partial_p = [], []
    for _, row in out.iterrows():
        r, p = partial_spearman(relative_level, row["cluster_x"], row["cluster_y"], timing_rank)
        partial_r.append(r)
        partial_p.append(p)
    out["partial_r"] = partial_r
    out["partial_p"] = partial_p
    return out


def circular_permutation_null(relative_level, cluster_x, cluster_y,
                               division_map=CENSUS_DIVISIONS, n_permutations=1000, seed=0):
    """Null distribution of Spearman r built by permuting Y's z-scores among
    states *within the same Census division* (regional structure preserved,
    the specific X-Y pairing destroyed) -- a stricter null than shuffling
    states uniformly at random."""
    rng = np.random.default_rng(seed)
    x = relative_level[relative_level["cluster"] == cluster_x].set_index("geo")["zscore"]
    y = relative_level[relative_level["cluster"] == cluster_y].set_index("geo")["zscore"]
    common = list(x.index.intersection(y.index))
    x_vals = x.loc[common].values
    y_vals = y.loc[common].values

    def region_of(geo):
        if geo.startswith("REGION:"):
            return geo  # already a pooled region -- its own singleton group
        return division_map.get(geo, "Unknown")

    regions = np.array([region_of(g) for g in common])
    idx_by_region = {r: np.where(regions == r)[0] for r in np.unique(regions)}

    null_rs = np.empty(n_permutations)
    for i in range(n_permutations):
        permuted = y_vals.copy()
        for idxs in idx_by_region.values():
            if len(idxs) > 1:
                permuted[idxs] = y_vals[rng.permutation(idxs)]
        null_rs[i], _ = spearmanr(x_vals, permuted)
    return null_rs


def null_calibrated_pvalue(observed_r, null_rs):
    return float(np.mean(np.abs(null_rs) >= abs(observed_r)))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build_variant_wave_panel(alias_path=PANGO_ALIAS_KEY_PATH,
                              significance_threshold=DEFAULT_SIGNIFICANCE_THRESHOLD,
                              max_climbs=MAX_CLIMBS, other_threshold=DEFAULT_OTHER_THRESHOLD,
                              weak_state_mode="pool", volume_threshold=DEFAULT_VOLUME_THRESHOLD,
                              half_max_fraction=WAVE_HALF_MAX_FRACTION):
    """End-to-end pipeline for one parameter setting: load -> weekly ->
    prevalence-gated clustering -> weak-state handling -> wave detection ->
    relative-level z-scores. The notebook calls this once per point in the
    significance-threshold sweep."""
    aliasor = load_aliasor(alias_path)
    raw = load_variant_states_long()
    _, state_df = split_national_and_state_rows(raw)

    weekly_counts = weekly_state_lineage_counts(state_df)
    weekly_totals = weekly_state_totals(weekly_counts)

    weekly_counts_clustered, cluster_map, climb_count = build_clustered_panel(
        aliasor, weekly_counts, weekly_totals,
        max_climbs=max_climbs, significance_threshold=significance_threshold,
        other_threshold=other_threshold,
    )

    state_panel, weak_state_report = build_state_panel(
        weekly_counts_clustered, weekly_totals, mode=weak_state_mode,
        volume_threshold=volume_threshold,
    )

    national_cluster_shares = national_weekly_cluster_shares(
        assign_clusters(weekly_counts, cluster_map), weekly_totals, group_col="cluster"
    )
    wave_windows = build_wave_windows(national_cluster_shares, half_max_fraction=half_max_fraction)
    relative_level = relative_level_zscore(state_panel, wave_windows)
    timing_rank = state_timing_rank(state_panel, wave_windows)

    return {
        "cluster_map": cluster_map,
        "climb_count": climb_count,
        "weekly_counts_clustered": weekly_counts_clustered,
        "state_panel": state_panel,
        "weak_state_report": weak_state_report,
        "national_cluster_shares": national_cluster_shares,
        "wave_windows": wave_windows,
        "relative_level": relative_level,
        "timing_rank": timing_rank,
    }


# --------------------------------------------------------------------------
# Vaccination coverage (observed sources only)
# --------------------------------------------------------------------------

VAX_DIR = f"{WORKING_DIR}/data/vaccination"

#: Prefix marking a pseudo-cluster that is a vaccination campaign rather than a
#: variant, so downstream code -- and anyone reading a plot legend -- can tell
#: the two apart at a glance.
VAX_PREFIX = "VAX:"

#: Only *observed* sources are used.  The SMH scenario coverage curves that were
#: here previously (round 14's flu-derived fall-2022 assumption, and the round
#: 17/18 season schedules) were assumptions fed into models, not measurements,
#: and have been dropped.  What remains:
#:
#:   CDC admin   - administrative dose counts, the ground truth for the primary
#:                 series, but stops 2022-06.
#:   Delphi CTIS - self-reported, huge samples, 51 states, stops 2022-06-25 when
#:                 the survey ended.
#:   NIS-ACM     - CDC's phone survey, 50 states + DC, monthly, and the only
#:                 source here that covers the bivalent booster period.
#:
#: Campaigns are kept as separate pseudo-clusters rather than spliced into one
#: line: they measure different things (cumulative-ever primary series vs. a
#: single season's booster restarting from zero), and two of them are
#: self-reported while one is administrative.
VAX_FILES = {
    "cdc_admin": "smh_rd14_cdc_vaccination_trend.csv",
    "delphi": "delphi_fb_survey_covid_vaccinated_state.csv",
    "nis_acm": "nis_acm_trends_jurisdiction.csv",
}

_VAX_LONG_COLUMNS = ["state", "week_ending", "campaign", "covered", "denominator", "coverage"]


def _vax_path(key):
    path = f"{VAX_DIR}/{VAX_FILES[key]}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- see data/vaccination/SOURCES.md for how it was fetched")
    return path


def state_population():
    """Total resident population per state, backed out of the CDC administrative
    file as (people fully vaccinated) / (percent fully vaccinated).

    Used only as a *pooling weight* when low-volume states are combined into a
    Census division, so that a region's coverage is population-weighted rather
    than a plain mean of its states.  It is deliberately not used as a coverage
    denominator: each survey source reports its own base (all adults, or adults
    who completed the primary series) and those are respected as-is.
    """
    df = pd.read_csv(_vax_path("cdc_admin"), low_memory=False,
                     usecols=["Date", "Location", "People Fully Vaccinated Cumulative",
                              "Percent of Total Pop Fully Vaccinated"])
    df = df[df["Location"].isin(STATE_FIPS)]
    df = df[df["Percent of Total Pop Fully Vaccinated"] > 0]
    late = df[df["Date"] == df["Date"].max()]
    pop = (late["People Fully Vaccinated Cumulative"]
           / (late["Percent of Total Pop Fully Vaccinated"] / 100.0))
    return pd.Series(pop.values, index=late["Location"].values).round()


def _weekly_dense(df, value_col="coverage"):
    """Reindex each state's series onto a continuous weekly grid and interpolate.

    Necessary because `state_wave_level` zero-fills weeks a cluster is missing
    from -- correct for a variant (no sequences that week really does mean zero
    share) but badly wrong for a monthly survey series, where a missing week
    means "not surveyed", not "coverage fell to zero".  Interpolating first
    turns NIS-ACM's monthly estimates into the weekly series the rest of the
    pipeline expects.
    """
    out = []
    for state, sub in df.groupby("state"):
        s = sub.set_index("week_ending")[value_col].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        grid = pd.date_range(s.index.min(), s.index.max(), freq=WEEK_CONVENTION)
        s = s.reindex(s.index.union(grid)).interpolate(method="time").reindex(grid)
        out.append(pd.DataFrame({"state": state, "week_ending": grid, value_col: s.values}))
    return pd.concat(out, ignore_index=True)


def _finish_campaign(df, campaign, pop):
    """Attach the campaign label and pooling weights to a state/week/coverage frame."""
    df = df.copy()
    df["campaign"] = VAX_PREFIX + campaign
    df["denominator"] = df["state"].map(pop)
    df["covered"] = df["coverage"] * df["denominator"]
    return df.dropna(subset=["coverage", "denominator"])[_VAX_LONG_COLUMNS]


def load_vax_cdc_admin():
    """Observed CDC administrative primary-series coverage, 2020-12 to 2022-06.

    Dose counts reported by jurisdictions -- the only non-survey source here, so
    the natural reference when comparing against the two self-reported series
    (both of which run noticeably higher).
    """
    df = pd.read_csv(_vax_path("cdc_admin"), low_memory=False,
                     usecols=["Date", "Location", "Percent of Total Pop Fully Vaccinated"])
    df = df[df["Location"].isin(STATE_FIPS)].copy()
    df["week_ending"] = assign_week_ending(df["Date"])
    df["coverage"] = df["Percent of Total Pop Fully Vaccinated"] / 100.0
    weekly = (df.groupby(["Location", "week_ending"])["coverage"].max()
                .reset_index().rename(columns={"Location": "state"}))
    return _finish_campaign(_weekly_dense(weekly), "primary series (CDC admin)", state_population())


def load_vax_delphi_ctis():
    """Delphi COVIDcast / CTIS self-reported >=1 dose, 51 states, 2021-01 to 2022-06-25.

    The survey was discontinued on 2022-06-25, so this cannot reach the
    bivalent period -- it is the deep-history source, covering the Alpha,
    Delta and Omicron BA.1/BA.2 waves at weekly-or-better resolution.
    Self-reported coverage runs well above administrative counts (a known CTIS
    bias), so compare its *between-state* pattern rather than its level.
    """
    df = pd.read_csv(_vax_path("delphi"))
    df["state"] = df["geo_value"].str.upper()
    df = df[df["state"].isin(STATE_FIPS)].copy()
    df["week_ending"] = assign_week_ending(pd.to_datetime(df["time_value"], format="%Y%m%d"))
    df["coverage"] = df["value"] / 100.0
    weekly = df.groupby(["state", "week_ending"])["coverage"].mean().reset_index()
    return _finish_campaign(_weekly_dense(weekly), "vaccinated >=1 dose (Delphi CTIS)", state_population())


#: The two NIS-ACM indicator rows this analysis uses, as
#: campaign label -> (indicator_name, indicator_category).
NIS_ACM_INDICATORS = {
    "vaccinated >=1 dose (NIS-ACM)": (
        "Vaccination and intent 4 level grouping",
        "Vaccinated (>=1 dose)"),
    "bivalent booster (NIS-ACM)": (
        "Bivalent Booster Uptake and Intention",
        "Received updated bivalent booster dose (among adults who completed primary series)"),
}


def _parse_nis_period(period, year):
    """'October 30 - November 26', 2022 -> midpoint Timestamp.

    Formats are inconsistent across rows (full and abbreviated month names,
    doubled spaces, stray trailing spaces, and "Sept" which strptime doesn't
    accept), so normalise before parsing.  The midpoint of the field window is
    used as the observation date, which is the usual reference point for an
    estimate collected over a survey window.
    """
    text = " ".join(str(period).replace("Sept", "Sep").split())
    try:
        lo_txt, hi_txt = [p.strip() for p in text.split("-")]
    except ValueError:
        return pd.NaT

    def one(part):
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                return pd.Timestamp(datetime.strptime(f"{part} {year}", fmt))
            except ValueError:
                continue
        return pd.NaT

    lo, hi = one(lo_txt), one(hi_txt)
    if pd.isna(lo) or pd.isna(hi):
        return pd.NaT
    if hi < lo:                      # window crosses the new year
        hi = hi.replace(year=hi.year + 1)
    return lo + (hi - lo) / 2


def load_vax_nis_acm(time_type="Monthly"):
    """CDC National Immunization Survey - Adult COVID Module, jurisdictional.

    Covers all 50 states + DC (plus territories and a few substate areas, which
    are dropped).  Two campaigns come from this file: cumulative >=1 dose
    (2021-04 to 2023-06) and updated bivalent booster uptake (2022-08 to
    2023-06) -- the latter being the only *observed* state-level source
    available for the bivalent period.

    Note the bases differ: >=1 dose is a percentage of all adults 18+, while
    bivalent uptake is a percentage of adults who completed the primary series.
    Both are self-reported.
    """
    df = pd.read_csv(_vax_path("nis_acm"))
    df = df[(df["Geography Type"] == "Jurisdictional Estimates")
            & (df["Time Type"] == time_type)].copy()
    name_to_abbrev = {v: k for k, v in _STATE_NAMES.items()}
    df["state"] = df["Geography"].map(name_to_abbrev)   # drops territories/substate rows
    df = df[df["state"].notna()]
    df["week_ending"] = assign_week_ending(
        df.apply(lambda r: _parse_nis_period(r["Time Period"], r["Year"]), axis=1))
    df["coverage"] = pd.to_numeric(df["Estimate (%)"], errors="coerce") / 100.0

    pop = state_population()
    frames = []
    for campaign, (indicator, category) in NIS_ACM_INDICATORS.items():
        sub = df[(df["Indicator Name"] == indicator) & (df["Indicator Category"] == category)]
        sub = sub.dropna(subset=["coverage", "week_ending"])
        if sub.empty:
            continue
        weekly = sub.groupby(["state", "week_ending"])["coverage"].mean().reset_index()
        frames.append(_finish_campaign(_weekly_dense(weekly), campaign, pop))
    return pd.concat(frames, ignore_index=True)


def load_vaccination_campaigns():
    """All observed vaccination campaigns as one tidy long frame:
    state, week_ending, campaign, covered, denominator, coverage."""
    return pd.concat([load_vax_cdc_admin(), load_vax_delphi_ctis(), load_vax_nis_acm()],
                     ignore_index=True)


def build_vaccination_panel(vax_long, weak_states, mode="pool", division_map=CENSUS_DIVISIONS):
    """Reshape coverage into the schema `build_state_panel` emits, so a campaign
    flows through the wave / z-score / correlation machinery exactly like a
    variant cluster.

    `count` is people covered and `total_count` the pooling-weight population,
    so combining weak states into a Census division is a plain sum on both and
    yields population-weighted regional coverage -- the same reason the variant
    panel pools counts rather than shares.  `weak_states` must be the set
    `build_state_panel` used, or the geo labels won't line up when the two
    panels are concatenated.
    """
    df = vax_long.rename(columns={"campaign": "cluster", "covered": "count",
                                  "denominator": "total_count"}).copy()
    if mode == "drop":
        df = df[~df["state"].isin(weak_states)]
    elif mode == "pool":
        def remap(state):
            if state in weak_states:
                region = division_map.get(state)
                return f"REGION:{region}" if region else state
            return state
        df["state"] = df["state"].map(remap)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    panel = (df.groupby(["state", "cluster", "week_ending"], as_index=False)[["count", "total_count"]]
               .sum().rename(columns={"state": "geo"}))
    panel["share"] = panel["count"] / panel["total_count"]
    return panel


def vaccination_campaign_windows(vax_long, ramp_lo=0.01, ramp_hi=0.95):
    """Wave-window rows (same schema as `build_wave_windows`) for each campaign.

    Coverage is cumulative and monotone, so the half-max peak detection used
    for variants is meaningless here -- a "peak" of the level would always be
    the final week.  Two different quantities are therefore derived separately:

    * `window_start`/`window_end` bracket the uptake ramp, from the week
      national coverage first clears `ramp_lo` of its eventual level to the
      week it first reaches `ramp_hi`.  This is the window a state's coverage
      level is averaged over, so it captures the campaign's realised coverage.
    * `peak_week` is the week of maximum *weekly increment* -- peak
      vaccination rate, the campaign's centre of mass in time.  This is the
      timing analogue of a variant's peak-prevalence week and is what any lag
      comparison should measure from.  Using the end of the ramp instead would
      date a campaign months late: these curves accrue a long slow tail, so a
      season's ramp can close months after most of its doses were given, which
      would misleadingly place a campaign *after* a wave it actually preceded.

    `peak_share` is the campaign's eventual national coverage.
    """
    nat = (vax_long.groupby(["campaign", "week_ending"])[["covered", "denominator"]].sum())
    nat["coverage"] = nat["covered"] / nat["denominator"]
    rows = []
    for campaign, series in nat.groupby(level="campaign")["coverage"]:
        s = series.droplevel(0).sort_index()
        final = s.max()
        if not np.isfinite(final) or final <= 0:
            continue
        start = s[s >= ramp_lo * final].index.min()
        end = s[s >= ramp_hi * final].index.min()
        increments = s.diff()
        ramp = increments.loc[start:end]
        peak_week = ramp.idxmax() if ramp.notna().any() else end
        rows.append({"cluster": campaign, "peak_week": peak_week, "peak_share": final,
                     "window_start": start, "window_end": end})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Administrative reference and self-report correction
# --------------------------------------------------------------------------

#: Survey series consistently run above administrative dose counts.  Rather
#: than assume a single national adjustment, a *per-state* multiplicative
#: factor is estimated wherever a survey series overlaps an administrative
#: series measuring the same thing on the same base, giving
#: `administrative ~= factor * reported`.
VAX_ADMIN_FILE = "cdc_admin_jurisdiction.csv"
VAX_2023_24_FILE = "nis_acm_updated_2023_24_weekly.csv"

#: Which administrative metric each self-reported campaign is measured against.
#: `None` means no administrative series overlaps it, so the factor has to be
#: inherited from another campaign (see `VAX_FACTOR_DONOR`).
VAX_CORRECTION_REFERENCE = {
    "vaccinated >=1 dose (Delphi CTIS)": "dose1_18plus",
    "vaccinated >=1 dose (NIS-ACM)": "dose1_18plus",
    "bivalent booster (NIS-ACM)": "bivalent_among_completers",
    "updated 2023-24 (NIS-ACM)": None,
}

#: The 2023-24 season series starts 2023-09-30, months after administrative
#: reporting stopped (2023-05-10), so its factor is inherited from the bivalent
#: campaign -- the nearest-in-time series from the same survey programme.
VAX_FACTOR_DONOR = {"updated 2023-24 (NIS-ACM)": "bivalent booster (NIS-ACM)"}

#: Bases differ between series.  Everything is ultimately expressed as a share
#: of the adult population; a series reported "among adults who completed the
#: primary series" has to be multiplied by primary-series coverage to get there.
VAX_BASE = {
    "primary series (CDC admin)": "population",
    "vaccinated >=1 dose (Delphi CTIS)": "population",
    "vaccinated >=1 dose (NIS-ACM)": "population",
    "bivalent booster (NIS-ACM)": "primary_completers",
    "updated 2023-24 (NIS-ACM)": "population",
}

VAX_SOURCE_TYPE = {
    "primary series (CDC admin)": "administrative",
    "vaccinated >=1 dose (Delphi CTIS)": "self-report",
    "vaccinated >=1 dose (NIS-ACM)": "self-report",
    "bivalent booster (NIS-ACM)": "self-report",
    "updated 2023-24 (NIS-ACM)": "self-report",
}


def load_cdc_admin_jurisdiction():
    """Weekly administrative coverage by state from CDC's jurisdiction file.

    Returns long rows of (state, week_ending, metric, coverage) for:

    * `dose1_18plus` -- >=1 dose among adults, the like-for-like comparator for
      the survey ">=1 dose" questions.  **CDC censors this at 95%**, so capped
      observations are flagged and excluded when fitting a correction factor.
    * `series_complete_18plus` -- primary series complete among adults; barely
      censored (~1% of rows), so it is what rebases the bivalent numbers.
    * `bivalent_among_completers` -- administrative bivalent coverage divided by
      primary-series coverage, putting it on the same base NIS-ACM reports
      bivalent uptake on ("among adults who completed the primary series").
      Both parts are administrative, so the rebasing does not smuggle survey
      data into the reference.
    """
    df = pd.read_csv(f"{VAX_DIR}/{VAX_ADMIN_FILE}")
    df = df[df["location"].isin(STATE_FIPS)].copy()
    df["week_ending"] = assign_week_ending(pd.to_datetime(df["date"]))
    df = df.rename(columns={"location": "state"})

    weekly = (df.groupby(["state", "week_ending"])
                .agg(dose1=("administered_dose1_recip_18pluspop_pct", "max"),
                     series_complete=("series_complete_18pluspop", "max"),
                     bivalent=("bivalent_booster_18plus_pop_pct", "max"))
                .reset_index())

    out = []
    d1 = weekly[["state", "week_ending", "dose1"]].dropna().copy()
    d1["metric"] = "dose1_18plus"
    d1["capped"] = d1["dose1"] >= 95.0          # CDC truncates this metric at 95%
    d1["coverage"] = d1["dose1"] / 100.0
    out.append(d1[["state", "week_ending", "metric", "coverage", "capped"]])

    sc = weekly[["state", "week_ending", "series_complete"]].dropna().copy()
    sc["metric"] = "series_complete_18plus"
    sc["capped"] = False
    sc["coverage"] = sc["series_complete"] / 100.0
    out.append(sc[["state", "week_ending", "metric", "coverage", "capped"]])

    biv = weekly.dropna(subset=["bivalent", "series_complete"]).copy()
    biv = biv[biv["series_complete"] > 0]
    biv["metric"] = "bivalent_among_completers"
    biv["capped"] = False
    biv["coverage"] = (biv["bivalent"] / 100.0) / (biv["series_complete"] / 100.0)
    out.append(biv[["state", "week_ending", "metric", "coverage", "capped"]])

    return pd.concat(out, ignore_index=True)


def load_vax_nis_acm_2023_24():
    """NIS-ACM weekly coverage with the updated 2023-24 vaccine, adults 18+.

    Fills the tail of the analysis window that the monthly NIS-ACM trends file
    stops short of (it ends 2023-06); this one runs 2023-09-30 to 2024-05-11.
    """
    df = pd.read_csv(f"{VAX_DIR}/{VAX_2023_24_FILE}")
    df = df[df["Geographic Level"] == "State"].copy()
    name_to_abbrev = {v: k for k, v in _STATE_NAMES.items()}
    df["state"] = df["Geographic Name"].map(name_to_abbrev)
    df = df[df["state"].notna()]
    df["week_ending"] = assign_week_ending(
        pd.to_datetime(df["Week_ending"], format="%m/%d/%Y %I:%M:%S %p"))
    df["coverage"] = pd.to_numeric(df["Estimate"], errors="coerce") / 100.0
    weekly = df.dropna(subset=["coverage"]).groupby(["state", "week_ending"])["coverage"].mean().reset_index()
    return _finish_campaign(_weekly_dense(weekly), "updated 2023-24 (NIS-ACM)", state_population())


def estimate_correction_factors(vax_long, admin_long, min_weeks=8, tail_weeks=None):
    """Per-state factor mapping a self-reported series onto the administrative
    scale, i.e. `reported * factor ~= administrative`.

    The factor is the **median weekly ratio** admin/reported over the overlap.
    Median rather than a fitted slope because a handful of weeks at the start of
    a rollout, where the two series are ramping with different reporting lags,
    otherwise drag the estimate badly; the median is indifferent to them.
    Administrative observations flagged `capped` (CDC truncates >=1-dose
    coverage at 95%) are excluded, since a censored numerator would bias every
    high-coverage state's factor upward.

    States with fewer than `min_weeks` usable overlapping weeks fall back to the
    campaign's national median factor, and campaigns with no administrative
    overlap at all inherit their donor campaign's per-state factors (see
    `VAX_FACTOR_DONOR`).  The `method` column records which of the three applied
    so an inherited estimate is never mistaken for a measured one.
    """
    rows = []
    for campaign in sorted(vax_long["campaign"].unique()):
        label = campaign.replace(VAX_PREFIX, "")
        if VAX_SOURCE_TYPE.get(label) != "self-report":
            continue
        metric = VAX_CORRECTION_REFERENCE.get(label)
        if metric is None:
            continue
        rep = vax_long[vax_long["campaign"] == campaign][["state", "week_ending", "coverage"]]
        adm = admin_long[(admin_long["metric"] == metric) & (~admin_long["capped"])]
        adm = adm[["state", "week_ending", "coverage"]].rename(columns={"coverage": "admin"})
        merged = rep.merge(adm, on=["state", "week_ending"], how="inner")
        merged = merged[(merged["coverage"] > 0.01) & (merged["admin"] > 0.01)]
        if tail_weeks:
            cutoff = merged["week_ending"].max() - pd.Timedelta(weeks=tail_weeks)
            merged = merged[merged["week_ending"] >= cutoff]
        merged["ratio"] = merged["admin"] / merged["coverage"]
        for state, sub in merged.groupby("state"):
            rows.append({"campaign": campaign, "state": state,
                         "factor": sub["ratio"].median(), "n_weeks": len(sub),
                         "median_reported": sub["coverage"].median(),
                         "median_admin": sub["admin"].median(),
                         "method": "measured"})
    factors = pd.DataFrame(rows)

    # thin per-state estimates -> campaign's national median
    out = []
    for campaign, sub in factors.groupby("campaign"):
        national = sub.loc[sub["n_weeks"] >= min_weeks, "factor"].median()
        sub = sub.copy()
        thin = sub["n_weeks"] < min_weeks
        sub.loc[thin, "factor"] = national
        sub.loc[thin, "method"] = "national_fallback"
        out.append(sub)
    factors = pd.concat(out, ignore_index=True) if out else factors

    # campaigns with no administrative overlap inherit a donor's factors
    for label, donor_label in VAX_FACTOR_DONOR.items():
        campaign, donor = VAX_PREFIX + label, VAX_PREFIX + donor_label
        if campaign not in set(vax_long["campaign"]) or donor not in set(factors["campaign"]):
            continue
        inherited = factors[factors["campaign"] == donor].copy()
        inherited["campaign"] = campaign
        inherited["method"] = "inherited from " + donor_label
        inherited[["n_weeks", "median_reported", "median_admin"]] = np.nan
        factors = pd.concat([factors, inherited], ignore_index=True)

    return factors


def apply_correction(vax_long, factors, admin_long=None, smooth_weeks=5):
    """Attach corrected, population-based and smoothed coverage to `vax_long`.

    Adds:
      `correction_factor` / `correction_method` -- as estimated above (1.0 for
          administrative series, which need no correction).
      `coverage_corrected` -- reported * factor, clipped to [0, 1].
      `population_share`   -- corrected coverage expressed as a share of the
          adult population.  Series reported "among primary-series completers"
          (bivalent) are multiplied by administrative primary-series coverage to
          move them onto that common base; everything else is already there.
      `population_share_smoothed` -- centred rolling mean over `smooth_weeks`,
          which takes the edge off survey sampling noise and off the stair-steps
          left by interpolating monthly estimates onto a weekly grid.
    """
    df = vax_long.copy()
    key = factors.set_index(["campaign", "state"])["factor"] if len(factors) else pd.Series(dtype=float)
    meth = factors.set_index(["campaign", "state"])["method"] if len(factors) else pd.Series(dtype=object)
    idx = pd.MultiIndex.from_arrays([df["campaign"], df["state"]])
    df["correction_factor"] = key.reindex(idx).values if len(key) else np.nan
    df["correction_method"] = meth.reindex(idx).values if len(meth) else None
    admin_mask = df["campaign"].str.replace(VAX_PREFIX, "", regex=False).map(VAX_SOURCE_TYPE) == "administrative"
    df.loc[admin_mask, ["correction_factor", "correction_method"]] = [1.0, "administrative (no correction)"]
    df["correction_factor"] = df["correction_factor"].fillna(1.0)
    df["correction_method"] = df["correction_method"].fillna("uncorrected")
    df["coverage_corrected"] = (df["coverage"] * df["correction_factor"]).clip(0, 1)

    # move "among primary-series completers" series onto the population base
    df["population_share"] = df["coverage_corrected"]
    if admin_long is not None:
        sc = (admin_long[admin_long["metric"] == "series_complete_18plus"]
              [["state", "week_ending", "coverage"]].rename(columns={"coverage": "primary_complete"}))
        needs = df["campaign"].str.replace(VAX_PREFIX, "", regex=False).map(VAX_BASE) == "primary_completers"
        if needs.any():
            sub = df[needs].merge(sc, on=["state", "week_ending"], how="left")
            # administrative reporting stops mid-2023; carry the last known
            # primary-series level forward rather than dropping later weeks
            sub["primary_complete"] = (sub.sort_values("week_ending")
                                          .groupby("state")["primary_complete"].ffill().bfill())
            df.loc[needs, "population_share"] = (
                sub["coverage_corrected"] * sub["primary_complete"]).values

    df = df.sort_values(["campaign", "state", "week_ending"])
    df["population_share_smoothed"] = (
        df.groupby(["campaign", "state"])["population_share"]
          .transform(lambda s: s.rolling(smooth_weeks, center=True, min_periods=1).mean()))
    df["source_type"] = df["campaign"].str.replace(VAX_PREFIX, "", regex=False).map(VAX_SOURCE_TYPE)
    return df


VAX_EXPORT_PATH = f"{WORKING_DIR}/data/vaccination/vaccination_state_weekly_corrected.csv"
VAX_FACTORS_PATH = f"{WORKING_DIR}/data/vaccination/vaccination_correction_factors.csv"


def build_corrected_vaccination_series(min_weeks=8, smooth_weeks=5):
    """Load every source, estimate per-state correction factors, and return
    (corrected_long, factors, admin_long) ready to export."""
    vax = pd.concat([load_vaccination_campaigns(), load_vax_nis_acm_2023_24()], ignore_index=True)
    admin = load_cdc_admin_jurisdiction()
    factors = estimate_correction_factors(vax, admin, min_weeks=min_weeks)
    corrected = apply_correction(vax, factors, admin_long=admin, smooth_weeks=smooth_weeks)
    return corrected, factors, admin


def export_vaccination_series(corrected, factors,
                              path=VAX_EXPORT_PATH, factors_path=VAX_FACTORS_PATH):
    """Write the weekly corrected series and the per-state factors to CSV, so
    other notebooks (e.g. the ascertainment analysis) can read them without
    re-running this pipeline."""
    # `denominator` travels with the rows so a reader can population-weight up
    # to national or regional aggregates without needing this module.
    cols = ["state", "week_ending", "campaign", "source_type", "coverage",
            "correction_factor", "correction_method", "coverage_corrected",
            "population_share", "population_share_smoothed", "denominator"]
    corrected[cols].to_csv(path, index=False)
    factors.to_csv(factors_path, index=False)
    return path, factors_path


def load_corrected_vaccination_series(path=VAX_EXPORT_PATH):
    """Read back what `export_vaccination_series` wrote."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run section 8b of variant_wave_analysis.ipynb "
            "to build and export the corrected vaccination series first.")
    df = pd.read_csv(path, parse_dates=["week_ending"])
    return df
