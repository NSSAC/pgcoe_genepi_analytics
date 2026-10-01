"""Loaders for the CDC surveillance and genomic-sequence series used in the
ascertainment analysis (fraction of infections sequenced / reported as cases).

Everything here reads from the shared COVID-19 commons on Rivanna.  The one
exception is `fetch_blood_donor_2022_2023`, which pulls from data.cdc.gov and
caches under `data/cdc_cache/` because the 2022+ blood-donor survey is not in
the commons snapshot.

Focus window for the current analysis is 2020-03 through 2022-06.
"""

import functools
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

COMMONS = "/home/bl4zc/biocomplexity/COVID-19_commons/data"
CDC = f"{COMMONS}/CDC"
VARIANTS = f"{COMMONS}/variants"
VDH_GENOMICS = f"{COMMONS}/VDH_genomics"
VDH_RECONCILED = (f"{VDH_GENOMICS}/ncbi_source_results/"
                  f"genomic_update_2024_05_18.extra.csv")

WORKING_DIR = "/sfs/gpfs/tardis/project/bii_nssac/people/bl4zc/pgcoe_genepi_analytics"
CACHE_DIR = f"{WORKING_DIR}/data/cdc_cache"

# The analysis window: pandemic period with the least intervention/behavioural
# control, ending before the post-Omicron collapse of case reporting.
WINDOW_START = pd.Timestamp("2020-03-01")
WINDOW_END = pd.Timestamp("2022-06-30")

NHSN_FILE = (
    "Weekly_United_States_Hospitalization_Metrics_by_Jurisdiction,_During_"
    "Mandatory_Reporting_Period_from_August_1,_2020_to_April_30,_2024,_and_"
    "for_Data_Reported_Voluntarily_Beginning_May_1,_2024,_National_Healthcare_"
    "Safety_Network_(NHSN).csv"
)

STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
}
FIPS_STATE = {v: k for k, v in STATE_FIPS.items()}

# NSSP identifies geographies by full state name, not code.
_STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
    "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}


def _cached_frame(fn):
    """Memoise a loader that parses a large file.

    Several of these files are hundreds of MB, and the panel builders read the
    same ones repeatedly -- the seroreversion sensitivity alone rebuilds twelve
    panels.  Results are cached per argument tuple and handed back as copies, so
    a caller that mutates its frame cannot corrupt the cache.

    Call `fn.cache_clear()` after editing a source file on disk.
    """
    cache = functools.lru_cache(maxsize=None)(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if kwargs:                      # only positional calls are memoised
            return fn(*args, **kwargs)
        return cache(*args).copy()

    wrapper.cache_clear = cache.cache_clear
    return wrapper


# --------------------------------------------------------------------------
# Genomic sequencing volume  (the numerator)
# --------------------------------------------------------------------------

@_cached_frame
def load_sequence_counts(geo="US"):
    """Daily count of SARS-CoV-2 genomes by specimen collection date.

    Source: `data/variants/all_variants.csv` in the commons, built from
    outbreak.info / GISAID.  Columns are lineage buckets plus `other`; the row
    sum is the total number of sequences collected in that state on that day.

    The file already carries a `fips == "US"` row that is the exact sum of the
    state rows, so never sum across all fips values.

    geo : "US" for the national row, or a two-letter state code.
    """
    av = pd.read_csv(f"{VARIANTS}/all_variants.csv", dtype={"fips": str})
    av["date"] = pd.to_datetime(av["date"])
    lineage_cols = [c for c in av.columns if c not in ("fips", "date")]

    fips = "US" if geo == "US" else STATE_FIPS[geo]
    sub = av[av.fips == fips].copy()
    sub["n_seq"] = sub[lineage_cols].sum(axis=1)
    out = (
        sub.set_index("date")["n_seq"]
        .sort_index()
        .rename("n_sequences")
        .to_frame()
    )
    out["geo"] = geo
    return out


def load_sequence_counts_by_lineage(geo="US"):
    """Same source as `load_sequence_counts`, kept wide by lineage bucket."""
    av = pd.read_csv(f"{VARIANTS}/all_variants.csv", dtype={"fips": str})
    av["date"] = pd.to_datetime(av["date"])
    fips = "US" if geo == "US" else STATE_FIPS[geo]
    sub = av[av.fips == fips].drop(columns="fips").set_index("date").sort_index()
    return sub


def load_vdh_genomic_linelist(with_age=False, with_ids=False):
    """VA sequenced-specimen line list (VDH/DCLS, NCBI-linked).

    One row per sequenced specimen with a `collection_date`.  Used as an
    independent check on the GISAID-derived VA sequence counts, since it is
    the state's own tally rather than a deposition-dependent one.

    `with_age=True` also pulls `Patient Age In Years` (a numeric age) and the
    decade-banded `Age Group`.  The commons GISAID extract has no age field at
    all, so this line list is the only route to age-stratified sequence counts
    -- and only for Virginia.
    """
    path = f"{VDH_GENOMICS}/Genomic Data NCBI_2025-01-03.csv"
    cols = ["collection_date", "COVID_19_VARIANT", "Variant upd", "NCBI_Lineage",
            "PATIENT_COUNTY", "HealthDistrict", "Variant Count"]
    if with_age:
        cols += ["Patient Age In Years", "Age Group"]
    if with_ids:
        cols += _VDH_ID_COLS
    df = pd.read_csv(path, usecols=lambda c: c.strip() in cols,
                     low_memory=False, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["collection_date"] = pd.to_datetime(df["collection_date"],
                                           format="%m/%d/%Y", errors="coerce")
    return df


def _link_genome_ids(df, id_cols):
    """Group rows that refer to the same sequenced genome.

    Both VDH extracts are merge outputs in which one genome can span several
    rows, and no single identifier column is reliable on its own: a GenBank
    accession can carry up to 5 different GISAID ids and a GISAID id up to 11
    accessions, because the underlying join fanned out.  Picking any one column
    as the key therefore either over- or under-counts.

    Instead, two rows are treated as the same genome when they share *any*
    identifier, and genomes are the connected components of that row/identifier
    graph.  On the reconciled extract this collapses 109,387 rows to 97,626
    genomes, with 88,885 singletons and a largest component of 156 rows -- the
    linkage stays local rather than chaining everything together.

    Rows carrying no identifier at all cannot be linked and each count as their
    own genome.  In the reconciled file that is 10,934 rows, and they do carry
    variant calls (`covid_19_variant` / `variant_upd`), so they are sequenced
    specimens that simply have no deposited accession recorded -- exactly the
    material a reconciled list is meant to add over a deposition-based count.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    # Work positionally, but hand the labels back on the caller's own index --
    # callers assign this straight onto a frame whose index is rarely 0..n-1.
    original_index = df.index
    work = df.reset_index(drop=True)
    rows, cols, offset = [], [], 0
    for c in id_cols:
        if c not in work.columns:
            continue
        col = work[c].dropna().astype(str)
        codes, uniq = pd.factorize(col)
        rows.extend(col.index)
        cols.extend(codes + offset)
        offset += len(uniq)

    n = len(work)
    if offset == 0:
        return pd.Series(np.arange(n), index=original_index)
    m = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, offset))
    big = coo_matrix((np.ones(m.nnz), (m.row, m.col + n)), shape=(n + offset,) * 2)
    _, labels = connected_components(big + big.T, directed=False)
    return pd.Series(labels[:n], index=original_index)


#: Identifier columns used to link rows into genomes, per source.
_VDH_ID_COLS = ["GISAID_ACCESSION_ID", "Specimen_Id", "Sequence_ID", "VTCRIID"]
_RECONCILED_ID_COLS = ["accession", "gisaid_accession_id", "isolate",
                       "specimen_id", "sequence_id", "vendor_id"]


@_cached_frame
def load_vdh_reconciled_linelist():
    """Virginia reconciled genome list (`genomic_update_2024_05_18.extra.csv`).

    An alternative to both the GISAID-derived counts and the
    `Genomic Data NCBI` extract: VDH's reconciliation of NCBI, GISAID and
    in-state sequencing records, so it picks up genomes that were sequenced but
    never deposited publicly.  Virginia only.

    Three things are cleaned on load:

    - a stray repeated header row sits at line 100,001 (the file is two blocks
      concatenated; the second block shares no rows with the first);
    - `Unnamed: 0` is a stale row index that defeats `drop_duplicates`, so it is
      dropped before de-duplicating;
    - rows are linked into genomes with `_link_genome_ids`.

    Returns one row per *record* with a `genome_id` column; use
    `vdh_reconciled_sequences_monthly` for genome counts.
    """
    df = pd.read_csv(VDH_RECONCILED, low_memory=False)
    df = df[df.variant_count.astype(str) != "variant_count"]
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])
    df = df.drop_duplicates()
    df["collection_date"] = pd.to_datetime(df["collection_date"], errors="coerce")
    df["genome_id"] = _link_genome_ids(df, _RECONCILED_ID_COLS)
    return df


def _genomes_one_row_each(df, genome_col="genome_id", date_col="collection_date",
                          age_col=None):
    """Collapse a linked record frame to one row per genome (earliest collection)."""
    d = df.dropna(subset=[date_col]).sort_values(date_col)
    agg = {date_col: (date_col, "first")}
    if age_col:
        agg["age"] = (age_col, "first")
    return d.groupby(genome_col).agg(**agg)


def vdh_reconciled_sequences_monthly():
    """Monthly Virginia genome count from the reconciled list, by collection date."""
    g = _genomes_one_row_each(load_vdh_reconciled_linelist())
    return (g.set_index("collection_date").resample("MS").size()
            .rename("n_sequences_reconciled"))


def vdh_reconciled_sequences_by_age(freq="MS"):
    """Reconciled Virginia genome counts by harmonised age band.

    Age (`patient_age_in_years`) is missing for ~6% of linked genomes; those are
    dropped, so bands sum slightly below the total.
    """
    g = _genomes_one_row_each(load_vdh_reconciled_linelist(),
                              age_col="patient_age_in_years").dropna(subset=["age"])
    g["age_band"] = g["age"].astype(float).map(_band_from_years)
    return (g.set_index("collection_date").groupby("age_band")
            .resample(freq).size().unstack("age_band")
            .reindex(columns=AGE_BANDS))


def vdh_sequences_monthly(dedup=True):
    """Monthly count of VA sequenced genomes by collection date (VDH NCBI extract).

    `dedup=True` links rows into genomes with `_link_genome_ids` first.  This
    extract carries 868 exactly duplicated rows plus identifier-level repeats
    (2,860 duplicated GISAID ids, 4,186 duplicated specimen ids), so counting
    rows overstates genomes by a few percent.  Pass False for the raw row count.
    """
    df = load_vdh_genomic_linelist(with_ids=dedup).dropna(subset=["collection_date"])
    if not dedup:
        return df.set_index("collection_date").resample("MS").size().rename("n_sequences_vdh")
    df = df.drop_duplicates()
    df["genome_id"] = _link_genome_ids(df, _VDH_ID_COLS)
    g = _genomes_one_row_each(df)
    return (g.set_index("collection_date").resample("MS").size()
            .rename("n_sequences_vdh"))


#: The genome-count sources available as an ascertainment numerator.
#:   "gisaid"     - outbreak.info/GISAID deposits, any state or US (deposition-based)
#:   "vdh"        - VDH `Genomic Data NCBI` extract, Virginia only
#:   "reconciled" - VDH reconciled NCBI/GISAID/in-state list, Virginia only
SEQUENCE_SOURCES = ("gisaid", "vdh", "reconciled")


def monthly_sequence_counts(geo="US", source="gisaid"):
    """Monthly genome counts for `geo` from one of `SEQUENCE_SOURCES`.

    "gisaid" counts genomes *deposited* to GISAID and is the only source with
    national coverage.  The two VDH sources are Virginia-only and count genomes
    the state records as sequenced, whether or not they were ever deposited, so
    they are not deposition-limited -- but their 2020 linkage is sparse, and
    over 2020-03 to 2021-06 they sit well below GISAID (VDH 4,298 and reconciled
    5,909 against GISAID's 9,845).  Take GISAID as the better early number and
    the VDH sources as the better ones from Delta onward.
    """
    if source not in SEQUENCE_SOURCES:
        raise ValueError(f"source must be one of {SEQUENCE_SOURCES}")
    if source == "gisaid":
        return load_sequence_counts(geo)["n_sequences"].resample("MS").sum()
    if geo != "VA":
        raise ValueError(f'source "{source}" is Virginia only, got geo={geo!r}')
    return (vdh_sequences_monthly() if source == "vdh"
            else vdh_reconciled_sequences_monthly())


def sequence_counts_by_age(geo="VA", source="vdh", freq="MS"):
    """Age-banded genome counts; Virginia only, since GISAID carries no age."""
    if source == "gisaid":
        raise ValueError("the GISAID extract has no age field")
    if geo != "VA":
        raise ValueError(f'source "{source}" is Virginia only, got geo={geo!r}')
    return (vdh_sequences_by_age(freq) if source == "vdh"
            else vdh_reconciled_sequences_by_age(freq))


# --------------------------------------------------------------------------
# Seroprevalence  (the infection denominator)
# --------------------------------------------------------------------------

_MONTHS = ("Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec")
_RANGE_RE = re.compile(
    rf"^\s*({_MONTHS})\s+(\d+)(?:,\s*(\d{{4}}))?\s*-\s*"
    rf"({_MONTHS})\s+(\d+),\s*(\d{{4}})\s*$"
)


def _parse_specimen_range(text):
    """'Dec 24, 2020 -  Jan 7, 2021' -> (start, end) Timestamps."""
    if not isinstance(text, str):
        return pd.NaT, pd.NaT
    m = _RANGE_RE.match(re.sub(r"\s+", " ", text))
    if not m:
        return pd.NaT, pd.NaT
    smon, sday, syear, emon, eday, eyear = m.groups()
    end = pd.Timestamp(datetime.strptime(f"{emon} {eday} {eyear}", "%b %d %Y"))
    if syear is None:
        syear = eyear
        start = pd.Timestamp(datetime.strptime(f"{smon} {sday} {syear}", "%b %d %Y"))
        if start > end:  # range crosses new year, e.g. Dec 24 - Jan 7
            start = pd.Timestamp(
                datetime.strptime(f"{smon} {sday} {int(syear) - 1}", "%b %d %Y")
            )
    else:
        start = pd.Timestamp(datetime.strptime(f"{smon} {sday} {syear}", "%b %d %Y"))
    return start, end


@_cached_frame
def load_commercial_lab_seroprevalence():
    """CDC Nationwide Commercial Laboratory Seroprevalence Survey (d2tw-32xv).

    Anti-nucleocapsid (infection-induced) *cumulative* seroprevalence, by state
    and by round.  53 sites = 50 states + DC + PR + a `US` national row.

    Rounds 1-30 (Jul 2020 - Feb 2022) carry `rate_cumulative_prevalence`; CDC
    stopped publishing that column at round 31, which is why this series alone
    cannot cover Mar-Jun 2022.

    Returns one row per site/round with parsed specimen dates and the midpoint
    used for time-aligning the cumulative curve.
    """
    df = pd.read_csv(
        f"{CDC}/Nationwide_Commercial_Laboratory_Seroprevalence_Survey.csv",
        low_memory=False,
    )
    keep = [
        "site", "round", "date_range_of_specimen", "catchment_population",
        "n_cumulative_prevalence", "rate_cumulative_prevalence",
        "lower_ci_cumulative_prevalence", "upper_ci_cumulative_prevalence",
        "estimated_cumulative_infections_all_count",
        "estimated_cumulative_infections_all_lower_ci",
        "estimated_cumulative_infections_all_upper_ci",
    ]
    df = df[keep].copy()

    parsed = df["date_range_of_specimen"].map(_parse_specimen_range)
    df["specimen_start"] = [p[0] for p in parsed]
    df["specimen_end"] = [p[1] for p in parsed]

    # Round 25 has no date range for any site; fall back to the round's
    # neighbours so the round still lands on the time axis.
    mid = df["specimen_start"] + (df["specimen_end"] - df["specimen_start"]) / 2
    df["specimen_mid"] = mid
    round_mid = df.groupby("round")["specimen_mid"].median()
    round_mid = round_mid.interpolate()
    df["specimen_mid"] = df["specimen_mid"].fillna(df["round"].map(round_mid))

    df = df.rename(columns={"rate_cumulative_prevalence": "anti_n_pct"})
    df["anti_n_pct"] = df["anti_n_pct"].mask(df["anti_n_pct"].isin(SERO_SENTINELS))
    return df.sort_values(["site", "round"]).reset_index(drop=True)


def load_blood_donor_infection_induced():
    """CDC 2020-2021 Nationwide Blood Donor Survey, infection-induced (mtc3-kq6r).

    Monthly anti-N seroprevalence, Jul 2020 - Dec 2021, by donor-catchment
    region.  `region == "Multi-region estimate (study-wide)"` is the national
    estimate.  Virginia is split across two regions:
      - "Eastern Virginia Region"
      - "West Virginia, Western Maryland and Northern Virginia Region"
    so neither is a clean statewide VA estimate.
    """
    path = (f"{CDC}/Nationwide__Blood_Donor_Seroprevalence_Survey_"
            f"Infection-Induced_Seroprevalence_Estimates.csv")
    df = pd.read_csv(path, low_memory=False)
    keep = ["region", "region_abbreviation", "year_and_month",
            "median_donation_date", "n_total_prevalence",
            "rate_total_prevalence", "lower_ci_total_prevalence",
            "upper_ci_total_prevalence"]
    df = df[keep].copy()
    df["median_donation_date"] = pd.to_datetime(df["median_donation_date"])
    df["month"] = pd.to_datetime(df["year_and_month"], format="%Y-%m",
                                 errors="coerce")
    df = df.rename(columns={"rate_total_prevalence": "anti_n_pct"})
    return df.dropna(subset=["month"]).sort_values(["region", "month"])


@_cached_frame
def fetch_blood_donor_2022_2023(refresh=False):
    """CDC 2022-2023 Nationwide Blood Donor Survey (ar8q-3jhn), from data.cdc.gov.

    Not in the commons snapshot.  Quarterly, by state FIPS plus a `USA` row.
    Use `indicator == "Past infection with or without vaccination"` for the
    infection-induced (anti-N) estimate; this is the only source here that
    covers 2022 Q1-Q2, and it is quarterly rather than monthly.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = f"{CACHE_DIR}/blood_donor_2022_2023_ar8q-3jhn.csv"
    if refresh or not os.path.exists(cache):
        url = "https://data.cdc.gov/resource/ar8q-3jhn.csv?$limit=100000"
        with urllib.request.urlopen(url, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
        with open(cache, "w") as fh:
            fh.write(raw)
    df = pd.read_csv(cache, dtype={"geographic_identifier": str})
    df["geographic_identifier"] = df["geographic_identifier"].str.zfill(2)
    q = df["time_period"].str.extract(r"(\d{4}) Quarter (\d)")
    df["quarter_start"] = pd.to_datetime(
        q[0] + "-" + ((q[1].astype(float) - 1) * 3 + 1).astype("Int64").astype(str) + "-01",
        errors="coerce",
    )
    df["quarter_mid"] = df["quarter_start"] + pd.Timedelta(days=45)
    return df


#: Median delay from infection to detectable anti-N antibody.  The
#: seroprevalence curve is measured on seroconversion time; shifting it back by
#: this much puts cumulative infections on infection time, which is what the
#: case and sequence series are indexed by.
SEROCONVERSION_LAG_DAYS = 21

#: A cumulative-infection curve can be anchored at zero before its first survey
#: round only if the survey caught the epidemic near its start.  Every series
#: here that legitimately qualifies opens between 2.2% and 7.9% seropositive;
#: the one that does not (Virginia 0-17, suppressed until Nov 2021) opens at
#: 28.2%.  Above this threshold the pre-survey period is left unestimated rather
#: than assumed empty.
ZERO_ANCHOR_MAX_PCT = 10.0


def seroprevalence_series(geo="US", splice_2022=True):
    """Single cumulative anti-N seroprevalence curve (percent) for `geo`.

    The backbone is the Commercial Laboratory Survey, which is state-resolved
    and roughly biweekly but stops reporting anti-N at round 30 (Feb 2022).  To
    reach mid-2022 the 2022-2023 Blood Donor Survey is spliced on.

    The two surveys are not interchangeable in level -- at the join the
    commercial-lab estimate is about 9 points higher nationally (57.7% vs
    48.8%), because clinical-lab remnant specimens come from people with more
    healthcare contact than volunteer blood donors.  So only the donor series'
    *increments* are carried over, added onto the commercial-lab level.  That
    uses the donor survey for the one thing it is needed for -- how much
    cumulative infection rose after Feb 2022 -- without importing its different
    baseline.  Rescaling multiplicatively instead would push the national curve
    over 100% by 2023.

    Returns a Series indexed by specimen-collection midpoint, plus a companion
    Series naming the source of each point.
    """
    cl = load_commercial_lab_seroprevalence()
    site = "US" if geo == "US" else geo
    sub = cl[(cl.site == site) & cl.anti_n_pct.notna()]
    base = (sub.set_index("specimen_mid")["anti_n_pct"]
            .groupby(level=0).mean().sort_index().astype(float))
    source = pd.Series("commercial_lab", index=base.index)

    if not splice_2022 or base.empty:
        return base, source

    bd = fetch_blood_donor_2022_2023()
    gid = "USA" if geo == "US" else STATE_FIPS[geo]
    donor = bd[
        (bd.indicator == "Past infection with or without vaccination")
        & (bd.geographic_identifier == gid)
        & (bd.age == "Overall") & (bd.sex == "Overall") & (bd.race == "Overall")
    ]
    donor = (donor.set_index("quarter_mid")["estimate_weighted"]
             .sort_index().astype(float).dropna())
    if donor.empty:
        return base, source

    # Align on the donor point nearest the end of the commercial-lab series,
    # then carry that point's forward increments onto the commercial-lab level.
    join = base.index.max()
    nearest = donor.index[np.argmin(np.abs(donor.index - join))]
    forward = donor[donor.index > nearest]
    if forward.empty:
        return base, source
    tail = (base.loc[join] + (forward - donor.loc[nearest])).clip(upper=100.0)

    combined = pd.concat([base, tail]).sort_index()
    source = pd.concat([source, pd.Series("blood_donor_2022", index=tail.index)])
    return combined, source.sort_index()


def cumulative_infections_from_seroprevalence(
    sero,
    population,
    index,
    seroreversion_halflife_days=None,
    seroconversion_lag_days=SEROCONVERSION_LAG_DAYS,
):
    """Turn a cumulative anti-N seroprevalence curve into cumulative infections.

    sero  : Series indexed by date, values in percent (0-100).
    index : DatetimeIndex to interpolate onto (the analysis time grid).

    Four things are done deliberately:

    1.  The round midpoints are shifted back by `seroconversion_lag_days` so the
        curve is indexed by infection date rather than seroconversion date.
        Without this the denominator lags the numerator by roughly three weeks,
        which is enough to push case ascertainment above 1 at wave onsets.

    2.  Interpolation onto the monthly grid is monotone cubic (PCHIP) rather
        than linear.  Survey rounds are irregular -- there is a three-month gap
        between rounds 24 and 26 -- and linear interpolation turns those gaps
        into flat stretches followed by step jumps, i.e. spurious zero-infection
        months next to spurious spikes.

    3.  The curve is forced non-decreasing with `cummax`.  Measured cumulative
        anti-N prevalence can dip between rounds from sampling noise and from
        seroreversion; a dip would otherwise produce negative incidence.

    4.  If `seroreversion_halflife_days` is given, the curve is inflated to undo
        antibody waning: each cohort's newly seropositive fraction is assumed to
        decay with that half-life, so the observed prevalence understates
        cumulative infection.  Leave as None for the raw, unadjusted
        (conservative, downward-biased) estimate.  Published anti-N half-life
        estimates span roughly 3-12 months, so treat this as a sensitivity knob
        rather than a calibrated correction.

    Values outside the observed seroprevalence range are left as NaN rather than
    extrapolated -- flat extrapolation would read as "zero infections".
    """
    s = sero.dropna().sort_index().astype(float)
    if s.empty:
        return pd.Series(np.nan, index=index, name="cumulative_infections")

    s.index = s.index - pd.Timedelta(days=seroconversion_lag_days)
    s = s.groupby(level=0).mean().sort_index()
    s = pd.Series(np.maximum.accumulate(s.to_numpy()), index=s.index)

    from scipy.interpolate import PchipInterpolator

    x = (s.index - s.index[0]).days.to_numpy(dtype=float)
    if len(x) < 2:
        return pd.Series(np.nan, index=index, name="cumulative_infections")
    interp = PchipInterpolator(x, s.to_numpy(), extrapolate=False)

    # Always work on a daily grid, then sample.  The seroreversion
    # reconstruction is a deconvolution and its answer depends on the grid it
    # runs on, so doing it on whatever index the caller asked for would make
    # the monthly panel and the era panel disagree.
    daily = pd.date_range(s.index[0], s.index[-1], freq="D")
    xd = (daily - s.index[0]).days.to_numpy(dtype=float)
    curve = pd.Series(interp(xd), index=daily) / 100.0

    if seroreversion_halflife_days:
        lam = np.log(2) / seroreversion_halflife_days
        obs = curve.to_numpy()
        true_inc = np.zeros(len(daily))
        decay_step = np.exp(-lam)
        carried = 0.0  # sum of earlier cohorts, decayed to the current day
        for i in range(len(daily)):
            carried *= decay_step
            true_inc[i] = max(obs[i] - carried, 0.0)
            carried += true_inc[i]
        curve = pd.Series(np.cumsum(true_inc), index=daily)

    # limit_area="inside" matters: the default would forward-fill past the last
    # survey round, turning "no data" into "no new infections".
    out = (curve.reindex(curve.index.union(index))
           .interpolate(method="time", limit_area="inside"))
    return (out.reindex(index) * population).rename("cumulative_infections")


# --------------------------------------------------------------------------
# Reported cases  (the case denominator)
# --------------------------------------------------------------------------

@_cached_frame
def load_weekly_cases_deaths():
    """CDC weekly aggregate cases/deaths by state (pwn4-m3yp), 2020-01 - 2023-05.

    Week ends on Wednesday.  Small and complete; preferred over the 15 GB case
    surveillance line lists when only counts are needed.
    """
    df = pd.read_csv(
        f"{CDC}/Weekly_United_States_COVID-19_Cases_and_Deaths_by_State_-_ARCHIVED.csv"
    )
    for c in ("date_updated", "start_date", "end_date"):
        df[c] = pd.to_datetime(df[c])
    return df.sort_values(["state", "end_date"])


@_cached_frame
def load_case_surveillance_monthly():
    """Monthly case counts aggregated from the CDC case surveillance line list.

    `PU_linelist_agg_cases.csv` is a pre-aggregation of the public-use case
    surveillance data (vbim-akqf / n8mc-b4w4) by case month, county, age, race
    and ethnicity.  Covers 2020-01 through 2022-07.  Use this rather than the
    raw 15-16 GB CSVs unless demographic strata are actually needed.
    """
    df = pd.read_csv(
        f"{CDC}/PU_linelist_agg_cases.csv",
        dtype={"state_fips_code": str, "county_fips_code": str},
    )
    df["state_fips_code"] = df["state_fips_code"].str.zfill(2)
    df["month"] = pd.to_datetime(df["case_month"], format="%Y-%m", errors="coerce")
    return df.dropna(subset=["month"])


# --------------------------------------------------------------------------
# Severity / healthcare-seeking context
# --------------------------------------------------------------------------

def load_hospitalizations(jurisdictions=("USA", "VA")):
    """NHSN weekly COVID hospital metrics by jurisdiction (aemt-mg7g).

    Week ends on Saturday, 2020-08-08 onward, so the first ~5 months of the
    pandemic have no NHSN coverage.
    """
    cols = [
        "week_end_date", "jurisdiction",
        "total_admissions_all_covid_confirmed",
        "total_admissions_adult_covid_confirmed",
        "total_admissions_pediatric_covid_confirmed",
        "avg_total_patients_hospitalized_covid_confirmed",
        "avg_staff_icu_patients_covid_confirmed",
        "avg_percent_inpatient_beds_covid",
    ]
    df = pd.read_csv(f"{CDC}/{NHSN_FILE}", usecols=cols, low_memory=False)
    df["week_end_date"] = pd.to_datetime(df["week_end_date"])
    if jurisdictions:
        df = df[df.jurisdiction.isin(jurisdictions)]
    return df.sort_values(["jurisdiction", "week_end_date"])


def load_nssp_ed_visits():
    """NSSP ED visits, percent of visits by pathogen (vutn-jzwm) -- state level.

    This is the *summary* NSSP dataset and it begins 2023-10-07, well after the
    2020 - mid-2022 analysis window.  Use `load_nssp_ed_trajectories` instead
    when earlier NSSP data is wanted: the trajectories dataset carries a full
    extra year of history.
    """
    path = (f"{CDC}/2023_Respiratory_Virus_Response_-_NSSP_Emergency_Department_"
            f"Visits_-_COVID-19,_Flu,_RSV,_Combined.csv")
    df = pd.read_csv(path)
    df["week_end"] = pd.to_datetime(df["week_end"])
    return df.sort_values(["geography", "pathogen", "week_end"])


def load_nssp_ed_trajectories(state_level=True):
    """NSSP ED visit trajectories by state and sub-state region (rdmq-nq56).

    `percent_visits_covid` is the share of emergency department visits with
    diagnosed COVID-19.  Begins **2022-10-01** -- a year earlier than the
    summary NSSP file -- and runs to 2024-12-28.

    That still starts after the 2020 - mid-2022 ascertainment window, so it
    cannot supply an ED signal for that period.  It does however overlap the
    CDC weekly case series (which ends 2023-05-10) by 33 weeks, which is enough
    to calibrate the two against each other.

    `state_level=True` keeps the `county == "All"` rows, i.e. one series per
    state plus a `United States` national row; False returns the full
    county/sub-state detail.

    Note the filename in the commons contains U+202F narrow no-break spaces,
    so it is resolved by glob rather than written out literally.
    """
    import glob

    matches = glob.glob(
        f"{CDC}/NSSP_Emergency_Department_Visit_Trajectories*.csv"
    )
    if not matches:
        raise FileNotFoundError("NSSP trajectories file not found in the commons")
    df = pd.read_csv(
        sorted(matches)[0],
        usecols=["week_end", "geography", "county", "percent_visits_covid",
                 "percent_visits_smoothed_covid", "percent_visits_combined"],
        low_memory=False,
    )
    df["week_end"] = pd.to_datetime(df["week_end"])
    if state_level:
        df = df[df.county == "All"].drop(columns="county")
    return df.sort_values(["geography", "week_end"])


def nssp_vs_reported_cases(geo="US"):
    """Weekly NSSP COVID ED-visit share against CDC reported cases, where they overlap.

    NSSP weeks end Saturday and the CDC case weeks end Wednesday, so the case
    index is shifted forward three days to close the same seven-day span.

    Returns only the overlapping weeks (2022-10-01 to 2023-05-13, 33 weeks).
    """
    nssp = load_nssp_ed_trajectories(state_level=True)
    name = "United States" if geo == "US" else _STATE_NAMES[geo]
    ed = nssp[nssp.geography == name].set_index("week_end")["percent_visits_covid"]

    cases = load_weekly_cases_deaths()
    if geo == "US":
        # Territories are outside the NSSP catchment used for the national row.
        drop = {"PR", "VI", "GU", "AS", "MP", "FSM", "RMI", "PW"}
        c = cases[~cases.state.isin(drop)].groupby("end_date")["new_cases"].sum()
    else:
        states = [geo] + (["NYC"] if geo == "NY" else [])
        c = cases[cases.state.isin(states)].groupby("end_date")["new_cases"].sum()
    c.index = c.index + pd.Timedelta(days=3)

    out = pd.DataFrame({"ed_pct_covid": ed, "reported_cases": c}).dropna()
    out["cases_per_1pct_ed"] = out.reported_cases / out.ed_pct_covid
    return out


def load_covidcast_doctor_visits():
    """Delphi COVIDcast outpatient CLI (`doctor-visits_smoothed_adj_cli`), county.

    Stands in for NSSP during 2020 - mid-2022, where NSSP has no data.  This is
    percent of outpatient visits with COVID-like illness, not ED visits, so it
    is a healthcare-seeking proxy rather than a like-for-like substitute.
    """
    path = f"{COMMONS}/COVIDcast/doctor-visits_smoothed_adj_cli_county.csv"
    df = pd.read_csv(path, dtype={"geo_value": str})
    for c in df.columns:
        if "date" in c:
            df[c] = pd.to_datetime(df[c], errors="ignore")
    return df


# --------------------------------------------------------------------------
# Variant context
# --------------------------------------------------------------------------

def load_cdc_variant_proportions(region="USA", modeltype="weighted"):
    """CDC SARS-CoV-2 Variant Proportions (jr58-6ysp).

    Weighted/nowcast lineage *shares* for USA and the 10 HHS regions, 2021-01
    onward.  There are no sequence counts in this file, and no state
    resolution, so it supports variant attribution but not the sequencing
    fraction itself.  Virginia sits in HHS Region 3.

    Multiple `creation_date` vintages are stacked; the most recent vintage for
    each week is kept.
    """
    # 3M rows x 10 columns, and we keep one region and one model type.  Reading
    # it whole needs several GB transiently, which is enough to fail partway
    # through a notebook run, so filter chunk by chunk instead.
    usecols = ["usa_or_hhsregion", "week_ending", "variant", "share",
               "share_hi", "share_lo", "modeltype", "creation_date"]
    keep = []
    for chunk in pd.read_csv(f"{CDC}/SARS-CoV-2_Variant_Proportions.csv",
                             usecols=usecols,
                             dtype={"usa_or_hhsregion": str, "variant": str,
                                    "modeltype": str},
                             chunksize=500_000):
        keep.append(chunk[(chunk.usa_or_hhsregion == region)
                          & (chunk.modeltype == modeltype)])
    df = pd.concat(keep, ignore_index=True)
    df["week_ending"] = pd.to_datetime(df["week_ending"])
    df["creation_date"] = pd.to_datetime(df["creation_date"])
    latest = df.groupby("week_ending")["creation_date"].transform("max")
    return (df[df.creation_date == latest]
            .sort_values(["week_ending", "share"], ascending=[True, False]))


# --------------------------------------------------------------------------
# Age stratification
# --------------------------------------------------------------------------

#: The bands every source can be harmonised onto.  The commercial-lab
#: seroprevalence survey and the case-surveillance aggregate both publish
#: exactly these; NSSP matches on the three adult bands (see
#: `fetch_nssp_by_age` for why its paediatric bands cannot be collapsed).
AGE_BANDS = ["0-17", "18-49", "50-64", "65+"]

#: Suppression sentinels used in every `rate_*` column of the commercial-lab
#: survey.  They are ordinary numbers in the CSV, so anything that does not mask
#: them will silently read 777 as "777% seropositive".
#:   666 = no specimens were collected
#:   777 = cell size below 75, estimate not shown
SERO_SENTINELS = (666, 777)

#: Commercial-lab column suffixes -> harmonised band.
_SERO_AGE_COLS = {
    "rate_0_17_prevalence": "0-17",
    "rate_18_49_prevalence": "18-49",
    "rate_50_64_prevalence": "50-64",
    "rate_65_prevalence": "65+",
}

#: Case-surveillance `age_group` labels -> harmonised band.
_CASE_AGE_MAP = {
    "0 - 17 years": "0-17",
    "18 to 49 years": "18-49",
    "50 to 64 years": "50-64",
    "65+ years": "65+",
}

_CENSUS_AGESEX_URL = (
    "https://www2.census.gov/programs-surveys/popest/datasets/2020-2021/"
    "state/asrh/sc-est2021-agesex-civ.csv"
)


def _band_from_years(age):
    """Single-year age -> harmonised band."""
    if age < 18:
        return "0-17"
    if age < 50:
        return "18-49"
    if age < 65:
        return "50-64"
    return "65+"


@_cached_frame
def fetch_age_populations(refresh=False, rescale_to_catchment=True):
    """Population by age band for US and each state, from Census PEP.

    Source is the Census Bureau's single-year-of-age civilian population
    estimates (vintage 2021), downloaded as a plain CSV -- the Census *API*
    needs a key, this bulk file does not.  Single years let the bands be cut
    exactly at 18/50/65 rather than approximated from published groupings.

    `rescale_to_catchment` scales each geography's bands so they sum to the
    same `catchment_population` the all-ages analysis uses.  The Census figure
    is civilian noninstitutionalised and runs ~1-2% above the survey catchment;
    rescaling keeps the age-stratified infection counts reconcilable with the
    all-ages ones instead of drifting a percent or two apart.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = f"{CACHE_DIR}/census_sc-est2021-agesex-civ.csv"
    if refresh or not os.path.exists(cache):
        with urllib.request.urlopen(_CENSUS_AGESEX_URL, timeout=180) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        with open(cache, "w") as fh:
            fh.write(raw)

    d = pd.read_csv(cache)
    d = d[(d.SEX == 0) & (d.AGE < 999)].copy()      # both sexes, drop the total row
    d["band"] = d.AGE.map(_band_from_years)
    d["geo"] = np.where(d.SUMLEV == 10, "US", d.NAME.map(
        {v: k for k, v in _STATE_NAMES.items()}))
    d = d.dropna(subset=["geo"])

    pop = (d.groupby(["geo", "band"])["POPEST2021_CIV"].sum()
           .unstack("band")[AGE_BANDS].astype(float))

    if rescale_to_catchment:
        for geo in pop.index:
            try:
                target = catchment_population(geo)
            except (KeyError, ValueError):
                continue
            if np.isfinite(target) and pop.loc[geo].sum() > 0:
                pop.loc[geo] *= target / pop.loc[geo].sum()
    return pop


@_cached_frame
def load_commercial_lab_seroprevalence_by_age():
    """Commercial-lab anti-N seroprevalence reshaped long by age band.

    Same survey as `load_commercial_lab_seroprevalence`, with `SERO_SENTINELS`
    masked to NaN.

    Usable (non-sentinel) coverage over rounds 1-30, across all sites, is 94% for
    18-49, 93% for 50-64 and 93% for 65+, but only 64% for 0-17 -- paediatric
    cell sizes fall below the suppression threshold far more often.  That bites
    hardest in single states: Virginia's 0-17 band is suppressed in 27 of its 30
    rounds and only becomes usable from Nov 2021, whereas the US 0-17 row has no
    suppressed rounds at all.

    Coverage also differs by band after round 30 -- see
    `seroprevalence_series_by_age`.
    """
    df = pd.read_csv(
        f"{CDC}/Nationwide_Commercial_Laboratory_Seroprevalence_Survey.csv",
        low_memory=False,
    )
    keep = ["site", "round", "date_range_of_specimen"] + list(_SERO_AGE_COLS)
    df = df[keep].copy()

    parsed = df["date_range_of_specimen"].map(_parse_specimen_range)
    df["specimen_start"] = [p[0] for p in parsed]
    df["specimen_end"] = [p[1] for p in parsed]
    mid = df["specimen_start"] + (df["specimen_end"] - df["specimen_start"]) / 2
    df["specimen_mid"] = mid
    round_mid = df.groupby("round")["specimen_mid"].median().interpolate()
    df["specimen_mid"] = df["specimen_mid"].fillna(df["round"].map(round_mid))

    long = df.melt(
        id_vars=["site", "round", "specimen_mid"],
        value_vars=list(_SERO_AGE_COLS),
        var_name="col", value_name="anti_n_pct",
    )
    long["age_band"] = long["col"].map(_SERO_AGE_COLS)
    long["anti_n_pct"] = long["anti_n_pct"].mask(long["anti_n_pct"].isin(SERO_SENTINELS))
    return (long.drop(columns="col")
            .sort_values(["site", "age_band", "round"])
            .reset_index(drop=True))


def load_case_surveillance_by_age(geo="US", scale_to_weekly=True):
    """Monthly reported cases by harmonised age band.

    `PU_linelist_agg_cases.csv` carries an `age_group` column whose four
    categories map one-to-one onto `AGE_BANDS`.  Rows with age "Missing"
    (~0.8% of cases) are dropped rather than redistributed.

    The line list is badly incomplete as a *count*: it holds about 58% of the
    weekly-series cases nationally and 47% in Virginia, and the shortfall drifts
    over time (VA runs 55% in 2020 and 40% in 2022).  Using it raw would halve
    case ascertainment relative to the all-ages panel for reasons that have
    nothing to do with age.

    So with `scale_to_weekly=True` (the default) the line list supplies only the
    age *composition* of each month, and that composition is applied to the
    month's total from the weekly case series.  The band counts then sum to the
    same `reported_cases` the all-ages panel uses.  Pass False for the raw
    line-list counts.
    """
    df = load_case_surveillance_monthly()
    if geo != "US":
        df = df[df.state_fips_code == STATE_FIPS[geo]]
    df = df[df.age_group.isin(_CASE_AGE_MAP)].copy()
    df["age_band"] = df.age_group.map(_CASE_AGE_MAP)
    counts = (df.groupby(["month", "age_band"])["count"].sum()
              .unstack("age_band").reindex(columns=AGE_BANDS))
    if not scale_to_weekly:
        return counts

    wc = load_weekly_cases_deaths()
    if geo == "US":
        wc = wc[wc.state.isin(set(STATE_FIPS) | {"NYC"})]
    else:
        wc = wc[wc.state.isin([geo] + (["NYC"] if geo == "NY" else []))]
    total = wc.set_index("end_date")["new_cases"].resample("MS").sum()

    shares = counts.div(counts.sum(axis=1), axis=0)
    return shares.mul(total.reindex(counts.index), axis=0)


#: Blood-donor age bands -> harmonised band.  The donor survey starts at 16, so
#: it has nothing for 0-17; 18-49 has to be pooled from two donor bands and is
#: therefore the least clean of the three.
_DONOR_AGE_MAP = {
    "18-49": ["16 to 29", "30 to 49"],
    "50-64": ["50 to 64"],
    "65+": ["65 and over"],
}


def seroprevalence_series_by_age(geo="US", band="18-49", splice_2022=True):
    """Cumulative anti-N seroprevalence (percent) for one age band.

    Coverage differs by band, because CDC changed what the commercial-lab survey
    collected at round 31:

    - **0-17** continues in the commercial-lab survey all the way to round 35
      (Dec 2022), so it needs no splice and covers the whole window natively.
    - **18-49 / 50-64 / 65+** stop at round 30 (Feb 2022), same as the all-ages
      series, so the 2022-23 Blood Donor Survey is spliced on by increments
      exactly as in `seroprevalence_series`.

    The donor bands do not line up perfectly: 18-49 is pooled from the donor
    "16 to 29" and "30 to 49" bands, population-weighted, so it carries 16-17
    year olds it should not. 50-64 and 65+ map exactly.
    """
    sero = load_commercial_lab_seroprevalence_by_age()
    site = "US" if geo == "US" else geo
    base = (sero[(sero.site == site) & (sero.age_band == band)
                 & sero.anti_n_pct.notna()]
            .set_index("specimen_mid")["anti_n_pct"]
            .groupby(level=0).mean().sort_index().astype(float))
    source = pd.Series("commercial_lab", index=base.index)
    if not splice_2022 or base.empty or band not in _DONOR_AGE_MAP:
        return base, source

    bd = fetch_blood_donor_2022_2023()
    gid = "USA" if geo == "US" else STATE_FIPS[geo]
    sub = bd[(bd.indicator == "Past infection with or without vaccination")
             & (bd.geographic_identifier == gid)
             & (bd.sex == "Overall") & (bd.race == "Overall")
             & (bd.age.isin(_DONOR_AGE_MAP[band]))]
    if sub.empty:
        return base, source

    if len(_DONOR_AGE_MAP[band]) == 1:
        donor = sub.set_index("quarter_mid")["estimate_weighted"]
    else:
        # Pool the donor bands by population weight rather than averaging them.
        pops = fetch_age_populations(rescale_to_catchment=False).loc[geo]
        w = {"16 to 29": pops["18-49"] * 12 / 32, "30 to 49": pops["18-49"] * 20 / 32}
        sub = sub.assign(w=sub.age.map(w))
        donor = (sub.assign(num=sub.estimate_weighted * sub.w)
                 .groupby("quarter_mid")[["num", "w"]].sum()
                 .pipe(lambda d: d.num / d.w))
    donor = donor.sort_index().astype(float).dropna()
    if donor.empty:
        return base, source

    join = base.index.max()
    nearest = donor.index[np.argmin(np.abs(donor.index - join))]
    forward = donor[donor.index > nearest]
    if forward.empty:
        return base, source
    tail = (base.loc[join] + (forward - donor.loc[nearest])).clip(upper=100.0)
    combined = pd.concat([base, tail]).sort_index()
    source = pd.concat([source, pd.Series("blood_donor_2022", index=tail.index)])
    return combined, source.sort_index()


def fetch_nssp_by_age(refresh=False):
    """NSSP ED visits by age band (7xva-uux8), from data.cdc.gov.

    Not in the commons.  **National only** -- there is no state breakdown, so
    this cannot be done for Virginia -- and it starts 2022-10-01, outside the
    2020 - mid-2022 window.

    `percent_visits` here is the share of ED visits *within* each age group
    that were COVID-related, not each group's share of all COVID visits.  The
    bands therefore cannot be summed: for the week ending 2023-01-07 they run
    1.3% (5-17) to 6.9% (<1) and add to 21.5%, against an all-ages figure near
    3.8%.  That also means the paediatric bands (<1, 1-4, 5-17) cannot be
    combined into a 0-17 band without ED visit volumes, which this dataset does
    not publish -- so NSSP joins the other sources on 18-49, 50-64 and 65+ only.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = f"{CACHE_DIR}/nssp_by_age_7xva-uux8.csv"
    if refresh or not os.path.exists(cache):
        url = ("https://data.cdc.gov/resource/7xva-uux8.csv?"
               + urllib.parse.urlencode({
                   "$where": "demographics_type='Age Group'",
                   "$limit": 200000,
               }))
        with urllib.request.urlopen(url, timeout=180) as resp:
            raw = resp.read().decode("utf-8")
        with open(cache, "w") as fh:
            fh.write(raw)
    df = pd.read_csv(cache)
    df["week_end"] = pd.to_datetime(df["week_end"])
    df = df.rename(columns={"demographics_values": "age_group"})
    # NSSP labels carry a " years" suffix; the three adult bands then match
    # AGE_BANDS exactly.  The paediatric ones deliberately do not map -- see the
    # docstring on why they cannot be pooled into 0-17.
    nssp_to_band = {"18-49 years": "18-49", "50-64 years": "50-64", "65+ years": "65+"}
    df["age_band"] = df.age_group.map(nssp_to_band)
    return df.sort_values(["pathogen", "age_group", "week_end"])


def vdh_sequences_by_age(freq="MS"):
    """Virginia genomes by harmonised age band, from the VDH/DCLS line list.

    The commons GISAID extract carries no age, but the VDH line list does --
    `Patient Age In Years` is a numeric age, so it bins exactly onto
    `AGE_BANDS`.  Age is missing for ~6.4% of sequenced specimens; those are
    dropped, so band counts sum slightly below the VDH total.

    This is Virginia only. There is no equivalent age field for the national
    sequence counts, so `frac_infections_sequenced` can only be age-stratified
    for VA.
    """
    df = load_vdh_genomic_linelist(with_age=True)
    df = df.dropna(subset=["collection_date", "Patient Age In Years"]).copy()
    df["age_band"] = df["Patient Age In Years"].map(_band_from_years)
    return (df.set_index("collection_date")
            .groupby("age_band").resample(freq).size()
            .unstack("age_band")[AGE_BANDS])


def build_age_panel(geo="US", seroreversion_halflife_days=365,
                    start=WINDOW_START, end=WINDOW_END,
                    seroconversion_lag_days=SEROCONVERSION_LAG_DAYS,
                    sequence_source="vdh"):
    """Monthly panel by age band: infections, cases and case ascertainment.

    Same construction as `build_monthly_panel`, run once per age band with that
    band's own seroprevalence curve and its own population denominator.

    Sequences are included only for `geo == "VA"`, where the VDH line list
    supplies age; the national GISAID extract has no age field.

    Returns a long frame indexed by (month, age_band).
    """
    index = pd.date_range(pd.Timestamp(start).normalize().replace(day=1), end, freq="MS")
    bounds = index.union([index[-1] + pd.offsets.MonthBegin(1)])

    pops = fetch_age_populations().loc[geo]
    cases = load_case_surveillance_by_age(geo).reindex(index)
    seqs = (sequence_counts_by_age(geo, sequence_source).reindex(index)
            if geo == "VA" else None)

    frames = []
    for band in AGE_BANDS:
        s, _ = seroprevalence_series_by_age(geo, band)
        cum = cumulative_infections_from_seroprevalence(
            s, float(pops[band]), bounds,
            seroreversion_halflife_days=seroreversion_halflife_days,
            seroconversion_lag_days=seroconversion_lag_days,
        )
        f = pd.DataFrame(index=index)
        f["age_band"] = band
        f["population"] = float(pops[band])
        f["cum_infections"] = cum.reindex(index)
        f["infections"] = cum.diff().shift(-1).reindex(index)
        f["reported_cases"] = cases[band] if band in cases else np.nan
        f["n_sequences"] = seqs[band] if seqs is not None and band in seqs else np.nan
        frames.append(f)

    panel = pd.concat(frames).set_index("age_band", append=True)
    panel.index.names = ["month", "age_band"]
    ok = panel["infections"].notna()
    panel["case_ascertainment"] = (panel.reported_cases / panel.infections).where(ok)
    panel["infections_per_case"] = (panel.infections / panel.reported_cases).where(ok)
    panel["attack_rate"] = panel.cum_infections / panel.population
    panel["frac_infections_sequenced"] = (panel.n_sequences / panel.infections).where(ok)
    panel["frac_cases_sequenced"] = panel.n_sequences / panel.reported_cases
    panel["geo"] = geo
    return panel.sort_index()


def build_age_era_panel(geo="US", eras=None, seroreversion_halflife_days=365,
                        seroconversion_lag_days=SEROCONVERSION_LAG_DAYS,
                        sequence_source="vdh"):
    """Variant-era totals and ratios by age band.

    Era infections are read off each band's cumulative curve at the era
    boundaries, for the same reason as in `build_era_panel`.
    """
    eras = eras if eras is not None else VARIANT_ERAS
    monthly = build_age_panel(geo, seroreversion_halflife_days=seroreversion_halflife_days,
                              seroconversion_lag_days=seroconversion_lag_days,
                              sequence_source=sequence_source)
    pops = fetch_age_populations().loc[geo]

    bounds = pd.DatetimeIndex(sorted({pd.Timestamp(b) for _, lo, hi in eras
                                      for b in (lo, pd.Timestamp(hi) + pd.Timedelta(days=1))}))
    rows = []
    for band in AGE_BANDS:
        s, _ = seroprevalence_series_by_age(geo, band)
        cum_at = cumulative_infections_from_seroprevalence(
            s, float(pops[band]), bounds,
            seroreversion_halflife_days=seroreversion_halflife_days,
            seroconversion_lag_days=seroconversion_lag_days,
        )
        if not s.empty and s.iloc[0] <= ZERO_ANCHOR_MAX_PCT:
            first_obs = s.index.min() - pd.Timedelta(days=seroconversion_lag_days)
            pre = cum_at.index < first_obs
            cum_at.loc[pre] = cum_at.loc[pre].fillna(0.0)

        sub = monthly.xs(band, level="age_band")
        for name, lo, hi in eras:
            lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
            m = sub.loc[(sub.index >= lo) & (sub.index <= hi)]
            rows.append({
                "era": name, "age_band": band,
                "infections": cum_at.get(hi + pd.Timedelta(days=1), np.nan) - cum_at.get(lo, np.nan),
                "reported_cases": m["reported_cases"].sum(min_count=1),
                "n_sequences": m["n_sequences"].sum(min_count=1),
                "population": float(pops[band]),
            })
    out = pd.DataFrame(rows).set_index(["era", "age_band"])
    out["case_ascertainment"] = out.reported_cases / out.infections
    out["infections_per_case"] = out.infections / out.reported_cases
    out["attack_rate_in_era"] = out.infections / out.population
    out["frac_infections_sequenced"] = out.n_sequences / out.infections
    out["frac_cases_sequenced"] = out.n_sequences / out.reported_cases
    out["geo"] = geo
    return out


# --------------------------------------------------------------------------
# Assembled panel
# --------------------------------------------------------------------------

def catchment_population(geo="US"):
    """Population denominator, taken from the seroprevalence survey catchment."""
    cl = load_commercial_lab_seroprevalence()
    site = "US" if geo == "US" else geo
    # Rounds 31+ shrank the catchments; use the full-state figure from the
    # rounds that actually carry the anti-N estimates.
    return float(cl[(cl.site == site) & (cl["round"] <= 30)]
                 ["catchment_population"].max())


def build_monthly_panel(geo="US", seroreversion_halflife_days=None,
                        start=WINDOW_START, end=WINDOW_END,
                        seroconversion_lag_days=SEROCONVERSION_LAG_DAYS,
                        splice_2022=True, sequence_source="gisaid"):
    """Monthly panel of sequences, cases, hospitalisations and infections.

    Monthly is the honest resolution: the seroprevalence rounds that anchor the
    infection denominator are roughly monthly, and the case surveillance
    aggregate is monthly.

    geo : "US" or a two-letter state code (the commercial-lab survey has a
          matching site for every state, plus a `US` national row).

    Columns
    -------
    n_sequences            genomes collected that month (GISAID / outbreak.info)
    reported_cases         CDC weekly aggregate cases, summed to month
    cases_surveillance     CDC case surveillance line-list cases by case month
    hosp_admissions        NHSN confirmed COVID admissions
    cum_infections         cumulative infections from anti-N seroprevalence
    infections             monthly incident infections (difference of the above)
    frac_infections_sequenced   n_sequences / infections
    frac_cases_sequenced        n_sequences / reported_cases
    case_ascertainment          reported_cases / infections
    """
    index = pd.date_range(start=pd.Timestamp(start).normalize().replace(day=1),
                          end=end, freq="MS")
    panel = pd.DataFrame(index=index)
    panel.index.name = "month"

    # --- sequences ---
    panel["n_sequences"] = monthly_sequence_counts(geo, sequence_source).reindex(index)
    panel["sequence_source"] = sequence_source

    # --- reported cases ---
    wc = load_weekly_cases_deaths()
    if geo == "US":
        # NYC is reported separately from NY in this file; both are real, and
        # the territories are excluded to match the seroprevalence catchment.
        keep = set(STATE_FIPS) | {"NYC"}
        wc = wc[wc.state.isin(keep)]
        cases = wc.set_index("end_date")["new_cases"].resample("MS").sum()
    else:
        states = [geo] + (["NYC"] if geo == "NY" else [])
        cases = (wc[wc.state.isin(states)]
                 .set_index("end_date")["new_cases"].resample("MS").sum())
    panel["reported_cases"] = cases.reindex(index)

    cs = load_case_surveillance_monthly()
    if geo != "US":
        cs = cs[cs.state_fips_code == STATE_FIPS[geo]]
    panel["cases_surveillance"] = (
        cs.groupby("month")["count"].sum().reindex(index)
    )

    # --- hospitalisations ---
    juris = "USA" if geo == "US" else geo
    hosp = load_hospitalizations(jurisdictions=(juris,))
    panel["hosp_admissions"] = (
        hosp.set_index("week_end_date")["total_admissions_all_covid_confirmed"]
        .resample("MS").sum().reindex(index)
    )

    # --- infections from seroprevalence ---
    population = catchment_population(geo)
    sero, _ = seroprevalence_series(geo, splice_2022=splice_2022)

    # Evaluate the cumulative curve at month *boundaries* so that a month's
    # infections are the rise across that month, not the rise into it.
    bounds = index.union([index[-1] + pd.offsets.MonthBegin(1)])
    cum = cumulative_infections_from_seroprevalence(
        sero, population, bounds,
        seroreversion_halflife_days=seroreversion_halflife_days,
        seroconversion_lag_days=seroconversion_lag_days,
    )
    panel["cum_infections"] = cum.reindex(index)
    panel["infections"] = cum.diff().shift(-1).reindex(index)
    panel["population"] = population

    # --- derived ascertainment ratios ---
    # Only defined where the seroprevalence curve actually covers both ends of
    # the month; elsewhere `infections` is a difference against a NaN.
    ok = panel["infections"].notna()
    panel["frac_infections_sequenced"] = (panel.n_sequences / panel.infections).where(ok)
    panel["frac_cases_sequenced"] = panel.n_sequences / panel.reported_cases
    panel["case_ascertainment"] = (panel.reported_cases / panel.infections).where(ok)
    panel["infections_per_case"] = (panel.infections / panel.reported_cases).where(ok)
    panel["geo"] = geo
    return panel


#: Variant eras for the analysis window, by specimen collection date.  Cut
#: points are where the CDC national weighted proportions cross ~50%, rounded to
#: month boundaries so they line up with the monthly panel.
VARIANT_ERAS = [
    ("Ancestral / pre-VOC", "2020-03-01", "2021-01-31"),
    ("Alpha (B.1.1.7)",     "2021-02-01", "2021-06-30"),
    ("Delta (B.1.617.2)",   "2021-07-01", "2021-11-30"),
    ("Omicron BA.1",        "2021-12-01", "2022-02-28"),
    ("Omicron BA.2 / BA.5", "2022-03-01", "2022-06-30"),
]


def build_era_panel(monthly, geo=None, eras=VARIANT_ERAS,
                    seroreversion_halflife_days=None,
                    seroconversion_lag_days=SEROCONVERSION_LAG_DAYS,
                    splice_2022=True):
    """Collapse a monthly panel to variant eras.

    Monthly ascertainment ratios built off a seroprevalence curve are noisy:
    the survey rounds are irregular and differencing amplifies sampling error,
    so a single month's ratio can swing by a factor of several for reasons that
    have nothing to do with surveillance effort.  Summing counts over a whole
    variant era first, then taking the ratio, is far more stable and is the
    level at which these numbers should be quoted.

    Era infections come from the cumulative curve evaluated at the era
    boundaries, not from summing the monthly differences.  That keeps the first
    era usable: cumulative infections at 2020-03-01 are effectively zero, so the
    curve's level at the era end *is* the era's infection count, even though the
    survey did not start until July 2020 and the monthly differences are
    therefore missing.
    """
    geo = geo if geo is not None else monthly["geo"].iloc[0]
    population = catchment_population(geo)
    sero, _ = seroprevalence_series(geo, splice_2022=splice_2022)

    bounds = pd.DatetimeIndex(sorted({pd.Timestamp(b) for _, lo, hi in eras
                                      for b in (lo, pd.Timestamp(hi) + pd.Timedelta(days=1))}))
    cum_at = cumulative_infections_from_seroprevalence(
        sero, population, bounds,
        seroreversion_halflife_days=seroreversion_halflife_days,
        seroconversion_lag_days=seroconversion_lag_days,
    )
    # Before the survey began, cumulative infection is ~0 rather than unknown --
    # but only if the survey actually started near the beginning of the epidemic.
    if not sero.empty and sero.iloc[0] <= ZERO_ANCHOR_MAX_PCT:
        first_obs = sero.index.min() - pd.Timedelta(days=seroconversion_lag_days)
        pre = cum_at.index < first_obs
        cum_at.loc[pre] = cum_at.loc[pre].fillna(0.0)

    rows = []
    for name, lo, hi in eras:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        m = monthly.loc[(monthly.index >= lo) & (monthly.index <= hi)]
        if m.empty:
            continue
        c0, c1 = cum_at.get(lo, np.nan), cum_at.get(hi + pd.Timedelta(days=1), np.nan)
        infections = c1 - c0
        rows.append({
            "era": name,
            "start": lo,
            "end": hi,
            "months": len(m),
            "n_sequences": m["n_sequences"].sum(),
            "reported_cases": m["reported_cases"].sum(),
            "cases_surveillance": m["cases_surveillance"].sum(),
            "hosp_admissions": m["hosp_admissions"].sum(min_count=1),
            "infections": infections,
            "sero_complete": bool(np.isfinite(infections)),
        })
    out = pd.DataFrame(rows).set_index("era")
    out["frac_infections_sequenced"] = out.n_sequences / out.infections
    out["frac_cases_sequenced"] = out.n_sequences / out.reported_cases
    out["case_ascertainment"] = out.reported_cases / out.infections
    out["infections_per_case"] = out.infections / out.reported_cases
    return out
