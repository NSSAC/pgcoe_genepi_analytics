"""Analysis 1 -- ascertainment_over_time.

How much of the epidemic did genomic surveillance actually see, and how did
that change over the run? Per simulation day: infections that occurred,
infections that were sequenced, and the ratio.

Outputs
    01_ascertainment_over_time.csv          per-day counts and rate
    01_ascertainment_over_time.png          incidence + sequenced, and rate over time
    01_ascertainment_over_time_summary.csv  tidy scalars for cross-run pooling
"""

import numpy as np

from pipeline_common import cc, COL, apply_plot_style, log

INDEX = 1
NAME = "ascertainment_over_time"
DESCRIPTION = "Per-day infections, sequenced infections, and ascertainment rate"


def run(ctx):
    log(f"[{INDEX}] {NAME}: building per-day ascertainment series")
    df = cc.build_ascertainment_timeseries(ctx.epi_tick, ctx.captured_epi_nodes,
                                            window=ctx.seq_window)
    ctx.write_table(df, INDEX, NAME)

    in_win = df[df["in_seq_window"]]
    total_inf = int(df["n_infections"].sum())
    total_seq = int(df["n_sequenced"].sum())
    win_inf = int(in_win["n_infections"].sum())
    win_seq = int(in_win["n_sequenced"].sum())

    add = lambda m, v: ctx.add_metric(INDEX, NAME, m, v)
    add("n_days_observed", int(len(df)))
    add("first_infection_tick", int(df["tick"].min()))
    add("last_infection_tick", int(df["tick"].max()))
    add("seq_window_start_tick", int(ctx.seq_window[0]))
    add("seq_window_end_tick", int(ctx.seq_window[1]))
    add("seq_window_days", int(ctx.seq_window[1] - ctx.seq_window[0]))
    add("n_infections_total", total_inf)
    add("n_sequenced_total", total_seq)
    add("ascertainment_rate_overall", total_seq / total_inf if total_inf else np.nan)
    add("n_infections_in_window", win_inf)
    add("n_sequenced_in_window", win_seq)
    add("ascertainment_rate_in_window", win_seq / win_inf if win_inf else np.nan)
    add("ascertainment_rate_in_window_daily_median",
        float(in_win["ascertainment_rate"].median()) if len(in_win) else np.nan)
    add("ascertainment_rate_in_window_daily_max",
        float(in_win["ascertainment_rate"].max()) if len(in_win) else np.nan)
    add("peak_daily_incidence", int(df["n_infections"].max()))
    add("peak_daily_incidence_tick", int(df.loc[df["n_infections"].idxmax(), "tick"]))

    _plot(ctx, df)
    ctx.write_summary(INDEX, NAME, NAME)


def _plot(ctx, df):
    plt = apply_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    lo, hi = ctx.seq_window

    ax = axes[0]
    ax.fill_between(df["tick"], df["n_infections"], color=COL["total"], alpha=.35,
                    label="infections")
    ax.plot(df["tick"], df["n_sequenced"], color=COL["captured"], lw=1.2, label="sequenced")
    ax.axvspan(lo, hi, color=COL["window"], zorder=0)
    ax.set_yscale("log")
    ax.set_ylabel("count per day (log)")
    ax.set_title(f"Incidence vs. sequencing over time -- {ctx.run_label}")
    ax.legend(fontsize=9, loc="upper right")

    ax = axes[1]
    ax.plot(df["tick"], 100 * df["ascertainment_rate"], color=COL["zero"], lw=.8, alpha=.55)
    roll = (100 * df["ascertainment_rate"]).rolling(7, center=True, min_periods=1).mean()
    ax.plot(df["tick"], roll, color=COL["zero"], lw=2, label="7-day mean")
    ax.axvspan(lo, hi, color=COL["window"], zorder=0, label="sequencing window")
    ax.set_ylabel("ascertainment rate (% of infections sequenced)")
    ax.set_xlabel("simulation day (tick)")
    ax.set_title("Ascertainment rate over time")
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    path = ctx.path_for(INDEX, NAME, "png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {path.name}")
