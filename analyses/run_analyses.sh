#!/usr/bin/env bash
#
# run_analyses.sh -- orchestrate the genomic-vs-simulation analysis pipeline.
#
# WHAT THIS DOES
#   Compares an EpiHiper agent-based transmission forest (the ground truth:
#   who infected whom, and when) against a Nextstrain/augur phylogeny built
#   from sequences sampled out of that same simulated epidemic, and writes
#   figures plus machine-readable CSV summaries.
#
#   Each analysis is numbered and named. Outputs are written as
#       <NN>_<semantic_name>.csv   detail tables
#       <NN>_<semantic_name>.png   figures
#       <NN>_<semantic_name>_summary.csv   tidy (metric, value) scalars
#   plus one 00_run_manifest.json per run recording inputs and parameters.
#
#   Every CSV carries a `run_label` column, and the *_summary.csv files share
#   a single tidy schema (run_label, analysis_index, analysis, metric, value).
#   To compare experiments later, concatenate them:
#       cat data/results/*/[0-9]*_summary.csv | ...      # or in pandas:
#       pd.concat(pd.read_csv(p) for p in glob("data/results/*/*_summary.csv"))
#
# THE ANALYSES
#   1  ascertainment_over_time
#        Per-day infections, sequenced infections, and the ascertainment rate.
#        Descriptive: how much of the epidemic surveillance actually saw.
#
#   2  generation_dating_bias
#        Error in the phylogeny's estimate of the time between two infections,
#        binned by how many transmission generations really separate them.
#        Reported in days and as a percent of the true duration, chain-clustered.
#
#   3  cross_chain_linkage
#        Where the tree groups infections from transmission chains that share
#        no transmission edge at all, and how those spurious pairings differ
#        from genuine within-chain pairs in timing, age, and county.
#        (The age/county part is skipped if --metadata is unavailable.)
#
# USAGE
#   ./run_analyses.sh [options]
#
#   -e, --epi-graph PATH     pickled EpiHiper transmission forest
#   -g, --gen-graph PATH     pickled Nextstrain/augur tree
#   -m, --metadata PATH      augur input metadata TSV (tip age/county)
#   -l, --run-label NAME     identifier stamped into every output row
#                            (default: gen-graph filename stem)
#   -o, --out-dir DIR        output directory
#                            (default: data/results/<run-label>)
#   -a, --analyses LIST      'all', or comma-separated indices/names, e.g.
#                            "1,3" or "ascertainment_over_time,3"
#   -p, --max-pairs N        cap on sampled pairs per chain (default 500)
#   -s, --seed N             seed for pair sampling (default 42)
#       --python PATH        python interpreter to use
#       --list               list available analyses and exit
#       --dry-run            print what would run, then exit
#   -h, --help               this message
#
# EXAMPLES
#   ./run_analyses.sh                            # everything, default inputs
#   ./run_analyses.sh --analyses 1               # just ascertainment
#   ./run_analyses.sh --analyses 2,3             # skip the descriptive pass
#   ./run_analyses.sh \
#       --gen-graph data/other_tree.gpickle \
#       --run-label expt_b --analyses all
#
# NOTES
#   * Loading the transmission forest takes several minutes and every analysis
#     needs it, so all selected analyses run in ONE python process and share
#     that load. Prefer "--analyses 1,2,3" over three separate invocations.
#   * A failing analysis is reported but does not abort the others; the exit
#     status is non-zero if any analysis failed.
#   * Console output is tee'd to <out-dir>/run_analyses.log.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- defaults ---------------------------------------------------------------
EPI_GRAPH="${REPO_ROOT}/data/epi_graph.gpickle"
GEN_GRAPH="${REPO_ROOT}/data/nextstrain_tree_graph_300day.gpickle"
METADATA="${REPO_ROOT}/data/nextstrain_tree_sample/run_03_vadelta_2026_03_22_128to428_SURS/metadata_with_index.tsv"
RUN_LABEL=""
OUT_DIR=""
ANALYSES="all"
MAX_PAIRS=500
SEED=42
PYTHON_BIN="${PYTHON_BIN:-/sfs/gpfs/tardis/home/bl4zc/.venv/bin/python}"
DRY_RUN=0
LIST_ONLY=0

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//;$d'; }

# ---- argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--epi-graph)   EPI_GRAPH="$2"; shift 2 ;;
    -g|--gen-graph)   GEN_GRAPH="$2"; shift 2 ;;
    -m|--metadata)    METADATA="$2";  shift 2 ;;
    -l|--run-label)   RUN_LABEL="$2"; shift 2 ;;
    -o|--out-dir)     OUT_DIR="$2";   shift 2 ;;
    -a|--analyses)    ANALYSES="$2";  shift 2 ;;
    -p|--max-pairs)   MAX_PAIRS="$2"; shift 2 ;;
    -s|--seed)        SEED="$2";      shift 2 ;;
    --python)         PYTHON_BIN="$2"; shift 2 ;;
    --list)           LIST_ONLY=1; shift ;;
    --dry-run)        DRY_RUN=1;   shift ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "error: unknown option '$1' (try --help)" >&2; exit 2 ;;
  esac
done

if [[ ! -x "${PYTHON_BIN}" ]] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: python interpreter not found: ${PYTHON_BIN}" >&2
  echo "       pass --python /path/to/python or set PYTHON_BIN" >&2
  exit 2
fi

if [[ ${LIST_ONLY} -eq 1 ]]; then
  exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_analyses.py" --list
fi

# ---- derive defaults that depend on other options ---------------------------
if [[ -z "${RUN_LABEL}" ]]; then
  RUN_LABEL="$(basename "${GEN_GRAPH}")"
  RUN_LABEL="${RUN_LABEL%.gpickle}"
fi
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${REPO_ROOT}/data/results/${RUN_LABEL}"
fi

# ---- validate inputs --------------------------------------------------------
for pair in "epi graph:${EPI_GRAPH}" "gen graph:${GEN_GRAPH}"; do
  label="${pair%%:*}"; path="${pair#*:}"
  if [[ ! -f "${path}" ]]; then
    echo "error: ${label} not found: ${path}" >&2
    exit 2
  fi
done
if [[ ! -f "${METADATA}" ]]; then
  echo "note: metadata not found (${METADATA})"
  echo "      analysis 3 will run without its age/county comparison."
fi

cat <<EOF
============================================================
 genomic-vs-simulation analysis pipeline
============================================================
 run label   : ${RUN_LABEL}
 epi graph   : ${EPI_GRAPH}
 gen graph   : ${GEN_GRAPH}
 metadata    : ${METADATA}
 analyses    : ${ANALYSES}
 max pairs   : ${MAX_PAIRS} per chain (seed ${SEED})
 output dir  : ${OUT_DIR}
 python      : ${PYTHON_BIN}
============================================================
EOF

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "(dry run -- nothing executed)"
  exit 0
fi

mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/run_analyses.log"

set +e
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_analyses.py" \
  --epi-graph "${EPI_GRAPH}" \
  --gen-graph "${GEN_GRAPH}" \
  --metadata  "${METADATA}" \
  --run-label "${RUN_LABEL}" \
  --out-dir   "${OUT_DIR}" \
  --analyses  "${ANALYSES}" \
  --max-pairs-per-chain "${MAX_PAIRS}" \
  --seed      "${SEED}" 2>&1 | tee "${LOG_FILE}"
status="${PIPESTATUS[0]}"
set -e

echo
if [[ "${status}" -eq 0 ]]; then
  echo "SUCCESS -- results in ${OUT_DIR}"
else
  echo "FINISHED WITH ERRORS (exit ${status}) -- see ${LOG_FILE}" >&2
fi
ls -1 "${OUT_DIR}" | sed 's/^/  /'
exit "${status}"
