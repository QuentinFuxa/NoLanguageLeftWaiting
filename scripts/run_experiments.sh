#!/usr/bin/env bash
# run_experiments.sh -- Automated experiment runner for NLLW SimulMT
#
# Usage:
#   ./scripts/run_experiments.sh [PHASE] [--lang LANG] [--model PATH]
#
# Phases:
#   1  Basic validation (fast, ~5 min)
#   2  Signal sweeps (medium, ~30 min)
#   3  Full Pareto sweep (long, ~2 hours)
#   4  Fusion experiments (medium, ~30 min)
#   5  All combined (everything above)
#
# Requirements:
#   - HYMT_MODEL_PATH or --model must point to a GGUF model
#   - Web server running: python web_debug/server.py (port 8777)
#   - Or use --model flag for direct model loading
#
# Machine mapping:
#   A40:  All experiments (primary)
#   L4_1: Phases 1-2 (cheap experiments)
#   L4_2: Phases 1-2 (parallel, different directions)

set -euo pipefail

# Defaults
PHASE="${1:-1}"
LANG="en-zh"
MODEL="${HYMT_MODEL_PATH:-}"
RESULTS_DIR="results/$(date +%Y%m%d)"
COMET_FLAG=""

# Parse optional args
shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lang) LANG="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --comet) COMET_FLAG="--comet"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

MODEL_FLAG=""
if [[ -n "$MODEL" ]]; then
    MODEL_FLAG="--model $MODEL"
fi

log() {
    echo "[$(date +%H:%M:%S)] $*"
}

run_bench() {
    local desc="$1"
    shift
    log "START: $desc"
    local start=$(date +%s)
    python -m nllw.bench $MODEL_FLAG --lang "$LANG" $COMET_FLAG --save "$@" \
        2>&1 | tee -a "$RESULTS_DIR/log_${LANG}.txt"
    local end=$(date +%s)
    log "DONE: $desc ($(( end - start ))s)"
    echo "---" >> "$RESULTS_DIR/log_${LANG}.txt"
}

# ===== Phase 1: Basic Validation =====
run_phase_1() {
    log "=== PHASE 1: Basic Validation ==="

    # 1a. Baseline (standard AlignAtt, no signals)
    run_bench "baseline (bd=3, wb=3)" -n 20

    # 1b. AlignAtt vs AlignAtt-LA comparison
    run_bench "AlignAtt vs AlignAtt-LA" --compare alignatt alignatt-la

    # 1c. AlignAtt vs baselines
    run_bench "wait-k baseline" --backend wait-k
    run_bench "full-sentence baseline" --backend full-sentence

    log "=== PHASE 1 COMPLETE ==="
}

# ===== Phase 2: Signal Sweeps =====
run_phase_2() {
    log "=== PHASE 2: Signal Sweeps ==="

    # 2a. Core parameter sweep
    run_bench "bd/wb Pareto mini" --sweep "bd=2,3,4 wb=2,3"

    # 2b. Aggregation methods
    run_bench "aggregation sweep" --sweep "agg=ts_vote,softmax_mean,entropy_weighted,consensus,geomean,gaussian_kernel,cumulative"

    # 2c. Shift-k border
    run_bench "shift-k sweep" --sweep "shiftk=0.3,0.4,0.5,0.6"

    # 2d. Info gain
    run_bench "info gain sweep" --sweep "infogain=0.2,0.3,0.5"

    # 2e. LSG logit KL
    run_bench "LSG KL sweep" --sweep "lsg=5.0,7.0,9.0"
    run_bench "LSG K sweep" --sweep "lsg=7.0 lsgk=1,3,5"

    # 2f. Cross-step signals
    run_bench "entropy change sweep" --sweep "entchg=-0.3,-0.5,-1.0"
    run_bench "prediction stability" --prediction-stability
    run_bench "attention shift" --attention-shift

    # 2g. Coverage + monotonicity
    run_bench "coverage sweep" --sweep "cov=0.2,0.3,0.4"
    run_bench "monotonicity" --attention-monotonicity

    # 2h. Repetition detection
    run_bench "repetition sweep" --sweep "rep=2,3,4"

    # 2i. Combined signals
    run_bench "shift-k + info gain" --shift-k 0.4 --info-gain 0.3
    run_bench "coverage + monotonicity" --coverage-threshold 0.3 --attention-monotonicity
    run_bench "entropy + stability" --entropy-change -0.5 --prediction-stability

    log "=== PHASE 2 COMPLETE ==="
}

# ===== Phase 3: Full Pareto Sweep =====
run_phase_3() {
    log "=== PHASE 3: Full Pareto Sweep ==="

    # 3a. Full bd x wb grid
    run_bench "full Pareto" --sweep "bd=2,3,4,5 wb=1,2,3"

    # 3b. SSBD sweep (LA backend)
    run_bench "SSBD sweep" --backend alignatt-la --sweep "ssbd=0.0,0.1,0.2,0.3"

    # 3c. Multi-direction
    for dir in en-zh en-de en-it cs-en; do
        LANG="$dir" run_bench "baseline $dir" -n 20
    done

    # 3d. AMS + temp norm
    run_bench "AMS + temp norm" --sweep "ams=0,1 tempnorm=0,1"

    # 3e. Full signal stack
    run_bench "all signals" \
        --coverage-threshold 0.3 --attention-monotonicity \
        --entropy-change -0.5 --prediction-stability \
        --shift-k 0.4 --lsg-kl 7.0 --repetition-halt 2 \
        --attention-shift

    log "=== PHASE 3 COMPLETE ==="
}

# ===== Phase 4: Fusion Experiments =====
run_phase_4() {
    log "=== PHASE 4: Fusion Experiments ==="

    # 4a. Fusion vs boolean cascade (same signals enabled)
    run_bench "fusion: standard only" --signal-fusion
    run_bench "cascade: standard only" # baseline for comparison

    # 4b. Fusion with all signals
    run_bench "fusion: all signals" \
        --signal-fusion \
        --shift-k 0.4 --coverage-threshold 0.3 \
        --entropy-change -0.5 --prediction-stability \
        --attention-monotonicity --attention-shift

    # 4c. Fusion threshold sweep
    run_bench "fusion threshold sweep" --signal-fusion --sweep "fthr=-0.2,0.0,0.2,0.4"

    # 4d. Fusion + LSG
    run_bench "fusion + LSG" --signal-fusion --lsg-kl 7.0 --shift-k 0.4

    # 4e. Fusion vs cascade head-to-head (identical config)
    run_bench "cascade: shift-k + coverage" \
        --shift-k 0.4 --coverage-threshold 0.3
    run_bench "fusion: shift-k + coverage" \
        --signal-fusion --shift-k 0.4 --coverage-threshold 0.3

    # 4f. Per-direction fusion
    for dir in en-zh en-de en-it cs-en; do
        LANG="$dir" run_bench "fusion $dir" \
            --signal-fusion --shift-k 0.4 --coverage-threshold 0.3 \
            --entropy-change -0.5 --prediction-stability
    done

    log "=== PHASE 4 COMPLETE ==="
}

# ===== Dispatch =====
case "$PHASE" in
    1) run_phase_1 ;;
    2) run_phase_2 ;;
    3) run_phase_3 ;;
    4) run_phase_4 ;;
    5)
        run_phase_1
        run_phase_2
        run_phase_3
        run_phase_4
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Usage: $0 [1|2|3|4|5] [--lang LANG] [--model PATH] [--comet]"
        exit 1
        ;;
esac

log "=== ALL DONE. Results in $RESULTS_DIR ==="
