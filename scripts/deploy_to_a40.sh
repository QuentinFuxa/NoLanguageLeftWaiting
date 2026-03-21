#!/bin/bash
# Deploy NLLW to A40 for GPU experiments
#
# Usage:
#   ./scripts/deploy_to_a40.sh              # Deploy code only
#   ./scripts/deploy_to_a40.sh --run        # Deploy + run iteration 21 experiments
#   ./scripts/deploy_to_a40.sh --quick      # Deploy + run quick experiments
#
# IMPORTANT: Uses rsync WITHOUT --delete to preserve GPU-generated configs
# and model files on the remote machine.

set -e

A40_HOST="fuxa@quest.ms.mff.cuni.cz"
A40_PORT=3622
REMOTE_DIR="nllw_deploy"
MODEL_PATH="/home/fuxa/HY-MT1.5-7B.Q8_0.gguf"

echo "=== Deploying NLLW to A40 ==="
echo "Remote: ${A40_HOST}:${REMOTE_DIR} (port ${A40_PORT})"

# Sync code (without --delete to preserve remote files)
rsync -avz \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '.pytest_cache' \
    --exclude 'results/' \
    --exclude 'models/' \
    --exclude '.claude/' \
    -e "ssh -p ${A40_PORT}" \
    . "${A40_HOST}:${REMOTE_DIR}/"

echo "=== Deploy complete ==="

# Optionally run experiments
if [[ "$1" == "--run" ]]; then
    echo "=== Running iteration 21 experiments (full) ==="
    ssh -p ${A40_PORT} ${A40_HOST} \
        "cd ${REMOTE_DIR} && python scripts/run_iteration21_experiments.py --model ${MODEL_PATH}" 2>&1 | tee results/a40_iteration21.log
elif [[ "$1" == "--quick" ]]; then
    echo "=== Running iteration 21 experiments (quick) ==="
    ssh -p ${A40_PORT} ${A40_HOST} \
        "cd ${REMOTE_DIR} && python scripts/run_iteration21_experiments.py --model ${MODEL_PATH} --quick" 2>&1 | tee results/a40_iteration21_quick.log
elif [[ "$1" == "--phase" ]] && [[ -n "$2" ]]; then
    echo "=== Running phase $2 ==="
    ssh -p ${A40_PORT} ${A40_HOST} \
        "cd ${REMOTE_DIR} && python scripts/run_iteration21_experiments.py --model ${MODEL_PATH} --phase $2" 2>&1 | tee results/a40_phase$2.log
fi
