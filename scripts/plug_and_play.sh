#!/usr/bin/env bash
set -euo pipefail

# ── Plug-and-Play evaluation script ─────────────────────────────
# Reads trained PAP model checkpoints and evaluates on test data.
# All denoisers should already be trained; this script loads the
# full chain from a checkpoint and runs inference.

# ── Data ────────────────────────────────────────────────────────
TEST_GLOB="/inspire/hdd/global_user/gexinmu-253108100065/Repos/waitlist/unrolling_deblur/datasets/DIV2K_valid_256_random_5/*.png"

# ── PAP Config ──────────────────────────────────────────────────
PAP_CONFIG="configs/pap_example.yaml"

# ── Checkpoints to evaluate ─────────────────────────────────────
# Add paths to trained PAP checkpoints here
CHECKPOINTS=(
    # "results/DIV2K/pap-.../train/best.pth"
)

# ── Testing options ─────────────────────────────────────────────
TEST_BATCH_SIZE=8
TEST_NUM_WORKERS=8
SAVE_IMAGES=true
NUM_VIS_STAGES=6

# ── Evaluate each checkpoint ────────────────────────────────────
if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "No checkpoints specified. Searching for PAP checkpoints in results/..."
    # Auto-discover PAP checkpoints
    mapfile -t CHECKPOINTS < <(find results/ -path "*/pap-*/train/best.pth" 2>/dev/null | sort)
    if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
        echo "No PAP checkpoints found in results/. Please specify checkpoint paths."
        exit 1
    fi
    echo "Found ${#CHECKPOINTS[@]} PAP checkpoint(s)."
fi

for CKPT in "${CHECKPOINTS[@]}"; do
    echo "================================================================"
    echo "Evaluating: ${CKPT}"
    echo "================================================================"

    python pap/pap_evaluate.py \
        --config "${PAP_CONFIG}" \
        --checkpoint "${CKPT}" \
        --prefer_ckpt_config \
        --test.batch_size "${TEST_BATCH_SIZE}" \
        --test.num_workers "${TEST_NUM_WORKERS}"

    echo ""
done

echo "All PAP evaluations done."
