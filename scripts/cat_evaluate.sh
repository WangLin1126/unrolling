#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

TEST_BATCH_SIZE=8
TEST_NUM_WORKERS=8

CHECKPOINTS=(
)

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "ERROR: No checkpoints specified. Edit CHECKPOINTS array in this script."
    echo ""
    echo "After training with cats_train.sh, find your checkpoints with:"
    echo "  find results/ -name 'best.pth' -path '*cats*'"
    exit 1
fi

# ── Evaluate each checkpoint ────────────────────────────────────
for CKPT in "${CHECKPOINTS[@]}"; do
    if [[ ! -f "${CKPT}" ]]; then
        echo "Skip missing checkpoint: ${CKPT}"
        continue
    fi

    TRAIN_DIR="$(dirname "${CKPT}")"
    EXP_DIR="$(dirname "${TRAIN_DIR}")"
    TEST_DIR="${EXP_DIR}/test"

    echo "============================================================"
    echo "Evaluating CATS checkpoint"
    echo "  Checkpoint : ${CKPT}"
    echo "  Output dir : ${TEST_DIR}"
    echo "  Batch size : ${TEST_BATCH_SIZE}"
    echo "============================================================"

    # --prefer_ckpt_config ensures the correct loss_mode and cts_kwargs
    # are loaded from the checkpoint, matching the training configuration.
    python evaluate.py \
        --config configs/default.yaml \
        --checkpoint "${CKPT}" \
        --prefer_ckpt_config \
        --test.batch_size "${TEST_BATCH_SIZE}" \
        --test.num_workers "${TEST_NUM_WORKERS}"

    echo ""
    echo "Results saved to: ${TEST_DIR}"
    echo "CATS analysis figures: ${TEST_DIR}/figures/cats_analysis/"
    echo ""

done

echo "============================================================"
echo "All CATS evaluations finished."
echo ""
echo "Key output files per experiment:"
echo "  test/summary.txt                       — PSNR/SSIM summary"
echo "  test/test_results.json                 — machine-readable metrics"
echo "  test/figures/cats_analysis/*_spectral.png      — spectral convergence"
echo "  test/figures/cats_analysis/*_psnr_trajectory.png — stage PSNR curve"
echo "  test/figures/cats_analysis/*_specialization.png  — stage specialization"
echo "============================================================"