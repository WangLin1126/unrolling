#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/default.yaml"

export CUDA_VISIBLE_DEVICES=0

TEST_BATCH_SIZE=8
TEST_NUM_WORKERS=8

CHECKPOINTS=(
/inspire/hdd/global_user/gexinmu-253108100065/wl/unrolling_deblur/results/DIV2K/front_light-T_5-solver_hqs-denoiser_dncnn-depth_15-hidden_64-inner_1-sigma_geom-beta_geom-lossw_uniform-lmode_last/20260306_203636/train/best.pth
)

for CKPT in "${CHECKPOINTS[@]}"; do
    if [[ ! -f "${CKPT}" ]]; then
        echo "Skip missing checkpoint: ${CKPT}"
        continue
    fi

    TRAIN_DIR="$(dirname "${CKPT}")"
    EXP_DIR="$(dirname "${TRAIN_DIR}")"
    TEST_DIR="${EXP_DIR}/test"

    echo "=================================================="
    echo "Testing checkpoint: ${CKPT}"
    echo "Output dir        : ${TEST_DIR}"
    echo "Batch size        : ${TEST_BATCH_SIZE}"
    echo "=================================================="

    python evaluate.py \
        --config "${CONFIG}" \
        --checkpoint "${CKPT}" \
        --prefer_ckpt_config \
        --test.batch_size "${TEST_BATCH_SIZE}" \
        --test.num_workers "${TEST_NUM_WORKERS}"

done

echo ""
echo "All requested tests finished."