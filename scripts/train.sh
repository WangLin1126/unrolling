#!/usr/bin/env bash
set -euo pipefail

# ── Runtime / DDP ───────────────────────────────────────────────
GPUS="0,1,2,3"
IFS=',' read -ra GPU_ARR <<< "${GPUS}"
NPROC_PER_NODE="${#GPU_ARR[@]}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

# ── Data ────────────────────────────────────────────────────────
TRAIN_GLOB="/inspire/hdd/global_user/gexinmu-253108100065/Repos/waitlist/unrolling_deblur/datasets/DIV2K_train_256_random_5/*.png"
TEST_GLOB="/inspire/hdd/global_user/gexinmu-253108100065/Repos/waitlist/unrolling_deblur/datasets/DIV2K_valid_256_random_5/*.png"
VAL_RATIO=0.15
PAD_BORDER=32

# blur generation
SIGMA_LIST="4"
NOISE_PROB=1.0
NOISE_SIGMA_MIN=0.05
NOISE_SIGMA_MAX=0.2

# ── Model ───────────────────────────────────────────────────────
CHECKPOINT=None
TS=(10)
# hqs | admm | pg
SOLVERS=("hqs")
# geom | power | uniform | trainable
SIGMA_SCHEDULES=("uniform")
FRONT_HEAVY=true
# dncnn | unet | resblock | drunet
DENOISERS=("drunet")
SHARE_DENOISERS=false
INNER_ITERS=(1)

# denoiser architecture
# DnCNN
MID_CHANNELS=32
DEPTH=15
# Unet
BASE_CH=32
NUM_LEVELS=2
NUM_BLOCKS=5

# schedule & loss
LEARNABLE_LOSS_WEIGHTS=(false)
# all: gradual change | last: all compare last stage | one_stage: only compute last stage loss
LOSS_MODES=("last")
# constant | geom | geom_inc | geom_dec | dpir
BETA_MODES=("dpir")

# ── Training ────────────────────────────────────────────────────
EPOCHS=200
BATCH_SIZE_PER_GPU=56
LR=2e-4
WEIGHT_DECAY=0.05
SCHEDULER="cosine"
STEP_SIZE=50
GAMMA=0.5
GRAD_CLIP=1.0
SEED=42
NUM_WORKERS=12
LOG_EVERY=10
VAL_EVERY=1
EARLY_STOP_PATIENCE=20
RUN_TEST_AFTER=true
USE_COMPILE=false

# ── Testing ─────────────────────────────────────────────────────
TEST_BATCH_SIZE=8
TEST_NUM_WORKERS=8
SAVE_IMAGES=false
NUM_VIS_STAGES=6

for T in "${TS[@]}"; do
for SOLVER in "${SOLVERS[@]}"; do
for SIGMA_SCHEDULE in "${SIGMA_SCHEDULES[@]}"; do
for DENOISER in "${DENOISERS[@]}"; do
for INNER_ITER in "${INNER_ITERS[@]}"; do
for LEARNABLE_LOSS_WEIGHT in "${LEARNABLE_LOSS_WEIGHTS[@]}"; do
for LOSS_MODE in "${LOSS_MODES[@]}"; do
for BETA_MODE in "${BETA_MODES[@]}"; do
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train.py \
    --data.train_glob "${TRAIN_GLOB}" \
    --data.test_glob "${TEST_GLOB}" \
    --data.val_ratio "${VAL_RATIO}" \
    --data.pad_border "${PAD_BORDER}" \
    --data.blur.sigma_list "${SIGMA_LIST}" \
    --data.blur.noise_prob "${NOISE_PROB}" \
    --data.blur.noise_sigma_min "${NOISE_SIGMA_MIN}" \
    --data.blur.noise_sigma_max "${NOISE_SIGMA_MAX}" \
    --model.T "${T}" \
    --model.solver "${SOLVER}" \
    --model.blur_sigma_schedule "${SIGMA_SCHEDULE}" \
    --model.denoiser "${DENOISER}" \
    --model.share_denoisers "${SHARE_DENOISERS}" \
    --model.inner_iters "${INNER_ITER}" \
    --model.learnable_loss_weights "${LEARNABLE_LOSS_WEIGHT}" \
    --model.blur_sigma_schedule_kwargs.front_heavy "${FRONT_HEAVY}" \
    --model.beta_schedule "${BETA_MODE}" \
    --model.checkpoint "${CHECKPOINT}" \
    --train.epochs "${EPOCHS}" \
    --train.batch_size "${BATCH_SIZE_PER_GPU}" \
    --train.lr "${LR}" \
    --train.weight_decay "${WEIGHT_DECAY}" \
    --train.scheduler "${SCHEDULER}" \
    --train.step_size "${STEP_SIZE}" \
    --train.gamma "${GAMMA}" \
    --train.grad_clip "${GRAD_CLIP}" \
    --train.seed "${SEED}" \
    --train.num_workers "${NUM_WORKERS}" \
    --train.log_every "${LOG_EVERY}" \
    --train.val_every "${VAL_EVERY}" \
    --train.early_stop_patience "${EARLY_STOP_PATIENCE}" \
    --train.run_test_after_train "${RUN_TEST_AFTER}" \
    --train.loss_mode "${LOSS_MODE}" \
    --train.use_compile "${USE_COMPILE}" \
    --test.batch_size "${TEST_BATCH_SIZE}" \
    --test.num_workers "${TEST_NUM_WORKERS}" \
    --test.save_images "${SAVE_IMAGES}" \
    --test.num_vis_stages "${NUM_VIS_STAGES}"

done
done
done
done
done
done
done
done


echo ""
echo "Done."