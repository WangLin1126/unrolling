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
NOISE_SIGMA_MIN=0.1
NOISE_SIGMA_MAX=0.1

# ── Model ───────────────────────────────────────────────────────
# Set to "null" to train from scratch, or provide a path to resume
CHECKPOINT="/inspire/hdd/global_user/gexinmu-253108100065/Repos/waitlist/unrolling_deblur/results/DIV2K/cats_freq-df-geomT10-hqs-dncnn-inner1-blur_4-noise_0.1_0.1-beta_constant-filter_gaussian/20260323_144649/train/last.pth"
TS=(10)
# hqs | admm | pg | ista | fista
SOLVERS=("hqs")
# geom | power | uniform | trainable
SIGMA_SCHEDULES=("uniform")
FRONT_HEAVY=true
# dncnn | unet | resblock | drunet | uformer | restormer
DENOISERS=("dncnn")
SHARE_DENOISERS=false
INNER_ITERS=1

# constant | geom | geom_inc | geom_dec | dpir
BETA_MODES=("constant")
# ("cats_freq" "cats_operator" "cats_residual" "cats_combined")
LOSS_MODES=("cats_freq")
# Difficulty schedule: how d(t) grows from 0 → 1 across stages
DIFFICULTY_SCHEDULE="geom"      # "linear" | "power" | "geom" | "trainable"
DIFFICULTY_GAMMA=2.0             # exponent for "power" schedule (>1 = slow start)
DIFFICULTY_R=0.7                 # ratio for "geom" schedule
# Frequency filter type for cats_freq / cats_combined
FILTER_TYPES=("gaussian")           # "gaussian" | "butterworth" | "ideal"
# Residual weight for cats_combined mode (0 = freq-only)
RESIDUAL_WEIGHT=0.5
# Extra weight on final-stage clean-GT loss (0 = disabled)
LAMBDA_FINAL=0.1

# ── Training ────────────────────────────────────────────────────
EPOCHS=200
BATCH_SIZE_PER_GPU=24
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
LEARNABLE_LOSS_WEIGHTS=false

# ── Testing ─────────────────────────────────────────────────────
TEST_BATCH_SIZE=8
TEST_NUM_WORKERS=8
SAVE_IMAGES=true                 # true to generate CATS analysis figures
NUM_VIS_STAGES=6

# ── Run ─────────────────────────────────────────────────────────
for T in "${TS[@]}"; do
for SOLVER in "${SOLVERS[@]}"; do
for SIGMA_SCHEDULE in "${SIGMA_SCHEDULES[@]}"; do
for DENOISER in "${DENOISERS[@]}"; do
for FILTER_TYPE in "${FILTER_TYPES[@]}"; do
for LEARNABLE_LOSS_WEIGHT in "${LEARNABLE_LOSS_WEIGHTS[@]}"; do
for LOSS_MODE in "${LOSS_MODES[@]}"; do
for BETA_MODE in "${BETA_MODES[@]}"; do
echo "============================================================"
echo "Training CATS with loss_mode=${LOSS_MODE}"
echo "  difficulty_schedule=${DIFFICULTY_SCHEDULE}, gamma=${DIFFICULTY_GAMMA}"
echo "  filter_type=${FILTER_TYPE}, residual_weight=${RESIDUAL_WEIGHT}"
echo "============================================================"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train.py \
    --config configs/default.yaml \
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
    --model.inner_iters "${INNER_ITERS}" \
    --model.learnable_loss_weights "${LEARNABLE_LOSS_WEIGHTS}" \
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
    --train.cts_kwargs.difficulty_schedule "${DIFFICULTY_SCHEDULE}" \
    --train.cts_kwargs.gamma "${DIFFICULTY_GAMMA}" \
    --train.cts_kwargs.r "${DIFFICULTY_R}" \
    --train.cts_kwargs.filter_type "${FILTER_TYPE}" \
    --train.cts_kwargs.residual_weight "${RESIDUAL_WEIGHT}" \
    --train.cts_kwargs.lambda_final "${LAMBDA_FINAL}" \
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
echo "CATS training done."