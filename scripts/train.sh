set -euo pipefail

# ── Data ────────────────────────────────────────────────────────
TRAIN_GLOB="/inspire/hdd/global_user/gexinmu-253108100065/wl/unrolling_deblur/datasets/DIV2K_train_256_random_5/*.png"
TEST_GLOB="/inspire/hdd/global_user/gexinmu-253108100065/wl/unrolling_deblur/datasets/DIV2K_valid_256_random_5/*.png"
VAL_RATIO=0.15
PAD_BORDER=32

# blur generation
SIGMA_LIST="4"
NOISE_PROB=1.0
NOISE_SIGMA_MIN=0.039
NOISE_SIGMA_MAX=0.04

# ── Model ───────────────────────────────────────────────────────
TS=(5)                          # unrolling stages
SOLVERS=("hqs")                  # hqs | admm | pg
SCHEDULES=("uniform")            # uniform | trainable
DENOISERS=("dncnn")              # dncnn | unet_small | resblock
SHARE_DENOISERS=false
INNER_ITERS=(1)

# denoiser architecture
MID_CHANNELS=64               # DnCNN / ResBlock hidden channels
DEPTH=15                       # DnCNN conv layers
BASE_CH=32                    # SmallUNet base channels
NUM_LEVELS=2                  # SmallUNet downsampling levels
NUM_BLOCKS=5                  # ResBlock count

# schedule & loss
LEARNABLE_SCHEDULES=(false)
LEARNABLE_LOSS_WEIGHTS=(false)

# ── Training ────────────────────────────────────────────────────
EPOCHS=200
BATCH_SIZE=24
LR=1e-4
WEIGHT_DECAY=0.05
SCHEDULER="cosine"            # cosine | step
STEP_SIZE=50                  # for step scheduler
GAMMA=0.5                     # for step scheduler
GRAD_CLIP=1.0
SEED=42
NUM_WORKERS=12
LOG_EVERY=20
VAL_EVERY=1
EARLY_STOP_PATIENCE=20
RUN_TEST_AFTER=true
GPUS="0,1,2,3"
LOSS_MODES=("all")              # last | all | one_stage
# ── Testing ─────────────────────────────────────────────────────
TEST_BATCH_SIZE=1
SAVE_IMAGES=true
NUM_VIS_STAGES=5

for T in ${TS[@]}; do
for SOLVER in ${SOLVERS[@]}; do
for SCHEDULE in ${SCHEDULES[@]}; do
for DENOISER in ${DENOISERS[@]}; do
for INNER_ITER in ${INNER_ITERS[@]}; do
for LEARNABLE_LOSS_WEIGHT in ${LEARNABLE_LOSS_WEIGHTS[@]}; do
for LEARNABLE_SCHEDULE in ${LEARNABLE_SCHEDULES[@]}; do
for LOSS_MODE in ${LOSS_MODES[@]}; do
python train.py  \
    --data.train_glob "${TRAIN_GLOB}" \
    --data.test_glob "${TEST_GLOB}" \
    --data.val_ratio ${VAL_RATIO} \
    --data.pad_border ${PAD_BORDER} \
    --data.blur.sigma_list "${SIGMA_LIST}" \
    --data.blur.noise_prob ${NOISE_PROB} \
    --data.blur.noise_sigma_min ${NOISE_SIGMA_MIN} \
    --data.blur.noise_sigma_max ${NOISE_SIGMA_MAX} \
    --model.T ${T} \
    --model.solver ${SOLVER} \
    --model.schedule ${SCHEDULE} \
    --model.denoiser ${DENOISER} \
    --model.share_denoisers ${SHARE_DENOISERS} \
    --model.inner_iters ${INNER_ITER} \
    --model.denoiser_kwargs.mid_channels ${MID_CHANNELS} \
    --model.denoiser_kwargs.depth ${DEPTH} \
    --model.denoiser_kwargs.base_ch ${BASE_CH} \
    --model.denoiser_kwargs.num_levels ${NUM_LEVELS} \
    --model.denoiser_kwargs.num_blocks ${NUM_BLOCKS} \
    --model.learnable_schedule ${LEARNABLE_SCHEDULE} \
    --model.learnable_loss_weights ${LEARNABLE_LOSS_WEIGHT} \
    --train.epochs ${EPOCHS} \
    --train.batch_size ${BATCH_SIZE} \
    --train.lr ${LR} \
    --train.weight_decay ${WEIGHT_DECAY} \
    --train.scheduler ${SCHEDULER} \
    --train.step_size ${STEP_SIZE} \
    --train.gamma ${GAMMA} \
    --train.grad_clip ${GRAD_CLIP} \
    --train.seed ${SEED} \
    --train.num_workers ${NUM_WORKERS} \
    --train.log_every ${LOG_EVERY} \
    --train.val_every ${VAL_EVERY} \
    --train.early_stop_patience ${EARLY_STOP_PATIENCE} \
    --train.run_test_after_train ${RUN_TEST_AFTER} \
    --train.gpus "${GPUS}" \
    --train.loss_mode ${LOSS_MODE} \
    --test.batch_size ${TEST_BATCH_SIZE} \
    --test.save_images ${SAVE_IMAGES} \
    --test.num_vis_stages ${NUM_VIS_STAGES}

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
# bash scripts/collect_results.sh