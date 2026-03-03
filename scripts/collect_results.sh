#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Collect all experiment test results into a summary CSV
#
# Scans results/ for test_results.json files and generates:
#   results/summary.csv    — sorted by test PSNR (descending)
#
# Usage:
#   bash scripts/collect_results.sh
#   bash scripts/collect_results.sh results/       # custom results dir
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

RESULTS_DIR="${1:-results}"
SUMMARY="${RESULTS_DIR}/summary.csv"

echo "Scanning: ${RESULTS_DIR}/"
echo ""

# header
echo "rank,experiment,timestamp,T,solver,denoiser,depth,hidden,inner_iters,schedule,lr,batch_size,epochs_trained,test_psnr,test_ssim" > "$SUMMARY"

# temp file for unsorted rows
TMP=$(mktemp)

for TEST_JSON in $(find "$RESULTS_DIR" -name "test_results.json" -path "*/test/*" 2>/dev/null | sort); do

    # directory structure: results/{dataset}/{param_tag}/{timestamp}/test/test_results.json
    TEST_DIR=$(dirname "$TEST_JSON")
    TS_DIR=$(dirname "$TEST_DIR")
    PARAM_DIR=$(dirname "$TS_DIR")

    TIMESTAMP=$(basename "$TS_DIR")
    PARAM_TAG=$(basename "$PARAM_DIR")

    TRAIN_DIR="${TS_DIR}/train"
    CFG_FILE="${TRAIN_DIR}/config.yaml"
    HIST_FILE="${TRAIN_DIR}/history.json"

    if [ ! -f "$CFG_FILE" ]; then
        echo "  [SKIP] No config: ${CFG_FILE}"
        continue
    fi

    python3 -c "
import yaml, json, sys

with open('${CFG_FILE}') as f:
    cfg = yaml.safe_load(f)
with open('${TEST_JSON}') as f:
    test = json.load(f)

# training epochs
n_epochs = 0
try:
    with open('${HIST_FILE}') as f:
        hist = json.load(f)
    n_epochs = len(hist)
except:
    pass

mc = cfg['model']
dk = mc.get('denoiser_kwargs', {})
tc = cfg['train']
denoiser = mc['denoiser']

if denoiser == 'dncnn':
    depth = dk.get('depth', 8)
    hidden = dk.get('mid_channels', 64)
elif denoiser == 'unet_small':
    depth = dk.get('num_levels', 2)
    hidden = dk.get('base_ch', 32)
elif denoiser == 'resblock':
    depth = dk.get('num_blocks', 5)
    hidden = dk.get('mid_channels', 64)
else:
    depth = 'NA'
    hidden = 'NA'

schedule = 'trainable' if mc.get('learnable_schedule') else mc.get('schedule', 'uniform')

print(
    f'${PARAM_TAG},'
    f'${TIMESTAMP},'
    f'{mc[\"T\"]},'
    f'{mc[\"solver\"]},'
    f'{denoiser},'
    f'{depth},'
    f'{hidden},'
    f'{mc.get(\"inner_iters\", 1)},'
    f'{schedule},'
    f'{tc[\"lr\"]},'
    f'{tc[\"batch_size\"]},'
    f'{n_epochs},'
    f'{test[\"avg_psnr\"]:.4f},'
    f'{test[\"avg_ssim\"]:.6f}'
)
" >> "$TMP" 2>/dev/null || echo "  [WARN] Failed to parse: ${TEST_JSON}"

done

# sort by test_psnr (column 13) descending, add rank
RANK=0
sort -t',' -k13 -rn "$TMP" | while IFS= read -r line; do
    RANK=$((RANK + 1))
    echo "${RANK},${line}" >> "$SUMMARY"
done

rm -f "$TMP"

N_RESULTS=$(( $(wc -l < "$SUMMARY") - 1 ))
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Collected ${N_RESULTS} experiment(s)"
echo "  Saved to: ${SUMMARY}"
echo "═══════════════════════════════════════════════════"
echo ""

if [ "$N_RESULTS" -gt 0 ]; then
    echo "Top results by TEST PSNR:"
    echo "─────────────────────────────────────────────────"
    # print header + top 15
    head -1 "$SUMMARY"
    tail -n +2 "$SUMMARY" | head -15
    echo ""
fi