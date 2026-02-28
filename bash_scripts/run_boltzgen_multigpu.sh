#!/bin/bash
#SBATCH --job-name=boltzgen
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=a100:4

source "${HOME}/gigabase-hack-berlin/bash_scripts/setup_env.sh"

RUN_NAME="${1:-default_run}"
YAML_FILE="${2:-${HOME}/gigabase-hack-berlin/yaml/nanobody_2gdz.yaml}"
NUM_DESIGNS="${3:-500}"
BUDGET="${4:-50}"
BATCH_SIZE="${5:-10}"
NUM_GPUS="${6:-4}"

OUTPUT_DIR="${BOLTZGEN_RUNS}/${RUN_NAME}"

echo "=== BoltzGen Multi-GPU Run ==="
echo "Run name: ${RUN_NAME}"
echo "YAML: ${YAML_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "Designs: ${NUM_DESIGNS}, Budget: ${BUDGET}, Batch: ${BATCH_SIZE}"
echo "GPUs: ${NUM_GPUS}"
echo "=============================="

mkdir -p "${OUTPUT_DIR}"

boltzgen run "${YAML_FILE}" \
    --output "${OUTPUT_DIR}" \
    --protocol nanobody-anything \
    --num_designs "${NUM_DESIGNS}" \
    --budget "${BUDGET}" \
    --cache "${BOLTZGEN_CACHE}" \
    --diffusion_batch_size "${BATCH_SIZE}" \
    --devices "${NUM_GPUS}" \
    --num_workers 16
