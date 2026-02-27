#!/bin/bash
#SBATCH --job-name=ipsae_full
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=a100:1

source "${HOME}/gigabase-hack-berlin/bash_scripts/setup_env.sh"

RUN_NAME="${1:-nanobody_2gdz_v1}"
MODE="${2:-msa_template}"
TARGET_CIF="${3:-${GIGABASE_PROJECT}/2GDZ-dimer.cif}"
TOP_N="${4:-10}"

RUN_DIR="${BOLTZGEN_RUNS}/${RUN_NAME}"

echo "=== Full ipSAE Pipeline ==="
echo "Run: ${RUN_NAME}"
echo "Mode: ${MODE}"
echo "Target: ${TARGET_CIF}"
echo "Top N designs: ${TOP_N}"
echo "Output: ${RUN_DIR}/ipsae_analysis_${MODE}/"
echo "==========================="

python "${HOME}/gigabase-hack-berlin/scripts/run_ipsae_full.py" \
    --run-dir "${RUN_DIR}" \
    --target-cif "${TARGET_CIF}" \
    --ipsae-script "${IPSAE_SCRIPT}" \
    --cache "${BOLTZGEN_CACHE}" \
    --top-n "${TOP_N}" \
    --mode "${MODE}"
