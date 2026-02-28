#!/bin/bash
#SBATCH --job-name=rfdiff_refine
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --output=logs/rfdiff_refine_%A_%a.out
#SBATCH --error=logs/rfdiff_refine_%A_%a.err

# RFdiffusion refinement: partial diffusion from a SPECIFIC backbone PDB.
# Used to explore local backbone variants around a winning design.
#
# Usage:
#   sbatch --array=0-49 scripts/run_rfdiff_refine.sh <input_pdb> <partial_T>
#
# Env vars (set by launcher):
#   RFDIFF_OUTPUT_DIR  - output directory for PDB files

set -euo pipefail

export PGDH_ROOT="/cluster/home/csageder/gigabase-hack-berlin"
source "${PGDH_ROOT}/scripts/setup_env.sh"

INPUT_PDB="${1:?Usage: $0 <input_pdb> <partial_T>}"
PARTIAL_T="${2:-5}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

module load eth_proxy stack/2024-06 gcc/12.2.0 cuda/12.8.0

set +u
source "${STRUCT_CONDA_BASE}/bin/activate" "${RFDIFF_CONDA_ENV}"
set -u

export PYTHONPATH="${RFDIFF_DIR}:${PYTHONPATH:-}"

RFDIFF_PARAMS="${PGDH_ROOT}/targets/rfdiff_params.sh"
if [ -f "${RFDIFF_PARAMS}" ]; then
    source "${RFDIFF_PARAMS}"
else
    echo "ERROR: ${RFDIFF_PARAMS} not found."
    exit 1
fi

OUTPUT_PREFIX="${RFDIFF_OUTPUT_DIR:?RFDIFF_OUTPUT_DIR must be set}"
mkdir -p "${OUTPUT_PREFIX}"

echo "=== RFdiffusion Refinement ==="
echo "Input PDB:  ${INPUT_PDB}"
echo "Partial T:  ${PARTIAL_T}"
echo "Task ID:    ${TASK_ID}"
echo "Output:     ${OUTPUT_PREFIX}"

if [ ! -f "${INPUT_PDB}" ]; then
    echo "ERROR: Input PDB not found: ${INPUT_PDB}"
    exit 1
fi

cd "${RFDIFF_DIR}"

PROVIDE_SEQ_ARG=""
if [ -n "${PROVIDE_SEQ:-}" ]; then
    PROVIDE_SEQ_ARG="contigmap.provide_seq=${PROVIDE_SEQ}"
fi

python scripts/run_inference.py \
    inference.output_prefix="${OUTPUT_PREFIX}/design_${TASK_ID}" \
    inference.input_pdb="${INPUT_PDB}" \
    inference.ckpt_override_path="${RFDIFF_DIR}/models/Complex_base_ckpt.pt" \
    "contigmap.contigs=${CONTIGS}" \
    "contigmap.inpaint_str=${INPAINT_STR}" \
    "ppi.hotspot_res=${HOTSPOTS}" \
    diffuser.partial_T="${PARTIAL_T}" \
    inference.num_designs=1 \
    ${PROVIDE_SEQ_ARG:+"${PROVIDE_SEQ_ARG}"}

echo "=== RFdiffusion refinement task ${TASK_ID} complete ==="
