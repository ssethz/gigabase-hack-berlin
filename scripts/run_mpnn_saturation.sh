#!/bin/bash
#SBATCH --job-name=mpnn_sat_d23
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --output=logs/mpnn_sat_%j.out
#SBATCH --error=logs/mpnn_sat_%j.err

# MPNN sequence saturation on a single backbone (design_23).
# Generates many sequences at multiple temperatures for maximum diversity.

set -euo pipefail

export PGDH_ROOT="/cluster/home/csageder/gigabase-hack-berlin"
source "${PGDH_ROOT}/scripts/setup_env.sh"

BACKBONE_MPNN_DIR="${1:?Usage: $0 <mpnn_input_dir> [num_seqs] [temperature]}"
NUM_SEQS="${2:-100}"
TEMPERATURE="${3:-0.1}"

module load eth_proxy stack/2024-06 gcc/12.2.0 cuda/12.8.0

set +u
source "${STRUCT_CONDA_BASE}/bin/activate" "${STRUCT_CONDA_ENV}"
set -u

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${PGDH_SCRATCH}/mpnn_saturation/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}/sequences"

echo "=== MPNN Sequence Saturation ==="
echo "Input MPNN dir: ${BACKBONE_MPNN_DIR}"
echo "Num sequences: ${NUM_SEQS}"
echo "Temperature: ${TEMPERATURE}"
echo "Output: ${OUTPUT_DIR}"

cp "${BACKBONE_MPNN_DIR}/chains.jsonl" "${OUTPUT_DIR}/"
cp "${BACKBONE_MPNN_DIR}/chain_ids.jsonl" "${OUTPUT_DIR}/"
cp "${BACKBONE_MPNN_DIR}/fixed_positions.jsonl" "${OUTPUT_DIR}/"

cd "${MPNN_DIR}"
python protein_mpnn_run.py \
    --jsonl_path "${OUTPUT_DIR}/chains.jsonl" \
    --chain_id_jsonl "${OUTPUT_DIR}/chain_ids.jsonl" \
    --fixed_positions_jsonl "${OUTPUT_DIR}/fixed_positions.jsonl" \
    --out_folder "${OUTPUT_DIR}/sequences" \
    --num_seq_per_target "${NUM_SEQS}" \
    --sampling_temp "${TEMPERATURE}" \
    --batch_size 1 \
    --use_soluble_model

echo "=== MPNN saturation complete ==="
echo "Output: ${OUTPUT_DIR}/sequences"
N_SEQS=$(grep -c "^>" "${OUTPUT_DIR}/sequences/seqs/"*.fa 2>/dev/null || echo 0)
echo "${N_SEQS} sequences generated"
echo "${OUTPUT_DIR}" > "${PGDH_SCRATCH}/mpnn_saturation_outdir.txt"
