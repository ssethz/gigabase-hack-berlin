#!/bin/bash
#SBATCH --job-name=mpnn_15pgdh
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --output=logs/mpnn_%A_%a.out
#SBATCH --error=logs/mpnn_%A_%a.err

# Run ProteinMPNN sequence design on RFdiffusion backbones.
#
# Supports SLURM job arrays: submit with --array=0-(N-1) to parallelize
# across N GPUs, each processing 1 backbone PDB.
#
# Usage:
#   sbatch --array=0-9 scripts/run_proteinmpnn.sh <backbone_dir> [seqs_per_target] [temperature]
#   sbatch scripts/run_proteinmpnn.sh <backbone_dir> [seqs_per_target] [temperature]

set -euo pipefail

export PGDH_ROOT="/cluster/home/csageder/gigabase-hack-berlin"
source "${PGDH_ROOT}/scripts/setup_env.sh"

BACKBONE_DIR="${1:?Usage: $0 <backbone_dir> [seqs_per_target] [temperature]}"
SEQS_PER_TARGET="${2:-8}"
TEMPERATURE="${3:-0.2}"

module load eth_proxy
module load stack/2024-06 gcc/12.2.0
module load cuda/12.8.0

set +u
source "${STRUCT_CONDA_BASE}/bin/activate" "${STRUCT_CONDA_ENV}"
set -u

# Use MPNN_OUTPUT_DIR env var if set by pipeline, otherwise generate timestamp-based path
OUTPUT_DIR="${MPNN_OUTPUT_DIR:-${PGDH_SCRATCH}/mpnn/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"

echo "=== ProteinMPNN Sequence Design ==="
echo "Backbones: ${BACKBONE_DIR}"
echo "Seqs per target: ${SEQS_PER_TARGET}"
echo "Temperature: ${TEMPERATURE}"
echo "Output: ${OUTPUT_DIR}"
echo "Array task ID: ${TASK_ID:-none (all backbones)}"

if [ ! -d "${BACKBONE_DIR}" ]; then
    echo "ERROR: Backbone directory not found: ${BACKBONE_DIR}"
    exit 1
fi

PDB_FILES=($(find "${BACKBONE_DIR}" -maxdepth 1 \( -name "*.pdb" -o -name "*.cif" \) | sort))
N_BACKBONES=${#PDB_FILES[@]}
if [ "${N_BACKBONES}" -eq 0 ]; then
    echo "ERROR: No backbone PDB/CIF files found in ${BACKBONE_DIR}"
    exit 1
fi

echo "=== GPU + dependency preflight ==="
nvidia-smi -L
python - <<'PY'
import sys
from pathlib import Path
import torch

print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if (torch.version.cuda is None) or (not torch.cuda.is_available()):
    print("ERROR: CUDA-enabled PyTorch not available in this job environment.", file=sys.stderr)
    sys.exit(1)
print(f"gpu_count={torch.cuda.device_count()}")
print(f"gpu_name={torch.cuda.get_device_name(0)}")

mpnn_dir = Path("/cluster/home/csageder/ProteinMPNN")
required = [
    mpnn_dir / "protein_mpnn_run.py",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    print("ERROR: Missing required ProteinMPNN files:", file=sys.stderr)
    for p in missing:
        print(f"  - {p}", file=sys.stderr)
    sys.exit(1)
PY

# ── Array mode: create a temp dir with just the assigned backbone ──
if [ -n "${TASK_ID}" ]; then
    if [ "${TASK_ID}" -ge "${N_BACKBONES}" ]; then
        echo "SKIP: Task ID ${TASK_ID} >= total backbones ${N_BACKBONES}"
        exit 0
    fi

    SINGLE_FILE="${PDB_FILES[${TASK_ID}]}"
    FILE_EXT="${SINGLE_FILE##*.}"
    FILE_NAME=$(basename "${SINGLE_FILE}" ".${FILE_EXT}")
    TASK_BACKBONE_DIR="${OUTPUT_DIR}/backbones_task${TASK_ID}"
    TASK_OUTPUT_DIR="${OUTPUT_DIR}/task_${TASK_ID}_${FILE_NAME}"
    mkdir -p "${TASK_BACKBONE_DIR}" "${TASK_OUTPUT_DIR}"
    cp "${SINGLE_FILE}" "${TASK_BACKBONE_DIR}/"

    echo "--- [task ${TASK_ID}/${N_BACKBONES}] ${FILE_NAME} (${FILE_EXT}) ---"

    # CIF files with is_motif_atom_with_fixed_seq handle their own fixed positions;
    # PDB files from the DARPin pipeline use --fix_binder_helices
    MPNN_PREP_ARGS="--backbone_dir ${TASK_BACKBONE_DIR} --output_dir ${TASK_OUTPUT_DIR}"
    if [ "${FILE_EXT}" = "pdb" ]; then
        MPNN_PREP_ARGS="${MPNN_PREP_ARGS} --fix_binder_helices"
    fi
    python "${PGDH_ROOT}/pipeline/prepare_mpnn_inputs.py" ${MPNN_PREP_ARGS}

    cd "${MPNN_DIR}"
    python protein_mpnn_run.py \
        --jsonl_path "${TASK_OUTPUT_DIR}/chains.jsonl" \
        --chain_id_jsonl "${TASK_OUTPUT_DIR}/chain_ids.jsonl" \
        --fixed_positions_jsonl "${TASK_OUTPUT_DIR}/fixed_positions.jsonl" \
        --out_folder "${TASK_OUTPUT_DIR}/sequences" \
        --num_seq_per_target "${SEQS_PER_TARGET}" \
        --sampling_temp "${TEMPERATURE}" \
        --batch_size 1 \
        --use_soluble_model

    rm -rf "${TASK_BACKBONE_DIR}"

    echo "=== ProteinMPNN task ${TASK_ID} complete: ${FILE_NAME} ==="
    exit 0
fi

# ── Sequential fallback (no array): process all backbones ──
python "${PGDH_ROOT}/pipeline/prepare_mpnn_inputs.py" \
    --backbone_dir "${BACKBONE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --fix_binder_helices

for f in chains.jsonl chain_ids.jsonl fixed_positions.jsonl; do
    if [ ! -s "${OUTPUT_DIR}/${f}" ]; then
        echo "ERROR: Missing or empty MPNN input file: ${OUTPUT_DIR}/${f}"
        exit 1
    fi
done

if [ "${PRECHECK_ONLY:-0}" = "1" ]; then
    echo "PRECHECK_ONLY=1 set; ProteinMPNN preflight passed."
    exit 0
fi

cd "${MPNN_DIR}"
python protein_mpnn_run.py \
    --jsonl_path "${OUTPUT_DIR}/chains.jsonl" \
    --chain_id_jsonl "${OUTPUT_DIR}/chain_ids.jsonl" \
    --fixed_positions_jsonl "${OUTPUT_DIR}/fixed_positions.jsonl" \
    --out_folder "${OUTPUT_DIR}/sequences" \
    --num_seq_per_target "${SEQS_PER_TARGET}" \
    --sampling_temp "${TEMPERATURE}" \
    --batch_size 1 \
    --use_soluble_model

echo "=== ProteinMPNN complete ==="
echo "Output: ${OUTPUT_DIR}/sequences"
N_FASTA=$(find "${OUTPUT_DIR}/sequences" -name "*.fa" | wc -l)
echo "${N_FASTA} FASTA files generated"
if [ "${N_FASTA}" -eq 0 ]; then
    echo "ERROR: ProteinMPNN produced zero FASTA files."
    exit 1
fi
