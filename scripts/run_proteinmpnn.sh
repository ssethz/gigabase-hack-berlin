#!/bin/bash
#SBATCH --job-name=mpnn_15pgdh
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --output=logs/mpnn_%j.out
#SBATCH --error=logs/mpnn_%j.err

# Run ProteinMPNN sequence design on RFdiffusion backbones.
# Usage: sbatch scripts/run_proteinmpnn.sh <backbone_dir> [seqs_per_target] [temperature]

set -euo pipefail

export PGDH_ROOT="/cluster/home/csageder/gigabase-hack-berlin"
source "${PGDH_ROOT}/scripts/setup_env.sh"

BACKBONE_DIR="${1:?Usage: $0 <backbone_dir> [seqs_per_target] [temperature]}"
SEQS_PER_TARGET="${2:-8}"
TEMPERATURE="${3:-0.1}"

module load eth_proxy
module load stack/2024-06 gcc/12.2.0
module load cuda/12.8.0

set +u
source "${STRUCT_CONDA_BASE}/bin/activate" "${STRUCT_CONDA_ENV}"
set -u

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${PGDH_SCRATCH}/mpnn/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo "=== ProteinMPNN Sequence Design ==="
echo "Backbones: ${BACKBONE_DIR}"
echo "Seqs per target: ${SEQS_PER_TARGET}"
echo "Temperature: ${TEMPERATURE}"
echo "Output: ${OUTPUT_DIR}"

if [ ! -d "${BACKBONE_DIR}" ]; then
    echo "ERROR: Backbone directory not found: ${BACKBONE_DIR}"
    exit 1
fi

N_BACKBONES=$(find "${BACKBONE_DIR}" -maxdepth 1 -name "*.pdb" | wc -l)
if [ "${N_BACKBONES}" -eq 0 ]; then
    echo "ERROR: No backbone PDB files found in ${BACKBONE_DIR}"
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

# Step 1: Parse PDB files to create MPNN input JSONLs
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

# Step 2: Run ProteinMPNN
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
echo "${N_FASTA}"
echo "FASTA files generated"
if [ "${N_FASTA}" -eq 0 ]; then
    echo "ERROR: ProteinMPNN produced zero FASTA files."
    exit 1
fi
