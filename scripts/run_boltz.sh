#!/bin/bash
#SBATCH --job-name=boltz_15pgdh
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --output=logs/boltz_%j.out
#SBATCH --error=logs/boltz_%j.err

# Run Boltz-2 complex structure prediction on designed binders.
# Usage: sbatch scripts/run_boltz.sh <yaml_dir> [diffusion_samples]

set -euo pipefail

export PGDH_ROOT="/cluster/home/csageder/gigabase-hack-berlin"
source "${PGDH_ROOT}/scripts/setup_env.sh"

YAML_DIR="${1:?Usage: $0 <yaml_dir> [diffusion_samples]}"
DIFFUSION_SAMPLES="${2:-5}"

module load eth_proxy
module load stack/2024-06 gcc/12.2.0
module load cuda/12.8.0

set +u
source "${STRUCT_CONDA_BASE}/bin/activate" "${STRUCT_CONDA_ENV}"
set -u

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${PGDH_SCRATCH}/boltz/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

export NVIDIA_TF32_OVERRIDE=1

echo "=== Boltz-2 Complex Prediction ==="
echo "YAML dir: ${YAML_DIR}"
echo "Diffusion samples: ${DIFFUSION_SAMPLES}"
echo "Output: ${OUTPUT_DIR}"

if [ ! -d "${YAML_DIR}" ]; then
    echo "ERROR: YAML directory not found: ${YAML_DIR}"
    exit 1
fi

echo "=== GPU + dependency preflight ==="
nvidia-smi -L
if ! command -v boltz >/dev/null 2>&1; then
    echo "ERROR: boltz command not found in current environment."
    exit 1
fi
python - <<'PY'
import sys
import torch
print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if (torch.version.cuda is None) or (not torch.cuda.is_available()):
    print("ERROR: CUDA-enabled PyTorch not available in this job environment.", file=sys.stderr)
    sys.exit(1)
print(f"gpu_count={torch.cuda.device_count()}")
print(f"gpu_name={torch.cuda.get_device_name(0)}")
PY

YAML_FILES=($(find "${YAML_DIR}" -name "*.yaml" | sort))
TOTAL=${#YAML_FILES[@]}
echo "Processing ${TOTAL} YAML files..."
if [ "${TOTAL}" -eq 0 ]; then
    echo "ERROR: No YAML files found in ${YAML_DIR}"
    exit 1
fi

if [ "${PRECHECK_ONLY:-0}" = "1" ]; then
    echo "PRECHECK_ONLY=1 set; Boltz preflight passed."
    exit 0
fi

FAILED=0

for i in "${!YAML_FILES[@]}"; do
    YAML="${YAML_FILES[$i]}"
    NAME=$(basename "${YAML}" .yaml)
    echo ""
    echo "--- [${i}/${TOTAL}] ${NAME} ---"

    boltz predict "${YAML}" \
        --out_dir "${OUTPUT_DIR}/${NAME}" \
        --diffusion_samples "${DIFFUSION_SAMPLES}" \
        --max_parallel_samples 1 \
        --output_format pdb \
        --write_full_pae \
        --override || {
            echo "FAILED: ${NAME}"
            FAILED=$((FAILED + 1))
        }
done

echo ""
echo "=== Boltz-2 complete ==="
echo "Output: ${OUTPUT_DIR}"
N_CONF=$(find "${OUTPUT_DIR}" -name "confidence_*.json" | wc -l)
echo "${N_CONF}"
echo "predictions generated"
if [ "${N_CONF}" -eq 0 ]; then
    echo "ERROR: Boltz produced zero confidence JSON files."
    exit 1
fi
if [ "${FAILED}" -gt 0 ]; then
    echo "ERROR: ${FAILED} Boltz predictions failed."
    exit 1
fi
