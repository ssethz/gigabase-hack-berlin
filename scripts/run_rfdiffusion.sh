#!/bin/bash
#SBATCH --job-name=rfdiff_15pgdh
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --output=logs/rfdiff_%j.out
#SBATCH --error=logs/rfdiff_%j.err

# Run RFdiffusion DARPin partial diffusion for 15-PGDH interface binder design.
# Usage: sbatch scripts/run_rfdiffusion.sh [n_designs] [partial_T]
#
# Scaffold: 2XEH NI3C Mut6 DARPin (157 residues, chain B)
# Target: 15-PGDH chain A, residues 0-265 (266 residues)
# Strategy: Fix target + DARPin helices, diffuse only binding loops
# Hotspots: alpha-9 helix region (A143, A144, A165)

set -euo pipefail

export PGDH_ROOT="/cluster/home/csageder/gigabase-hack-berlin"
source "${PGDH_ROOT}/scripts/setup_env.sh"

N_DESIGNS="${1:-3}"
PARTIAL_T="${2:-10}"

module load eth_proxy
module load stack/2024-06 gcc/12.2.0
module load cuda/12.8.0

# RFdiffusion dependencies (including DGL) are in SE3nv.
set +u
source "${STRUCT_CONDA_BASE}/bin/activate" "${RFDIFF_CONDA_ENV}"
set -u

# Ensure RFdiffusion package is importable when running from source checkout.
export PYTHONPATH="${RFDIFF_DIR}:${PYTHONPATH:-}"

echo "=== GPU preflight ==="
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

try:
    import rfdiffusion  # noqa: F401
except Exception as e:
    print(f"ERROR: Cannot import rfdiffusion in current env: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import dgl  # noqa: F401
except Exception as e:
    print(f"ERROR: Cannot import dgl in current env: {e}", file=sys.stderr)
    sys.exit(1)

rfdiff_dir = Path("/cluster/home/csageder/RFdiffusion")
required = [
    rfdiff_dir / "scripts" / "run_inference.py",
    rfdiff_dir / "models" / "Complex_base_ckpt.pt",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    print("ERROR: Missing required RFdiffusion files:", file=sys.stderr)
    for p in missing:
        print(f"  - {p}", file=sys.stderr)
    sys.exit(1)
PY

cd "${RFDIFF_DIR}"

DARPIN_COMPLEX="${PGDH_ROOT}/targets/darpin_on_15pgdh.pdb"
HOTSPOTS="[A143,A144,A165]"
INPAINT_STR="[A0-265/B1-13/B14-24/B37-46/B47-57/B70-79/B80-90/B103-112/B113-123/B136-145/B146-157]"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PREFIX="${PGDH_SCRATCH}/rfdiff/darpin_pT${PARTIAL_T}_${TIMESTAMP}"

echo "=== RFdiffusion DARPin partial diffusion (2XEH NI3C Mut6) ==="
echo "Input complex: ${DARPIN_COMPLEX}"
echo "Partial T: ${PARTIAL_T}"
echo "Hotspots: ${HOTSPOTS}"
echo "Inpaint:  fix target + DARPin helices, diffuse loops only"
echo "N designs: ${N_DESIGNS}"

if [ ! -f "${DARPIN_COMPLEX}" ]; then
    echo "ERROR: DARPin complex PDB not found: ${DARPIN_COMPLEX}"
    echo "Run 'bash run_pipeline.sh prepare' first."
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PREFIX}")"
if [ ! -w "$(dirname "${OUTPUT_PREFIX}")" ]; then
    echo "ERROR: Output directory is not writable: $(dirname "${OUTPUT_PREFIX}")"
    exit 1
fi

if [ "${PRECHECK_ONLY:-0}" = "1" ]; then
    echo "PRECHECK_ONLY=1 set; RFdiffusion preflight passed."
    exit 0
fi

python scripts/run_inference.py \
    inference.output_prefix="${OUTPUT_PREFIX}/design" \
    inference.input_pdb="${DARPIN_COMPLEX}" \
    inference.ckpt_override_path="${RFDIFF_DIR}/models/Complex_base_ckpt.pt" \
    "contigmap.contigs=[A0-265/0 B1-157]" \
    "contigmap.inpaint_str=${INPAINT_STR}" \
    "ppi.hotspot_res=${HOTSPOTS}" \
    diffuser.partial_T="${PARTIAL_T}" \
    inference.num_designs="${N_DESIGNS}"

echo "=== RFdiffusion complete ==="
echo "Output: ${OUTPUT_PREFIX}"
ls -la "${OUTPUT_PREFIX}"/*.pdb 2>/dev/null | wc -l
echo "designs generated"
