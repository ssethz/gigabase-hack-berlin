#!/bin/bash
#SBATCH --job-name=ipsae_full
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_3090:1
#
# Usage: sbatch run_ipsae_full_3090.sh <RUN_NAME> [OPTIONS]
#
# Options are passed directly to python script:
#   --use-msa                  Use MSA server for target sequence
#   --use-template <CIF>       Use template forcing (requires CIF path)
#   --top-n N                  Number of top designs (default: 10)

source "${HOME}/gigabase-hack-berlin/bash_scripts/setup_env.sh"

RUN_NAME="${1}"
shift

echo "=== Full ipSAE Pipeline (3090) ==="
echo "Run: ${RUN_NAME}"
echo "Args: $@"
echo "==================================="

python "${HOME}/gigabase-hack-berlin/scripts/run_ipsae_full.py" \
    --run-dir "${BOLTZGEN_RUNS}/${RUN_NAME}" \
    --ipsae-script "${IPSAE_SCRIPT}" \
    --cache "${BOLTZGEN_CACHE}" \
    "$@"
