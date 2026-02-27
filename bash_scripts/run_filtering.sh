#!/bin/bash
#SBATCH --job-name=boltzgen_filter
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

source "${HOME}/gigabase-hack-berlin/bash_scripts/setup_env.sh"

RUN_NAME="${1:-nanobody_2gdz_v1}"
YAML_FILE="${2:-${HOME}/gigabase-hack-berlin/yaml/nanobody_2gdz.yaml}"

OUTPUT_DIR="${BOLTZGEN_RUNS}/${RUN_NAME}"

echo "=== Boltzgen Filtering ==="
echo "Run: ${RUN_NAME}"
echo "YAML: ${YAML_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================="

boltzgen run "${YAML_FILE}" \
    --output "${OUTPUT_DIR}" \
    --steps filtering
