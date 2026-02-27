#!/bin/bash
#SBATCH --job-name=boltzgen_nanobody
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=a100:1

source "${HOME}/gigabase-hack-berlin/bash_scripts/setup_env.sh"

OUTPUT_DIR="/cluster/project/krause/ssussex/gigabase/boltzgen_output"
YAML_FILE="${HOME}/gigabase-hack-berlin/yaml/nanobody_2gdz.yaml"
CACHE_DIR="/cluster/project/krause/ssussex/gigabase/.boltzgen_cache"

boltzgen run "${YAML_FILE}" \
    --output "${OUTPUT_DIR}" \
    --protocol nanobody-anything \
    --num_designs 50 \
    --budget 10 \
    --cache "${CACHE_DIR}" \
    --diffusion_batch_size 10 \
    --num_workers 8
