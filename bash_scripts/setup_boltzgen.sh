#!/bin/bash
#SBATCH --job-name=setup_boltzgen
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_3090:1

source "${HOME}/gigabase-hack-berlin/bash_scripts/setup_env.sh"

pip install boltzgen
