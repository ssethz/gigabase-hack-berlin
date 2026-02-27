#!/bin/bash
# Common environment setup - source this from other scripts

if [ -f "${HOME}/structure_sandbox/.env.local" ]; then
    source "${HOME}/structure_sandbox/.env.local"
else
    echo "ERROR: ${HOME}/structure_sandbox/.env.local not found."
    echo "Please copy .env.template to .env.local and customize it:"
    echo "  cp .env.template .env.local"
    exit 1
fi

module load eth_proxy
module load ${STRUCT_STACK_MODULE} ${STRUCT_GCC_MODULE}
module load ${STRUCT_CUDA_MODULE}

source ${STRUCT_CONDA_BASE}/bin/activate ${STRUCT_CONDA_ENV}

cd ${HOME}/gigabase-hack-berlin

# Hugging Face token for model downloads
if [ -f "${HOME}/.huggingface_token" ]; then
    export HF_TOKEN=$(cat "${HOME}/.huggingface_token")
fi
