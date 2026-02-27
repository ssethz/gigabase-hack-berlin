#!/bin/bash
# Environment setup for the 15-PGDH interface binder design pipeline.
# Source this from SLURM scripts and interactive sessions.

export PGDH_ROOT="/cluster/home/csageder/gigabase-hack-berlin"
export PGDH_SCRATCH="/cluster/scratch/csageder/pgdh_outputs"
export RFDIFF_DIR="/cluster/home/csageder/RFdiffusion"
export MPNN_DIR="/cluster/home/csageder/ProteinMPNN"
export IPSAE_SCRIPT="/cluster/home/csageder/IPSAE/ipsae.py"
export STRUCT_SANDBOX="/cluster/home/csageder/structure_sandbox"

export STRUCT_CONDA_BASE="/cluster/home/csageder/miniforge3"
export STRUCT_CONDA_ENV="structure"
export RFDIFF_CONDA_ENV="SE3nv"

export BOLTZ_CACHE="/cluster/project/krause/csageder/struct/boltz_cache"

mkdir -p "${PGDH_SCRATCH}"
mkdir -p "${PGDH_ROOT}/outputs"
mkdir -p "${PGDH_ROOT}/logs"
