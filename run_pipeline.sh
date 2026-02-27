#!/bin/bash
# =============================================================================
# 15-PGDH Interface Binder Pipeline - Master Orchestrator
# =============================================================================
#
# DARPin-scaffold binder design pipeline targeting the 15-PGDH alpha-9
# dimer interface surface via RFdiffusion partial diffusion.
#
# Usage:
#   bash run_pipeline.sh [step]
#
#   Steps: prepare | rfdiff | mpnn | boltz | score | select | status
#   Default: status (prints pipeline step guide)
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/scripts/setup_env.sh"

STEP="${1:-status}"

echo "============================================================"
echo " 15-PGDH Interface Binder Design Pipeline"
echo "============================================================"
echo " Strategy: DARPin partial diffusion (RFdiffusion)"
echo " Target:   15-PGDH alpha-9 dimer interface surface"
echo " Root:     ${PGDH_ROOT}"
echo " Scratch:  ${PGDH_SCRATCH}"
echo " Step:     ${STEP}"
echo "============================================================"

# ─── Step 1: Target Preparation ──────────────────────────────────────
run_prepare() {
    echo ""
    echo ">>> Step 1: Target Preparation"
    echo "────────────────────────────────────────────────────────"

    module load eth_proxy stack/2024-06 gcc/12.2.0 cuda/12.8.0
    source "${STRUCT_CONDA_BASE}/bin/activate" "${STRUCT_CONDA_ENV}"

    cd "${PGDH_ROOT}"

    # Download PDB if not present
    if [ ! -f "2GDZ.pdb" ]; then
        curl -sO https://files.rcsb.org/download/2GDZ.pdb
    fi

    # Prepare target (monomer, dimer, hotspots)
    python pipeline/prepare_target.py

    # Prepare DARPin scaffold
    python pipeline/prepare_darpin.py

    echo ">>> Target preparation complete."
}

# ─── Step 2: RFdiffusion Backbone Generation ─────────────────────────
run_rfdiff() {
    echo ""
    echo ">>> Step 2: RFdiffusion DARPin Partial Diffusion"
    echo "────────────────────────────────────────────────────────"

    cd "${PGDH_ROOT}"
    mkdir -p logs

    echo "Submitting DARPin partial diffusion designs (2XEH NI3C Mut6)..."
    JOB_1=$(sbatch --parsable scripts/run_rfdiffusion.sh 3 10)
    echo "  Job ${JOB_1}: DARPin partial_T=10 (3 designs, loop-only diffusion)"

    echo ""
    echo "Submitted 1 SLURM job. Monitor with: squeue -u ${USER}"
    echo "Wait for completion before running: bash run_pipeline.sh mpnn"
    echo ""
    echo "Job IDs: ${JOB_1}"

    # Save job IDs for dependency tracking
    echo "${JOB_1}" > "${PGDH_SCRATCH}/rfdiff_jobs.txt"
}

# ─── Step 3: ProteinMPNN Sequence Design ─────────────────────────────
run_mpnn() {
    echo ""
    echo ">>> Step 3: ProteinMPNN Sequence Design"
    echo "────────────────────────────────────────────────────────"

    cd "${PGDH_ROOT}"

    # Find all RFdiffusion output directories
    RFDIFF_DIRS=$(find "${PGDH_SCRATCH}/rfdiff" -maxdepth 1 -type d -name "darpin_*" 2>/dev/null | sort)

    if [ -z "${RFDIFF_DIRS}" ]; then
        echo "ERROR: No RFdiffusion outputs found in ${PGDH_SCRATCH}/rfdiff/"
        echo "Run 'bash run_pipeline.sh rfdiff' first."
        exit 1
    fi

    echo "Found RFdiffusion output directories:"
    for DIR in ${RFDIFF_DIRS}; do
        N_PDB=$(find "${DIR}" -name "*.pdb" | wc -l)
        echo "  ${DIR} (${N_PDB} PDBs)"

        JOB=$(sbatch --parsable scripts/run_proteinmpnn.sh "${DIR}" 8 0.1)
        echo "    Submitted MPNN job: ${JOB}"
    done

    echo ""
    echo "Wait for completion before running: bash run_pipeline.sh boltz"
}

# ─── Step 4: Boltz-2 Complex Prediction ──────────────────────────────
run_boltz() {
    echo ""
    echo ">>> Step 4: Boltz-2 Complex Prediction"
    echo "────────────────────────────────────────────────────────"

    cd "${PGDH_ROOT}"

    module load eth_proxy stack/2024-06 gcc/12.2.0 cuda/12.8.0
    source "${STRUCT_CONDA_BASE}/bin/activate" "${STRUCT_CONDA_ENV}"

    # Generate Boltz YAMLs from MPNN outputs
    MPNN_DIRS=$(find "${PGDH_SCRATCH}/mpnn" -maxdepth 1 -type d -name "20*" 2>/dev/null | sort)

    if [ -z "${MPNN_DIRS}" ]; then
        echo "ERROR: No ProteinMPNN outputs found."
        exit 1
    fi

    YAML_DIR="${PGDH_SCRATCH}/boltz_yamls"
    mkdir -p "${YAML_DIR}"

    for DIR in ${MPNN_DIRS}; do
        echo "Generating Boltz YAMLs from: ${DIR}"
        python pipeline/prepare_boltz_inputs.py \
            --mpnn_dir "${DIR}/sequences" \
            --config configs/pipeline_config.json \
            --output_dir "${YAML_DIR}" \
            --max_per_backbone 4
    done

    # Submit Boltz prediction job
    N_YAMLS=$(find "${YAML_DIR}" -name "*.yaml" | wc -l)
    echo "Total Boltz YAML files: ${N_YAMLS}"

    JOB=$(sbatch --parsable scripts/run_boltz.sh "${YAML_DIR}" 5)
    echo "Submitted Boltz job: ${JOB}"
    echo ""
    echo "Wait for completion before running: bash run_pipeline.sh score"
}

# ─── Step 5: Scoring ─────────────────────────────────────────────────
run_score() {
    echo ""
    echo ">>> Step 5: Scoring (ipSAE + Boltz metrics)"
    echo "────────────────────────────────────────────────────────"

    cd "${PGDH_ROOT}"

    module load eth_proxy stack/2024-06 gcc/12.2.0 cuda/12.8.0
    source "${STRUCT_CONDA_BASE}/bin/activate" "${STRUCT_CONDA_ENV}"

    BOLTZ_DIR=$(find "${PGDH_SCRATCH}/boltz" -maxdepth 1 -type d -name "20*" | sort | tail -1)

    if [ -z "${BOLTZ_DIR}" ]; then
        echo "ERROR: No Boltz outputs found."
        exit 1
    fi

    RESULTS_DIR="${PGDH_ROOT}/results"
    mkdir -p "${RESULTS_DIR}"

    python pipeline/score_designs.py \
        --boltz_dir "${BOLTZ_DIR}" \
        --output "${RESULTS_DIR}/scored_designs.csv" \
        --run_ipsae \
        --pae_cutoff 10 \
        --dist_cutoff 10

    echo ">>> Scoring complete: ${RESULTS_DIR}/scored_designs.csv"
}

# ─── Step 6: Selection ───────────────────────────────────────────────
run_select() {
    echo ""
    echo ">>> Step 6: Filtering and Selection"
    echo "────────────────────────────────────────────────────────"

    cd "${PGDH_ROOT}"

    module load eth_proxy stack/2024-06 gcc/12.2.0 cuda/12.8.0
    source "${STRUCT_CONDA_BASE}/bin/activate" "${STRUCT_CONDA_ENV}"

    RESULTS_DIR="${PGDH_ROOT}/results"

    python pipeline/filter_and_select.py \
        --scores_csv "${RESULTS_DIR}/scored_designs.csv" \
        --output "${RESULTS_DIR}/selected_designs.json" \
        --n_select 20 \
        --min_iptm 0.3 \
        --min_iplddt 0.5

    echo ""
    echo "============================================================"
    echo " Pipeline Complete!"
    echo "============================================================"
    echo " Results: ${RESULTS_DIR}/selected_designs.json"
    echo " Scores:  ${RESULTS_DIR}/scored_designs.csv"
    echo "============================================================"
}

# ─── Dispatch ─────────────────────────────────────────────────────────
case "${STEP}" in
    prepare)
        run_prepare
        ;;
    rfdiff)
        run_rfdiff
        ;;
    mpnn)
        run_mpnn
        ;;
    boltz)
        run_boltz
        ;;
    score)
        run_score
        ;;
    select)
        run_select
        ;;
    status)
        echo ""
        echo "Pipeline steps (run sequentially, wait for SLURM jobs between steps):"
        echo "  1. bash run_pipeline.sh prepare   (target + scaffold preparation)"
        echo "  2. bash run_pipeline.sh rfdiff     (submit RFdiffusion SLURM jobs)"
        echo "  3. bash run_pipeline.sh mpnn       (after rfdiff completes)"
        echo "  4. bash run_pipeline.sh boltz      (after mpnn completes)"
        echo "  5. bash run_pipeline.sh score      (after boltz completes)"
        echo "  6. bash run_pipeline.sh select     (final selection)"
        echo ""
        echo "Monitor SLURM jobs with: squeue -u ${USER}"
        ;;
    *)
        echo "Unknown step: ${STEP}"
        echo "Usage: bash run_pipeline.sh [prepare|rfdiff|mpnn|boltz|score|select|status]"
        exit 1
        ;;
esac
