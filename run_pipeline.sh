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
    echo ">>> Step 2: RFdiffusion DARPin Partial Diffusion (PARALLEL)"
    echo "────────────────────────────────────────────────────────"

    cd "${PGDH_ROOT}"
    mkdir -p logs

    N_DESIGNS="${RFDIFF_N_DESIGNS:-10}"
    PARTIAL_T="${RFDIFF_PARTIAL_T:-10}"
    MAX_CONCURRENT="${RFDIFF_MAX_CONCURRENT:-10}"
    ARRAY_SPEC="0-$((N_DESIGNS - 1))%${MAX_CONCURRENT}"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RFDIFF_OUT="${PGDH_SCRATCH}/rfdiff/darpin_pT${PARTIAL_T}_${TIMESTAMP}"
    mkdir -p "${RFDIFF_OUT}"

    echo "Submitting ${N_DESIGNS} DARPin partial diffusion designs in parallel..."
    echo "  partial_T=${PARTIAL_T}, array=${ARRAY_SPEC}"
    echo "  Output: ${RFDIFF_OUT}"

    JOB_1=$(sbatch --parsable \
        --array="${ARRAY_SPEC}" \
        --export="ALL,RFDIFF_OUTPUT_DIR=${RFDIFF_OUT}" \
        scripts/run_rfdiffusion.sh "${PARTIAL_T}")
    echo "  Array job ${JOB_1}: ${N_DESIGNS} designs across ${MAX_CONCURRENT} concurrent GPUs"

    echo ""
    echo "Monitor with: squeue -u ${USER}"
    echo "Wait for completion before running: bash run_pipeline.sh mpnn"
    echo ""

    # Save for dependency tracking
    echo "${JOB_1}" > "${PGDH_SCRATCH}/rfdiff_jobs.txt"
    echo "${RFDIFF_OUT}" > "${PGDH_SCRATCH}/rfdiff_outdir.txt"
}

# ─── Step 3: ProteinMPNN Sequence Design ─────────────────────────────
run_mpnn() {
    echo ""
    echo ">>> Step 3: ProteinMPNN Sequence Design (PARALLEL)"
    echo "────────────────────────────────────────────────────────"

    cd "${PGDH_ROOT}"

    MAX_CONCURRENT="${MPNN_MAX_CONCURRENT:-10}"

    # Find all RFdiffusion output directories
    RFDIFF_DIRS=$(find "${PGDH_SCRATCH}/rfdiff" -maxdepth 1 -type d -name "darpin_*" 2>/dev/null | sort)

    if [ -z "${RFDIFF_DIRS}" ]; then
        echo "ERROR: No RFdiffusion outputs found in ${PGDH_SCRATCH}/rfdiff/"
        echo "Run 'bash run_pipeline.sh rfdiff' first."
        exit 1
    fi

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    MPNN_OUT="${PGDH_SCRATCH}/mpnn/${TIMESTAMP}"
    mkdir -p "${MPNN_OUT}"

    echo "Found RFdiffusion output directories:"
    MPNN_JOBS=()
    for DIR in ${RFDIFF_DIRS}; do
        N_PDB=$(find "${DIR}" -maxdepth 1 -name "*.pdb" | wc -l)
        echo "  ${DIR} (${N_PDB} PDBs)"

        if [ "${N_PDB}" -eq 0 ]; then
            echo "    Skipping: no PDBs"
            continue
        fi

        ARRAY_MAX=$((N_PDB - 1))
        ARRAY_SPEC="0-${ARRAY_MAX}%${MAX_CONCURRENT}"

        JOB=$(sbatch --parsable \
            --array="${ARRAY_SPEC}" \
            --export="ALL,MPNN_OUTPUT_DIR=${MPNN_OUT}" \
            scripts/run_proteinmpnn.sh "${DIR}" 8 0.2)
        echo "    Array job ${JOB}: ${N_PDB} backbones in parallel"
        MPNN_JOBS+=("${JOB}")
    done

    echo ""
    echo "Submitted ${#MPNN_JOBS[@]} MPNN array jobs"
    echo "Monitor with: squeue -u ${USER}"
    echo "Wait for completion before running: bash run_pipeline.sh boltz"

    # Save for dependency tracking
    printf "%s\n" "${MPNN_JOBS[@]}" > "${PGDH_SCRATCH}/mpnn_jobs.txt"
    echo "${MPNN_OUT}" > "${PGDH_SCRATCH}/mpnn_outdir.txt"
}

# ─── Step 4: Boltz-2 Complex Prediction ──────────────────────────────
run_boltz() {
    echo ""
    echo ">>> Step 4: Boltz-2 Complex Prediction (PARALLEL)"
    echo "────────────────────────────────────────────────────────"

    cd "${PGDH_ROOT}"

    module load eth_proxy stack/2024-06 gcc/12.2.0 cuda/12.8.0
    source "${STRUCT_CONDA_BASE}/bin/activate" "${STRUCT_CONDA_ENV}"

    MAX_CONCURRENT="${BOLTZ_MAX_CONCURRENT:-20}"

    # Find MPNN sequence outputs: look for FASTA files in task-based or legacy layout
    MPNN_SEQ_DIRS=$(find "${PGDH_SCRATCH}/mpnn" -type d -name "sequences" 2>/dev/null | sort)

    if [ -z "${MPNN_SEQ_DIRS}" ]; then
        echo "ERROR: No ProteinMPNN sequence outputs found."
        exit 1
    fi

    YAML_DIR="${PGDH_SCRATCH}/boltz_yamls"
    mkdir -p "${YAML_DIR}"

    for DIR in ${MPNN_SEQ_DIRS}; do
        echo "Generating Boltz YAMLs from: ${DIR}"
        python pipeline/prepare_boltz_inputs.py \
            --mpnn_dir "${DIR}" \
            --config configs/pipeline_config.json \
            --output_dir "${YAML_DIR}" \
            --max_per_backbone 4
    done

    N_YAMLS=$(find "${YAML_DIR}" -name "*.yaml" | wc -l)
    echo "Total Boltz YAML files: ${N_YAMLS}"

    if [ "${N_YAMLS}" -eq 0 ]; then
        echo "ERROR: No YAML files generated."
        exit 1
    fi

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BOLTZ_OUT="${PGDH_SCRATCH}/boltz/${TIMESTAMP}"
    mkdir -p "${BOLTZ_OUT}"

    ARRAY_MAX=$((N_YAMLS - 1))
    ARRAY_SPEC="0-${ARRAY_MAX}%${MAX_CONCURRENT}"

    echo "Submitting ${N_YAMLS} Boltz predictions in parallel (max ${MAX_CONCURRENT} concurrent)..."
    JOB=$(sbatch --parsable \
        --array="${ARRAY_SPEC}" \
        --export="ALL,BOLTZ_OUTPUT_DIR=${BOLTZ_OUT}" \
        scripts/run_boltz.sh "${YAML_DIR}" 1)
    echo "  Array job ${JOB}: ${N_YAMLS} predictions across ${MAX_CONCURRENT} concurrent GPUs"

    echo ""
    echo "Monitor with: squeue -u ${USER}"
    echo "Wait for completion before running: bash run_pipeline.sh score"

    echo "${JOB}" > "${PGDH_SCRATCH}/boltz_jobs.txt"
    echo "${BOLTZ_OUT}" > "${PGDH_SCRATCH}/boltz_outdir.txt"
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
        --pae_cutoff 15 \
        --dist_cutoff 15

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

# ─── Auto: Full pipeline with SLURM dependency chaining ──────────────
run_auto() {
    echo ""
    echo ">>> AUTO MODE: Full pipeline with SLURM dependency chaining"
    echo "────────────────────────────────────────────────────────"

    cd "${PGDH_ROOT}"
    mkdir -p logs

    N_DESIGNS="${RFDIFF_N_DESIGNS:-10}"
    PARTIAL_T="${RFDIFF_PARTIAL_T:-10}"
    RFDIFF_CONCURRENT="${RFDIFF_MAX_CONCURRENT:-10}"
    MPNN_CONCURRENT="${MPNN_MAX_CONCURRENT:-10}"
    BOLTZ_CONCURRENT="${BOLTZ_MAX_CONCURRENT:-20}"
    SEQS_PER_TARGET="${MPNN_SEQS_PER_TARGET:-8}"
    MAX_PER_BACKBONE="${BOLTZ_MAX_PER_BACKBONE:-4}"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    echo ""
    echo "Configuration:"
    echo "  RFdiffusion designs:    ${N_DESIGNS} (max ${RFDIFF_CONCURRENT} concurrent)"
    echo "  partial_T:              ${PARTIAL_T}"
    echo "  MPNN seqs/target:       ${SEQS_PER_TARGET} (max ${MPNN_CONCURRENT} concurrent)"
    echo "  Boltz max/backbone:     ${MAX_PER_BACKBONE} (max ${BOLTZ_CONCURRENT} concurrent)"
    echo "  Timestamp:              ${TIMESTAMP}"
    echo ""

    RFDIFF_OUT="${PGDH_SCRATCH}/rfdiff/darpin_pT${PARTIAL_T}_${TIMESTAMP}"
    MPNN_OUT="${PGDH_SCRATCH}/mpnn/${TIMESTAMP}"
    YAML_DIR="${PGDH_SCRATCH}/boltz_yamls_${TIMESTAMP}"
    BOLTZ_OUT="${PGDH_SCRATCH}/boltz/${TIMESTAMP}"
    mkdir -p "${RFDIFF_OUT}" "${MPNN_OUT}" "${YAML_DIR}" "${BOLTZ_OUT}"

    # ── Step 1: RFdiffusion array ──
    RFDIFF_ARRAY="0-$((N_DESIGNS - 1))%${RFDIFF_CONCURRENT}"
    RFDIFF_JOB=$(sbatch --parsable \
        --array="${RFDIFF_ARRAY}" \
        --export="ALL,RFDIFF_OUTPUT_DIR=${RFDIFF_OUT}" \
        scripts/run_rfdiffusion.sh "${PARTIAL_T}")
    echo "[1/4] RFdiffusion: job ${RFDIFF_JOB} (${N_DESIGNS} designs, array=${RFDIFF_ARRAY})"

    # ── Step 2: MPNN array (depends on all RFdiffusion completing) ──
    MPNN_JOB=$(sbatch --parsable \
        --dependency="afterok:${RFDIFF_JOB}" \
        --array="0-$((N_DESIGNS - 1))%${MPNN_CONCURRENT}" \
        --export="ALL,MPNN_OUTPUT_DIR=${MPNN_OUT}" \
        scripts/run_proteinmpnn.sh "${RFDIFF_OUT}" "${SEQS_PER_TARGET}" 0.2)
    echo "[2/4] ProteinMPNN: job ${MPNN_JOB} (depends on ${RFDIFF_JOB})"

    # ── Step 3: Boltz prep + submission (depends on all MPNN completing) ──
    # We need a wrapper job that generates YAMLs then submits the Boltz array.
    BOLTZ_LAUNCHER="${PGDH_SCRATCH}/boltz_launcher_${TIMESTAMP}.sh"
    cat > "${BOLTZ_LAUNCHER}" <<LAUNCHER_EOF
#!/bin/bash
#SBATCH --job-name=boltz_prep
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/boltz_prep_%j.out
#SBATCH --error=logs/boltz_prep_%j.err

set -euo pipefail
export PGDH_ROOT="${PGDH_ROOT}"
source "\${PGDH_ROOT}/scripts/setup_env.sh"

module load eth_proxy stack/2024-06 gcc/12.2.0 cuda/12.8.0
set +u
source "\${STRUCT_CONDA_BASE}/bin/activate" "\${STRUCT_CONDA_ENV}"
set -u

cd "\${PGDH_ROOT}"

# Find all sequence outputs from the MPNN array jobs
MPNN_SEQ_DIRS=\$(find "${MPNN_OUT}" -type d -name "sequences" 2>/dev/null | sort)

if [ -z "\${MPNN_SEQ_DIRS}" ]; then
    echo "ERROR: No MPNN sequence dirs found in ${MPNN_OUT}"
    exit 1
fi

for DIR in \${MPNN_SEQ_DIRS}; do
    echo "Generating Boltz YAMLs from: \${DIR}"
    python pipeline/prepare_boltz_inputs.py \\
        --mpnn_dir "\${DIR}" \\
        --config configs/pipeline_config.json \\
        --output_dir "${YAML_DIR}" \\
        --max_per_backbone ${MAX_PER_BACKBONE}
done

N_YAMLS=\$(find "${YAML_DIR}" -name "*.yaml" | wc -l)
echo "Total Boltz YAML files: \${N_YAMLS}"

if [ "\${N_YAMLS}" -eq 0 ]; then
    echo "ERROR: No YAML files generated."
    exit 1
fi

ARRAY_MAX=\$((N_YAMLS - 1))
ARRAY_SPEC="0-\${ARRAY_MAX}%${BOLTZ_CONCURRENT}"

BOLTZ_JOB=\$(sbatch --parsable \\
    --array="\${ARRAY_SPEC}" \\
    --export="ALL,BOLTZ_OUTPUT_DIR=${BOLTZ_OUT}" \\
    scripts/run_boltz.sh "${YAML_DIR}" 1)
echo "Submitted Boltz array: \${BOLTZ_JOB} (\${N_YAMLS} predictions)"
echo "\${BOLTZ_JOB}" > "${PGDH_SCRATCH}/boltz_jobs.txt"
LAUNCHER_EOF

    PREP_JOB=$(sbatch --parsable \
        --dependency="afterok:${MPNN_JOB}" \
        "${BOLTZ_LAUNCHER}")
    echo "[3/4] Boltz prep+launch: job ${PREP_JOB} (depends on ${MPNN_JOB})"

    echo ""
    echo "============================================================"
    echo " Pipeline submitted! Job dependency chain:"
    echo "   RFdiffusion ${RFDIFF_JOB} → MPNN ${MPNN_JOB} → Boltz prep ${PREP_JOB} → Boltz array (TBD)"
    echo ""
    echo " Output directories:"
    echo "   RFdiffusion: ${RFDIFF_OUT}"
    echo "   MPNN:        ${MPNN_OUT}"
    echo "   Boltz YAMLs: ${YAML_DIR}"
    echo "   Boltz:       ${BOLTZ_OUT}"
    echo ""
    echo " Monitor: squeue -u ${USER}"
    echo " After Boltz completes, run:"
    echo "   bash run_pipeline.sh score"
    echo "   bash run_pipeline.sh select"
    echo "============================================================"

    # Save metadata
    echo "${RFDIFF_JOB}" > "${PGDH_SCRATCH}/rfdiff_jobs.txt"
    echo "${RFDIFF_OUT}" > "${PGDH_SCRATCH}/rfdiff_outdir.txt"
    echo "${MPNN_JOB}" > "${PGDH_SCRATCH}/mpnn_jobs.txt"
    echo "${MPNN_OUT}" > "${PGDH_SCRATCH}/mpnn_outdir.txt"
    echo "${BOLTZ_OUT}" > "${PGDH_SCRATCH}/boltz_outdir.txt"
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
    auto)
        run_auto
        ;;
    status)
        echo ""
        echo "Pipeline steps (run sequentially, wait for SLURM jobs between steps):"
        echo "  1. bash run_pipeline.sh prepare   (target + scaffold preparation)"
        echo "  2. bash run_pipeline.sh rfdiff     (submit RFdiffusion array jobs)"
        echo "  3. bash run_pipeline.sh mpnn       (after rfdiff completes)"
        echo "  4. bash run_pipeline.sh boltz      (after mpnn completes)"
        echo "  5. bash run_pipeline.sh score      (after boltz completes)"
        echo "  6. bash run_pipeline.sh select     (final selection)"
        echo ""
        echo "  OR: bash run_pipeline.sh auto      (chain ALL steps with SLURM deps)"
        echo ""
        echo "Parallelism controls (env vars):"
        echo "  RFDIFF_N_DESIGNS=10        Number of RFdiffusion designs"
        echo "  RFDIFF_PARTIAL_T=10        Diffusion noise steps"
        echo "  RFDIFF_MAX_CONCURRENT=10   Max concurrent RFdiffusion GPUs"
        echo "  MPNN_MAX_CONCURRENT=10     Max concurrent MPNN GPUs"
        echo "  BOLTZ_MAX_CONCURRENT=20    Max concurrent Boltz GPUs"
        echo ""
        echo "Example: RFDIFF_N_DESIGNS=50 BOLTZ_MAX_CONCURRENT=30 bash run_pipeline.sh auto"
        echo ""
        echo "Monitor SLURM jobs with: squeue -u ${USER}"
        ;;
    *)
        echo "Unknown step: ${STEP}"
        echo "Usage: bash run_pipeline.sh [prepare|rfdiff|mpnn|boltz|score|select|status]"
        exit 1
        ;;
esac
