#!/bin/bash
# =============================================================================
# Design-23 Backbone Refinement Pipeline
# =============================================================================
# Re-diffuses design_23 backbone at multiple partial_T values in parallel,
# then chains MPNN sequence design and Boltz structure prediction.
#
# Architecture (all SLURM dependency-chained):
#   RFdiff T=3 (50 designs) ─┐
#   RFdiff T=5 (50 designs) ─┼─> MPNN (parallel per backbone) ─> Boltz prep ─> Boltz (parallel)
#   RFdiff T=8 (50 designs) ─┘
#
# Usage: bash scripts/run_refinement_pipeline.sh

set -euo pipefail

export PGDH_ROOT="/cluster/home/csageder/gigabase-hack-berlin"
source "${PGDH_ROOT}/scripts/setup_env.sh"

# ── Configuration ──
INPUT_PDB="/cluster/scratch/csageder/pgdh_outputs/rfdiff/darpin_pT15_20260228_035623/design_23_0.pdb"
PARTIAL_T_VALUES=(3 5 8)
DESIGNS_PER_T=50
RFDIFF_CONCURRENT=50
MPNN_CONCURRENT=20
MPNN_SEQS=8
MPNN_TEMP=0.1
BOLTZ_CONCURRENT=50
BOLTZ_MAX_PER_BB=4

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo " Design-23 Backbone Refinement Pipeline"
echo "============================================================"
echo " Input:      ${INPUT_PDB}"
echo " partial_T:  ${PARTIAL_T_VALUES[*]}"
echo " Designs/T:  ${DESIGNS_PER_T}"
echo " Total:      $((${#PARTIAL_T_VALUES[@]} * DESIGNS_PER_T)) backbones"
echo " MPNN:       ${MPNN_SEQS} seqs @ T=${MPNN_TEMP}"
echo " Boltz:      top ${BOLTZ_MAX_PER_BB}/backbone, ${BOLTZ_CONCURRENT} concurrent"
echo " Timestamp:  ${TIMESTAMP}"
echo "============================================================"

mkdir -p logs

# ── Step 1: Submit RFdiffusion array jobs in parallel ──
RFDIFF_JOBS=()
RFDIFF_DIRS=()

for PT in "${PARTIAL_T_VALUES[@]}"; do
    RFDIFF_OUT="${PGDH_SCRATCH}/rfdiff/refine_d23_pT${PT}_${TIMESTAMP}"
    mkdir -p "${RFDIFF_OUT}"
    RFDIFF_DIRS+=("${RFDIFF_OUT}")

    ARRAY_SPEC="0-$((DESIGNS_PER_T - 1))%${RFDIFF_CONCURRENT}"

    JOB=$(sbatch --parsable \
        --array="${ARRAY_SPEC}" \
        --export="ALL,RFDIFF_OUTPUT_DIR=${RFDIFF_OUT}" \
        scripts/run_rfdiff_refine.sh "${INPUT_PDB}" "${PT}")
    RFDIFF_JOBS+=("${JOB}")
    echo "[RFdiff] T=${PT}: job ${JOB} (${DESIGNS_PER_T} designs, array=${ARRAY_SPEC})"
done

# Build dependency string: afterok on ALL RFdiffusion jobs
RFDIFF_DEP=$(IFS=:; echo "${RFDIFF_JOBS[*]}")
echo ""
echo "All RFdiffusion jobs: ${RFDIFF_DEP}"

# ── Step 2: MPNN array jobs (one per T value, depends on its RFdiffusion) ──
MPNN_JOBS=()
MPNN_OUT="${PGDH_SCRATCH}/mpnn/refine_d23_${TIMESTAMP}"
mkdir -p "${MPNN_OUT}"

for i in "${!PARTIAL_T_VALUES[@]}"; do
    PT="${PARTIAL_T_VALUES[$i]}"
    RFDIFF_DIR="${RFDIFF_DIRS[$i]}"
    RFDIFF_JOB="${RFDIFF_JOBS[$i]}"

    ARRAY_SPEC="0-$((DESIGNS_PER_T - 1))%${MPNN_CONCURRENT}"

    JOB=$(sbatch --parsable \
        --dependency="afterok:${RFDIFF_JOB}" \
        --array="${ARRAY_SPEC}" \
        --export="ALL,MPNN_OUTPUT_DIR=${MPNN_OUT}" \
        scripts/run_proteinmpnn.sh "${RFDIFF_DIR}" "${MPNN_SEQS}" "${MPNN_TEMP}")
    MPNN_JOBS+=("${JOB}")
    echo "[MPNN]   T=${PT}: job ${JOB} (depends on RFdiff ${RFDIFF_JOB})"
done

MPNN_DEP=$(IFS=:; echo "${MPNN_JOBS[*]}")

# ── Step 3: Boltz prep + launch (depends on all MPNN completing) ──
YAML_DIR="${PGDH_SCRATCH}/boltz_yamls_refine_${TIMESTAMP}"
BOLTZ_OUT="${PGDH_SCRATCH}/boltz/refine_d23_${TIMESTAMP}"

BOLTZ_LAUNCHER="${PGDH_SCRATCH}/boltz_refine_launcher_${TIMESTAMP}.sh"
cat > "${BOLTZ_LAUNCHER}" <<LAUNCHER_EOF
#!/bin/bash
#SBATCH --job-name=boltz_ref_prep
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/boltz_ref_prep_%j.out
#SBATCH --error=logs/boltz_ref_prep_%j.err

set -euo pipefail
export PGDH_ROOT="${PGDH_ROOT}"
source "\${PGDH_ROOT}/scripts/setup_env.sh"

module load eth_proxy stack/2024-06 gcc/12.2.0 cuda/12.8.0
set +u
source "\${STRUCT_CONDA_BASE}/bin/activate" "\${STRUCT_CONDA_ENV}"
set -u

cd "\${PGDH_ROOT}"

MPNN_SEQ_DIRS=\$(find "${MPNN_OUT}" -type d -name "sequences" 2>/dev/null | sort)

if [ -z "\${MPNN_SEQ_DIRS}" ]; then
    echo "ERROR: No MPNN sequence dirs found in ${MPNN_OUT}"
    exit 1
fi

mkdir -p "${YAML_DIR}"

for DIR in \${MPNN_SEQ_DIRS}; do
    echo "Generating Boltz YAMLs from: \${DIR}"
    python pipeline/prepare_boltz_inputs.py \\
        --mpnn_dir "\${DIR}" \\
        --config configs/pipeline_config.json \\
        --output_dir "${YAML_DIR}" \\
        --max_per_backbone ${BOLTZ_MAX_PER_BB}
done

N_YAMLS=\$(find "${YAML_DIR}" -name "*.yaml" | wc -l)
echo "Total Boltz YAML files: \${N_YAMLS}"

if [ "\${N_YAMLS}" -eq 0 ]; then
    echo "ERROR: No YAML files generated."
    exit 1
fi

mkdir -p "${BOLTZ_OUT}"
ARRAY_MAX=\$((N_YAMLS - 1))
ARRAY_SPEC="0-\${ARRAY_MAX}%${BOLTZ_CONCURRENT}"

BOLTZ_JOB=\$(sbatch --parsable \\
    --array="\${ARRAY_SPEC}" \\
    --export="ALL,BOLTZ_OUTPUT_DIR=${BOLTZ_OUT}" \\
    scripts/run_boltz.sh "${YAML_DIR}" 1)
echo "Submitted Boltz array: \${BOLTZ_JOB} (\${N_YAMLS} predictions, max ${BOLTZ_CONCURRENT} concurrent)"
echo "\${BOLTZ_JOB}" > "${PGDH_SCRATCH}/boltz_refine_jobs.txt"
LAUNCHER_EOF

PREP_JOB=$(sbatch --parsable \
    --dependency="afterok:${MPNN_DEP}" \
    "${BOLTZ_LAUNCHER}")
echo ""
echo "[Boltz]  Prep+launch: job ${PREP_JOB} (depends on all MPNN: ${MPNN_DEP})"

echo ""
echo "============================================================"
echo " Refinement Pipeline Submitted!"
echo "============================================================"
echo " Job chain:"
for i in "${!PARTIAL_T_VALUES[@]}"; do
    echo "   RFdiff T=${PARTIAL_T_VALUES[$i]}: ${RFDIFF_JOBS[$i]} -> MPNN: ${MPNN_JOBS[$i]}"
done
echo "   All MPNN -> Boltz prep: ${PREP_JOB} -> Boltz array (TBD)"
echo ""
echo " Output directories:"
for i in "${!PARTIAL_T_VALUES[@]}"; do
    echo "   RFdiff T=${PARTIAL_T_VALUES[$i]}: ${RFDIFF_DIRS[$i]}"
done
echo "   MPNN:        ${MPNN_OUT}"
echo "   Boltz YAMLs: ${YAML_DIR}"
echo "   Boltz:       ${BOLTZ_OUT}"
echo ""
echo " After Boltz completes, run scoring:"
echo "   python pipeline/score_designs.py \\"
echo "     --boltz_dir ${BOLTZ_OUT} \\"
echo "     --output results/scored_refine_design23.csv \\"
echo "     --run_ipsae --pae_cutoff 15 --dist_cutoff 15"
echo "============================================================"

echo "${BOLTZ_OUT}" > "${PGDH_SCRATCH}/boltz_refine_outdir.txt"
