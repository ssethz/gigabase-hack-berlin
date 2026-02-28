#!/bin/bash
# Launch the full MPNN saturation pipeline for design_23:
#   MPNN (100 seqs) -> Boltz prep -> Boltz predict (parallel) -> Score
#
# Usage: bash scripts/run_saturation_pipeline.sh

set -euo pipefail

export PGDH_ROOT="/cluster/home/csageder/gigabase-hack-berlin"
source "${PGDH_ROOT}/scripts/setup_env.sh"

MPNN_INPUT="/cluster/scratch/csageder/pgdh_outputs/mpnn/20260228_035623/task_16_design_23_0"
NUM_SEQS=100
TEMPERATURE=0.1
BOLTZ_CONCURRENT="${BOLTZ_MAX_CONCURRENT:-20}"
MAX_PER_BACKBONE=100

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo " Design-23 MPNN Saturation Pipeline"
echo "============================================================"
echo " Backbone:  design_23_0 (ipSAE=0.39 winner)"
echo " Sequences: ${NUM_SEQS} @ temp ${TEMPERATURE}"
echo " Timestamp: ${TIMESTAMP}"
echo "============================================================"

mkdir -p logs

# ── Step 1: MPNN sequence saturation ──
MPNN_JOB=$(sbatch --parsable \
    scripts/run_mpnn_saturation.sh "${MPNN_INPUT}" "${NUM_SEQS}" "${TEMPERATURE}")
echo "[1/3] MPNN saturation: job ${MPNN_JOB} (${NUM_SEQS} seqs @ T=${TEMPERATURE})"

# ── Step 2: Boltz prep + launch (depends on MPNN) ──
YAML_DIR="${PGDH_SCRATCH}/boltz_yamls_sat_${TIMESTAMP}"
BOLTZ_OUT="${PGDH_SCRATCH}/boltz/sat_${TIMESTAMP}"

BOLTZ_LAUNCHER="${PGDH_SCRATCH}/boltz_sat_launcher_${TIMESTAMP}.sh"
cat > "${BOLTZ_LAUNCHER}" <<LAUNCHER_EOF
#!/bin/bash
#SBATCH --job-name=boltz_sat_prep
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/boltz_sat_prep_%j.out
#SBATCH --error=logs/boltz_sat_prep_%j.err

set -euo pipefail
export PGDH_ROOT="${PGDH_ROOT}"
source "\${PGDH_ROOT}/scripts/setup_env.sh"

module load eth_proxy stack/2024-06 gcc/12.2.0 cuda/12.8.0
set +u
source "\${STRUCT_CONDA_BASE}/bin/activate" "\${STRUCT_CONDA_ENV}"
set -u

cd "\${PGDH_ROOT}"

MPNN_SAT_DIR=\$(cat "${PGDH_SCRATCH}/mpnn_saturation_outdir.txt")
MPNN_SEQ_DIR="\${MPNN_SAT_DIR}/sequences"

echo "MPNN saturation output: \${MPNN_SAT_DIR}"

if [ ! -d "\${MPNN_SEQ_DIR}" ]; then
    echo "ERROR: No sequences dir found at \${MPNN_SEQ_DIR}"
    exit 1
fi

mkdir -p "${YAML_DIR}"

python pipeline/prepare_boltz_inputs.py \\
    --mpnn_dir "\${MPNN_SEQ_DIR}" \\
    --config configs/pipeline_config.json \\
    --output_dir "${YAML_DIR}" \\
    --max_per_backbone ${MAX_PER_BACKBONE}

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
echo "\${BOLTZ_JOB}" > "${PGDH_SCRATCH}/boltz_sat_jobs.txt"
LAUNCHER_EOF

PREP_JOB=$(sbatch --parsable \
    --dependency="afterok:${MPNN_JOB}" \
    "${BOLTZ_LAUNCHER}")
echo "[2/3] Boltz prep+launch: job ${PREP_JOB} (depends on ${MPNN_JOB})"

echo ""
echo "============================================================"
echo " Pipeline submitted!"
echo "   MPNN:       ${MPNN_JOB}"
echo "   Boltz prep: ${PREP_JOB} (chains after MPNN)"
echo "   Boltz out:  ${BOLTZ_OUT}"
echo ""
echo " After Boltz completes, run scoring:"
echo "   python pipeline/score_designs.py \\"
echo "     --boltz_dir ${BOLTZ_OUT} \\"
echo "     --output results/scored_sat_design23.csv \\"
echo "     --run_ipsae --pae_cutoff 15 --dist_cutoff 15"
echo "============================================================"

echo "${BOLTZ_OUT}" > "${PGDH_SCRATCH}/boltz_sat_outdir.txt"
