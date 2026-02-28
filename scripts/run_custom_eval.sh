#!/bin/bash
# =============================================================================
# Custom Evaluation Pipeline: MPNN (50 seqs) → Boltz → Score
# =============================================================================
#
# Evaluates externally provided PDB backbones by:
#   1. Running ProteinMPNN (50 sequences per backbone)
#   2. Preparing Boltz-2 YAMLs for all designed sequences
#   3. Running Boltz-2 structure prediction
#   4. Scoring with ipSAE + Boltz confidence metrics
#
# Usage:
#   bash scripts/run_custom_eval.sh <pdb_folder> [run_name]
#
# Example:
#   bash scripts/run_custom_eval.sh inputs/friend_designs my_eval_v1
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PGDH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${PGDH_ROOT}/scripts/setup_env.sh"

INPUT_FOLDER="${1:?Usage: $0 <input_folder> [run_name]}"
RUN_NAME="${2:-custom_eval_$(date +%Y%m%d_%H%M%S)}"

SEQS_PER_TARGET="${CUSTOM_SEQS_PER_TARGET:-50}"
MPNN_TEMP="${CUSTOM_MPNN_TEMP:-0.2}"
MAX_PER_BACKBONE="${CUSTOM_MAX_PER_BACKBONE:-50}"
BOLTZ_SAMPLES="${CUSTOM_BOLTZ_SAMPLES:-1}"
MPNN_CONCURRENT="${MPNN_MAX_CONCURRENT:-10}"
BOLTZ_CONCURRENT="${BOLTZ_MAX_CONCURRENT:-20}"

# Resolve input folder to absolute path
if [[ "${INPUT_FOLDER}" != /* ]]; then
    INPUT_FOLDER="${PGDH_ROOT}/${INPUT_FOLDER}"
fi

if [ ! -d "${INPUT_FOLDER}" ]; then
    echo "ERROR: Input folder not found: ${INPUT_FOLDER}"
    exit 1
fi

N_PDB=$(find "${INPUT_FOLDER}" -maxdepth 1 \( -name "*.pdb" -o -name "*.cif" \) | wc -l)
if [ "${N_PDB}" -eq 0 ]; then
    echo "ERROR: No PDB or CIF files found in ${INPUT_FOLDER}"
    exit 1
fi

EVAL_DIR="${PGDH_SCRATCH}/custom_eval/${RUN_NAME}"
MPNN_OUT="${EVAL_DIR}/mpnn"
YAML_DIR="${EVAL_DIR}/boltz_yamls"
BOLTZ_OUT="${EVAL_DIR}/boltz"
RESULTS_DIR="${EVAL_DIR}/results"
mkdir -p "${MPNN_OUT}" "${YAML_DIR}" "${BOLTZ_OUT}" "${RESULTS_DIR}" "${PGDH_ROOT}/logs"

echo "============================================================"
echo " Custom Evaluation Pipeline"
echo "============================================================"
echo " Input files:    ${INPUT_FOLDER} (${N_PDB} PDB/CIF files)"
echo " Run name:       ${RUN_NAME}"
echo " Seqs/backbone:  ${SEQS_PER_TARGET}"
echo " MPNN temp:      ${MPNN_TEMP}"
echo " Boltz samples:  ${BOLTZ_SAMPLES}"
echo " Output:         ${EVAL_DIR}"
echo "============================================================"
echo ""

# ── Step 1: ProteinMPNN array ──────────────────────────────────────────

MPNN_ARRAY="0-$((N_PDB - 1))%${MPNN_CONCURRENT}"

MPNN_JOB=$(sbatch --parsable \
    --array="${MPNN_ARRAY}" \
    --export="ALL,MPNN_OUTPUT_DIR=${MPNN_OUT}" \
    "${PGDH_ROOT}/scripts/run_proteinmpnn.sh" "${INPUT_FOLDER}" "${SEQS_PER_TARGET}" "${MPNN_TEMP}")
echo "[1/3] ProteinMPNN: job ${MPNN_JOB} (${N_PDB} backbones × ${SEQS_PER_TARGET} seqs, array=${MPNN_ARRAY})"

# ── Step 2: Boltz prep + launch (depends on MPNN) ─────────────────────

BOLTZ_LAUNCHER="${EVAL_DIR}/boltz_launcher.sh"
cat > "${BOLTZ_LAUNCHER}" <<LAUNCHER_EOF
#!/bin/bash
#SBATCH --job-name=boltz_prep_${RUN_NAME}
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/custom_boltz_prep_%j.out
#SBATCH --error=logs/custom_boltz_prep_%j.err

set -euo pipefail
export PGDH_ROOT="${PGDH_ROOT}"
source "\${PGDH_ROOT}/scripts/setup_env.sh"

module load eth_proxy stack/2024-06 gcc/12.2.0 cuda/12.8.0
set +u
source "\${STRUCT_CONDA_BASE}/bin/activate" "\${STRUCT_CONDA_ENV}"
set -u

cd "\${PGDH_ROOT}"

# ── MPNN score extraction (immediate result) ──
echo "=== Extracting MPNN sequence scores ==="
python pipeline/score_mpnn.py \\
    --mpnn_dir "${MPNN_OUT}" \\
    --output "${RESULTS_DIR}/mpnn_scores.csv" \\
    --skip_original
echo ""
echo "MPNN scores ready: ${RESULTS_DIR}/mpnn_scores.csv"
echo ""

MPNN_SEQ_DIRS=\$(find "${MPNN_OUT}" -type d -name "sequences" 2>/dev/null | sort)

if [ -z "\${MPNN_SEQ_DIRS}" ]; then
    echo "ERROR: No MPNN sequence dirs found in ${MPNN_OUT}"
    exit 1
fi

# Merge all chain_info.json files from MPNN task dirs into one
MERGED_CI="${EVAL_DIR}/chain_info_merged.json"
python -c "
import json, sys
from pathlib import Path
merged = {}
for f in sorted(Path('${MPNN_OUT}').glob('**/chain_info.json')):
    with open(f) as fh:
        merged.update(json.load(fh))
with open('\${MERGED_CI}', 'w') as fh:
    json.dump(merged, fh, indent=2)
print(f'Merged chain_info: {len(merged)} backbones')
"

for DIR in \${MPNN_SEQ_DIRS}; do
    echo "Generating Boltz YAMLs from: \${DIR}"
    if [ -f "\${MERGED_CI}" ]; then
        python pipeline/prepare_boltz_inputs.py \\
            --mpnn_dir "\${DIR}" \\
            --chain_info "\${MERGED_CI}" \\
            --output_dir "${YAML_DIR}" \\
            --max_per_backbone ${MAX_PER_BACKBONE}
    else
        python pipeline/prepare_boltz_inputs.py \\
            --mpnn_dir "\${DIR}" \\
            --config configs/pipeline_config.json \\
            --output_dir "${YAML_DIR}" \\
            --max_per_backbone ${MAX_PER_BACKBONE}
    fi
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
    "\${PGDH_ROOT}/scripts/run_boltz.sh" "${YAML_DIR}" ${BOLTZ_SAMPLES})
echo "Submitted Boltz array: \${BOLTZ_JOB} (\${N_YAMLS} predictions)"
echo "\${BOLTZ_JOB}" > "${EVAL_DIR}/boltz_job.txt"

# ── Step 3: Score (depends on Boltz) ─────────────────────────────────

SCORE_SCRIPT="${EVAL_DIR}/score_launcher.sh"
cat > "\${SCORE_SCRIPT}" <<'SCORE_INNER_EOF'
#!/bin/bash
#SBATCH --job-name=score_${RUN_NAME}
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=${PGDH_ROOT}/logs/custom_score_%j.out
#SBATCH --error=${PGDH_ROOT}/logs/custom_score_%j.err

set -euo pipefail
export PGDH_ROOT="${PGDH_ROOT}"
source "\${PGDH_ROOT}/scripts/setup_env.sh"

module load eth_proxy stack/2024-06 gcc/12.2.0 cuda/12.8.0
set +u
source "\${STRUCT_CONDA_BASE}/bin/activate" "\${STRUCT_CONDA_ENV}"
set -u

cd "\${PGDH_ROOT}"

python pipeline/score_designs.py \\
    --boltz_dir "${BOLTZ_OUT}" \\
    --output "${RESULTS_DIR}/scored_designs.csv" \\
    --run_ipsae \\
    --pae_cutoff 15 \\
    --dist_cutoff 15 \\
    --chain_pairs AB,AC

echo "Scoring complete: ${RESULTS_DIR}/scored_designs.csv"
SCORE_INNER_EOF

SCORE_JOB=\$(sbatch --parsable \\
    --dependency="afterok:\${BOLTZ_JOB}" \\
    "\${SCORE_SCRIPT}")
echo "Submitted scoring job: \${SCORE_JOB} (depends on Boltz \${BOLTZ_JOB})"
echo "\${SCORE_JOB}" > "${EVAL_DIR}/score_job.txt"
LAUNCHER_EOF

PREP_JOB=$(sbatch --parsable \
    --dependency="afterok:${MPNN_JOB}" \
    "${BOLTZ_LAUNCHER}")
echo "[2/3] Boltz prep+launch: job ${PREP_JOB} (depends on MPNN ${MPNN_JOB})"
echo "[3/3] Scoring will be submitted automatically after Boltz completes"

echo ""
echo "============================================================"
echo " Custom evaluation submitted!"
echo "============================================================"
echo " Job chain: MPNN ${MPNN_JOB} → Boltz prep ${PREP_JOB} → Boltz array (TBD) → Score (TBD)"
echo ""
echo " Output directories:"
echo "   MPNN:        ${MPNN_OUT}"
echo "   Boltz YAMLs: ${YAML_DIR}"
echo "   Boltz:       ${BOLTZ_OUT}"
echo "   Results:     ${RESULTS_DIR}"
echo ""
echo " Monitor: squeue -u ${USER}"
echo " Final output: ${RESULTS_DIR}/scored_designs.csv"
echo "============================================================"

# Save metadata
cat > "${EVAL_DIR}/metadata.json" <<META_EOF
{
    "run_name": "${RUN_NAME}",
    "input_folder": "${INPUT_FOLDER}",
    "n_pdb": ${N_PDB},
    "seqs_per_target": ${SEQS_PER_TARGET},
    "mpnn_temp": ${MPNN_TEMP},
    "boltz_samples": ${BOLTZ_SAMPLES},
    "max_per_backbone": ${MAX_PER_BACKBONE},
    "mpnn_job": "${MPNN_JOB}",
    "boltz_prep_job": "${PREP_JOB}",
    "eval_dir": "${EVAL_DIR}",
    "timestamp": "$(date -Iseconds)"
}
META_EOF
