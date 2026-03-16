#!/bin/bash
# =============================================================================
# submit_tune.sh  —  Submit Willmore genus-2 hyperparameter search to PBS.
#
# Submits N_TRIALS independent jobs as a PBS array (one trial per job), then
# submits a final merge+report job that runs after all trials complete.
#
# Usage (from the parent of the github directory, same place you run qsub from):
#   bash github/tuning/submit_tune.sh
#
# Edit the "Configuration" block below to match your cluster.
# =============================================================================
set -euo pipefail

# ---- Configuration (edit these) ---------------------------------------------
N_TRIALS=20
EPOCHS=100
SEED=42

QUEUE="serial"
MEM="10G"
MODULE="python/3.8.11-intel-2021.3.0"
VENV="env/bin/activate"   # relative to PBS_O_WORKDIR (the dir you submit from)
# -----------------------------------------------------------------------------

LAST_IDX=$((N_TRIALS - 1))

echo "============================================================"
echo "Willmore genus-2 hyperparameter search"
echo "  Trials : $N_TRIALS  (indices 0–${LAST_IDX})"
echo "  Epochs : $EPOCHS per trial"
echo "  Queue  : $QUEUE  mem=$MEM"
echo "============================================================"
echo ""

# ---- Submit trial array job --------------------------------------------------
# Each subjob writes to its own tuning/tune_results_trial_NNN.json to avoid
# concurrent-write races between parallel jobs.
ARRAY_JOB_ID=$(qsub - << EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -l mem=${MEM}
#PBS -N w_g2_tune
#PBS -m abe
#PBS -k oe
#PBS -j oe
#PBS -r y
#PBS -J 0-${LAST_IDX}

cd "\$PBS_O_WORKDIR" || exit 1

module load ${MODULE}
source ${VENV}

cd github || exit 1

echo "=== Trial \${PBS_ARRAY_INDEX}/${LAST_IDX} starting on \$(hostname) at \$(date) ==="
echo "Job: \${PBS_JOBID}"
echo ""

python -u tuning/tune_genus2.py --trial-idx "\${PBS_ARRAY_INDEX}" --n-trials ${N_TRIALS} --epochs ${EPOCHS} --seed ${SEED}

echo ""
echo "=== Trial \${PBS_ARRAY_INDEX} finished at \$(date) ==="
EOF
)

echo "Array job submitted: ${ARRAY_JOB_ID}"

# Keep [] but strip .hostname suffix — PBS needs JOBID[] to reference a whole array
BARE_ID="${ARRAY_JOB_ID%%\[*}"
ARRAY_DEP="${BARE_ID}[]"

# ---- Submit merge+report job (runs after all array elements finish) ----------
MERGE_JOB_ID=$(qsub - << EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -l mem=2G
#PBS -N w_g2_report
#PBS -m abe
#PBS -k oe
#PBS -j oe
#PBS -r y
#PBS -W depend=afterok:${ARRAY_DEP}

cd "\$PBS_O_WORKDIR" || exit 1

module load ${MODULE}
source ${VENV}

cd github || exit 1

echo "=== Merging results at \$(date) ==="
echo ""

python -u tuning/tune_genus2.py --n-trials ${N_TRIALS} --seed ${SEED} --merge

echo ""
echo "=== Report complete at \$(date) ==="
EOF
)

echo "Report job submitted: ${MERGE_JOB_ID}  (depends on ${ARRAY_JOB_ID})"
echo "  Early report  : cd github && python -u tuning/tune_genus2.py --merge --n-trials ${N_TRIALS} --seed ${SEED}"
