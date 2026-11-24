#!/bin/bash
#SBATCH --job-name=direction2-h100
#SBATCH --account=edu
#SBATCH --partition=burst
#SBATCH --qos=burst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/insomnia001/depts/edu/users/wt2358/projects/allm-dgsc/outputs/direction2-h100-%j.log

set -euo pipefail
cd /insomnia001/depts/edu/users/wt2358/projects/allm-dgsc
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/direction2-matplotlib-${SLURM_JOB_ID}"
D2_PY=/insomnia001/depts/edu/users/wt2358/envs/allm-direction2/bin/python
D2_RUN_LABEL="${D2_RUN_LABEL:-h100}"
D2_MIN_GPU_MIB="${D2_MIN_GPU_MIB:-75000}"
D2_OUTPUT="outputs/direction2-${D2_RUN_LABEL}-${SLURM_JOB_ID}"
# Shared with the R6000 adapter, guarding the validation as well as collection.
exec 9>outputs/.direction2-suite.lock
flock -n 9 || { echo 'Another Direction 2 workflow holds the execution lock'; exit 1; }
mkdir "$D2_OUTPUT"
scontrol show job -o "$SLURM_JOB_ID" > "$D2_OUTPUT/allocation.txt"
cp scripts/direction2_h100.sbatch "$D2_OUTPUT/workflow_script.sbatch"
if [[ "$D2_RUN_LABEL" == r6000 ]]; then
  cp scripts/direction2_r6000.sbatch "$D2_OUTPUT/submission_script.sbatch"
fi
D2_STAGE=starting
write_status() {
  printf '{"stage":"%s","exit_code":%s,"job_id":"%s","updated_utc":"%s"}\n' \
    "$D2_STAGE" "$1" "$SLURM_JOB_ID" "$(date -u +%FT%TZ)" > "$D2_OUTPUT/workflow_status.json.tmp"
  mv "$D2_OUTPUT/workflow_status.json.tmp" "$D2_OUTPUT/workflow_status.json"
}
trap 'D2_EXIT=$?; write_status "$D2_EXIT"' EXIT

# Separate validation process exits and releases its one parameter set first.
# A failed capacity/load/generation check stops this script before collection.
D2_STAGE=loading_and_native_generation
write_status null
"$D2_PY" -u -m exp_code.positional_intervention.check_loading \
  --config configs/direction2_pilot.json \
  --minimum-gpu-mib "$D2_MIN_GPU_MIB" --output "$D2_OUTPUT/loading"

# New hardware/source identity requires a new session, preserving the A6000
# failure. This process loads once and still runs every original parity gate.
D2_STAGE=scientific_gates_and_pilot
write_status null
"$D2_PY" -u -m exp_code.positional_intervention.run_batch \
  --config configs/direction2_pilot.json --output "$D2_OUTPUT/batch"
D2_STAGE=complete
write_status 0
