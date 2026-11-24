#!/bin/bash
#SBATCH --job-name=direction2-r6000
#SBATCH --account=edu
#SBATCH --partition=burst
#SBATCH --qos=burst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:R6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=72G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/insomnia001/depts/edu/users/wt2358/projects/allm-dgsc/outputs/direction2-r6000-%j.out
#SBATCH --error=/insomnia001/depts/edu/users/wt2358/projects/allm-dgsc/outputs/direction2-r6000-%j.err

set -euo pipefail
cd /insomnia001/depts/edu/users/wt2358/projects/allm-dgsc
# Slurm's R6000 is RTX PRO 6000 on ins046/047. Verify actual GPU capacity and
# CUDA compatibility at runtime; no model-name shortcut can bypass validation.
export D2_RUN_LABEL=r6000
export D2_MIN_GPU_MIB=90000
# Bash ignores the shared script's H100 SBATCH comments; only the directives
# above determine this allocation. It runs loading + native generation, then
# all original scientific gates and the frozen six-item pilot automatically.
exec bash scripts/direction2_h100.sbatch
