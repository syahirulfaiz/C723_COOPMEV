#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=cosc723_coopmev
#SBATCH --time=12:00:00
##SBATCH --partition=prod           # adjust for your cluster, or pass: sbatch --partition=...
##SBATCH --account=YOUR_ACCOUNT     # adjust for your cluster, or pass: sbatch --account=...
#SBATCH --output=logs/cosc723.%j.out.txt
#SBATCH --error=logs/cosc723.%j.err.txt

# COSC723 CoopMEV-Lite on Slurm (one command pipeline: checker self-test -> offline plans ->
# frozen evaluation -> COSC723_* figures, driven by driver.sh).
#
#   mkdir -p logs                        # Slurm opens the log files before this script runs
#   export OPENROUTER_API_KEY='...'      # or put it (chmod 600) into ./.openrouter.env
#   sbatch --export=ALL slurm_driver.sh  # final campaign: 23 settings x seeds 40-49 x 5 conditions
#
# Allocation smoke test (no API key needed; dev fallback plans on one gas setting, ~5 minutes):
#   SMOKE=1 sbatch --export=ALL slurm_driver.sh

set -eo pipefail
module purge || true
module load miniconda/3 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ -n "${COSC723_ENV:-}" ]]; then
  conda activate "$COSC723_ENV"
else
  echo "note: set COSC723_ENV=/path/to/conda-env (or activate it before sbatch); continuing with the current python" >&2
fi
set -u

cd "$(dirname "$(readlink -f "$0")")"
PROJECT="$(pwd)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1

if [[ -f "$PROJECT/.openrouter.env" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT/.openrouter.env"
fi
export OPENROUTER_MODEL="${OPENROUTER_MODEL:-openai/gpt-oss-120b}"
export OPENROUTER_MAX_COMPLETION_TOKENS="${OPENROUTER_MAX_COMPLETION_TOKENS:-512}"
export OPENROUTER_REASONING_EFFORT="${OPENROUTER_REASONING_EFFORT:-low}"

python --version
echo "[slurm] project=$PROJECT smoke=${SMOKE:-0} openrouter_key=$([[ -n "${OPENROUTER_API_KEY:-}" ]] && echo SET || echo MISSING)"

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "[slurm] allocation smoke: checker self-test + dev-plan subset + figures"
  srun --ntasks=1 --cpus-per-task=1 python -u driver.py --self-test
  srun --ntasks=1 --cpus-per-task=1 python -u driver.py \
    --conditions PPO_no_plan B2 Proposed --seeds 40 41 \
    --only-dimensions gas --max-scenarios 1 --allow-dev-plans
  srun --ntasks=1 --cpus-per-task=1 python -u chart.py \
    --input results/raw_results.csv --output-dir results/figures
  echo "[slurm] smoke complete"
else
  if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "[slurm] warning: OPENROUTER_API_KEY is unset; cached OpenRouter plans are reused and missing plans abort the run." >&2
  fi
  srun --ntasks=1 --cpus-per-task=1 bash driver.sh "$PROJECT"
  echo "[slurm] full campaign complete: results/raw_results.csv and results/figures/COSC723_*.png"
fi
conda deactivate || true
