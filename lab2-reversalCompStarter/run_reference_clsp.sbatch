#!/bin/bash -l
#SBATCH --account=a100acct
#SBATCH --partition=gpu-a100
#SBATCH --job-name=reversal-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.log
#SBATCH --error=slurm-%x-%j.log

# Submits the full reference pipeline (SFT -> eval -> GRPO -> eval) as one
# GPU job. Override knobs via environment variables at submit time, e.g.:
#   GRPO_STEPS=600 sbatch run_reference_clsp.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ ! -f train_sft.py || ! -f train_grpo.py || ! -f evaluate.py ]]; then
  echo "Submit this script from the reversalCompStarter repo root."
  exit 2
fi

COMP_ENV="${COMP_ENV:-reversal-comp}"
RUN_NAME="${RUN_NAME:-ref_${SLURM_JOB_ID:-manual}}"
SFT_OUT="${SFT_OUT:-models/sft-lora-${RUN_NAME}}"
GRPO_OUT="${GRPO_OUT:-models/grpo-lora-${RUN_NAME}}"
GRPO_STEPS="${GRPO_STEPS:-300}"

mkdir -p models predictions

echo "=== Slurm ==="
echo "job_id=${SLURM_JOB_ID:-}"
echo "submit_dir=${SLURM_SUBMIT_DIR:-$PWD}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "partition=${SLURM_JOB_PARTITION:-}"
echo "job_gpus=${SLURM_JOB_GPUS:-${SLURM_GPUS:-unset}}"
echo "note=CLSP gpu-a100 jobs require account a100acct; GPU work runs inside srun below."
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job "${SLURM_JOB_ID}" | grep -E 'JobState|NodeList|ReqTRES|AllocTRES|TresPerNode|CpusPerTres' || true
fi
nvidia-smi

if command -v module >/dev/null 2>&1; then
  module load conda/25 2>/dev/null || module load conda/24 2>/dev/null || module load conda/23.2.1 2>/dev/null || true
fi

CONDA_INSTALL_DIR="${CONDA_INSTALL_DIR:-$HOME/miniforge3}"

install_conda() {
  local arch installer tmp_url

  if [[ -x "$CONDA_INSTALL_DIR/bin/conda" ]]; then
    return 0
  fi
  if [[ -e "$CONDA_INSTALL_DIR" ]]; then
    echo "CONDA_INSTALL_DIR exists but does not contain conda: $CONDA_INSTALL_DIR"
    echo "Set CONDA_INSTALL_DIR to a different path or remove the incomplete install."
    return 1
  fi

  case "$(uname -m)" in
    x86_64|amd64) arch="x86_64" ;;
    aarch64|arm64) arch="aarch64" ;;
    *)
      echo "Unsupported architecture for automatic Miniforge install: $(uname -m)"
      return 1
      ;;
  esac

  installer="${TMPDIR:-/tmp}/miniforge3-${USER:-user}-$$.sh"
  tmp_url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${arch}.sh"

  echo "Installing Miniforge conda to $CONDA_INSTALL_DIR"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$tmp_url" -o "$installer"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$installer" "$tmp_url"
  else
    echo "Need curl or wget to install conda automatically."
    return 1
  fi

  bash "$installer" -b -p "$CONDA_INSTALL_DIR"
  rm -f "$installer"
}

if ! command -v conda >/dev/null 2>&1; then
  for conda_sh in \
    "$CONDA_INSTALL_DIR/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"
  do
    if [[ -f "$conda_sh" ]]; then
      # shellcheck source=/dev/null
      source "$conda_sh"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  install_conda
  # shellcheck source=/dev/null
  source "$CONDA_INSTALL_DIR/etc/profile.d/conda.sh"
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

resolve_pybin() {
  # parse the env prefix from `conda env list` — `conda run` is flaky in
  # batch shells, and an absolute path needs no activation at all
  local prefix
  prefix="$(conda env list | awk -v e="$COMP_ENV" '$1 == e {print $NF}')"
  if [[ -n "$prefix" && -x "$prefix/bin/python" ]]; then
    echo "$prefix/bin/python"
  fi
}

if ! conda env list | awk 'NF {print $1}' | grep -Fxq "$COMP_ENV"; then
  echo "Creating conda env: $COMP_ENV"
  conda create -y -n "$COMP_ENV" python=3.12
fi

# the env can exist but be unusable (created bare, no python of its own —
# interactive shells mask this by falling through to base's python)
PYBIN="$(resolve_pybin || true)"
if [[ -z "$PYBIN" || ! -x "$PYBIN" ]]; then
  echo "Env '$COMP_ENV' has no usable python — installing python 3.12 into it"
  conda install -y -n "$COMP_ENV" python=3.12
  PYBIN="$(resolve_pybin)"
fi
if [[ -z "$PYBIN" || ! -x "$PYBIN" ]]; then
  echo "Could not resolve python for conda env '$COMP_ENV' (got: '$PYBIN')"
  exit 4
fi
echo "env_python=$PYBIN"

if ! "$PYBIN" -c "import mlx_lm" >/dev/null 2>&1; then
  echo "Installing mlx-lm into $COMP_ENV"
  "$PYBIN" -m pip install --upgrade pip
  "$PYBIN" -m pip install "mlx-lm[cuda13]==0.31.2"
fi

export PYBIN RUN_NAME SFT_OUT GRPO_OUT GRPO_STEPS

echo "=== Run Pipeline In GPU Job Step ==="
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-12}" --gpus=1 bash -lc '
set -euo pipefail

echo "=== GPU Step ==="
echo "hostname=$(hostname)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "python=$PYBIN"
nvidia-smi

CUDA_ENV=()
if [[ "${CUDA_VISIBLE_DEVICES:-}" == "-1" ]]; then
  # Some Slurm setups expose ONLY the allocated GPU through the job-step
  # cgroup while leaving CUDA_VISIBLE_DEVICES at a readonly -1. In that case,
  # logical device 0 means "the GPU this job was given."
  echo "CUDA_VISIBLE_DEVICES is readonly -1 in this job step; using logical device 0 (the cgroup-allocated GPU)."
  CUDA_ENV=(env CUDA_VISIBLE_DEVICES=0)
fi

"${CUDA_ENV[@]}" "$PYBIN" - <<PY
import os
import mlx.core as mx
print("process_cuda_visible_devices=" + str(os.environ.get("CUDA_VISIBLE_DEVICES")), flush=True)
print("mlx_version=" + mx.__version__, flush=True)
print("mlx_device=" + str(mx.default_device()), flush=True)
x = (mx.ones((256, 256)) @ mx.ones((256, 256))).sum()
mx.eval(x)
if str(mx.default_device()) != "Device(gpu, 0)":
    raise SystemExit("MLX cannot see the GPU inside the Slurm job step.")
PY

echo "=== Train SFT ==="
"${CUDA_ENV[@]}" "$PYBIN" train_sft.py --out "$SFT_OUT"

echo "=== Evaluate SFT (greedy) ==="
"${CUDA_ENV[@]}" "$PYBIN" evaluate.py --adapter-path "$SFT_OUT" \
  --out "predictions/${RUN_NAME}_sft.jsonl"

echo "=== Train GRPO ==="
"${CUDA_ENV[@]}" "$PYBIN" train_grpo.py --adapter-path "$SFT_OUT" \
  --steps "$GRPO_STEPS" --out "$GRPO_OUT"

echo "=== Evaluate GRPO (on-policy, temp 0.7) ==="
"${CUDA_ENV[@]}" "$PYBIN" evaluate.py --adapter-path "$GRPO_OUT" --temp 0.7 \
  --out "predictions/${RUN_NAME}_grpo.jsonl"

echo "=== Outputs ==="
echo "sft_adapter=$SFT_OUT"
echo "grpo_adapter=$GRPO_OUT"
echo "predictions=predictions/${RUN_NAME}_sft.jsonl predictions/${RUN_NAME}_grpo.jsonl"
'
