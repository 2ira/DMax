#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSET_ROOT="${ASSET_ROOT:-${ROOT_DIR}/research/phase1/assets}"
MODEL_ROOT="${MODEL_ROOT:-${ASSET_ROOT}/models}"
DATA_ROOT="${DATA_ROOT:-${ASSET_ROOT}/datasets}"

DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"
PREPARE_SUBSETS="${PREPARE_SUBSETS:-1}"
DOWNLOAD_DREAM_REFERENCES="${DOWNLOAD_DREAM_REFERENCES:-0}"
MODEL_DOWNLOAD_MAX_WORKERS="${MODEL_DOWNLOAD_MAX_WORKERS:-8}"
MODEL_REPOS="${MODEL_REPOS:-inclusionAI/LLaDA2.0-mini Zigeng/DMax-Math-16B Zigeng/DMax-Coder-16B}"

echo "[phase1] repo root: ${ROOT_DIR}"
echo "[phase1] asset root: ${ASSET_ROOT}"

mkdir -p "${MODEL_ROOT}" "${DATA_ROOT}"

download_model() {
  local repo_id="$1"
  echo "[phase1] downloading model: ${repo_id}"
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
  export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1800}"
  export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-1800}"
  python "${ROOT_DIR}/dInfer/evaluations/download_hf_model.py" \
    --repo_id "${repo_id}" \
    --local_dir "${MODEL_ROOT}" \
    --max_workers "${MODEL_DOWNLOAD_MAX_WORKERS}"
}

if [[ "${DOWNLOAD_MODELS}" == "1" ]]; then
  for repo_id in ${MODEL_REPOS}; do
    download_model "${repo_id}"
  done

  if [[ "${DOWNLOAD_DREAM_REFERENCES}" == "1" ]]; then
    echo "[phase1] Dream models are optional second-phase references."
    echo "[phase1] Add the public repo IDs you want to compare once you confirm the exact checkpoints."
  fi
else
  echo "[phase1] skipping model downloads"
fi

if [[ "${PREPARE_SUBSETS}" == "1" ]]; then
  echo "[phase1] preparing deterministic dataset subsets"
  python "${ROOT_DIR}/research/phase1/prepare_phase1_subsets.py" \
    --output-root "${DATA_ROOT}" \
    --seed 42
else
  echo "[phase1] skipping subset preparation"
fi

echo "[phase1] done"
