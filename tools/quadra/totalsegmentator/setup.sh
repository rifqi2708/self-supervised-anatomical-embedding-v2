#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.env"
fi

WORK_ROOT="${QUADRA_WORK_ROOT:-/workspace/quadra-totalsegmentator}"
DATASET_URL="${QUADRA_DATASET_URL:-}"
DEMOGRAPHICS_URL="${QUADRA_DEMOGRAPHICS_URL:-}"
CURRENT_STAGE_DIR=""

cleanup_stage() {
  if [[ -n "${CURRENT_STAGE_DIR}" && "${CURRENT_STAGE_DIR}" == "${WORK_ROOT}/.staging/"* ]]; then
    rm -rf -- "${CURRENT_STAGE_DIR}"
  fi
}
trap cleanup_stage EXIT

usage() {
  echo "Usage: $0 [--work-root PATH] [--dataset-url URL] [--demographics-url URL]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-root)
      WORK_ROOT="$2"
      shift 2
      ;;
    --dataset-url)
      DATASET_URL="$2"
      shift 2
      ;;
    --demographics-url)
      DEMOGRAPHICS_URL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p \
  "${WORK_ROOT}/data" \
  "${WORK_ROOT}/metadata" \
  "${WORK_ROOT}/outputs" \
  "${WORK_ROOT}/model-cache" \
  "${WORK_ROOT}/logs" \
  "${WORK_ROOT}/.staging"

if [[ ! -d "${WORK_ROOT}" || ! -w "${WORK_ROOT}" ]]; then
  echo "Persistent work root is not writable: ${WORK_ROOT}" >&2
  exit 2
fi

VENV_DIR="${WORK_ROOT}/venv"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install \
  --requirement "${SCRIPT_DIR}/requirements-runpod.txt"

export TOTALSEG_WEIGHTS_PATH="${WORK_ROOT}/model-cache"

download_dataset() {
  local destination="${WORK_ROOT}/data/QUADRA_HC_WB"
  if [[ -f "${destination}/QUADRA_HC_021/test_CT-AC.nii.gz" ]]; then
    echo "Dataset already present: ${destination}"
    return
  fi
  if [[ -z "${DATASET_URL}" ]]; then
    echo "Dataset missing and no --dataset-url or QUADRA_DATASET_URL was provided" >&2
    exit 2
  fi

  local stage_dir
  stage_dir="$(mktemp -d "${WORK_ROOT}/.staging/dataset.XXXXXX")"
  CURRENT_STAGE_DIR="${stage_dir}"
  "${VENV_DIR}/bin/gdown" --fuzzy "${DATASET_URL}" --output "${stage_dir}/archive"
  mkdir -p "${stage_dir}/extract"
  if command -v unzip >/dev/null 2>&1 \
    && unzip -tqq "${stage_dir}/archive" >/dev/null 2>&1; then
    unzip -q "${stage_dir}/archive" -d "${stage_dir}/extract"
  elif "${VENV_DIR}/bin/python" -m zipfile -t \
    "${stage_dir}/archive" >/dev/null 2>&1; then
    # The validated RunPod PyTorch image does not include `unzip`. Python's
    # standard-library ZIP64 support keeps clean-Pod setup self-contained.
    "${VENV_DIR}/bin/python" -m zipfile -e \
      "${stage_dir}/archive" "${stage_dir}/extract"
  elif tar -tf "${stage_dir}/archive" >/dev/null 2>&1; then
    tar -xf "${stage_dir}/archive" -C "${stage_dir}/extract"
  else
    echo "Downloaded dataset is not a supported zip or tar archive" >&2
    exit 2
  fi

  local discovered
  discovered="$(find "${stage_dir}/extract" -type f -path '*/QUADRA_HC_021/test_CT-AC.nii.gz' -print -quit)"
  if [[ -z "${discovered}" ]]; then
    echo "Archive does not contain QUADRA_HC_021/test_CT-AC.nii.gz" >&2
    exit 2
  fi
  local dataset_root
  dataset_root="$(dirname -- "$(dirname -- "${discovered}")")"
  local subject_number
  local session
  for subject_number in $(seq -w 21 48); do
    for session in test retest; do
      if [[ ! -f "${dataset_root}/QUADRA_HC_0${subject_number}/${session}_CT-AC.nii.gz" ]]; then
        echo "Archive is missing QUADRA_HC_0${subject_number}/${session}_CT-AC.nii.gz" >&2
        exit 2
      fi
    done
  done
  if [[ -e "${destination}" ]]; then
    echo "Refusing to replace existing incomplete dataset: ${destination}" >&2
    exit 2
  fi
  mv -- "${dataset_root}" "${destination}"
  rm -rf -- "${stage_dir}"
  CURRENT_STAGE_DIR=""
}

download_demographics() {
  local destination="${WORK_ROOT}/metadata/Demographics (All).xlsx"
  if [[ -s "${destination}" ]]; then
    echo "Demographics already present: ${destination}"
    return
  fi
  if [[ -z "${DEMOGRAPHICS_URL}" ]]; then
    echo "Demographics missing and no --demographics-url or QUADRA_DEMOGRAPHICS_URL was provided" >&2
    exit 2
  fi

  local stage_dir
  stage_dir="$(mktemp -d "${WORK_ROOT}/.staging/demographics.XXXXXX")"
  CURRENT_STAGE_DIR="${stage_dir}"
  "${VENV_DIR}/bin/gdown" --fuzzy "${DEMOGRAPHICS_URL}" \
    --output "${stage_dir}/Demographics (All).xlsx"
  if [[ ! -s "${stage_dir}/Demographics (All).xlsx" ]]; then
    echo "Demographics download is empty" >&2
    exit 2
  fi
  "${VENV_DIR}/bin/python" -c \
    'import sys; from openpyxl import load_workbook; wb = load_workbook(sys.argv[1], read_only=True); assert "Demographics (All)" in wb.sheetnames; wb.close()' \
    "${stage_dir}/Demographics (All).xlsx"
  mv -- "${stage_dir}/Demographics (All).xlsx" "${destination}"
  rm -rf -- "${stage_dir}"
  CURRENT_STAGE_DIR=""
}

download_dataset
download_demographics

"${VENV_DIR}/bin/python" -m tools.quadra.totalsegmentator prepare \
  --dataset-root "${WORK_ROOT}/data/QUADRA_HC_WB" \
  --demographics "${WORK_ROOT}/metadata/Demographics (All).xlsx" \
  --output "${WORK_ROOT}/metadata/cohort_manifest.json"

cat <<EOF
Setup complete.
Activate with:
  source "${VENV_DIR}/bin/activate"
  export TOTALSEG_WEIGHTS_PATH="${WORK_ROOT}/model-cache"

Stage 2 preflight:
  python -m tools.quadra.totalsegmentator preflight \
    --dataset-root "${WORK_ROOT}/data/QUADRA_HC_WB" \
    --output-root "${WORK_ROOT}/outputs"
EOF
