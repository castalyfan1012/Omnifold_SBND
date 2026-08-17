#!/bin/bash
# Usage (run from Omnifold_SBND/ directory):
#   nohup bash sbnd/runOmnifold_sbnd_fakedata.sh tilt_alpha0.3     > fd03.log 2>&1 &
#   nohup bash sbnd/runOmnifold_sbnd_fakedata.sh tilt_alpha0.5     > fd05.log 2>&1 &
#   nohup bash sbnd/runOmnifold_sbnd_fakedata.sh tilt_alpha0.3 20  > fd03_n20.log 2>&1 &
#
# FIX: conda overrides PATH so python3/python resolve to /opt/conda, not the venv.
# Use the venv python by absolute path instead.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="${REPO_DIR}/venv_omnifold/bin/python3"

if [[ ! -f "${VENV_PYTHON}" ]]; then
    echo "ERROR: venv python not found at ${VENV_PYTHON}"
    echo "Run first: source setup.sh --install"
    exit 1
fi

echo "Using python: ${VENV_PYTHON}"
echo "Python version: $("${VENV_PYTHON}" --version 2>&1)"
echo "TF version: $("${VENV_PYTHON}" -c 'import tensorflow as tf; print(tf.__version__)')"

TAG="${1:-tilt_alpha0.5}"
NITER="${2:-10}"
DATA_DIR="../FormattedData_SBND/"
WEIGHTS_DIR="weights_sbnd_fakedata_${TAG}/"

echo "=== OmniFold fake-data run ==="
echo "  TAG:         ${TAG}"
echo "  NITER:       ${NITER}"
echo "  WEIGHTS_DIR: ${WEIGHTS_DIR}"
echo "  DATA_DIR:    ${DATA_DIR}"

mkdir -p "${WEIGHTS_DIR}"

CONFIG="sbnd/config_omnifold_sbnd_fakedata_${TAG}.json"
cat > "${CONFIG}" << JSONEOF
{
  "FILE_MC_RECO":        "mc_vals_reco.npy",
  "FILE_MC_GEN":         "mc_vals_truth.npy",
  "FILE_MC_FLAG_RECO":   "mc_pass_reco.npy",
  "FILE_MC_FLAG_GEN":    "mc_pass_truth.npy",
  "FILE_DATA_RECO":      "mc_vals_reco.npy",
  "FILE_DATA_FLAG_RECO": "mc_pass_reco.npy",
  "FILE_DATA_WEIGHT":    "data_weights_sbnd_fakedata_${TAG}.npy",
  "FILE_MC_RECO_WEIGHT": "mc_weights_reco.npy",
  "FILE_MC_GEN_WEIGHT":  "mc_weights_truth.npy",
  "NITER":    ${NITER},
  "NTRIAL":   3,
  "LR":       1e-3,
  "BATCH_SIZE": 512,
  "EPOCHS":   100,
  "NAME":     "sbnd_fakedata_${TAG}",
  "NPATIENCE": 10
}
JSONEOF

echo "  Config written: ${CONFIG}"
echo ""

"${VENV_PYTHON}" run_sbnd.py \
    --config "${CONFIG}" \
    --file_path "${DATA_DIR}" \
    --weights_folder "${WEIGHTS_DIR}" \
    --no_eff \
    --verbose

echo ""
echo "=== Done. Next steps: ==="
echo "  ${VENV_PYTHON} sbnd/MakePlots.py validation --tag ${TAG}"
echo "  ${VENV_PYTHON} sbnd/MakePlots.py paper --var both --tag ${TAG}"