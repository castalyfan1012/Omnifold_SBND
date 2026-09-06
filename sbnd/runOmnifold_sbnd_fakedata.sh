#!/bin/bash
# Usage (run from Omnifold_SBND/ directory):
#   nohup bash sbnd/runOmnifold_sbnd_fakedata.sh --var true_p --alpha 0.3        > fdt_p_03.log 2>&1 &
#   nohup bash sbnd/runOmnifold_sbnd_fakedata.sh --var true_costheta --alpha 0.3 > fdt_costheta_03.log 2>&1 &
#   nohup bash sbnd/runOmnifold_sbnd_fakedata.sh --var both --alpha 0.3          > fdt_both_03.log 2>&1 &
#   nohup bash sbnd/runOmnifold_sbnd_fakedata.sh --var true_p --alpha 0.3 --niter 20 > fdt_p_03_n20.log 2>&1 &
#
# The tag is derived from --var and --alpha automatically:
#   --var true_p        --alpha 0.3  =>  tag = tilt_p_alpha0.3
#   --var true_costheta --alpha 0.3  =>  tag = tilt_costheta_alpha0.3
#   --var both          --alpha 0.3  =>  tag = tilt_both_alpha0.3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="${REPO_DIR}/venv_omnifold/bin/python3"

if [[ ! -f "${VENV_PYTHON}" ]]; then
    echo "ERROR: venv python not found at ${VENV_PYTHON}"
    echo "Run first: source setup.sh --install"
    exit 1
fi

# ── Parse arguments ──────────────────────────────────────────────────────────
VAR="true_p"
ALPHA="0.3"
NITER="10"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --var)    VAR="$2";   shift 2 ;;
        --alpha)  ALPHA="$2"; shift 2 ;;
        --niter)  NITER="$2"; shift 2 ;;
        *)        echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Derive tag from --var and --alpha ────────────────────────────────────────
case "${VAR}" in
    true_p)        TAG="tilt_p_alpha${ALPHA}" ;;
    true_costheta) TAG="tilt_costheta_alpha${ALPHA}" ;;
    both)          TAG="tilt_both_alpha${ALPHA}" ;;
    *)             echo "ERROR: --var must be true_p, true_costheta, or both"; exit 1 ;;
esac

DATA_DIR="../FormattedData_SBND/"
WEIGHTS_DIR="weights_sbnd_fakedata_${TAG}/"

echo "Using python: ${VENV_PYTHON}"
echo "Python version: $("${VENV_PYTHON}" --version 2>&1)"
echo "TF version: $("${VENV_PYTHON}" -c 'import tensorflow as tf; print(tf.__version__)')"

echo ""
echo "=== OmniFold fake-data run ==="
echo "  --var:        ${VAR}"
echo "  --alpha:      ${ALPHA}"
echo "  TAG:          ${TAG}"
echo "  NITER:        ${NITER}"
echo "  WEIGHTS_DIR:  ${WEIGHTS_DIR}"
echo "  DATA_DIR:     ${DATA_DIR}"

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
echo "  ${VENV_PYTHON} sbnd/MakePlots.py validation --var ${VAR} --alpha ${ALPHA}"
echo "  ${VENV_PYTHON} sbnd/MakePlots.py paper --var both --tag ${TAG}"