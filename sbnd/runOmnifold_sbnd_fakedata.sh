#!/bin/bash
# Usage:
#   bash sbnd/runOmnifold_sbnd_fakedata.sh                  # default: tilt_alpha0.5
#   bash sbnd/runOmnifold_sbnd_fakedata.sh tilt_alpha0.3     # custom tag
#   bash sbnd/runOmnifold_sbnd_fakedata.sh tilt_alpha0.3 20  # custom tag + NITER

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

# Write config
CONFIG="sbnd/config_omnifold_sbnd_fakedata_${TAG}.json"
cat > "${CONFIG}" << EOF
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
EOF

echo "  Config written: ${CONFIG}"
echo ""

python3 t2k.py \
    --config "${CONFIG}" \
    --file_path "${DATA_DIR}" \
    --weights_folder "${WEIGHTS_DIR}" \
    --no_eff \
    --verbose

echo ""
echo "=== Done. Next steps: ==="
echo "  python3 sbnd/MakePlots.py validation --tag ${TAG}"
echo "  python3 sbnd/MakePlots.py paper --var both --tag ${TAG}"