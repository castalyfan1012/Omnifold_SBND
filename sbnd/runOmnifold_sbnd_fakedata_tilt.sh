#!/bin/bash
# Run from OmnifoldT2K/ directory
source setup.sh

TAG="tilt_alpha0.5"
FILE_PATH="../FormattedData_SBND/"

mkdir -p "weights_sbnd_fakedata_${TAG}" "plots_sbnd_fakedata_${TAG}"

python t2k.py \
    --config "sbnd/config_omnifold_sbnd_fakedata_tilt.json" \
    --file_path "$FILE_PATH" \
    --weights_folder "./weights_sbnd_fakedata_${TAG}/" \
    --plot_folder "./plots_sbnd_fakedata_${TAG}/" \
    --no_eff --verbose