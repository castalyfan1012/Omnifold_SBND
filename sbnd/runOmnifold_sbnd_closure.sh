#!/bin/bash
# Run from OmnifoldT2K/ directory
source setup.sh

FILE_PATH="../FormattedData_SBND/"
mkdir -p weights_sbnd_closure plots_sbnd_closure

python t2k.py \
    --config sbnd/config_omnifold_sbnd_closure.json \
    --file_path $FILE_PATH \
    --weights_folder ./weights_sbnd_closure/ \
    --plot_folder ./plots_sbnd_closure/ \
    --no_eff --verbose