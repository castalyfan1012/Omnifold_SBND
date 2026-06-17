#!/bin/bash
source setup.sh

python t2k.py \
    --config t2k/config_omnifold.json \
    --weights_folder weights_omnifold/ \
    --plot_folder plots_omnifold/ \
    --file_path ../FormattedData/ \
    --no_eff --verbose