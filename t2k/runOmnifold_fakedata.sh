#!/bin/bash
source setup.sh

FILE_PATH="../FormattedData/"
mkdir -p weights_fakedata plots_fakedata

python t2k.py \
    --config t2k/config_omnifold_fakedata.json \
    --file_path $FILE_PATH \
    --plot_folder ./plots_fakedata/ \
    --weights_folder ./weights_fakedata/ \
    --no_eff --verbose