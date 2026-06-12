#!/bin/bash
source test/bin/activate

FILE_PATH="../FormattedData/"   # same as the nominal runOmnifold.sh

mkdir -p weights_fakedata plots_fakedata

python t2k.py \
    --config config_omnifold_fakedata.json \
    --file_path $FILE_PATH \
    --plot_folder ./plots_fakedata/ \
    --weights_folder ./weights_fakedata/ \
    --no_eff --verbose