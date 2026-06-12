#!/bin/bash
export CUDA_VISIBLE_DEVICES=""
python t2k.py --config config_omnifold.json --weights_folder weights_omnifold/ --file_path ../FormattedData/ --no_eff --verbose