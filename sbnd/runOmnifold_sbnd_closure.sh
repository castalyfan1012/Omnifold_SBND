#!/bin/bash

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

FILE_PATH="../FormattedData_SBND/"
mkdir -p weights_sbnd_closure plots_sbnd_closure

"${VENV_PYTHON}" run_sbnd.py \
    --config sbnd/config_omnifold_sbnd_closure.json \
    --file_path $FILE_PATH \
    --weights_folder ./weights_sbnd_closure/ \
    --plot_folder ./plots_sbnd_closure/ \
    --no_eff --verbose