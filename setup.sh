#!/bin/bash
# ============================================================================
# setup.sh — OmniFold environment setup for FNAL machines
#
# Usage:
#   First time:  source setup.sh --install
#   Every time:  source setup.sh
#
# After sourcing, `python3` will resolve to the venv python (even on EAF
# where conda normally overrides PATH).
# ============================================================================

VENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/venv_omnifold"
export VENV_PYTHON="${VENV_DIR}/bin/python3"

_detect_platform() {
    if [[ -f /etc/redhat-release ]]; then
        if grep -q "Scientific Linux" /etc/redhat-release 2>/dev/null; then
            echo "SL7"
        else
            echo "EAF"
        fi
    else
        echo "UNKNOWN"
    fi
}
PLATFORM=$(_detect_platform)

if [[ "$1" == "--install" ]]; then
    echo "=== OmniFold Setup: Installing ($PLATFORM) ==="

    # Use /usr/bin/python3 to avoid conda contamination
    PY3=""
    for candidate in /usr/bin/python3 /usr/local/bin/python3 $(command -v python3 2>/dev/null); do
        if [[ -x "$candidate" ]]; then PY3="$candidate"; break; fi
    done
    if [[ -z "$PY3" ]]; then echo "ERROR: python3 not found."; return 1; fi
    echo "Using base python: $PY3 ($($PY3 --version 2>&1))"

    [[ -d "$VENV_DIR" ]] && rm -rf "$VENV_DIR"
    $PY3 -m venv "$VENV_DIR"

    VENV_PIP="${VENV_DIR}/bin/pip"
    "${VENV_PIP}" install --upgrade pip setuptools wheel
    "${VENV_PIP}" install \
        numpy scipy matplotlib scikit-learn pandas \
        tensorflow-cpu==2.15.0 \
        pyyaml tqdm h5py tables
    [[ "$PLATFORM" == "SL7" ]] && "${VENV_PIP}" install "urllib3<2"
    "${VENV_PIP}" install uproot awkward 2>/dev/null || true

    echo ""
    TF_VER=$("${VENV_PYTHON}" -c 'import tensorflow as tf; print(tf.__version__)' 2>&1)
    if [[ "$TF_VER" == 2* ]]; then
        echo "TF OK: ${TF_VER}"
    else
        echo "ERROR: TF import failed: ${TF_VER}"; return 1
    fi
    echo ""
    echo "=== Installation complete ==="
    echo "Next: source setup.sh"
    return 0
fi

# ── Activate mode (default) ──────────────────────────────────────────────────
if [[ ! -f "${VENV_PYTHON}" ]]; then
    echo "ERROR: venv not found. Run: source setup.sh --install"
    return 1
fi

source "${VENV_DIR}/bin/activate"

# Force venv bin FIRST in PATH, ahead of conda
export PATH="${VENV_DIR}/bin:${PATH}"
hash -r  # clear bash command cache

TF_VER=$("${VENV_PYTHON}" -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null)
echo "OmniFold env ($PLATFORM) — python3 → $(which python3)"
echo "TF: ${TF_VER:-NOT FOUND}"