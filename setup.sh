#!/bin/bash
# ============================================================================
# setup.sh — OmniFold environment setup for FNAL machines
#
# Usage:
#   First time:  source setup.sh --install
#   Every time:  source setup.sh
#
# Works on both EAF (AlmaLinux 9) and GPVM (SL7).
# Creates a Python 3.9+ venv with TensorFlow CPU, numpy, etc.
# ============================================================================

VENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/venv_omnifold"

# ── Helper: detect platform ─────────────────────────────────────────────────
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

# ── Install mode ─────────────────────────────────────────────────────────────
if [[ "$1" == "--install" ]]; then
    echo "=== OmniFold Setup: Installing ($PLATFORM) ==="

    # Find python3
    PY3=$(command -v python3 2>/dev/null)
    if [[ -z "$PY3" ]]; then
        echo "ERROR: python3 not found. On GPVM, try: setup python v3_9_2"
        return 1
    fi
    PY_VER=$($PY3 --version 2>&1)
    echo "Using: $PY3 ($PY_VER)"

    # Create venv
    if [[ -d "$VENV_DIR" ]]; then
        echo "Removing existing venv..."
        rm -rf "$VENV_DIR"
    fi
    $PY3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    pip install --upgrade pip setuptools wheel 2>/dev/null

    # Core packages
    pip install \
        numpy scipy matplotlib scikit-learn pandas \
        tensorflow-cpu==2.15.0 \
        pyyaml tqdm h5py tables

    # SL7-specific: urllib3 v2 requires OpenSSL 1.1.1+, SL7 has 1.0.2
    if [[ "$PLATFORM" == "SL7" ]]; then
        echo "SL7 detected: pinning urllib3<2 for OpenSSL compatibility"
        pip install "urllib3<2"
    fi

    # Optional: for T2K ROOT file processing
    pip install uproot awkward 2>/dev/null || true

    echo ""
    echo "=== Installation complete ==="
    echo "Venv location: $VENV_DIR"
    echo "Python: $(which python) ($(python --version 2>&1))"
    echo "TF version: $(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null)"
    echo ""
    echo "Next time, just run: source setup.sh"
    return 0
fi

# ── Activate mode (default) ──────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: venv not found at $VENV_DIR"
    echo "Run first: source setup.sh --install"
    return 1
fi

source "$VENV_DIR/bin/activate"
echo "OmniFold env activated ($PLATFORM) — $(python --version 2>&1)"