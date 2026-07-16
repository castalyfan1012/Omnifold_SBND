"""
MakeFakeDataWeights_SBND.py

Creates fake data weights for SBND OmniFold validation.
Primary variables: true_p (col 0), true_costheta (col 1) — matches FormatData_SBND.py.

Modes:
  --mode tilt       Synthetic log(true_p) tilt (default)
  --mode universe   Pre-saved BNB universe weights

Usage:
    python3 sbnd/MakeFakeDataWeights_SBND.py
    python3 sbnd/MakeFakeDataWeights_SBND.py --mode universe --universe-file ...
    python3 sbnd/MakeFakeDataWeights_SBND.py --check-closure
"""

import numpy as np
import os, glob, re, argparse
import matplotlib.pyplot as plt

OUTPUT_DIR = '../FormattedData_SBND/'
PLOT_DIR   = 'plots_sbnd_fakedata/'
os.makedirs(PLOT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['tilt', 'universe'], default='tilt')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--universe-file', type=str, default=None)
parser.add_argument('--universe-idx', type=int, default=0)
parser.add_argument('--check-closure', action='store_true')
parser.add_argument('--closure-dir', type=str, default='weights_sbnd_closure/')
flags = parser.parse_args()


def check_closure(weights_dir):
    def iter_num(p):
        m = re.search(r'Iter(\d+)', p)
        return int(m.group(1)) if m else -1
    push_files = sorted(glob.glob(os.path.join(weights_dir, 'Step2_Iter*_PushWeights.npy')), key=iter_num)
    pull_files = sorted(glob.glob(os.path.join(weights_dir, 'Step1_Iter*_PullWeights.npy')), key=iter_num)
    if not push_files or not pull_files:
        print(f"ERROR: No weight files in {weights_dir}. Run closure test first.")
        return False
    push = np.load(push_files[-1])
    pull = np.load(pull_files[-1])
    print(f"=== Closure Test Check ({weights_dir}) ===")
    print(f"  Push: mean={push.mean():.4f}, std={push.std():.4f}, range=[{push.min():.4f}, {push.max():.4f}]")
    print(f"  Pull: mean={pull.mean():.4f}, std={pull.std():.4f}")
    passed = abs(push.mean() - 1.0) < 0.05 and push.std() < 0.2
    print("  CLOSURE TEST PASSED" if passed else "  WARNING: closure test looks off")
    return passed


if flags.check_closure:
    check_closure(flags.closure_dir)
    exit(0)

# ── Load ──────────────────────────────────────────────────────────────────────
mc_weights_reco = np.load(OUTPUT_DIR + 'mc_weights_reco.npy')
truth_raw       = np.load(OUTPUT_DIR + 'mc_vals_truth_NoNorm.npy')
n_events = len(mc_weights_reco)

print(f"Loaded {n_events:,} events from {OUTPUT_DIR}")
print(f"  mc_weights_reco: mean={mc_weights_reco.mean():.4f}")
print(f"  truth_raw shape: {truth_raw.shape}  (cols: [0]=true_p, [1]=true_costheta)")

if os.path.exists(flags.closure_dir):
    print()
    check_closure(flags.closure_dir)
    print()

# Columns set by FormatData_SBND.py: [0]=true_p, [1]=true_costheta
true_p        = truth_raw[:, 0]
true_costheta = truth_raw[:, 1]

# ── Tilt mode ─────────────────────────────────────────────────────────────────
if flags.mode == 'tilt':
    ALPHA = flags.alpha
    print(f"=== Synthetic Tilt Mode (alpha={ALPHA}, variable=true_p) ===")

    log_p      = np.log(true_p.clip(1.0, None))
    log_p_mean = np.mean(log_p)
    log_p_std  = np.std(log_p)

    tilt = 1.0 + ALPHA * (log_p - log_p_mean) / log_p_std
    tilt = np.clip(tilt, 0.0, None)
    tilt = tilt * (mc_weights_reco.sum() / (mc_weights_reco * tilt).sum())

    data_weights = mc_weights_reco * tilt
    tag = f'tilt_alpha{ALPHA}'

    np.save(OUTPUT_DIR + f'data_weights_sbnd_fakedata_{tag}.npy', data_weights)
    np.save(OUTPUT_DIR + f'truth_weights_sbnd_fakedata_{tag}.npy', tilt)

    print(f"  Tilt range: [{tilt.min():.3f}, {tilt.max():.3f}]")
    print(f"  Data weights: mean={data_weights.mean():.4f}, std={data_weights.std():.4f}")

    # Diagnostic plots: p, costheta, and tilt function
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    bins_p = np.linspace(0, np.percentile(true_p, 99), 25)
    axes[0].hist(true_p, bins=bins_p, weights=mc_weights_reco, alpha=0.6, label='Nominal MC')
    axes[0].hist(true_p, bins=bins_p, weights=data_weights, alpha=0.6, label='Fake Data')
    axes[0].set_xlabel('true_p [MeV/c]')
    axes[0].set_ylabel('Weighted events')
    axes[0].legend()
    axes[0].set_title('Injected distortion: true_p')

    bins_cos = np.linspace(-1, 1, 20)
    axes[1].hist(true_costheta, bins=bins_cos, weights=mc_weights_reco, alpha=0.6, label='Nominal MC')
    axes[1].hist(true_costheta, bins=bins_cos, weights=data_weights, alpha=0.6, label='Fake Data')
    axes[1].set_xlabel(r'true $\cos\theta$')
    axes[1].set_ylabel('Weighted events')
    axes[1].legend()
    axes[1].set_title(r'Projected: true $\cos\theta$')

    axes[2].scatter(true_p[::5], tilt[::5], s=3, alpha=0.3)
    axes[2].axhline(1.0, color='r', linestyle='--')
    axes[2].set_xlabel('true_p [MeV/c]')
    axes[2].set_ylabel('Tilt weight')
    axes[2].set_title(f'Tilt function (alpha={ALPHA})')

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/sbnd_fakedata_injected_{tag}.png', dpi=150)
    print(f"  Plot: {PLOT_DIR}/sbnd_fakedata_injected_{tag}.png")

# ── Universe mode ─────────────────────────────────────────────────────────────
elif flags.mode == 'universe':
    print("=== BNB Universe Mode ===")
    if flags.universe_file is None:
        print("ERROR: --universe-file required. Re-export from notebook matching current FormatData_SBND.py.")
        exit(1)

    uni_all = np.load(flags.universe_file)
    if uni_all.ndim == 1:
        uni_all = uni_all[:, np.newaxis]

    assert uni_all.shape[0] == n_events, \
        f"Shape mismatch: universe file has {uni_all.shape[0]} events, expected {n_events}.\n" \
        f"Re-export from notebook using RECO_VARS = ['reco_p', 'reco_costheta']."

    idx = flags.universe_idx
    uni_vals = np.clip(uni_all[:, idx], 0.0, 10.0)
    data_weights = mc_weights_reco * uni_vals
    data_weights = data_weights * (mc_weights_reco.sum() / data_weights.sum())

    tag = f'bnb_univ{idx}'
    np.save(OUTPUT_DIR + f'data_weights_sbnd_fakedata_{tag}.npy', data_weights)
    np.save(OUTPUT_DIR + f'truth_weights_sbnd_fakedata_{tag}.npy', uni_vals)

    print(f"  Universe {idx}: mean={uni_vals.mean():.4f}, std={uni_vals.std():.4f}")
    print(f"  Data weights: mean={data_weights.mean():.4f}, std={data_weights.std():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins_p = np.linspace(0, np.percentile(true_p, 99), 25)
    axes[0].hist(true_p, bins=bins_p, weights=mc_weights_reco, alpha=0.6, label='Nominal MC')
    axes[0].hist(true_p, bins=bins_p, weights=data_weights, alpha=0.6, label=f'Fake Data (BNB univ {idx})')
    axes[0].set_xlabel('true_p [MeV/c]')
    axes[0].set_ylabel('Weighted events')
    axes[0].legend()
    axes[0].set_title('Injected distortion')

    axes[1].scatter(true_p[::5], uni_vals[::5], s=3, alpha=0.3)
    axes[1].axhline(1.0, color='r', linestyle='--')
    axes[1].set_xlabel('true_p [MeV/c]')
    axes[1].set_ylabel('Universe weight')
    axes[1].set_title(f'BNB universe {idx}')

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/sbnd_fakedata_injected_{tag}.png', dpi=150)
    print(f"  Plot: {PLOT_DIR}/sbnd_fakedata_injected_{tag}.png")