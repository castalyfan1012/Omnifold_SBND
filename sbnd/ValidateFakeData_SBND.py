"""
ValidateFakeData_SBND.py

Validates OmniFold push weight recovery against injected distortion.

Usage:
    python3 ValidateFakeData_SBND.py --tag tilt_alpha0.5
    python3 ValidateFakeData_SBND.py --tag bnb_univ0
"""

import numpy as np
import matplotlib.pyplot as plt
import glob, re, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, required=True,
                    help='Tag matching the fake data run (e.g. tilt_alpha0.5, bnb_univ0)')
parser.add_argument('--data-dir', type=str, default='../FormattedData_SBND/')
parser.add_argument('--weights-dir', type=str, default=None,
                    help='Override weights directory (default: weights_sbnd_fakedata_{tag}/)')
parser.add_argument('--plot-dir', type=str, default=None)
flags = parser.parse_args()

DATA_DIR    = flags.data_dir
TAG         = flags.tag
WEIGHTS_DIR = flags.weights_dir or f'weights_sbnd_fakedata_{TAG}/'
PLOT_DIR    = flags.plot_dir or f'plots_sbnd_fakedata_{TAG}/'
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
truth_raw      = np.load(DATA_DIR + 'mc_vals_truth_NoNorm.npy')
mc_weights     = np.load(DATA_DIR + 'mc_weights_reco.npy')
injected_tilt  = np.load(DATA_DIR + f'truth_weights_sbnd_fakedata_{TAG}.npy')

true_ke       = truth_raw[:, 0]
true_costheta = truth_raw[:, 1]

# ── Load final iteration push weights ─────────────────────────────────────────
def iter_num(p):
    m = re.search(r'Iter(\d+)', p)
    return int(m.group(1)) if m else -1

push_files = sorted(glob.glob(WEIGHTS_DIR + 'Step2_Iter*_PushWeights.npy'), key=iter_num)
pull_files = sorted(glob.glob(WEIGHTS_DIR + 'Step1_Iter*_PullWeights.npy'), key=iter_num)

if not push_files:
    print(f"ERROR: No push weight files in {WEIGHTS_DIR}")
    exit(1)

print(f"Push file: {push_files[-1]}")
print(f"Pull file: {pull_files[-1]}")

push = np.load(push_files[-1])
pull = np.load(pull_files[-1])
push_mean = push if push.ndim == 1 else push.mean(axis=0)
pull_mean = pull if pull.ndim == 1 else pull.mean(axis=0)

print(f"Push:     mean={push_mean.mean():.4f}, std={push_mean.std():.4f}")
print(f"Injected: mean={injected_tilt.mean():.4f}, std={injected_tilt.std():.4f}")

# ── Binning helper ────────────────────────────────────────────────────────────
def binned_mean(x, w, bins):
    out = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.sum() > 0:
            out[i] = np.average(w[mask])
    return out


# ── Plot 1: Push weight recovery vs true_ke ──────────────────────────────────
bins_ke = np.linspace(0, np.percentile(true_ke, 99), 20)
centers_ke = 0.5 * (bins_ke[:-1] + bins_ke[1:])

push_binned_ke = binned_mean(true_ke, push_mean,     bins_ke)
tilt_binned_ke = binned_mean(true_ke, injected_tilt, bins_ke)

fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                          gridspec_kw={'height_ratios': [3, 1]})
axes[0].plot(centers_ke, tilt_binned_ke, 'b-o', label='Injected')
axes[0].plot(centers_ke, push_binned_ke, 'r-s', label='OmniFold push weights')
axes[0].axhline(1.0, color='gray', linestyle='--')
axes[0].set_ylabel('Weight')
axes[0].legend()
axes[0].set_title(f'SBND Push Weight Recovery: true_ke ({TAG})')

ratio_ke = push_binned_ke / np.where(tilt_binned_ke > 0, tilt_binned_ke, 1.0)
axes[1].plot(centers_ke, ratio_ke, 'k-o')
axes[1].axhline(1.0, color='gray', linestyle='--')
axes[1].fill_between(centers_ke, 0.8, 1.2, alpha=0.15, color='green', label='+/-20% band')
axes[1].set_xlabel('true_ke [MeV]')
axes[1].set_ylabel('Push / Injected')
axes[1].set_ylim(0.5, 1.5)
axes[1].legend()
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/recovery_true_ke.png', dpi=150)
print(f"Saved {PLOT_DIR}/recovery_true_ke.png")


# ── Plot 2: Push weight recovery vs true_costheta ────────────────────────────
bins_cos = np.linspace(-1, 1, 15)
centers_cos = 0.5 * (bins_cos[:-1] + bins_cos[1:])

push_binned_cos = binned_mean(true_costheta, push_mean,     bins_cos)
tilt_binned_cos = binned_mean(true_costheta, injected_tilt, bins_cos)

fig2, axes2 = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})
axes2[0].plot(centers_cos, tilt_binned_cos, 'b-o', label='Injected')
axes2[0].plot(centers_cos, push_binned_cos, 'r-s', label='OmniFold push weights')
axes2[0].axhline(1.0, color='gray', linestyle='--')
axes2[0].set_ylabel('Weight')
axes2[0].legend()
axes2[0].set_title(f'SBND Push Weight Recovery: true costheta ({TAG})')

ratio_cos = push_binned_cos / np.where(tilt_binned_cos > 0, tilt_binned_cos, 1.0)
axes2[1].plot(centers_cos, ratio_cos, 'k-o')
axes2[1].axhline(1.0, color='gray', linestyle='--')
axes2[1].fill_between(centers_cos, 0.8, 1.2, alpha=0.15, color='green', label='+/-20% band')
axes2[1].set_xlabel(r'true $\cos\theta$')
axes2[1].set_ylabel('Push / Injected')
axes2[1].set_ylim(0.5, 1.5)
axes2[1].legend()
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/recovery_true_costheta.png', dpi=150)
print(f"Saved {PLOT_DIR}/recovery_true_costheta.png")


# ── Plot 3: Unfolded distributions ──────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))

axes3[0].hist(true_ke, bins=bins_ke, weights=mc_weights,            alpha=0.5, label='Nominal MC')
axes3[0].hist(true_ke, bins=bins_ke, weights=mc_weights * injected_tilt, alpha=0.5, label='Fake Data (truth)')
axes3[0].hist(true_ke, bins=bins_ke, weights=mc_weights * push_mean,     alpha=0.5, label='OmniFold unfolded')
axes3[0].set_xlabel('true_ke [MeV]')
axes3[0].set_ylabel('Weighted events')
axes3[0].legend(fontsize=9)
axes3[0].set_title('Unfolded distribution: true_ke')

axes3[1].hist(true_costheta, bins=bins_cos, weights=mc_weights,            alpha=0.5, label='Nominal MC')
axes3[1].hist(true_costheta, bins=bins_cos, weights=mc_weights * injected_tilt, alpha=0.5, label='Fake Data (truth)')
axes3[1].hist(true_costheta, bins=bins_cos, weights=mc_weights * push_mean,     alpha=0.5, label='OmniFold unfolded')
axes3[1].set_xlabel(r'true $\cos\theta$')
axes3[1].set_ylabel('Weighted events')
axes3[1].legend(fontsize=9)
axes3[1].set_title(r'Unfolded distribution: true $\cos\theta$')

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/unfolded_distributions.png', dpi=150)
print(f"Saved {PLOT_DIR}/unfolded_distributions.png")


# ── Iteration convergence ────────────────────────────────────────────────────
print(f"\n=== Iteration convergence ===")
for f in sorted(push_files, key=iter_num):
    w = np.load(f)
    w = w if w.ndim == 1 else w.mean(axis=0)
    it = iter_num(f)
    print(f"  Iter {it:2d}: mean={w.mean():.4f}, std={w.std():.4f}")