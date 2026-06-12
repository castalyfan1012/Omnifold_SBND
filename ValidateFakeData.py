import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os

# --- Config ---
DATA_DIR    = '../FormattedData/'
WEIGHTS_DIR = 'weights_fakedata/'
PLOT_DIR = 'plots_fakedata/'
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Load truth-level data ---
truth_nonorm    = np.load(DATA_DIR + 'mc_vals_truth_NoNorm.npy')
mc_pass_truth   = np.load(DATA_DIR + 'mc_pass_truth.npy')
mc_weights_reco = np.load(DATA_DIR + 'mc_weights_reco.npy')
injected_tilt   = np.load(DATA_DIR + 'truth_weights_fakedata_pmu_tilt.npy')

pmu_true = truth_nonorm[mc_pass_truth, 0]   # MeV/c, matched truth events
log_pmu  = np.log(pmu_true)

# --- Load final iteration weights (sort numerically, not lexicographically) ---
def iter_num(path):
    match = re.search(r'Iter(\d+)', path)
    return int(match.group(1)) if match else -1

push_files = sorted(glob.glob(WEIGHTS_DIR + 'Step2_Iter*_t2k_FakeData_pmuTilt_PushWeights.npy'), key=iter_num)
pull_files = sorted(glob.glob(WEIGHTS_DIR + 'Step1_Iter*_t2k_FakeData_pmuTilt_PullWeights.npy'), key=iter_num)
print("Using push file:", push_files[-1])
print("Using pull file:", pull_files[-1])

push = np.load(push_files[-1])
pull = np.load(pull_files[-1])

# Weights are 1D (N_events,) — no trial averaging needed
push_mean = push if push.ndim == 1 else push.mean(axis=0)
pull_mean = pull if pull.ndim == 1 else pull.mean(axis=0)

print(f"Push weights shape: {push.shape}, mean={push_mean.mean():.4f}, std={push_mean.std():.4f}")
print(f"Pull weights shape: {pull.shape}, mean={pull_mean.mean():.4f}, std={pull_mean.std():.4f}")
print(f"Injected tilt:      mean={injected_tilt.mean():.4f}, std={injected_tilt.std():.4f}")

# --- Binning in log(pmu) space ---
bins    = np.linspace(log_pmu.min(), np.percentile(log_pmu, 99), 25)
centers = 0.5 * (bins[:-1] + bins[1:])

def binned_mean(x, w, bins):
    out = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if mask.sum() > 0:
            out[i] = np.average(w[mask])
    return out

push_binned = binned_mean(log_pmu, push_mean,     bins)
tilt_binned = binned_mean(log_pmu, injected_tilt, bins)

# --- Plot 1: Push weight recovery ---
# Tests whether OmniFold Step 2 (unfolding) correctly recovers
# the truth-level distortion we injected. If push weights ≈ injected tilt,
# OmniFold is working correctly as an unfolding method.
fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                          gridspec_kw={'height_ratios': [3, 1]})
axes[0].plot(centers, tilt_binned, 'b-o', label='Injected tilt (truth)')
axes[0].plot(centers, push_binned, 'r-s', label='OmniFold push weights')
axes[0].axhline(1.0, color='gray', linestyle='--')
axes[0].set_ylabel('Weight')
axes[0].legend()
axes[0].set_title('Push weight recovery: log(pmu_true) tilt')

ratio = push_binned / np.where(tilt_binned > 0, tilt_binned, 1.0)
axes[1].plot(centers, ratio, 'k-o')
axes[1].axhline(1.0, color='gray', linestyle='--')
axes[1].fill_between(centers, 0.8, 1.2, alpha=0.15, color='green', label='+/-20% band')
axes[1].set_xlabel('log(pmu_true)')
axes[1].set_ylabel('Push / Injected')
axes[1].set_ylim(0.5, 1.5)
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fakedata_recovery_check.png'), dpi=150)
print("Saved fakedata_recovery_check.png")

# --- Plot 2: Step 1 reco-level check ---
# Tests whether OmniFold Step 1 (reweighting MC reco to look like data)
# worked. The green histogram (MC x pull weights) should overlap the orange
# (fake data). This is the reco-level closure before unfolding.
fig2, ax = plt.subplots(figsize=(7, 5))
ax.hist(log_pmu, bins=bins, weights=mc_weights_reco,
        alpha=0.5, label='Nominal MC')
ax.hist(log_pmu, bins=bins, weights=mc_weights_reco * injected_tilt,
        alpha=0.5, label='Fake Data (injected)')
ax.hist(log_pmu, bins=bins, weights=mc_weights_reco * pull_mean,
        alpha=0.5, label='MC x Pull weights (Step 1)')
ax.set_xlabel('log(pmu_true)')
ax.set_ylabel('Weighted events')
ax.legend()
ax.set_title('Step 1 check: reweighted MC vs fake data')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fakedata_step1_check.png'), dpi=150)
print("Saved fakedata_step1_check.png")