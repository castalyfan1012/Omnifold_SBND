"""
MakeFakeDataWeights.py

Injects a known truth-level distortion as fake "data" weights for OmniFold
closure validation. The reco feature arrays are unchanged, so the detector
response function is preserved (as required).

The injected function is a linear tilt in truth muon momentum:
    w(pmu_true) = 1 + alpha * (pmu_true - mean_pmu) / std_pmu

This is simple enough to predict analytically and verify against push weights.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# --- Config ---
DATA_DIR = '../FormattedData/'   # adjust if different
ALPHA = 0.5   # tilt strength; 0.5 means +/-50% variation across +-1 sigma range
OUTPUT_WEIGHT_FILE = os.path.join(DATA_DIR, 'data_weights_fakedata_pmu_tilt.npy')
OUTPUT_TRUE_WEIGHT_FILE = os.path.join(DATA_DIR, 'truth_weights_fakedata_pmu_tilt.npy')

# --- Load matched truth info (NoNorm has physical units) ---
# Columns: [0]=pmu_true, [1]=cosmu, [2]=phimu, [3]=pp_true,
#          [4]=cospr, [5]=phipr, [6]=sample, [7]=reaction,
#          [8]=topology, [9]=neutintcode, [10]=pt_imbalance,
#          [11]=mu_p_alpha, [12]=mu_p_angle
truth_nonorm = np.load(os.path.join(DATA_DIR, 'mc_vals_truth_NoNorm.npy'))
mc_pass_truth = np.load(os.path.join(DATA_DIR, 'mc_pass_truth.npy'))
mc_weights_reco = np.load(os.path.join(DATA_DIR, 'mc_weights_reco.npy'))
mc_weights_truth = np.load(os.path.join(DATA_DIR, 'mc_weights_truth.npy'))

# mc_pass_truth selects which truth events have a matched reco event.
# The matched subset, in order, corresponds 1-to-1 with the reco events.
matched_truth = truth_nonorm[mc_pass_truth]   # shape: (N_reco, 13)
pmu_true = matched_truth[:, 0]               # MeV/c

print(f"N matched reco events: {len(pmu_true)}")
print(f"pmu_true range: {pmu_true.min():.1f} -- {pmu_true.max():.1f} MeV/c")

# --- Define tilt weight in log(pmu) space ---
log_pmu = np.log(pmu_true)
log_pmu_mean = np.mean(log_pmu)
log_pmu_std  = np.std(log_pmu)
tilt = 1.0 + ALPHA * (log_pmu - log_pmu_mean) / log_pmu_std
tilt = np.clip(tilt, 0.0, None)
tilt = tilt * (np.sum(mc_weights_reco) / np.sum(mc_weights_reco * tilt))

data_weights = mc_weights_reco * tilt
np.save(OUTPUT_WEIGHT_FILE, data_weights)
print(f"Saved fake data weights to {OUTPUT_WEIGHT_FILE}")
print(f"  mean={data_weights.mean():.4f}, std={data_weights.std():.4f}")

np.save(OUTPUT_TRUE_WEIGHT_FILE, tilt)
print(f"Saved truth-level tilt weights to {OUTPUT_TRUE_WEIGHT_FILE}")

# --- Diagnostic plot ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(pmu_true, bins=50, weights=mc_weights_reco, alpha=0.6, label='Nominal MC')
axes[0].hist(pmu_true, bins=50, weights=data_weights,    alpha=0.6, label='Fake Data')
axes[0].set_xlabel('pmu_true [MeV/c]')
axes[0].set_ylabel('Weighted events')
axes[0].legend()
axes[0].set_title('Injected distortion (truth pmu)')

axes[1].scatter(pmu_true[::10], tilt[::10], s=1, alpha=0.3)
axes[1].axhline(1.0, color='r', linestyle='--')
axes[1].set_xlabel('pmu_true [MeV/c]')
axes[1].set_ylabel('Tilt weight')
axes[1].set_title(f'Tilt function (alpha={ALPHA})')

plt.tight_layout()
PLOT_DIR = 'plots_fakedata/'
os.makedirs(PLOT_DIR, exist_ok=True)
plt.savefig(os.path.join(PLOT_DIR, 'fakedata_injected_tilt.png'))
print(f"Saved diagnostic plot to {PLOT_DIR}fakedata_injected_tilt.png")
