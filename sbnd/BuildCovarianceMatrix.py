"""
BuildCovarianceMatrix.py

Collects push weights from all systematic universe runs and builds
the covariance matrix for the unfolded distribution.

Usage:
    python3 sbnd/BuildCovarianceMatrix.py --source bnb --var true_ke
    python3 sbnd/BuildCovarianceMatrix.py --source genie --var true_costheta
    python3 sbnd/BuildCovarianceMatrix.py --source all --var true_ke
"""

import numpy as np
import glob
import re
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--source', choices=['bnb', 'genie', 'all'], default='all')
parser.add_argument('--data-dir', type=str, default='../FormattedData_SBND/')
parser.add_argument('--weights-base', type=str, default='sbnd',
                    help='Base directory containing weights_{source}/ subdirs')
parser.add_argument('--plot-dir', type=str, default='sbnd/plots_systematics/')
parser.add_argument('--var', choices=['true_ke', 'true_costheta'], default='true_ke')
flags = parser.parse_args()

os.makedirs(flags.plot_dir, exist_ok=True)

# ── Auto-binning based on variable ────────────────────────────────────────────
BINNING = {
    'true_ke':       np.array([0, 200, 400, 600, 800, 1000, 1400, 2000]),
    'true_costheta': np.linspace(-1, 1, 11),
}
XLABEL = {
    'true_ke':       r'True electron KE [MeV]',
    'true_costheta': r'True $\cos\theta_e$',
}

bins   = BINNING[flags.var]
xlabel = XLABEL[flags.var]
n_bins = len(bins) - 1
bin_widths = np.diff(bins)
centers    = 0.5 * (bins[:-1] + bins[1:])

# ── Load data ─────────────────────────────────────────────────────────────────
truth_raw  = np.load(flags.data_dir + 'mc_vals_truth_NoNorm.npy')
mc_weights = np.load(flags.data_dir + 'mc_weights_reco.npy')

var_idx  = 0 if flags.var == 'true_ke' else 1
var_vals = truth_raw[:, var_idx]

# Nominal unfolded histogram (push weights = 1)
nom_hist, _ = np.histogram(var_vals, bins=bins, weights=mc_weights)

# ── Collect universe push weights ─────────────────────────────────────────────
sources = ['bnb', 'genie'] if flags.source == 'all' else [flags.source]
all_hists = []

for src in sources:
    pattern = f'{flags.weights_base}/weights_{src}/{src}_univ*/Step2_Iter*_PushWeights.npy'
    push_files = sorted(glob.glob(pattern))
    print(f"  Pattern: {pattern}")
    print(f"  Files matched: {len(push_files)}")

    # Group by universe, take last iteration
    univ_files = {}
    for f in push_files:
        m = re.search(r'univ(\d+)', f)
        if m:
            uid = int(m.group(1))
            it_m = re.search(r'Iter(\d+)', f)
            it = int(it_m.group(1)) if it_m else 0
            if uid not in univ_files or it > univ_files[uid][0]:
                univ_files[uid] = (it, f)

    print(f"{src}: {len(univ_files)} universes found")

    for uid in sorted(univ_files.keys()):
        _, fpath = univ_files[uid]
        push = np.load(fpath)
        push = push if push.ndim == 1 else push.mean(axis=0)
        h, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
        all_hists.append(h)

all_hists = np.array(all_hists)
n_univ = len(all_hists)
print(f"Total universes: {n_univ}")

if n_univ == 0:
    print("ERROR: No universe results found.")
    exit(1)

# ── Statistics ────────────────────────────────────────────────────────────────
mean_hist = all_hists.mean(axis=0)
diff = all_hists - mean_hist[np.newaxis, :]
cov  = (diff.T @ diff) / n_univ

frac_cov = np.zeros_like(cov)
for i in range(n_bins):
    for j in range(n_bins):
        denom = mean_hist[i] * mean_hist[j]
        if denom > 0:
            frac_cov[i, j] = cov[i, j] / denom

diag_unc = np.sqrt(np.diag(cov))
frac_unc = np.sqrt(np.diag(frac_cov))

# ── Save ──────────────────────────────────────────────────────────────────────
np.savez(f'{flags.plot_dir}/covariance_{flags.source}_{flags.var}.npz',
         cov=cov, frac_cov=frac_cov, bins=bins, mean_hist=mean_hist,
         nom_hist=nom_hist, all_hists=all_hists)

# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n{'Bin center':>10s} {'Nominal':>10s} {'Mean':>10s} "
      f"{'Abs unc':>10s} {'Frac unc':>10s}")
for i in range(n_bins):
    print(f"{centers[i]:10.1f} {nom_hist[i]:10.1f} {mean_hist[i]:10.1f} "
          f"{diag_unc[i]:10.2f} {frac_unc[i]:10.4f}")

# ── Plot 1: Covariance matrices ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].imshow(cov, origin='lower', aspect='auto',
                       extent=[bins[0], bins[-1], bins[0], bins[-1]])
axes[0].set_title(f'Covariance matrix ({flags.source})')
axes[0].set_xlabel(xlabel)
axes[0].set_ylabel(xlabel)
plt.colorbar(im0, ax=axes[0])

vmax = max(abs(frac_cov.min()), abs(frac_cov.max()), 0.01)
im1 = axes[1].imshow(frac_cov, origin='lower', aspect='auto',
                       extent=[bins[0], bins[-1], bins[0], bins[-1]],
                       vmin=-vmax, vmax=vmax, cmap='RdBu_r')
axes[1].set_title(f'Fractional covariance ({flags.source})')
axes[1].set_xlabel(xlabel)
axes[1].set_ylabel(xlabel)
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig(f'{flags.plot_dir}/cov_matrix_{flags.source}_{flags.var}.png', dpi=150)

# ── Plot 2: Unfolded spectrum with systematic band ────────────────────────────
fig2, axes2 = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})

# Top panel: distributions with syst band
# Nominal (step histogram)
axes2[0].step(bins, np.append(nom_hist, nom_hist[-1]),
              where='post', color='blue', linewidth=1.5, label='Nominal MC')

# Mean of universes with ±1σ band
axes2[0].step(bins, np.append(mean_hist, mean_hist[-1]),
              where='post', color='red', linewidth=1.5, label=f'Mean ({flags.source})')

# Systematic band: fill between mean ± 1σ
upper = mean_hist + diag_unc
lower = mean_hist - diag_unc
for i in range(n_bins):
    axes2[0].fill_between([bins[i], bins[i+1]], lower[i], upper[i],
                           color='red', alpha=0.2,
                           label=(r'$\pm 1\sigma$ syst' if i == 0 else None))

axes2[0].set_ylabel('Unfolded weighted events')
axes2[0].legend(fontsize=10)
axes2[0].set_title(f'OmniFold unfolded: {flags.var} ({flags.source} systematics)')

# Bottom panel: fractional uncertainty per bin
axes2[1].step(bins, np.append(frac_unc, frac_unc[-1]),
              where='post', color='red', linewidth=1.5)
for i in range(n_bins):
    axes2[1].fill_between([bins[i], bins[i+1]], 0, frac_unc[i],
                           color='red', alpha=0.2)
axes2[1].set_xlabel(xlabel)
axes2[1].set_ylabel('Fractional uncertainty')
axes2[1].set_ylim(0, max(frac_unc) * 1.5)
axes2[1].axhline(0, color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(f'{flags.plot_dir}/unfolded_with_unc_{flags.source}_{flags.var}.png', dpi=150)

# ── Plot 3: Universe spread (spaghetti plot) ─────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 5))

# Plot each universe as a faint line
for i in range(min(n_univ, 50)):   # cap at 50 for readability
    ax3.step(bins, np.append(all_hists[i], all_hists[i][-1]),
             where='post', color='gray', alpha=0.15, linewidth=0.5)

ax3.step(bins, np.append(nom_hist, nom_hist[-1]),
         where='post', color='blue', linewidth=2, label='Nominal')
ax3.step(bins, np.append(mean_hist, mean_hist[-1]),
         where='post', color='red', linewidth=2, linestyle='--', label='Universe mean')

ax3.set_xlabel(xlabel)
ax3.set_ylabel('Unfolded weighted events')
ax3.set_title(f'Universe spread: {flags.var} ({flags.source}, {n_univ} universes)')
ax3.legend()
plt.tight_layout()
plt.savefig(f'{flags.plot_dir}/universe_spread_{flags.source}_{flags.var}.png', dpi=150)

print(f"\nPlots saved to {flags.plot_dir}/")