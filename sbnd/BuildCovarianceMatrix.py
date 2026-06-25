"""
BuildCovarianceMatrix.py

Collects push weights from all systematic universe runs, builds
covariance matrices, and computes chi2 vs iteration convergence.

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
parser.add_argument('--source', choices=['bnb', 'genie', 'mcstat', 'all'], default='all')
parser.add_argument('--data-dir', type=str, default='../FormattedData_SBND/')
parser.add_argument('--weights-base', type=str, default='sbnd',
                    help='Base directory containing weights_{source}/ subdirs')
parser.add_argument('--plot-dir', type=str, default='sbnd/plots_systematics/')
parser.add_argument('--var', choices=['true_ke', 'true_costheta'], default='true_ke')
flags = parser.parse_args()

os.makedirs(flags.plot_dir, exist_ok=True)

# ── Binning ───────────────────────────────────────────────────────────────────
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
centers = 0.5 * (bins[:-1] + bins[1:])

# ── Load data ─────────────────────────────────────────────────────────────────
truth_raw  = np.load(flags.data_dir + 'mc_vals_truth_NoNorm.npy')
mc_weights = np.load(flags.data_dir + 'mc_weights_reco.npy')

var_idx  = 0 if flags.var == 'true_ke' else 1
var_vals = truth_raw[:, var_idx]

nom_hist, _ = np.histogram(var_vals, bins=bins, weights=mc_weights)

# ── Helpers ───────────────────────────────────────────────────────────────────
def iter_num(p):
    m = re.search(r'Iter(\d+)', p)
    return int(m.group(1)) if m else -1

def chi2_poisson(observed, expected):
    mask = expected > 0
    return np.sum((observed[mask] - expected[mask])**2 / expected[mask])

# ── Determine sources ────────────────────────────────────────────────────────
if flags.source == 'all':
    sources = ['bnb', 'genie', 'mcstat']
    # Only include sources that actually have weight files
    sources = [s for s in sources
               if glob.glob(f'{flags.weights_base}/weights_{s}/{s}_univ*/Step2_Iter*_PushWeights.npy')]
else:
    sources = [flags.source]

# ── Collect universe push weights (final iteration) ──────────────────────────
all_hists = []
all_univ_dirs_by_source = {}   # for chi2 vs iteration later

for src in sources:
    pattern = f'{flags.weights_base}/weights_{src}/{src}_univ*/Step2_Iter*_PushWeights.npy'
    push_files = sorted(glob.glob(pattern))

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

    # Store universe dirs for chi2-vs-iteration analysis
    univ_dirs = sorted(glob.glob(f'{flags.weights_base}/weights_{src}/{src}_univ*/'))
    all_univ_dirs_by_source[src] = univ_dirs

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

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Covariance matrices
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Unfolded spectrum with systematic band
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})

axes2[0].step(bins, np.append(nom_hist, nom_hist[-1]),
              where='post', color='blue', linewidth=1.5, label='Nominal MC')
axes2[0].step(bins, np.append(mean_hist, mean_hist[-1]),
              where='post', color='red', linewidth=1.5, label=f'Mean ({flags.source})')

upper = mean_hist + diag_unc
lower = mean_hist - diag_unc
for i in range(n_bins):
    axes2[0].fill_between([bins[i], bins[i+1]], lower[i], upper[i],
                           color='red', alpha=0.2,
                           label=(r'$\pm 1\sigma$ syst' if i == 0 else None))

axes2[0].set_ylabel('Unfolded weighted events')
axes2[0].legend(fontsize=10)
axes2[0].set_title(f'OmniFold unfolded: {flags.var} ({flags.source} systematics)')

axes2[1].step(bins, np.append(frac_unc, frac_unc[-1]),
              where='post', color='red', linewidth=1.5)
for i in range(n_bins):
    axes2[1].fill_between([bins[i], bins[i+1]], 0, frac_unc[i], color='red', alpha=0.2)
axes2[1].set_xlabel(xlabel)
axes2[1].set_ylabel('Fractional uncertainty')
axes2[1].set_ylim(0, max(frac_unc) * 1.5)
axes2[1].axhline(0, color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(f'{flags.plot_dir}/unfolded_with_unc_{flags.source}_{flags.var}.png', dpi=150)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Universe spread (spaghetti plot)
# ══════════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(8, 5))

for i in range(min(n_univ, 50)):
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

# ══════════════════════════════════════════════════════════════════════════════
# Plot 4: Chi2 vs iteration (convergence diagnostic)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Chi2 vs iteration (covariance method)")
print(f"{'='*60}")

# Determine max iteration
sample_dirs = list(all_univ_dirs_by_source.values())[0]
if sample_dirs:
    sample_files = sorted(glob.glob(sample_dirs[0] + 'Step2_Iter*_PushWeights.npy'), key=iter_num)
    max_iter = max(iter_num(f) for f in sample_files) if sample_files else 0
else:
    max_iter = 0

if max_iter > 0:
    ndf = n_bins - 1
    iters_list = list(range(max_iter + 1))

    # Collect all universe directories across all sources
    all_udirs = []
    for src in sources:
        all_udirs.extend(all_univ_dirs_by_source.get(src, []))

    chi2_per_iter = []

    print(f"\n  {'Iter':>5s} {'chi2':>10s} {'chi2/ndf':>10s} {'ndf':>6s}")

    # Add "no unfolding" point (iteration -1): push weights = 1
    # Use same covariance from iteration 0 for comparison
    hists_iter0 = []
    for udir in all_udirs:
        push_file = glob.glob(udir + f'Step2_Iter0_*_PushWeights.npy')
        if not push_file:
            continue
        # For "no unfolding", use unit weights instead of push weights
        h_nom, _ = np.histogram(var_vals, bins=bins, weights=mc_weights)
        hists_iter0.append(h_nom)  # all identical = nominal

    # The "prior" chi2: how far is nominal from the mean of universe iter-0 results
    if len(hists_iter0) >= 2:
        hists_arr_0 = []
        for udir in all_udirs:
            push_file = glob.glob(udir + f'Step2_Iter0_*_PushWeights.npy')
            if not push_file:
                continue
            push = np.load(push_file[0])
            push = push if push.ndim == 1 else push.mean(axis=0)
            h, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
            hists_arr_0.append(h)
        hists_arr_0 = np.array(hists_arr_0)
        diff_0 = hists_arr_0 - hists_arr_0.mean(axis=0)
        cov_0  = (diff_0.T @ diff_0) / len(hists_arr_0)
        cov_0_reg = cov_0 + np.eye(n_bins) * 1e-6 * np.diag(cov_0).mean()
        try:
            cov_0_inv = np.linalg.inv(cov_0_reg)
            delta_prior = nom_hist - hists_arr_0.mean(axis=0)
            c2_prior = float(delta_prior @ cov_0_inv @ delta_prior)
        except np.linalg.LinAlgError:
            c2_prior = float('nan')
        print(f"  {'prior':>5s} {c2_prior:10.2f} {c2_prior/ndf:10.4f} {ndf:6d}  (no unfolding)")

    for it in iters_list:
        # Collect unfolded histogram for each universe at this iteration
        hists_this_iter = []
        for udir in all_udirs:
            push_file = glob.glob(udir + f'Step2_Iter{it}_*_PushWeights.npy')
            if not push_file:
                continue
            push = np.load(push_file[0])
            push = push if push.ndim == 1 else push.mean(axis=0)
            h, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
            hists_this_iter.append(h)

        if len(hists_this_iter) < 2:
            chi2_per_iter.append(0)
            print(f"  {it:5d} {'N/A':>10s} {'N/A':>10s} {ndf:6d}  (too few universes)")
            continue

        hists_arr = np.array(hists_this_iter)   # (N_univ, N_bins)
        mu   = hists_arr.mean(axis=0)           # mean unfolded result
        diff = hists_arr - mu[np.newaxis, :]
        cov_iter = (diff.T @ diff) / len(hists_arr)

        # Regularize: add small diagonal to prevent singular matrix
        cov_reg = cov_iter + np.eye(n_bins) * 1e-6 * np.diag(cov_iter).mean()

        try:
            cov_inv = np.linalg.inv(cov_reg)
            delta = mu - nom_hist
            c2 = float(delta @ cov_inv @ delta)
        except np.linalg.LinAlgError:
            c2 = float('nan')

        chi2_per_iter.append(c2)
        print(f"  {it:5d} {c2:10.2f} {c2/ndf:10.4f} {ndf:6d}")

    # Plot
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.plot(iters_list, chi2_per_iter, 'ro-', linewidth=2, markersize=6,
             label=r'$\chi^2 = (\mu - \mathrm{nom})^T\, C^{-1}\, (\mu - \mathrm{nom})$')
    # ax4.axhline(ndf, color='gray', linestyle=':', linewidth=1,
    #             label=f'ndf = {ndf}')
    ax4.set_xlabel('OmniFold iteration')
    ax4.set_ylabel(r'$\chi^2$')
    ax4.set_title(f'Convergence: {flags.var} ({flags.source}, {len(all_udirs)} universes)')
    ax4.legend(fontsize=9)
    ax4.set_xticks(iters_list)
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/chi2_vs_iter_{flags.source}_{flags.var}.png', dpi=150)
    print(f"\nSaved {flags.plot_dir}/chi2_vs_iter_{flags.source}_{flags.var}.png")
else:
    print("  Skipping chi2 vs iteration: could not determine max iteration.")

print(f"\nAll plots saved to {flags.plot_dir}/")
