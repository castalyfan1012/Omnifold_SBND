"""
ValidateFakeData_SBND.py

Validates OmniFold push weight recovery against injected distortion.
Variables: true_p (col 0), true_costheta (col 1).
Includes chi2 vs iteration convergence diagnostic.

Usage:
    python3 sbnd/ValidateFakeData_SBND.py --tag tilt_alpha0.5
"""

import numpy as np
import matplotlib.pyplot as plt
import glob, re, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, required=True)
parser.add_argument('--data-dir', type=str, default='../FormattedData_SBND/')
parser.add_argument('--weights-dir', type=str, default=None)
parser.add_argument('--plot-dir', type=str, default=None)
flags = parser.parse_args()

DATA_DIR    = flags.data_dir
TAG         = flags.tag
WEIGHTS_DIR = flags.weights_dir or f'weights_sbnd_fakedata_{TAG}/'
PLOT_DIR    = flags.plot_dir or f'plots_sbnd_fakedata_{TAG}/'
os.makedirs(PLOT_DIR, exist_ok=True)

# Binning: primary variables are p and costheta
BINNING = {
    'true_p':        np.array([0, 200, 400, 600, 800, 1000, 1400, 2000]),
    'true_costheta': np.linspace(-1, 1, 11),
}
XLABEL = {
    'true_p':        r'True electron momentum [MeV/c]',
    'true_costheta': r'True $\cos\theta_e$',
}
VAR_COLS = {'true_p': 0, 'true_costheta': 1}

# ── Load ──────────────────────────────────────────────────────────────────────
truth_raw     = np.load(DATA_DIR + 'mc_vals_truth_NoNorm.npy')
mc_weights    = np.load(DATA_DIR + 'mc_weights_reco.npy')
injected_tilt = np.load(DATA_DIR + f'truth_weights_sbnd_fakedata_{TAG}.npy')

# [0]=true_p, [1]=true_costheta (set by FormatData_SBND.py)
true_p        = truth_raw[:, 0]
true_costheta = truth_raw[:, 1]

def iter_num(p):
    m = re.search(r'Iter(\d+)', p)
    return int(m.group(1)) if m else -1

push_files = sorted(glob.glob(WEIGHTS_DIR + 'Step2_Iter*_PushWeights.npy'), key=iter_num)
pull_files = sorted(glob.glob(WEIGHTS_DIR + 'Step1_Iter*_PullWeights.npy'), key=iter_num)

if not push_files:
    print(f"ERROR: No push weight files in {WEIGHTS_DIR}")
    exit(1)

print(f"Push file (final): {push_files[-1]}")
push_final = np.load(push_files[-1])
push_mean  = push_final if push_final.ndim == 1 else push_final.mean(axis=0)

print(f"Push:     mean={push_mean.mean():.4f}, std={push_mean.std():.4f}")
print(f"Injected: mean={injected_tilt.mean():.4f}, std={injected_tilt.std():.4f}")

def binned_mean(x, w, bins):
    out = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.sum() > 0:
            out[i] = np.average(w[mask])
    return out

def chi2_simple(obs, exp):
    mask = exp > 0
    return np.sum((obs[mask] - exp[mask])**2 / exp[mask])

# ── Plot 1 & 2: Push weight recovery for each variable ──────────────────────
var_data = {'true_p': true_p, 'true_costheta': true_costheta}

for var_name, var_vals in var_data.items():
    bins    = BINNING[var_name]
    xlabel  = XLABEL[var_name]
    centers = 0.5 * (bins[:-1] + bins[1:])

    push_binned = binned_mean(var_vals, push_mean,     bins)
    tilt_binned = binned_mean(var_vals, injected_tilt, bins)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})
    axes[0].plot(centers, tilt_binned, 'b-o', label='Injected')
    axes[0].plot(centers, push_binned, 'r-s', label='OmniFold push weights')
    axes[0].axhline(1.0, color='gray', linestyle='--')
    axes[0].set_ylabel('Weight')
    axes[0].legend()
    axes[0].set_title(f'SBND Push Weight Recovery: {var_name} ({TAG})')

    ratio = push_binned / np.where(tilt_binned > 0, tilt_binned, 1.0)
    axes[1].plot(centers, ratio, 'k-o')
    axes[1].axhline(1.0, color='gray', linestyle='--')
    axes[1].fill_between(centers, 0.8, 1.2, alpha=0.15, color='green', label='+/-20% band')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('Push / Injected')
    axes[1].set_ylim(0.5, 1.5)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/recovery_{var_name}.png', dpi=150)
    print(f"Saved {PLOT_DIR}/recovery_{var_name}.png")

# ── Plot 3: Unfolded distributions (both variables side by side) ─────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
for ax, (var_name, var_vals) in zip(axes3, var_data.items()):
    bins   = BINNING[var_name]
    xlabel = XLABEL[var_name]
    ax.hist(var_vals, bins=bins, weights=mc_weights,                   alpha=0.5, label='Nominal MC')
    ax.hist(var_vals, bins=bins, weights=mc_weights * injected_tilt,   alpha=0.5, label='Fake Data (truth)')
    ax.hist(var_vals, bins=bins, weights=mc_weights * push_mean,       alpha=0.5, label='OmniFold unfolded')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weighted events')
    ax.legend(fontsize=9)
    ax.set_title(f'Unfolded: {var_name}')
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/unfolded_distributions.png', dpi=150)
print(f"Saved {PLOT_DIR}/unfolded_distributions.png")

# ── Plot 4: Chi2 vs iteration (both variables, paper convention) ─────────────
print(f"\n{'='*60}")
print(f"Chi2 vs iteration convergence")
print(f"{'='*60}")
print(f"(Iter 0 = prior, Iter N = after N-th OmniFold pass)")

fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))

for ax, (var_name, var_vals) in zip(axes4, var_data.items()):
    bins = BINNING[var_name]
    ndf  = len(bins) - 2

    truth_hist, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * injected_tilt)
    nom_hist, _   = np.histogram(var_vals, bins=bins, weights=mc_weights)

    chi2_prior = chi2_simple(nom_hist, truth_hist) / ndf

    paper_iters = [0]
    paper_chi2  = [chi2_prior]

    print(f"\n  {var_name}:")
    print(f"  {'Iter':>5s} {'chi2/ndf':>10s}")
    print(f"  {'0':>5s} {chi2_prior:10.4f}  (prior)")

    for f in push_files:
        it = iter_num(f)
        push = np.load(f)
        push = push if push.ndim == 1 else push.mean(axis=0)
        h, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
        c2 = chi2_simple(h, truth_hist) / ndf
        paper_iters.append(it + 1)
        paper_chi2.append(c2)
        print(f"  {it+1:5d} {c2:10.4f}")

    ax.plot(paper_iters, paper_chi2, 'ro-', linewidth=2, markersize=5,
            label=var_name.replace('_', ' '))
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('OmniFold Iteration')
    ax.set_ylabel(r'$\chi^2$/DoF')
    ax.set_title(var_name)
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.set_ylim(0.1, max(paper_chi2) * 2)
    ax.set_xlim(-0.5, max(paper_iters) + 0.5)

plt.suptitle(f'Convergence: {TAG}', fontsize=13)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/chi2_vs_iterations.png', dpi=150)
print(f"\nSaved {PLOT_DIR}/chi2_vs_iterations.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n=== Push weight stats per iteration ===")
for f in push_files:
    w = np.load(f)
    w = w if w.ndim == 1 else w.mean(axis=0)
    it = iter_num(f)
    print(f"  Iter {it+1:2d}: mean={w.mean():.4f}, std={w.std():.4f}")