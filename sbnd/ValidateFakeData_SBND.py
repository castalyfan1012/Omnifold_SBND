"""
ValidateFakeData_SBND.py

Validates OmniFold push weight recovery against injected distortion.
Includes chi2 vs iteration convergence diagnostic.

Usage:
    python3 sbnd/ValidateFakeData_SBND.py --tag tilt_alpha0.5
    python3 sbnd/ValidateFakeData_SBND.py --tag bnb_univ0
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

# ── Binning ───────────────────────────────────────────────────────────────────
BINNING = {
    'true_ke':       np.array([0, 200, 400, 600, 800, 1000, 1400, 2000]),
    'true_costheta': np.linspace(-1, 1, 11),
}
XLABEL = {
    'true_ke':       r'True electron KE [MeV]',
    'true_costheta': r'True $\cos\theta_e$',
}

# ── Load data ─────────────────────────────────────────────────────────────────
truth_raw      = np.load(DATA_DIR + 'mc_vals_truth_NoNorm.npy')
mc_weights     = np.load(DATA_DIR + 'mc_weights_reco.npy')
injected_tilt  = np.load(DATA_DIR + f'truth_weights_sbnd_fakedata_{TAG}.npy')

true_ke       = truth_raw[:, 0]
true_costheta = truth_raw[:, 1]

# ── Load ALL iteration push/pull weights ──────────────────────────────────────
def iter_num(p):
    m = re.search(r'Iter(\d+)', p)
    return int(m.group(1)) if m else -1

push_files = sorted(glob.glob(WEIGHTS_DIR + 'Step2_Iter*_PushWeights.npy'), key=iter_num)
pull_files = sorted(glob.glob(WEIGHTS_DIR + 'Step1_Iter*_PullWeights.npy'), key=iter_num)

if not push_files:
    print(f"ERROR: No push weight files in {WEIGHTS_DIR}")
    exit(1)

# Final iteration weights
push_final = np.load(push_files[-1])
pull_final = np.load(pull_files[-1])
push_mean = push_final if push_final.ndim == 1 else push_final.mean(axis=0)
pull_mean = pull_final if pull_final.ndim == 1 else pull_final.mean(axis=0)

print(f"Push file (final): {push_files[-1]}")
print(f"Pull file (final): {pull_files[-1]}")
print(f"Push:     mean={push_mean.mean():.4f}, std={push_mean.std():.4f}")
print(f"Injected: mean={injected_tilt.mean():.4f}, std={injected_tilt.std():.4f}")

# ── Helper functions ──────────────────────────────────────────────────────────
def binned_mean(x, w, bins):
    out = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.sum() > 0:
            out[i] = np.average(w[mask])
    return out

def chi2_poisson(observed, expected):
    """Pearson chi2 with Poisson-like variance."""
    mask = expected > 0
    return np.sum((observed[mask] - expected[mask])**2 / expected[mask])


# ── Plot 1: Push weight recovery vs true_ke ──────────────────────────────────
for var_name, var_vals in [('true_ke', true_ke), ('true_costheta', true_costheta)]:
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


# ── Plot 2: Unfolded distributions ──────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
for ax, var_name, var_vals in [(axes3[0], 'true_ke', true_ke),
                                (axes3[1], 'true_costheta', true_costheta)]:
    bins   = BINNING[var_name]
    xlabel = XLABEL[var_name]
    ax.hist(var_vals, bins=bins, weights=mc_weights,                   alpha=0.5, label='Nominal MC')
    ax.hist(var_vals, bins=bins, weights=mc_weights * injected_tilt,   alpha=0.5, label='Fake Data (truth)')
    ax.hist(var_vals, bins=bins, weights=mc_weights * push_mean,       alpha=0.5, label='OmniFold unfolded')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weighted events')
    ax.legend(fontsize=9)
    ax.set_title(f'Unfolded distribution: {var_name}')
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/unfolded_distributions.png', dpi=150)
print(f"Saved {PLOT_DIR}/unfolded_distributions.png")


# ── Plot 3: Chi2 vs iteration (convergence diagnostic) ──────────────────────
print(f"\n{'='*60}")
print(f"Chi2 vs iteration convergence")
print(f"{'='*60}")

fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))

for ax, var_name, var_vals in [(axes4[0], 'true_ke', true_ke),
                                (axes4[1], 'true_costheta', true_costheta)]:
    bins = BINNING[var_name]
    ndf  = len(bins) - 2   # nbins - 1 (constraint)

    # "Truth" = MC weighted by injected tilt
    truth_hist, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * injected_tilt)

    # Nominal chi2 (no unfolding at all)
    chi2_nom = chi2_poisson(
        np.histogram(var_vals, bins=bins, weights=mc_weights)[0],
        truth_hist
    )

    # Chi2 at each iteration
    iters  = []
    chi2s  = []
    for f in push_files:
        it = iter_num(f)
        push = np.load(f)
        push = push if push.ndim == 1 else push.mean(axis=0)
        unfolded, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
        c2 = chi2_poisson(unfolded, truth_hist)
        iters.append(it)
        chi2s.append(c2)

    # Print table
    print(f"\n  {var_name}:")
    print(f"  {'Iter':>5s} {'chi2':>10s} {'chi2/ndf':>10s}")
    print(f"  {'nom':>5s} {chi2_nom:10.2f} {chi2_nom/ndf:10.4f}  (no unfolding)")
    for it, c2 in zip(iters, chi2s):
        print(f"  {it:5d} {c2:10.2f} {c2/ndf:10.4f}")

    # Plot
    ax.plot(iters, chi2s, 'ro-', linewidth=2, markersize=6,
            label='OmniFold unfolded')
    ax.axhline(chi2_nom, color='blue', linestyle='--', linewidth=1.5,
               label=f'No unfolding: {chi2_nom:.1f}')
    ax.axhline(ndf, color='gray', linestyle=':', linewidth=1,
               label=f'ndf = {ndf}')
    ax.set_xlabel('OmniFold iteration')
    ax.set_ylabel(r'$\chi^2$ (unfolded vs injected truth)')
    ax.set_title(f'{var_name}')
    ax.legend(fontsize=9)
    ax.set_xticks(iters)

plt.suptitle(f'Convergence: {TAG}', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/chi2_vs_iterations.png', dpi=150)
print(f"\nSaved {PLOT_DIR}/chi2_vs_iterations.png")


# ── Iteration convergence summary ────────────────────────────────────────────
print(f"\n=== Iteration convergence (push weight stats) ===")
for f in push_files:
    w = np.load(f)
    w = w if w.ndim == 1 else w.mean(axis=0)
    it = iter_num(f)
    print(f"  Iter {it:2d}: mean={w.mean():.4f}, std={w.std():.4f}")