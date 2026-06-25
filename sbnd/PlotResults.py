"""
PlotResults.py

Produces publication-quality plots matching the style of
Huang et al. (arXiv:2504.06857) for presenting SBND OmniFold results.

Generates:
  1. Unfolded xsec vs Data Truth (Fig 9 style) — shows OmniFold recovers truth
  2. Ratio to Data Truth (Fig 10 style) — ratio should be ~1 within error bars
  3. Chi2 convergence (Fig 4 style) — shows stable convergence
  4. Uncertainty budget (Fig 8 style) — fractional uncertainty breakdown

Usage:
    python3 sbnd/PaperStylePlots.py
    python3 sbnd/PaperStylePlots.py --var true_costheta
"""

import numpy as np
import glob, re, os, argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

# Use a clean style
mpl.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'axes.grid': False,
})

parser = argparse.ArgumentParser()
parser.add_argument('--var', choices=['true_ke', 'true_costheta', 'both'], default='both')
parser.add_argument('--data-dir', type=str, default='../FormattedData_SBND/')
parser.add_argument('--weights-base', type=str, default='sbnd')
parser.add_argument('--export-dir', type=str, default='sbnd/exported_weights/')
parser.add_argument('--plot-dir', type=str, default='sbnd/plots_paper/')
parser.add_argument('--tag', type=str, default='tilt_alpha0.5',
                    help='Fake data tag for recovery plots')
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
YLABEL_XSEC = {
    'true_ke':       r'd$\sigma$/dKE [arb. units / MeV]',
    'true_costheta': r'd$\sigma$/d$\cos\theta$ [arb. units]',
}

# ── Load data ─────────────────────────────────────────────────────────────────
truth_raw  = np.load(flags.data_dir + 'mc_vals_truth_NoNorm.npy')
mc_weights = np.load(flags.data_dir + 'mc_weights_reco.npy')

def iter_num(p):
    m = re.search(r'Iter(\d+)', p)
    return int(m.group(1)) if m else -1

def chi2_cov(observed, expected, cov):
    """Chi2 using covariance matrix."""
    cov_reg = cov + np.eye(len(cov)) * 1e-6 * np.diag(cov).mean()
    try:
        cov_inv = np.linalg.inv(cov_reg)
        delta = observed - expected
        return float(delta @ cov_inv @ delta)
    except np.linalg.LinAlgError:
        return float('nan')


def make_plots(var_name):
    bins      = BINNING[var_name]
    xlabel    = XLABEL[var_name]
    n_bins    = len(bins) - 1
    bin_widths = np.diff(bins)
    centers   = 0.5 * (bins[:-1] + bins[1:])
    var_idx   = 0 if var_name == 'true_ke' else 1
    var_vals  = truth_raw[:, var_idx]

    # Load efficiency
    eff_file = flags.export_dir + f'efficiency_{var_name}.npy'
    if not os.path.exists(eff_file):
        print(f"  WARNING: {eff_file} not found, using flat efficiency")
        eff = np.ones(n_bins)
    else:
        eff = np.load(eff_file)

    # Load injected tilt weights
    tilt_file = flags.data_dir + f'truth_weights_sbnd_fakedata_{flags.tag}.npy'
    if not os.path.exists(tilt_file):
        print(f"  ERROR: {tilt_file} not found")
        return
    injected = np.load(tilt_file)

    # "Data Truth" = MC weighted by injected tilt (what we're trying to recover)
    truth_hist, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * injected)
    truth_xsec = truth_hist / (eff.clip(1e-6) * bin_widths)

    # "Prior" = nominal MC (before any unfolding)
    nom_hist, _ = np.histogram(var_vals, bins=bins, weights=mc_weights)
    nom_xsec = nom_hist / (eff.clip(1e-6) * bin_widths)

    # Load tilt fake data push weights (final iteration)
    tilt_weights_dir = f'weights_sbnd_fakedata_{flags.tag}/'
    push_files = sorted(glob.glob(tilt_weights_dir + 'Step2_Iter*_PushWeights.npy'), key=iter_num)
    if not push_files:
        print(f"  ERROR: No push files in {tilt_weights_dir}")
        return

    push_final = np.load(push_files[-1])
    push_final = push_final if push_final.ndim == 1 else push_final.mean(axis=0)

    # OmniFold unfolded
    unf_hist, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push_final)
    unf_xsec = unf_hist / (eff.clip(1e-6) * bin_widths)

    # Load systematic covariance for error bars
    cov_file = f'{flags.weights_base}/plots_systematics/covariance_all_{var_name}.npz'
    if os.path.exists(cov_file):
        cov_data = np.load(cov_file)
        cov_events = cov_data['cov']
        scale = eff.clip(1e-6) * bin_widths
        cov_xsec = cov_events / np.outer(scale, scale)
        xsec_unc = np.sqrt(np.diag(cov_xsec))
        # Also get mean from universes
        mean_hist = cov_data['mean_hist']
        mean_xsec = mean_hist / (eff.clip(1e-6) * bin_widths)
    else:
        xsec_unc = np.zeros(n_bins)
        mean_xsec = unf_xsec

    print(f"\n{'='*60}")
    print(f"Paper-style plots: {var_name}")
    print(f"{'='*60}")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 1: Unfolded xsec vs Data Truth (Fig 9 style)
    # Shows: Prior (gray band), Data Truth (black line), OmniFold (red points)
    # ══════════════════════════════════════════════════════════════════════════
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # Data Truth — solid black line (this is what we're trying to recover)
    ax1.step(bins, np.append(truth_xsec, truth_xsec[-1]),
             where='post', color='black', linewidth=2, label='Data Truth')

    # Prior — gray band (nominal MC before unfolding)
    ax1.step(bins, np.append(nom_xsec, nom_xsec[-1]),
             where='post', color='gray', linewidth=1.5, linestyle='--')
    for i in range(n_bins):
        ax1.fill_between([bins[i], bins[i+1]],
                          nom_xsec[i] * 0.95, nom_xsec[i] * 1.05,
                          color='gray', alpha=0.2,
                          label=('Prior' if i == 0 else None))

    # OmniFold — red points with error bars
    ax1.errorbar(centers, unf_xsec, yerr=xsec_unc,
                 fmt='o', color='red', markersize=5, capsize=3,
                 linewidth=1.5, label='OmniFold')

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(YLABEL_XSEC[var_name])
    ax1.set_title(r'SBND $\nu_e$ CC Inclusive')
    ax1.legend()
    ax1.set_xlim(bins[0], bins[-1])
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/xsec_vs_truth_{var_name}.png', dpi=150)
    print(f"  Saved {flags.plot_dir}/xsec_vs_truth_{var_name}.png")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 2: Ratio to Data Truth (Fig 10 style)
    # Shows: ratio of unfolded/truth — should be 1.0 within error bars
    # ══════════════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(8, 4))

    ratio_unf = unf_xsec / np.where(truth_xsec > 0, truth_xsec, 1)
    ratio_unc = xsec_unc / np.where(truth_xsec > 0, truth_xsec, 1)
    ratio_prior = nom_xsec / np.where(truth_xsec > 0, truth_xsec, 1)

    # Prior ratio — gray band
    ax2.fill_between(bins, 0.4, 0.4, color='gray')  # dummy for legend
    for i in range(n_bins):
        ax2.fill_between([bins[i], bins[i+1]],
                          ratio_prior[i] - 0.02, ratio_prior[i] + 0.02,
                          color='gray', alpha=0.3,
                          label=('Prior' if i == 0 else None))
    ax2.step(bins, np.append(ratio_prior, ratio_prior[-1]),
             where='post', color='gray', linewidth=1.5, linestyle='--')

    # OmniFold ratio — red points
    ax2.errorbar(centers, ratio_unf, yerr=ratio_unc,
                 fmt='o', color='red', markersize=5, capsize=3,
                 linewidth=1.5, label='OmniFold')

    ax2.axhline(1.0, color='black', linewidth=1)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Ratio to Data Truth')
    # ax2.set_ylim(0.4, 1.8)
    ax2.legend()
    ax2.set_xlim(bins[0], bins[-1])
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/ratio_to_truth_{var_name}.png', dpi=150)
    print(f"  Saved {flags.plot_dir}/ratio_to_truth_{var_name}.png")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 3: Chi2 convergence (Fig 4 style)
    # Shows: chi2/ndf as function of iteration number
    # ══════════════════════════════════════════════════════════════════════════
    ndf = n_bins - 1

    # Compute chi2 at each iteration using the covariance matrix
    iters = []
    chi2_vals = []

    # Prior chi2 (iteration -1: no unfolding)
    chi2_prior_simple = np.sum((nom_hist - truth_hist)**2 / truth_hist.clip(1))

    for f in push_files:
        it = iter_num(f)
        push = np.load(f)
        push = push if push.ndim == 1 else push.mean(axis=0)
        h, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
        # Simple Pearson chi2 against truth (like the paper's Fig 4)
        c2 = np.sum((h - truth_hist)**2 / truth_hist.clip(1))
        iters.append(it)
        chi2_vals.append(c2 / ndf)

    fig3, ax3 = plt.subplots(figsize=(7, 5))

    ax3.plot(iters, chi2_vals, 'o-', color='red', linewidth=2, markersize=5,
             label=var_name.replace('_', ' '))
    ax3.axhline(1.0, color='gray', linestyle=':', linewidth=1)

    # Mark the prior
    ax3.plot(-1, chi2_prior_simple / ndf, 's', color='blue', markersize=8,
             label=f'Prior: {chi2_prior_simple/ndf:.1f}', zorder=5)

    ax3.set_xlabel('OmniFold Iterations')
    ax3.set_ylabel(r'$\chi^2$/DoF')
    ax3.set_title(r'$\chi^2$ convergence: SBND $\nu_e$ CC')
    ax3.legend()

    # Set y-axis to show the convergence clearly
    # max_chi2 = max(chi2_vals[0], 5) * 1.2
    # ax3.set_ylim(0, min(max_chi2, 15))
    ax3.set_xlim(-2, max(iters) + 1)

    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/chi2_convergence_{var_name}.png', dpi=150)
    print(f"  Saved {flags.plot_dir}/chi2_convergence_{var_name}.png")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 4: Fractional uncertainty breakdown (Fig 8 style)
    # Shows: BNB, GENIE, MCstat, Total fractional uncertainties per bin
    # ══════════════════════════════════════════════════════════════════════════
    fig4, ax4 = plt.subplots(figsize=(8, 5))

    colors_src = {'bnb': 'blue', 'genie': 'red', 'mcstat': 'green'}
    labels_src = {'bnb': 'BNB Flux', 'genie': 'GENIE XSec', 'mcstat': 'MC Stat'}

    for src in ['bnb', 'genie', 'mcstat']:
        cov_src_file = f'{flags.weights_base}/plots_systematics/covariance_{src}_{var_name}.npz'
        if not os.path.exists(cov_src_file):
            continue
        cov_src = np.load(cov_src_file)
        mean_src = cov_src['mean_hist']
        diag_src = np.sqrt(np.diag(cov_src['cov']))
        frac_src = diag_src / mean_src.clip(1e-6)

        ax4.step(bins, np.append(frac_src, frac_src[-1]),
                 where='post', color=colors_src[src], linewidth=1.5,
                 label=f'{labels_src[src]} Uncertainty')

    # Total (from combined covariance)
    if os.path.exists(cov_file):
        cov_all = np.load(cov_file)
        diag_all = np.sqrt(np.diag(cov_all['cov']))
        frac_all = diag_all / cov_all['mean_hist'].clip(1e-6)
        ax4.step(bins, np.append(frac_all, frac_all[-1]),
                 where='post', color='black', linewidth=2,
                 label='Total Uncertainty')

    ax4.set_xlabel(xlabel)
    ax4.set_ylabel('Bin Fractional Uncertainty')
    ax4.set_title(r'Uncertainty budget: SBND $\nu_e$ CC')
    ax4.legend()
    ax4.set_xlim(bins[0], bins[-1])
    ax4.set_yscale('log')
    ax4.set_ylim(1e-3, 0.2)
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/uncertainty_budget_{var_name}.png', dpi=150)
    print(f"  Saved {flags.plot_dir}/uncertainty_budget_{var_name}.png")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 5: Correlation matrix (Appendix A style)
    # ══════════════════════════════════════════════════════════════════════════
    if os.path.exists(cov_file):
        cov_all = np.load(cov_file)
        cov_mat = cov_all['cov']
        diag = np.sqrt(np.diag(cov_mat))
        corr = cov_mat / np.outer(diag.clip(1e-10), diag.clip(1e-10))

        fig5, ax5 = plt.subplots(figsize=(6, 5))
        im = ax5.imshow(corr, origin='lower', aspect='auto',
                         extent=[0, n_bins, 0, n_bins],
                         vmin=-1, vmax=1, cmap='RdBu_r')

        # Bin edge labels
        tick_labels = [f'{bins[i]:.0f}' for i in range(n_bins)]
        ax5.set_xticks(np.arange(n_bins) + 0.5)
        ax5.set_xticklabels(tick_labels, fontsize=10)
        ax5.set_yticks(np.arange(n_bins) + 0.5)
        ax5.set_yticklabels(tick_labels, fontsize=10)
        ax5.set_xlabel(f'{var_name} Bin')
        ax5.set_ylabel(f'{var_name} Bin')
        ax5.set_title(f'Correlation Matrix ({var_name})')
        plt.colorbar(im, ax=ax5)
        plt.tight_layout()
        plt.savefig(f'{flags.plot_dir}/correlation_{var_name}.png', dpi=150)
        print(f"  Saved {flags.plot_dir}/correlation_{var_name}.png")


# ── Run ───────────────────────────────────────────────────────────────────────
vars_to_run = ['true_ke', 'true_costheta'] if flags.var == 'both' else [flags.var]
for v in vars_to_run:
    make_plots(v)

print(f"\nAll paper-style plots saved to {flags.plot_dir}/")