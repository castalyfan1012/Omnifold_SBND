"""
PlotResults.py

Produces publication-quality plots matching the style of
Huang et al. (arXiv:2504.06857) for presenting SBND OmniFold results.

Iteration convention (matching the paper, Fig 4):
  Iteration 0 = prior (no unfolding, push weights = 1)
  Iteration N = after N-th OmniFold pass
  File "Iter0" saved by omnifold.py = our iteration 1

Generates:
  1. Unfolded xsec vs Data Truth (Fig 9 style)
  2. Ratio to Data Truth (Fig 10 style)
  3. Chi2 convergence (Fig 4 style) — log y-axis, prior at iter 0
  4. Uncertainty budget (Fig 8 style)
  5. Correlation matrix (Appendix A style)

Usage:
    python3 sbnd/PlotResults.py
    python3 sbnd/PlotResults.py --var true_costheta
"""

import numpy as np
import glob, re, os, argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

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
parser.add_argument('--var', choices=['true_p', 'true_costheta', 'both'], default='both')
parser.add_argument('--data-dir', type=str, default='../FormattedData_SBND/')
parser.add_argument('--weights-base', type=str, default='sbnd')
parser.add_argument('--export-dir', type=str, default='sbnd/exported_weights/')
parser.add_argument('--plot-dir', type=str, default='sbnd/plots_xsec/')
parser.add_argument('--tag', type=str, default='tilt_alpha0.5',
                    help='Fake data tag for recovery plots')
flags = parser.parse_args()

os.makedirs(flags.plot_dir, exist_ok=True)

BINNING = {
    'true_p':       np.array([0, 200, 400, 600, 800, 1000, 1400, 2000]),
    'true_costheta': np.linspace(-1, 1, 11),
}
XLABEL = {
    'true_p':       r'True electron momentum [MeV]',
    'true_costheta': r'True $\cos\theta_e$',
}
YLABEL_XSEC = {
    'true_p':       r'd$\sigma$/dp [arb. units / MeV]',
    'true_costheta': r'd$\sigma$/d$\cos\theta$ [arb. units]',
}

truth_raw  = np.load(flags.data_dir + 'mc_vals_truth_NoNorm.npy')
mc_weights = np.load(flags.data_dir + 'mc_weights_reco.npy')

def iter_num(p):
    m = re.search(r'Iter(\d+)', p)
    return int(m.group(1)) if m else -1


def make_plots(var_name):
    bins       = BINNING[var_name]
    xlabel     = XLABEL[var_name]
    n_bins     = len(bins) - 1
    bin_widths = np.diff(bins)
    centers    = 0.5 * (bins[:-1] + bins[1:])
    var_idx    = 0 if var_name == 'true_p' else 1
    var_vals   = truth_raw[:, var_idx]

    # Load efficiency
    eff_file = flags.export_dir + f'efficiency_{var_name}.npy'
    eff = np.load(eff_file) if os.path.exists(eff_file) else np.ones(n_bins)

    # Load injected tilt weights
    tilt_file = flags.data_dir + f'truth_weights_sbnd_fakedata_{flags.tag}.npy'
    if not os.path.exists(tilt_file):
        print(f"  ERROR: {tilt_file} not found")
        return
    injected = np.load(tilt_file)

    # Histograms
    truth_hist, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * injected)
    nom_hist, _   = np.histogram(var_vals, bins=bins, weights=mc_weights)
    truth_xsec = truth_hist / (eff.clip(1e-6) * bin_widths)
    nom_xsec   = nom_hist   / (eff.clip(1e-6) * bin_widths)

    # Load push weights (all iterations)
    tilt_weights_dir = f'weights_sbnd_fakedata_{flags.tag}/'
    push_files = sorted(glob.glob(tilt_weights_dir + 'Step2_Iter*_PushWeights.npy'), key=iter_num)
    if not push_files:
        print(f"  ERROR: No push files in {tilt_weights_dir}")
        return

    push_final = np.load(push_files[-1])
    push_final = push_final if push_final.ndim == 1 else push_final.mean(axis=0)
    unf_hist, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push_final)
    unf_xsec = unf_hist / (eff.clip(1e-6) * bin_widths)

    # Load systematic covariance
    cov_file = f'{flags.weights_base}/plots_systematics/covariance_all_{var_name}.npz'
    if os.path.exists(cov_file):
        cov_data = np.load(cov_file)
        scale = eff.clip(1e-6) * bin_widths
        cov_xsec = cov_data['cov'] / np.outer(scale, scale)
        xsec_unc = np.sqrt(np.diag(cov_xsec))
    else:
        xsec_unc = np.zeros(n_bins)

    print(f"\n{'='*60}")
    print(f"Paper-style plots: {var_name}")
    print(f"{'='*60}")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 1: Unfolded xsec vs Data Truth (Fig 9 style)
    # ══════════════════════════════════════════════════════════════════════════
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    ax1.step(bins, np.append(truth_xsec, truth_xsec[-1]),
             where='post', color='black', linewidth=2, label='Data Truth')

    ax1.step(bins, np.append(nom_xsec, nom_xsec[-1]),
             where='post', color='gray', linewidth=1.5, linestyle='--')
    for i in range(n_bins):
        ax1.fill_between([bins[i], bins[i+1]],
                          nom_xsec[i] * 0.95, nom_xsec[i] * 1.05,
                          color='gray', alpha=0.2,
                          label=('Prior' if i == 0 else None))

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
    print(f"  Saved xsec_vs_truth_{var_name}.png")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 2: Ratio to Data Truth (Fig 10 style)
    # ══════════════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(8, 4))

    ratio_unf   = unf_xsec / np.where(truth_xsec > 0, truth_xsec, 1)
    ratio_unc   = xsec_unc / np.where(truth_xsec > 0, truth_xsec, 1)
    ratio_prior = nom_xsec / np.where(truth_xsec > 0, truth_xsec, 1)

    for i in range(n_bins):
        ax2.fill_between([bins[i], bins[i+1]],
                          ratio_prior[i] - 0.02, ratio_prior[i] + 0.02,
                          color='gray', alpha=0.3,
                          label=('Prior' if i == 0 else None))
    ax2.step(bins, np.append(ratio_prior, ratio_prior[-1]),
             where='post', color='gray', linewidth=1.5, linestyle='--')

    ax2.errorbar(centers, ratio_unf, yerr=ratio_unc,
                 fmt='o', color='red', markersize=5, capsize=3,
                 linewidth=1.5, label='OmniFold')

    ax2.axhline(1.0, color='black', linewidth=1)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Ratio to Data Truth')
    ax2.legend()
    ax2.set_xlim(bins[0], bins[-1])
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/ratio_to_truth_{var_name}.png', dpi=150)
    print(f"  Saved ratio_to_truth_{var_name}.png")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 3: Chi2 convergence (Fig 4 style)
    #
    # Paper convention:
    #   iter 0 = prior (push weights = 1)
    #   iter N = after N-th OmniFold pass (file Iter{N-1})
    # ══════════════════════════════════════════════════════════════════════════
    ndf = n_bins - 1

    # Prior chi2 (no unfolding)
    chi2_prior = np.sum((nom_hist - truth_hist)**2 / truth_hist.clip(1)) / ndf

    # Chi2 at each OmniFold pass
    file_iters = []
    chi2_vals  = []
    for f in push_files:
        it = iter_num(f)
        push = np.load(f)
        push = push if push.ndim == 1 else push.mean(axis=0)
        h, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
        c2 = np.sum((h - truth_hist)**2 / truth_hist.clip(1)) / ndf
        file_iters.append(it)
        chi2_vals.append(c2)

    # Build paper-convention arrays: prior at 0, file Iter0 at 1, etc.
    paper_iters = [0] + [it + 1 for it in file_iters]
    paper_chi2  = [chi2_prior] + chi2_vals

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.plot(paper_iters, paper_chi2, 'o-', color='red', linewidth=2, markersize=5,
             label=var_name.replace('_', ' '))
    ax3.axhline(1.0, color='gray', linestyle=':', linewidth=1, label=r'$\chi^2$/DoF = 1')

    ax3.set_xlabel('OmniFold Iteration')
    ax3.set_ylabel(r'$\chi^2$/DoF')
    ax3.set_title(r'$\chi^2$ convergence: SBND $\nu_e$ CC')
    ax3.legend()
    ax3.set_yscale('log')
    ax3.set_ylim(0.1, max(paper_chi2) * 2)
    ax3.set_xlim(-0.5, max(paper_iters) + 0.5)

    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/chi2_convergence_{var_name}.png', dpi=150)
    print(f"  Saved chi2_convergence_{var_name}.png")

    # Print table
    print(f"\n  Chi2 convergence ({var_name}):")
    print(f"  {'Iter':>5s} {'chi2/ndf':>10s}")
    for pi, c2 in zip(paper_iters, paper_chi2):
        note = '  (prior)' if pi == 0 else ''
        print(f"  {pi:5d} {c2:10.4f}{note}")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 4: Fractional uncertainty breakdown (Fig 8 style)
    # ══════════════════════════════════════════════════════════════════════════
    fig4, ax4 = plt.subplots(figsize=(8, 5))

    colors_src = {'bnb': 'blue', 'genie': 'red', 'mcstat': 'green'}
    labels_src = {'bnb': 'BNB Flux', 'genie': 'GENIE XSec', 'mcstat': 'MC Stat'}

    for src in ['bnb', 'genie', 'mcstat']:
        cov_src_file = f'{flags.weights_base}/plots_systematics/covariance_{src}_{var_name}.npz'
        if not os.path.exists(cov_src_file):
            continue
        cov_src = np.load(cov_src_file)
        diag_src = np.sqrt(np.diag(cov_src['cov']))
        frac_src = diag_src / cov_src['mean_hist'].clip(1e-6)
        ax4.step(bins, np.append(frac_src, frac_src[-1]),
                 where='post', color=colors_src[src], linewidth=1.5,
                 label=f'{labels_src[src]} Uncertainty')

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
    print(f"  Saved uncertainty_budget_{var_name}.png")

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
        print(f"  Saved correlation_{var_name}.png")
    



def make_combined_chi2_plot(flags):
    """Chi2/DoF convergence for all variables on one plot (Fig 4 style)."""
    truth_raw_  = np.load(flags.data_dir + 'mc_vals_truth_NoNorm.npy')
    mc_w        = np.load(flags.data_dir + 'mc_weights_reco.npy')

    tilt_file = flags.data_dir + f'truth_weights_sbnd_fakedata_{flags.tag}.npy'
    injected  = np.load(tilt_file)

    tilt_dir  = f'weights_sbnd_fakedata_{flags.tag}/'
    push_files = sorted(glob.glob(tilt_dir + 'Step2_Iter*_PushWeights.npy'), key=iter_num)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'true_p': 'red', 'true_costheta': 'blue'}
    labels = {'true_p': r'$(p_e)$', 'true_costheta': r'$(\cos\theta_e)$'}

    for var_name in ['true_p', 'true_costheta']:
        bins_  = BINNING[var_name]
        ndf    = len(bins_) - 2
        idx    = 0 if var_name == 'true_p' else 1
        vv     = truth_raw_[:, idx]

        truth_h, _ = np.histogram(vv, bins=bins_, weights=mc_w * injected)
        nom_h, _   = np.histogram(vv, bins=bins_, weights=mc_w)

        chi2_prior = np.sum((nom_h - truth_h)**2 / truth_h.clip(1)) / ndf

        paper_iters = [0]
        paper_chi2  = [chi2_prior]

        for f in push_files:
            it = iter_num(f)
            push = np.load(f)
            push = push if push.ndim == 1 else push.mean(axis=0)
            h, _ = np.histogram(vv, bins=bins_, weights=mc_w * push)
            c2 = np.sum((h - truth_h)**2 / truth_h.clip(1)) / ndf
            paper_iters.append(it + 1)
            paper_chi2.append(c2)

        ax.plot(paper_iters, paper_chi2, 'o-', color=colors[var_name],
                linewidth=2, markersize=5, label=labels[var_name])

    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('OmniFold Iteration')
    ax.set_ylabel(r'$\chi^2$/DoF')
    ax.set_title(r'$\chi^2$ convergence: SBND $\nu_e$ CC')
    ax.legend(fontsize=12)
    ax.set_yscale('log')
    ax.set_xlim(-0.5, max(paper_iters) + 0.5)
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/chi2_convergence_combined.png', dpi=150)
    print(f"  Saved chi2_convergence_combined.png")

def make_weight_change_plot(flags):
    """Average weight change distribution (Fig 5 style)."""
    tilt_dir = f'weights_sbnd_fakedata_{flags.tag}/'
    push_files = sorted(glob.glob(tilt_dir + 'Step2_Iter*_PushWeights.npy'), key=iter_num)

    N = 5  # average over last N iterations
    checkpoints = [5, 10, len(push_files)]  # "after 5, 10, and final iterations"

    fig, ax = plt.subplots(figsize=(7, 5))
    colors_cp = ['blue', 'orange', 'green']

    for ci, total_iter in enumerate(checkpoints):
        if total_iter > len(push_files):
            continue
        # Average weight change over last N iterations (or fewer if not enough)
        start = max(0, total_iter - N)
        weights_window = []
        for f in push_files[start:total_iter]:
            w = np.load(f)
            w = w if w.ndim == 1 else w.mean(axis=0)
            weights_window.append(w)

        if len(weights_window) < 2:
            continue

        # Per-event average change over the window
        w_arr = np.array(weights_window)   # (N_window, N_events)
        avg_change = np.diff(w_arr, axis=0).mean(axis=0)   # mean change per event

        ax.hist(avg_change, bins=100, range=(-0.15, 0.15),
                density=True, alpha=0.6, color=colors_cp[ci],
                label=f'After {total_iter} Iterations')

    ax.set_xlabel(f'Average Weight Change\nfrom Previous {N} Iterations')
    ax.set_ylabel('Fraction of Events')
    ax.set_title(r'Weight convergence: SBND $\nu_e$ CC')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/weight_change_distribution.png', dpi=150)
    print(f"  Saved weight_change_distribution.png")

def make_2d_xsec_slices(flags):
    """Double-differential xsec in costheta slices (Fig 9 style)."""
    truth_raw_ = np.load(flags.data_dir + 'mc_vals_truth_NoNorm.npy')
    mc_w       = np.load(flags.data_dir + 'mc_weights_reco.npy')
    tilt       = np.load(flags.data_dir + f'truth_weights_sbnd_fakedata_{flags.tag}.npy')

    tilt_dir   = f'weights_sbnd_fakedata_{flags.tag}/'
    push_files = sorted(glob.glob(tilt_dir + 'Step2_Iter*_PushWeights.npy'), key=iter_num)
    push = np.load(push_files[-1])
    push = push if push.ndim == 1 else push.mean(axis=0)

    true_p     = truth_raw_[:, 0]
    true_cos   = truth_raw_[:, 1]

    # costheta slices (adjust to match your analysis binning)
    cos_slices = [(-1, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.0)]
    p_bins = np.array([0, 200, 400, 600, 800, 1200, 2000])

    n_slices = len(cos_slices)
    ncols = 3
    nrows = (n_slices + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)

    for si, (cos_lo, cos_hi) in enumerate(cos_slices):
        ax = axes[si // ncols][si % ncols]
        mask = (true_cos >= cos_lo) & (true_cos < cos_hi)

        if mask.sum() == 0:
            ax.set_visible(False)
            continue

        p_slice = true_p[mask]
        w_slice = mc_w[mask]
        t_slice = tilt[mask]
        push_slice = push[mask]

        bin_w = np.diff(p_bins)
        cos_w = cos_hi - cos_lo

        truth_h, _ = np.histogram(p_slice, bins=p_bins, weights=w_slice * t_slice)
        nom_h, _   = np.histogram(p_slice, bins=p_bins, weights=w_slice)
        unf_h, _   = np.histogram(p_slice, bins=p_bins, weights=w_slice * push_slice)

        centers = 0.5 * (p_bins[:-1] + p_bins[1:])

        ax.step(p_bins, np.append(truth_h / (bin_w * cos_w), (truth_h / (bin_w * cos_w))[-1]),
                where='post', color='black', linewidth=2, label='Data Truth')
        ax.errorbar(centers, unf_h / (bin_w * cos_w), fmt='o', color='red',
                    markersize=4, capsize=2, label='OmniFold')
        ax.step(p_bins, np.append(nom_h / (bin_w * cos_w), (nom_h / (bin_w * cos_w))[-1]),
                where='post', color='gray', linewidth=1, linestyle='--', label='Prior')

        ax.set_title(f'{cos_lo:.1f} < cos$\\theta$ < {cos_hi:.1f}', fontsize=11)
        ax.set_xlabel('p [MeV/c]', fontsize=10)
        if si % ncols == 0:
            ax.set_ylabel('arb. units', fontsize=10)
        if si == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for si in range(n_slices, nrows * ncols):
        axes[si // ncols][si % ncols].set_visible(False)

    plt.suptitle(r'SBND $\nu_e$ CC: $d^2\sigma / dp\, d\cos\theta$', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/xsec_2d_slices.png', dpi=150)
    print(f"  Saved xsec_2d_slices.png")

def make_2d_correlation(flags):
    """2D (p, costheta) correlation matrix (Fig 12 style)."""
    truth_raw_ = np.load(flags.data_dir + 'mc_vals_truth_NoNorm.npy')
    mc_w       = np.load(flags.data_dir + 'mc_weights_reco.npy')

    true_p   = truth_raw_[:, 0]
    true_cos = truth_raw_[:, 1]

    p_bins   = np.array([0, 400, 800, 1400, 2000])
    cos_bins = np.array([-1, -0.5, 0, 0.5, 0.75, 1.0])
    n_p = len(p_bins) - 1
    n_c = len(cos_bins) - 1
    n_2d = n_p * n_c

    def hist_2d_flat(p_vals, cos_vals, weights):
        h, _, _ = np.histogram2d(cos_vals, p_vals,
                                  bins=[cos_bins, p_bins], weights=weights)
        return h.flatten()

    # Collect universe results
    all_flat = []
    for src in ['bnb', 'genie', 'mcstat']:
        pattern = f'{flags.weights_base}/weights_{src}/{src}_univ*/Step2_Iter*_PushWeights.npy'
        files = sorted(glob.glob(pattern))
        univ_files = {}
        for f in files:
            m = re.search(r'univ(\d+)', f)
            if m:
                uid = int(m.group(1))
                it_m = re.search(r'Iter(\d+)', f)
                it = int(it_m.group(1)) if it_m else 0
                if uid not in univ_files or it > univ_files[uid][0]:
                    univ_files[uid] = (it, f)
        for uid in sorted(univ_files.keys()):
            push = np.load(univ_files[uid][1])
            push = push if push.ndim == 1 else push.mean(axis=0)
            h = hist_2d_flat(true_p, true_cos, mc_w * push)
            all_flat.append(h)

    if len(all_flat) < 2:
        print("  Not enough universes for 2D correlation")
        return

    all_flat = np.array(all_flat)
    mean_flat = all_flat.mean(axis=0)
    diff = all_flat - mean_flat
    cov_2d = (diff.T @ diff) / len(all_flat)
    diag = np.sqrt(np.diag(cov_2d))
    corr_2d = cov_2d / np.outer(diag.clip(1e-10), diag.clip(1e-10))

    # Labels
    bin_labels = []
    for ic in range(n_c):
        for ip in range(n_p):
            bin_labels.append(f'c[{cos_bins[ic]:.1f},{cos_bins[ic+1]:.1f}]\n'
                            f'p[{p_bins[ip]:.0f},{p_bins[ip+1]:.0f}]')

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_2d, origin='lower', aspect='auto',
                    vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(n_2d))
    ax.set_xticklabels(range(n_2d), fontsize=8)
    ax.set_yticks(range(n_2d))
    ax.set_yticklabels(range(n_2d), fontsize=8)
    ax.set_xlabel(r'$(p, \cos\theta)$ Bin Index')
    ax.set_ylabel(r'$(p, \cos\theta)$ Bin Index')
    ax.set_title(r'Correlation Matrix: $(p, \cos\theta)$ 2D bins')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/correlation_2d_p_costheta.png', dpi=150)
    print(f"  Saved correlation_2d_p_costheta.png")


vars_to_run = ['true_p', 'true_costheta'] if flags.var == 'both' else [flags.var]
for v in vars_to_run:
    make_plots(v)
    
if flags.var == 'both':
    make_combined_chi2_plot(flags)

make_weight_change_plot(flags)
make_2d_xsec_slices(flags)
make_2d_correlation(flags)

print(f"\nAll plots saved to {flags.plot_dir}/")