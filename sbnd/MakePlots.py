"""
MakePlots.py — All SBND OmniFold plots in one script.

Consolidates: ValidateFakeData_SBND, PlotResults.
Adds new plots from Roger's suggestions: reweighting snapshots,
per-event weights vs observable, weight distributions, 2D weight map.

Actions:
  validation   Fake-data recovery, chi2 convergence, unfolded distributions
  paper        Publication-style plots (xsec, ratio, chi2, unc budget,
               correlation, 2D slices, 2D correlation, weight change,
               reweighting snapshots, per-event weights, weight map)
  all          Both validation and paper

Usage:
    python3 sbnd/MakePlots.py validation --tag tilt_alpha0.5
    python3 sbnd/MakePlots.py paper --var both --tag tilt_alpha0.5
    python3 sbnd/MakePlots.py all --var both --tag tilt_alpha0.5
"""

import numpy as np
import glob, re, os, argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
    'figure.dpi': 150, 'axes.grid': False,
})

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest='action', required=True)

p_val = sub.add_parser('validation')
p_val.add_argument('--tag', required=True)
p_val.add_argument('--data-dir', default='../FormattedData_SBND/')
p_val.add_argument('--weights-dir', default=None)
p_val.add_argument('--plot-dir', default=None)

p_pap = sub.add_parser('paper')
p_pap.add_argument('--var', choices=['true_p', 'true_costheta', 'both'], default='both')
p_pap.add_argument('--tag', default='tilt_alpha0.5')
p_pap.add_argument('--data-dir', default='../FormattedData_SBND/')
p_pap.add_argument('--weights-base', default='sbnd')
p_pap.add_argument('--export-dir', default='sbnd/exported_weights/')
p_pap.add_argument('--plot-dir', default='sbnd/plots_xsec/')
# FIX 1: separate cov-dir from plots_systematics
p_pap.add_argument('--cov-dir', default='sbnd/covariance/',
                   help='Directory where BuildResults.py covariance saved .npz files')

p_all = sub.add_parser('all')
p_all.add_argument('--var', choices=['true_p', 'true_costheta', 'both'], default='both')
p_all.add_argument('--tag', default='tilt_alpha0.5')
p_all.add_argument('--data-dir', default='../FormattedData_SBND/')
p_all.add_argument('--weights-base', default='sbnd')
p_all.add_argument('--export-dir', default='sbnd/exported_weights/')
p_all.add_argument('--plot-dir', default='sbnd/plots_xsec/')
p_all.add_argument('--cov-dir', default='sbnd/covariance/')

flags = parser.parse_args()

BINNING = {
    'true_p':        np.array([0, 200, 400, 600, 800, 1000, 1400, 2000]),
    'true_costheta': np.linspace(-1, 1, 11),
}
XLABEL = {
    'true_p':        r'True electron momentum [MeV]',
    'true_costheta': r'True $\cos\theta_e$',
}
YLABEL_XSEC = {
    'true_p':        r'd$\sigma$/dp [arb. / MeV]',
    'true_costheta': r'd$\sigma$/d$\cos\theta$ [arb.]',
}

def iter_num(p):
    m = re.search(r'Iter(\d+)', p)
    return int(m.group(1)) if m else -1


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
def do_validation():
    TAG = flags.tag
    DATA_DIR    = flags.data_dir
    WEIGHTS_DIR = flags.weights_dir or f'weights_sbnd_fakedata_{TAG}/'
    PLOT_DIR    = flags.plot_dir or f'plots_sbnd_fakedata_{TAG}/'
    os.makedirs(PLOT_DIR, exist_ok=True)

    truth_raw     = np.load(DATA_DIR + 'mc_vals_truth_NoNorm.npy')
    mc_weights    = np.load(DATA_DIR + 'mc_weights_reco.npy')
    injected_tilt = np.load(DATA_DIR + f'truth_weights_sbnd_fakedata_{TAG}.npy')
    true_p, true_costheta = truth_raw[:, 0], truth_raw[:, 1]

    push_files = sorted(glob.glob(WEIGHTS_DIR + 'Step2_Iter*_PushWeights.npy'), key=iter_num)
    if not push_files:
        print(f"ERROR: No push files in {WEIGHTS_DIR}"); return

    push_final = np.load(push_files[-1])
    push_mean  = push_final if push_final.ndim == 1 else push_final.mean(axis=0)
    print(f"Push: mean={push_mean.mean():.4f}, std={push_mean.std():.4f}")
    print(f"Injected: mean={injected_tilt.mean():.4f}")

    def binned_mean(x, w, bins):
        out = np.zeros(len(bins) - 1)
        for i in range(len(bins) - 1):
            mask = (x >= bins[i]) & (x < bins[i + 1])
            if mask.sum() > 0: out[i] = np.average(w[mask])
        return out

    def chi2_simple(obs, exp):
        mask = exp > 0
        return np.sum((obs[mask] - exp[mask])**2 / exp[mask])

    var_data = {'true_p': true_p, 'true_costheta': true_costheta}

    # ── Recovery plots ────────────────────────────────────────────────────────
    for var_name, var_vals in var_data.items():
        bins = BINNING[var_name]; centers = 0.5 * (bins[:-1] + bins[1:])
        push_binned = binned_mean(var_vals, push_mean, bins)
        tilt_binned = binned_mean(var_vals, injected_tilt, bins)
        fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1]})
        axes[0].plot(centers, tilt_binned, 'b-o', label='Injected')
        axes[0].plot(centers, push_binned, 'r-s', label='OmniFold push')
        axes[0].axhline(1.0, color='gray', linestyle='--')
        axes[0].set_ylabel('Weight'); axes[0].legend()
        axes[0].set_title(f'Recovery: {var_name} ({TAG})')
        ratio = push_binned / np.where(tilt_binned > 0, tilt_binned, 1.0)
        axes[1].plot(centers, ratio, 'k-o')
        axes[1].axhline(1.0, color='gray', linestyle='--')
        axes[1].fill_between(centers, 0.8, 1.2, alpha=0.15, color='green')
        axes[1].set_xlabel(XLABEL[var_name]); axes[1].set_ylabel('Push / Injected')
        axes[1].set_ylim(0.5, 1.5)
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/recovery_{var_name}.png', dpi=150); plt.close()
        print(f"  Saved recovery_{var_name}.png")

    # ── Unfolded distributions ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (vn, vv) in zip(axes, var_data.items()):
        bins = BINNING[vn]
        ax.hist(vv, bins=bins, weights=mc_weights, alpha=0.5, label='Nominal MC')
        ax.hist(vv, bins=bins, weights=mc_weights * injected_tilt, alpha=0.5, label='Fake Data')
        ax.hist(vv, bins=bins, weights=mc_weights * push_mean, alpha=0.5, label='OmniFold')
        ax.set_xlabel(XLABEL[vn]); ax.set_ylabel('Weighted events')
        ax.legend(fontsize=9); ax.set_title(f'Unfolded: {vn}')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/unfolded_distributions.png', dpi=150); plt.close()
    print(f"  Saved unfolded_distributions.png")

    # ── Chi2 vs iteration ─────────────────────────────────────────────────────
    print(f"\n  Fake-data recovery chi2/ndf vs OmniFold iteration:")
    print(f"  Measures: (unfolded - data_truth)^2 / data_truth per bin, summed / ndf.")
    print(f"  Should decrease from large (prior far from truth) toward ~1.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (vn, vv) in zip(axes, var_data.items()):
        bins = BINNING[vn]; ndf = len(bins) - 2  # n_bins - 1: one dof for normalization
        truth_h, _ = np.histogram(vv, bins=bins, weights=mc_weights * injected_tilt)
        nom_h, _   = np.histogram(vv, bins=bins, weights=mc_weights)
        pi, pc = [0], [chi2_simple(nom_h, truth_h) / ndf]
        for f in push_files:
            it = iter_num(f)
            push = np.load(f)
            push = push if push.ndim == 1 else push.mean(axis=0)
            h, _ = np.histogram(vv, bins=bins, weights=mc_weights * push)
            pi.append(it + 1); pc.append(chi2_simple(h, truth_h) / ndf)
        print(f"\n  {vn}  (ndf={ndf})")
        print(f"  {'Iter':>5s} {'chi2/ndf':>10s}")
        for i, c in zip(pi, pc):
            note = "  <- prior (no unfolding)" if i == 0 else ""
            print(f"  {i:5d} {c:10.4f}{note}")
        ax.plot(pi, pc, 'ro-', linewidth=2, markersize=5)
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel('OmniFold Iteration'); ax.set_ylabel(r'$\chi^2$/DoF')
        ax.set_title(vn); ax.set_yscale('log')
        ax.set_ylim(0.1, max(pc) * 2)
    plt.suptitle(f'Convergence: {TAG}', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/chi2_vs_iterations.png', dpi=150); plt.close()
    print(f"\n  Saved chi2_vs_iterations.png")

    print(f"\n=== Push weight stats per iteration ===")
    for f in push_files:
        w = np.load(f)
        w = w if w.ndim == 1 else w.mean(axis=0)
        print(f"  Iter {iter_num(f)+1:2d}: mean={w.mean():.4f}, std={w.std():.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAPER-STYLE PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
def do_paper():
    os.makedirs(flags.plot_dir, exist_ok=True)
    truth_raw  = np.load(flags.data_dir + 'mc_vals_truth_NoNorm.npy')
    mc_weights = np.load(flags.data_dir + 'mc_weights_reco.npy')

    tilt_file = flags.data_dir + f'truth_weights_sbnd_fakedata_{flags.tag}.npy'
    if not os.path.exists(tilt_file):
        print(f"ERROR: {tilt_file} not found"); return
    injected = np.load(tilt_file)

    tilt_dir = f'weights_sbnd_fakedata_{flags.tag}/'
    push_files = sorted(glob.glob(tilt_dir + 'Step2_Iter*_PushWeights.npy'), key=iter_num)
    if not push_files:
        print(f"ERROR: No push files in {tilt_dir}"); return

    push_final = np.load(push_files[-1])
    push_final = push_final if push_final.ndim == 1 else push_final.mean(axis=0)

    vars_to_run = (['true_p', 'true_costheta'] if flags.var == 'both' else [flags.var])

    for var_name in vars_to_run:
        _make_core_plots(var_name, truth_raw, mc_weights, injected, push_files, push_final)

    if flags.var == 'both':
        _make_combined_chi2(truth_raw, mc_weights, injected, push_files)

    _make_reweighting_snapshots(truth_raw, mc_weights, injected, push_files)
    _make_weight_vs_observable(truth_raw, mc_weights, injected, push_final)
    _make_weight_distributions(push_files)
    _make_weight_map_2d(truth_raw, mc_weights, push_final)
    _make_weight_change(mc_weights, push_files)
    _make_2d_xsec_slices(truth_raw, mc_weights, injected, push_final)
    _make_2d_correlation(truth_raw, mc_weights)

    print(f"\nAll paper plots saved to {flags.plot_dir}/")


def _make_core_plots(var_name, truth_raw, mc_weights, injected, push_files, push_final):
    """Plots 1-5: xsec, ratio, chi2, unc budget, correlation."""
    bins   = BINNING[var_name]
    xlabel = XLABEL[var_name]
    n_bins = len(bins) - 1; bin_widths = np.diff(bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    var_idx = 0 if var_name == 'true_p' else 1
    var_vals = truth_raw[:, var_idx]

    eff_file = flags.export_dir + f'efficiency_{var_name}.npy'
    eff = np.load(eff_file) if os.path.exists(eff_file) else np.ones(n_bins)

    truth_hist, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * injected)
    nom_hist, _   = np.histogram(var_vals, bins=bins, weights=mc_weights)
    unf_hist, _   = np.histogram(var_vals, bins=bins, weights=mc_weights * push_final)
    truth_xsec = truth_hist / (eff.clip(1e-6) * bin_widths)
    nom_xsec   = nom_hist   / (eff.clip(1e-6) * bin_widths)
    unf_xsec   = unf_hist   / (eff.clip(1e-6) * bin_widths)

    # FIX 1: read cov from --cov-dir, not plots_systematics
    cov_file = f'{flags.cov_dir}/covariance_all_{var_name}.npz'
    if os.path.exists(cov_file):
        cd = np.load(cov_file)
        scale = eff.clip(1e-6) * bin_widths
        xsec_unc = np.sqrt(np.diag(cd['cov'] / np.outer(scale, scale)))
    else:
        print(f"  WARNING: {cov_file} not found — no systematic band")
        xsec_unc = np.zeros(n_bins)

    print(f"\n  Paper plots: {var_name}")

    # Plot 1: xsec vs truth
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.step(bins, np.append(truth_xsec, truth_xsec[-1]),
            where='post', color='black', linewidth=2, label='Data Truth')
    ax.step(bins, np.append(nom_xsec, nom_xsec[-1]),
            where='post', color='gray', linewidth=1.5, linestyle='--')
    for i in range(n_bins):
        ax.fill_between([bins[i], bins[i+1]],
                         nom_xsec[i]*0.95, nom_xsec[i]*1.05,
                         color='gray', alpha=0.2,
                         label=('Prior' if i == 0 else None))
    ax.errorbar(centers, unf_xsec, yerr=xsec_unc,
                fmt='o', color='red', markersize=5, capsize=3, linewidth=1.5,
                label='OmniFold')
    ax.set_xlabel(xlabel); ax.set_ylabel(YLABEL_XSEC[var_name])
    ax.set_title(r'SBND $\nu_e$ CC Inclusive'); ax.legend()
    ax.set_xlim(bins[0], bins[-1])
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/xsec_vs_truth_{var_name}.png', dpi=150)
    print(f"    xsec_vs_truth_{var_name}.png"); plt.close()

    # Plot 2: ratio to truth
    fig, ax = plt.subplots(figsize=(8, 6))
    ratio_unf   = unf_xsec / np.where(truth_xsec > 0, truth_xsec, 1)
    ratio_unc   = xsec_unc / np.where(truth_xsec > 0, truth_xsec, 1)
    ratio_prior = nom_xsec / np.where(truth_xsec > 0, truth_xsec, 1)
    for i in range(n_bins):
        ax.fill_between([bins[i], bins[i+1]],
                         ratio_prior[i]-0.02, ratio_prior[i]+0.02,
                         color='gray', alpha=0.3,
                         label=('Prior' if i == 0 else None))
    ax.errorbar(centers, ratio_unf, yerr=ratio_unc,
                fmt='o', color='red', markersize=5, capsize=3, linewidth=1.5,
                label='OmniFold')
    ax.axhline(1.0, color='black', linewidth=1)
    ax.set_xlabel(xlabel); ax.set_ylabel('Ratio to Data Truth')
    ax.legend(); ax.set_xlim(bins[0], bins[-1])
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/ratio_to_truth_{var_name}.png', dpi=150)
    print(f"    ratio_to_truth_{var_name}.png"); plt.close()

    # Plot 3: chi2 convergence — same formula as do_validation (chi2_simple)
    def _chi2(obs, exp):
        mask = exp > 0
        return np.sum((obs[mask] - exp[mask])**2 / exp[mask])
    ndf = n_bins - 1
    pi, pc = [0], [_chi2(nom_hist, truth_hist) / ndf]
    for f in push_files:
        it = iter_num(f)
        push = np.load(f)
        push = push if push.ndim == 1 else push.mean(axis=0)
        h, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
        pi.append(it + 1)
        pc.append(_chi2(h, truth_hist) / ndf)
    print(f"\n  chi2/ndf ({var_name}, ndf={ndf}): prior={pc[0]:.4f}, final={pc[-1]:.4f}")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pi, pc, 'o-', color='red', linewidth=2, markersize=5)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('OmniFold Iteration'); ax.set_ylabel(r'$\chi^2$/DoF')
    ax.set_title(r'$\chi^2$ convergence: SBND $\nu_e$ CC')
    ax.set_yscale('log'); ax.set_ylim(min(v for v in pc if v > 0) * 0.5, max(pc) * 3)
    ax.set_xticks(pi)
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/chi2_convergence_{var_name}.png', dpi=150)
    print(f"    chi2_convergence_{var_name}.png"); plt.close()

    # Plot 4: uncertainty budget
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_src = {'bnb': 'blue', 'genie': 'red', 'mcstat': 'green', 'ml': 'purple'}
    labels_src = {'bnb': 'BNB Flux', 'genie': 'GENIE XSec',
                  'mcstat': 'MC Stat', 'ml': 'ML/NN Init'}
    for src in ['bnb', 'genie', 'mcstat', 'ml']:
        # FIX 1: read from --cov-dir
        cf = f'{flags.cov_dir}/covariance_{src}_{var_name}.npz'
        if not os.path.exists(cf): continue
        d = np.load(cf)
        frac = np.sqrt(np.diag(d['cov'])) / d['mean_hist'].clip(1e-6)
        ax.step(bins, np.append(frac, frac[-1]),
                where='post', color=colors_src[src], linewidth=1.5,
                label=labels_src[src])
    if os.path.exists(cov_file):
        cd = np.load(cov_file)
        frac_all = np.sqrt(np.diag(cd['cov'])) / cd['mean_hist'].clip(1e-6)
        ax.step(bins, np.append(frac_all, frac_all[-1]),
                where='post', color='black', linewidth=2, label='Total')
    ax.set_xlabel(xlabel); ax.set_ylabel('Bin Fractional Uncertainty')
    ax.set_title(r'Uncertainty budget: SBND $\nu_e$ CC')
    ax.legend(); ax.set_xlim(bins[0], bins[-1])
    ax.set_yscale('log'); ax.set_ylim(1e-3, 0.5)
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/uncertainty_budget_{var_name}.png', dpi=150)
    print(f"    uncertainty_budget_{var_name}.png"); plt.close()

    # Plot 5: correlation matrix
    if os.path.exists(cov_file):
        cd = np.load(cov_file); cov_mat = cd['cov']
        diag = np.sqrt(np.diag(cov_mat))
        corr = cov_mat / np.outer(diag.clip(1e-10), diag.clip(1e-10))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, origin='lower', aspect='auto',
                        extent=[0, n_bins, 0, n_bins], vmin=-1, vmax=1, cmap='RdBu_r')
        fmt = '.0f' if var_name == 'true_p' else '.1f'
        tl = [f'[{bins[i]:{fmt}},{bins[i+1]:{fmt}})' for i in range(n_bins)]
        ax.set_xticks(np.arange(n_bins) + 0.5)
        ax.set_xticklabels(tl, fontsize=8, rotation=45, ha='right')
        ax.set_yticks(np.arange(n_bins) + 0.5)
        ax.set_yticklabels(tl, fontsize=8)
        ax.set_title(f'Correlation ({var_name})')
        plt.colorbar(im, ax=ax); plt.tight_layout()
        plt.savefig(f'{flags.plot_dir}/correlation_{var_name}.png', dpi=150)
        print(f"    correlation_{var_name}.png"); plt.close()


def _make_combined_chi2(truth_raw, mc_weights, injected, push_files):
    """Overlay chi2/ndf for both variables. Same computation as individual plots."""
    colors = {'true_p': 'red', 'true_costheta': 'blue'}
    labels = {'true_p': r'$p_e$', 'true_costheta': r'$\cos\theta_e$'}

    curves = {}
    for vn in ['true_p', 'true_costheta']:
        bins   = BINNING[vn]
        n_bins = len(bins) - 1
        ndf    = n_bins - 1          # same as _make_core_plots and do_validation
        idx    = 0 if vn == 'true_p' else 1
        vv     = truth_raw[:, idx]
        th, _  = np.histogram(vv, bins=bins, weights=mc_weights * injected)
        nh, _  = np.histogram(vv, bins=bins, weights=mc_weights)
        pi = [0]
        def _chi2(obs, exp):
            mask = exp > 0
            return np.sum((obs[mask] - exp[mask])**2 / exp[mask])
        pc = [_chi2(nh, th) / ndf]
        for f in push_files:
            it   = iter_num(f)
            push = np.load(f)
            push = push if push.ndim == 1 else push.mean(axis=0)
            h, _ = np.histogram(vv, bins=bins, weights=mc_weights * push)
            pi.append(it + 1)
            pc.append(_chi2(h, th) / ndf)
        curves[vn] = (pi, pc)
        print(f"  Combined chi2/ndf ({vn}, ndf={ndf}): prior={pc[0]:.4f}, final={pc[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    all_vals = []
    for vn, (pi, pc) in curves.items():
        ax.plot(pi, pc, 'o-', color=colors[vn], linewidth=2, markersize=5, label=labels[vn])
        all_vals.extend(pc)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, label=r'$\chi^2$/DoF = 1')
    ax.set_xlabel('OmniFold Iteration'); ax.set_ylabel(r'$\chi^2$/DoF')
    ax.set_title(r'$\chi^2$ convergence: SBND $\nu_e$ CC')
    ax.legend(fontsize=12); ax.set_yscale('log')
    ax.set_ylim(min(v for v in all_vals if v > 0) * 0.5, max(all_vals) * 3)
    ax.set_xticks(pi)
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/chi2_convergence_combined.png', dpi=150)
    print(f"    chi2_convergence_combined.png"); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Reweighting snapshots
# ═══════════════════════════════════════════════════════════════════════════════
def _make_reweighting_snapshots(truth_raw, mc_weights, injected, push_files):
    n_iters = len(push_files)
    snap_paper = sorted(set(s for s in [1, 3, 5, n_iters] if s <= n_iters))
    colors_iter = plt.cm.viridis(np.linspace(0.2, 0.9, len(snap_paper)))

    for var_name in ['true_p', 'true_costheta']:
        bins = BINNING[var_name]; var_idx = 0 if var_name == 'true_p' else 1
        var_vals = truth_raw[:, var_idx]; centers = 0.5 * (bins[:-1] + bins[1:])

        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1]})
        prior_h, _ = np.histogram(var_vals, bins=bins, weights=mc_weights)
        truth_h, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * injected)
        axes[0].step(bins, np.append(prior_h, prior_h[-1]),
                     where='post', color='gray', linewidth=2, linestyle='--', label='Prior')
        axes[0].step(bins, np.append(truth_h, truth_h[-1]),
                     where='post', color='black', linewidth=2.5, label='Data Truth')

        for ci, paper_it in enumerate(snap_paper):
            push = np.load(push_files[paper_it - 1])
            push = push if push.ndim == 1 else push.mean(axis=0)
            h, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
            axes[0].step(bins, np.append(h, h[-1]),
                         where='post', color=colors_iter[ci], linewidth=1.5,
                         label=f'Iter {paper_it}')
            ratio = h / np.where(truth_h > 0, truth_h, 1)
            axes[1].plot(centers, ratio, 'o-', color=colors_iter[ci], markersize=4)

        ratio_prior = prior_h / np.where(truth_h > 0, truth_h, 1)
        axes[1].plot(centers, ratio_prior, 's--', color='gray', markersize=4)
        axes[0].set_ylabel('Weighted events')
        axes[0].set_title(rf'Reweighting snapshots: {var_name}')
        axes[0].legend(fontsize=9, ncol=2)
        axes[1].axhline(1.0, color='black', linewidth=1)
        axes[1].set_xlabel(XLABEL[var_name])
        axes[1].set_ylabel('Ratio to Truth'); axes[1].set_ylim(0.7, 1.3)
        plt.tight_layout()
        plt.savefig(f'{flags.plot_dir}/reweighting_snapshots_{var_name}.png', dpi=150)
        print(f"    reweighting_snapshots_{var_name}.png"); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Per-event weights vs observable
# ═══════════════════════════════════════════════════════════════════════════════
def _make_weight_vs_observable(truth_raw, mc_weights, injected, push_final):
    for var_name in ['true_p', 'true_costheta']:
        bins = BINNING[var_name]; var_idx = 0 if var_name == 'true_p' else 1
        var_vals = truth_raw[:, var_idx]; centers = 0.5 * (bins[:-1] + bins[1:])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        n_plot = min(len(var_vals), 5000)
        idx = np.random.default_rng(42).choice(len(var_vals), n_plot, replace=False)
        sc = axes[0].scatter(var_vals[idx], push_final[idx],
                             c=push_final[idx], cmap='RdBu_r', vmin=0.5, vmax=1.5,
                             s=8, alpha=0.4, edgecolors='none')
        axes[0].axhline(1.0, color='black', linewidth=1, linestyle='--')
        axes[0].set_xlabel(XLABEL[var_name]); axes[0].set_ylabel('Push weight')
        axes[0].set_title(f'Per-event weights: {var_name}')
        axes[0].set_ylim(0, max(3.0, np.percentile(push_final, 99.5) * 1.2))
        plt.colorbar(sc, ax=axes[0], label='Weight')

        bm = np.zeros(len(bins) - 1); bs = np.zeros(len(bins) - 1)
        bmed = np.zeros(len(bins) - 1)
        for i in range(len(bins) - 1):
            mask = (var_vals >= bins[i]) & (var_vals < bins[i + 1])
            if mask.sum() > 0:
                w = push_final[mask]
                bm[i] = w.mean(); bs[i] = w.std(); bmed[i] = np.median(w)
        axes[1].errorbar(centers, bm, yerr=bs, fmt='ro-', capsize=4,
                         linewidth=1.5, markersize=6, label=r'Mean $\pm$ std')
        axes[1].plot(centers, bmed, 'bs--', markersize=5, linewidth=1, label='Median')
        tm = np.zeros(len(bins) - 1)
        for i in range(len(bins) - 1):
            mask = (var_vals >= bins[i]) & (var_vals < bins[i + 1])
            if mask.sum() > 0: tm[i] = injected[mask].mean()
        axes[1].plot(centers, tm, 'g^-', markersize=5, linewidth=1, label='Injected tilt')
        axes[1].axhline(1.0, color='black', linewidth=1, linestyle='--')
        axes[1].set_xlabel(XLABEL[var_name]); axes[1].set_ylabel('Push weight')
        axes[1].set_title(f'Weight profile: {var_name}')
        axes[1].legend(); axes[1].set_ylim(0, 3)
        plt.tight_layout()
        plt.savefig(f'{flags.plot_dir}/weights_vs_observable_{var_name}.png', dpi=150)
        print(f"    weights_vs_observable_{var_name}.png"); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Push weight distributions per iteration
# ═══════════════════════════════════════════════════════════════════════════════
def _make_weight_distributions(push_files):
    n_iters = len(push_files)
    snap = sorted(set(s for s in [1, 3, 5, n_iters] if s <= n_iters))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(snap)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for ci, paper_it in enumerate(snap):
        push = np.load(push_files[paper_it - 1])
        push = push if push.ndim == 1 else push.mean(axis=0)
        ax.hist(push, bins=100, range=(0.5, 2.0), density=True, alpha=0.5,
                color=colors[ci], label=f'Iter {paper_it}')
    ax.set_xlabel('Push weight'); ax.set_ylabel('Density')
    ax.set_title(r'Push weight distributions: SBND $\nu_e$ CC')
    ax.legend(); ax.axvline(1.0, color='black', linewidth=1, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/weight_distributions.png', dpi=150)
    print(f"    weight_distributions.png"); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: 2D weight map
# ═══════════════════════════════════════════════════════════════════════════════
def _make_weight_map_2d(truth_raw, mc_weights, push_final):
    true_p, true_cos = truth_raw[:, 0], truth_raw[:, 1]
    p_bins = BINNING['true_p']; cos_bins = BINNING['true_costheta']
    stat = np.full((len(p_bins)-1, len(cos_bins)-1), np.nan)
    for ip in range(len(p_bins)-1):
        for ic in range(len(cos_bins)-1):
            mask = ((true_p >= p_bins[ip]) & (true_p < p_bins[ip+1]) &
                    (true_cos >= cos_bins[ic]) & (true_cos < cos_bins[ic+1]))
            if mask.sum() > 0:
                stat[ip, ic] = push_final[mask].mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(cos_bins, p_bins, stat, cmap='RdBu_r', vmin=0.8, vmax=1.2)
    ax.set_xlabel(r'True $\cos\theta_e$')
    ax.set_ylabel(r'True electron momentum [MeV/c]')
    ax.set_title('Mean OmniFold push weight (2D)')
    plt.colorbar(im, ax=ax, label='Mean weight')
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/weight_map_2d.png', dpi=150)
    print(f"    weight_map_2d.png"); plt.close()


def _make_weight_change(mc_weights, push_files):
    N = 5
    checkpoints = sorted(set([5, 10, len(push_files)]))
    fig, ax = plt.subplots(figsize=(7, 5))
    colors_cp = ['blue', 'orange', 'green']; ci = 0
    for total_iter in checkpoints:
        if total_iter > len(push_files): continue
        start = max(0, total_iter - N)
        ww = [np.load(f) if np.load(f).ndim == 1 else np.load(f).mean(axis=0)
              for f in push_files[start:total_iter]]
        if len(ww) < 2: continue
        avg_change = np.diff(np.array(ww), axis=0).mean(axis=0)
        ax.hist(avg_change, bins=100, range=(-0.15, 0.15), density=True,
                alpha=0.6, color=colors_cp[ci], label=f'After {total_iter} iters')
        ci += 1
    ax.set_xlabel(f'Avg weight change (last {N} iters)')
    ax.set_ylabel('Fraction'); ax.set_title(r'Weight convergence')
    ax.legend(); plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/weight_change_distribution.png', dpi=150)
    print(f"    weight_change_distribution.png"); plt.close()


def _make_2d_xsec_slices(truth_raw, mc_w, tilt, push):
    true_p, true_cos = truth_raw[:, 0], truth_raw[:, 1]
    cos_slices = [(-1, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.0)]
    p_bins = np.array([0, 200, 400, 600, 800, 1200, 2000])
    ncols = 3; nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 8), squeeze=False)
    for si, (clo, chi) in enumerate(cos_slices):
        ax = axes[si // ncols][si % ncols]
        mask = (true_cos >= clo) & (true_cos < chi)
        if mask.sum() == 0: ax.set_visible(False); continue
        bw = np.diff(p_bins); cw = chi - clo; cen = 0.5 * (p_bins[:-1] + p_bins[1:])
        th, _ = np.histogram(true_p[mask], bins=p_bins, weights=mc_w[mask] * tilt[mask])
        nh, _ = np.histogram(true_p[mask], bins=p_bins, weights=mc_w[mask])
        uh, _ = np.histogram(true_p[mask], bins=p_bins, weights=mc_w[mask] * push[mask])
        ax.step(p_bins, np.append(th/(bw*cw), (th/(bw*cw))[-1]),
                where='post', color='black', linewidth=2, label='Data Truth')
        ax.errorbar(cen, uh/(bw*cw), fmt='o', color='red', markersize=4, capsize=2,
                    label='OmniFold')
        ax.step(p_bins, np.append(nh/(bw*cw), (nh/(bw*cw))[-1]),
                where='post', color='gray', linewidth=1, linestyle='--', label='Prior')
        ax.set_title(f'{clo:.1f} < cos$\\theta$ < {chi:.1f}', fontsize=11)
        ax.set_xlabel('p [MeV/c]', fontsize=10)
        if si == 0: ax.legend(fontsize=8)
    plt.suptitle(r'SBND $\nu_e$ CC: $d^2\sigma / dp\, d\cos\theta$', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/xsec_2d_slices.png', dpi=150)
    print(f"    xsec_2d_slices.png"); plt.close()


def _make_2d_correlation(truth_raw, mc_w):
    true_p, true_cos = truth_raw[:, 0], truth_raw[:, 1]
    p_bins = np.array([0, 500, 1000, 2000])
    cos_bins = np.array([-1, 0, 0.5, 0.75, 1.0])
    n_p, n_c = len(p_bins)-1, len(cos_bins)-1; n_2d = n_p * n_c
    all_flat = []
    for src in ['bnb', 'genie', 'mcstat']:
        pattern = f'{flags.weights_base}/weights_{src}/{src}_univ*/Step2_Iter*_PushWeights.npy'
        files = sorted(glob.glob(pattern))
        ufiles = {}
        for f in files:
            m = re.search(r'univ(\d+)', f)
            if m:
                uid = int(m.group(1)); it = iter_num(f)
                if uid not in ufiles or it > ufiles[uid][0]:
                    ufiles[uid] = (it, f)
        for uid in sorted(ufiles.keys()):
            push = np.load(ufiles[uid][1])
            push = push if push.ndim == 1 else push.mean(axis=0)
            h, _, _ = np.histogram2d(true_cos, true_p, bins=[cos_bins, p_bins],
                                      weights=mc_w * push)
            all_flat.append(h.flatten())
    if len(all_flat) < 2:
        print("  Not enough universes for 2D correlation"); return
    all_flat = np.array(all_flat)
    diff = all_flat - all_flat.mean(axis=0)
    cov_2d = (diff.T @ diff) / len(all_flat)
    diag = np.sqrt(np.diag(cov_2d))
    corr = cov_2d / np.outer(diag.clip(1e-10), diag.clip(1e-10))
    bl = [f'c[{cos_bins[ic]:.1f},{cos_bins[ic+1]:.1f}]\np[{p_bins[ip]:.0f},{p_bins[ip+1]:.0f}]'
          for ic in range(n_c) for ip in range(n_p)]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.pcolormesh(np.arange(n_2d+1), np.arange(n_2d+1), corr,
                        vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(np.arange(n_2d) + 0.5)
    ax.set_xticklabels(bl, fontsize=7, rotation=90)
    ax.set_yticks(np.arange(n_2d) + 0.5)
    ax.set_yticklabels(bl, fontsize=7)
    ax.set_title(r'Correlation: $(p, \cos\theta)$ 2D bins')
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/correlation_2d_p_costheta.png', dpi=150)
    print(f"    correlation_2d_p_costheta.png"); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════════════════
if flags.action in ('validation', 'all'):
    do_validation()

if flags.action in ('paper', 'all'):
    if not hasattr(flags, 'weights_base'):
        flags.weights_base = 'sbnd'
    if not hasattr(flags, 'export_dir'):
        flags.export_dir = 'sbnd/exported_weights/'
    if not hasattr(flags, 'cov_dir'):
        flags.cov_dir = 'sbnd/covariance/'
    do_paper()