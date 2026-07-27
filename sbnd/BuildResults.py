"""
BuildResults.py — Build covariance matrices and extract cross-sections.

Consolidates: BuildCovarianceMatrix, ExtractXsec.

Actions:
  covariance   Build covariance matrices from systematic universes
  xsec         Extract differential cross-sections with uncertainty bands

Key fix vs old BuildCovarianceMatrix.py:
  When --source all, per-source covariance files (covariance_bnb_*.npz etc)
  are now saved automatically — PlotAll.py uncertainty_budget needs them.

Usage:
    python3 sbnd/BuildResults.py covariance --source all --var true_p
    python3 sbnd/BuildResults.py covariance --source ml  --var true_p
    python3 sbnd/BuildResults.py xsec --var both
"""

import numpy as np
import glob, re, os, argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest='action', required=True)

# ── covariance ────────────────────────────────────────────────────────────────
p_cov = sub.add_parser('covariance')
p_cov.add_argument('--source', choices=['bnb', 'genie', 'mcstat', 'ml', 'all'],
                    default='all')
p_cov.add_argument('--var', choices=['true_p', 'true_costheta', 'both'],
                    default='both')
p_cov.add_argument('--data-dir', default='../FormattedData_SBND/')
p_cov.add_argument('--weights-base', default='sbnd')
p_cov.add_argument('--ml-weights-dir', default='sbnd/weights_ml_unc/')
p_cov.add_argument('--plot-dir', default='sbnd/plots_systematics/')
p_cov.add_argument('--cov-dir', default='sbnd/covariance/',
                   help='Directory to save .npz covariance files (separate from plots)')

# ── xsec ──────────────────────────────────────────────────────────────────────
p_xs = sub.add_parser('xsec')
p_xs.add_argument('--var', choices=['true_p', 'true_costheta', 'both'],
                   default='both')
p_xs.add_argument('--data-dir', default='../FormattedData_SBND/')
p_xs.add_argument('--weights-base', default='sbnd')
p_xs.add_argument('--export-dir', default='sbnd/exported_weights/')
p_xs.add_argument('--plot-dir', default='sbnd/plots_xsec/')
p_xs.add_argument('--cov-dir', default='sbnd/covariance/')
p_xs.add_argument('--tag', default='tilt_alpha0.5')

flags = parser.parse_args()

# ═══════════════════════════════════════════════════════════════════════════════
# Shared
# ═══════════════════════════════════════════════════════════════════════════════
BINNING = {
    'true_p':        np.array([0, 200, 400, 600, 800, 1000, 1400, 2000]),
    'true_costheta': np.linspace(-1, 1, 11),
}
XLABEL = {
    'true_p':        r'True electron momentum [MeV/c]',
    'true_costheta': r'True $\cos\theta_e$',
}

def iter_num(p):
    m = re.search(r'Iter(\d+)', p)
    return int(m.group(1)) if m else -1


# ═══════════════════════════════════════════════════════════════════════════════
# covariance
# ═══════════════════════════════════════════════════════════════════════════════
def build_covariance_single_var(var_name):
    bins   = BINNING[var_name]
    xlabel = XLABEL[var_name]
    n_bins = len(bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])

    truth_raw  = np.load(flags.data_dir + 'mc_vals_truth_NoNorm.npy')
    mc_weights = np.load(flags.data_dir + 'mc_weights_reco.npy')
    var_idx  = 0 if var_name == 'true_p' else 1
    var_vals = truth_raw[:, var_idx]
    nom_hist, _ = np.histogram(var_vals, bins=bins, weights=mc_weights)

    os.makedirs(flags.plot_dir, exist_ok=True)
    cov_dir = flags.cov_dir
    os.makedirs(cov_dir, exist_ok=True)

    # ── Determine sources ─────────────────────────────────────────────────────
    if flags.source == 'all':
        sources = ['bnb', 'genie', 'mcstat']
        sources = [s for s in sources
                   if glob.glob(f'{flags.weights_base}/weights_{s}/{s}_univ*/Step2_Iter*_PushWeights.npy')]
        ml_dirs = sorted(glob.glob(flags.ml_weights_dir + 'replica_*/'))
        if ml_dirs:
            sources.append('ml')
    elif flags.source == 'ml':
        sources = ['ml']
        ml_dirs = sorted(glob.glob(flags.ml_weights_dir + 'replica_*/'))
    else:
        sources = [flags.source]
        ml_dirs = []

    # ── Collect per-source ────────────────────────────────────────────────────
    all_hists = []
    hists_by_source = {}
    all_univ_dirs_by_source = {}

    for src in sources:
        src_hists = []

        if src == 'ml':
            print(f"ml: {len(ml_dirs)} replicas found")
            all_univ_dirs_by_source['ml'] = ml_dirs
            for rdir in ml_dirs:
                pf = sorted(glob.glob(rdir + 'Step2_Iter*_PushWeights.npy'), key=iter_num)
                if not pf:
                    continue
                push = np.load(pf[-1])
                push = push if push.ndim == 1 else push.mean(axis=0)
                h, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
                all_hists.append(h)
                src_hists.append(h)
        else:
            pattern = f'{flags.weights_base}/weights_{src}/{src}_univ*/Step2_Iter*_PushWeights.npy'
            pfiles = sorted(glob.glob(pattern))
            univ_files = {}
            for f in pfiles:
                m = re.search(r'univ(\d+)', f)
                if m:
                    uid = int(m.group(1))
                    it = iter_num(f)
                    if uid not in univ_files or it > univ_files[uid][0]:
                        univ_files[uid] = (it, f)
            print(f"{src}: {len(univ_files)} universes found")
            udirs = sorted(glob.glob(f'{flags.weights_base}/weights_{src}/{src}_univ*/'))
            all_univ_dirs_by_source[src] = udirs
            for uid in sorted(univ_files.keys()):
                push = np.load(univ_files[uid][1])
                push = push if push.ndim == 1 else push.mean(axis=0)
                h, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
                all_hists.append(h)
                src_hists.append(h)

        hists_by_source[src] = np.array(src_hists) if src_hists else np.array([])

    all_hists = np.array(all_hists)
    n_univ = len(all_hists)
    print(f"Total universes ({var_name}): {n_univ}")
    if n_univ == 0:
        print("ERROR: No universe results found."); return

    # ── Combined statistics ───────────────────────────────────────────────────
    mean_hist = all_hists.mean(axis=0)
    diff = all_hists - mean_hist[np.newaxis, :]
    cov  = (diff.T @ diff) / n_univ
    frac_cov = np.zeros_like(cov)
    for i in range(n_bins):
        for j in range(n_bins):
            d = mean_hist[i] * mean_hist[j]
            if d > 0:
                frac_cov[i, j] = cov[i, j] / d
    diag_unc = np.sqrt(np.diag(cov))
    frac_unc = np.sqrt(np.diag(frac_cov))

    # ── Save combined ─────────────────────────────────────────────────────────
    np.savez(f'{cov_dir}/covariance_{flags.source}_{var_name}.npz',
             cov=cov, frac_cov=frac_cov, bins=bins,
             mean_hist=mean_hist, nom_hist=nom_hist, all_hists=all_hists)

    # ── BUG FIX: save per-source when --source all ────────────────────────────
    if flags.source == 'all':
        for src, src_arr in hists_by_source.items():
            if len(src_arr) < 2:
                continue
            sm = src_arr.mean(axis=0)
            sd = src_arr - sm[np.newaxis, :]
            sc = (sd.T @ sd) / len(src_arr)
            sf = np.zeros_like(sc)
            for i in range(n_bins):
                for j in range(n_bins):
                    d = sm[i] * sm[j]
                    if d > 0:
                        sf[i, j] = sc[i, j] / d
            np.savez(f'{cov_dir}/covariance_{src}_{var_name}.npz',
                     cov=sc, frac_cov=sf, bins=bins,
                     mean_hist=sm, nom_hist=nom_hist, all_hists=src_arr)
            print(f"  Per-source saved: covariance_{src}_{var_name}.npz "
                  f"({len(src_arr)} universes)")

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'Bin center':>10s} {'Nominal':>10s} {'Mean':>10s} "
          f"{'Abs unc':>10s} {'Frac unc':>10s}")
    for i in range(n_bins):
        print(f"{centers[i]:10.1f} {nom_hist[i]:10.1f} {mean_hist[i]:10.1f} "
              f"{diag_unc[i]:10.2f} {frac_unc[i]:10.4f}")

    # ── Plot 1: Covariance matrices ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].imshow(cov, origin='lower', aspect='auto',
                          extent=[bins[0], bins[-1], bins[0], bins[-1]])
    axes[0].set_title(f'Covariance ({flags.source})')
    axes[0].set_xlabel(xlabel); axes[0].set_ylabel(xlabel)
    plt.colorbar(im0, ax=axes[0])
    vmax = max(abs(frac_cov.min()), abs(frac_cov.max()), 0.01)
    im1 = axes[1].imshow(frac_cov, origin='lower', aspect='auto',
                          extent=[bins[0], bins[-1], bins[0], bins[-1]],
                          vmin=-vmax, vmax=vmax, cmap='RdBu_r')
    axes[1].set_title(f'Fractional covariance ({flags.source})')
    axes[1].set_xlabel(xlabel); axes[1].set_ylabel(xlabel)
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/cov_matrix_{flags.source}_{var_name}.png', dpi=150)
    plt.close()

    # ── Plot 2: Unfolded spectrum with systematic band ────────────────────────
    fig2, axes2 = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})
    axes2[0].step(bins, np.append(nom_hist, nom_hist[-1]),
                  where='post', color='blue', linewidth=1.5, label='Nominal MC')
    axes2[0].step(bins, np.append(mean_hist, mean_hist[-1]),
                  where='post', color='red', linewidth=1.5, label=f'Mean ({flags.source})')
    for i in range(n_bins):
        axes2[0].fill_between([bins[i], bins[i+1]],
                               mean_hist[i]-diag_unc[i], mean_hist[i]+diag_unc[i],
                               color='red', alpha=0.2,
                               label=(r'$\pm 1\sigma$' if i == 0 else None))
    axes2[0].set_ylabel('Unfolded weighted events'); axes2[0].legend(fontsize=10)
    axes2[0].set_title(f'OmniFold unfolded: {var_name} ({flags.source})')
    axes2[1].step(bins, np.append(frac_unc, frac_unc[-1]),
                  where='post', color='red', linewidth=1.5)
    for i in range(n_bins):
        axes2[1].fill_between([bins[i], bins[i+1]], 0, frac_unc[i], color='red', alpha=0.2)
    axes2[1].set_xlabel(xlabel); axes2[1].set_ylabel('Frac. unc.')
    axes2[1].set_ylim(0, max(frac_unc) * 1.5)
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/unfolded_with_unc_{flags.source}_{var_name}.png', dpi=150)
    plt.close()

    # ── Plot 3: Universe spread (spaghetti) ───────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for i in range(min(n_univ, 50)):
        ax3.step(bins, np.append(all_hists[i], all_hists[i][-1]),
                 where='post', color='gray', alpha=0.15, linewidth=0.5)
    ax3.step(bins, np.append(nom_hist, nom_hist[-1]),
             where='post', color='blue', linewidth=2, label='Nominal')
    ax3.step(bins, np.append(mean_hist, mean_hist[-1]),
             where='post', color='red', linewidth=2, linestyle='--', label='Mean')
    ax3.set_xlabel(xlabel); ax3.set_ylabel('Unfolded weighted events')
    ax3.set_title(f'Universe spread: {var_name} ({flags.source}, {n_univ} univ)')
    ax3.legend(); plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/universe_spread_{flags.source}_{var_name}.png', dpi=150)
    plt.close()

    # ── Plot 4: Chi2 vs iteration ─────────────────────────────────────────────
    all_udirs = []
    for src in sources:
        all_udirs.extend(all_univ_dirs_by_source.get(src, []))
    if not all_udirs:
        print(f"  Skipping chi2 vs iteration (no universe dirs)")
        return

    sample_files = sorted(glob.glob(all_udirs[0] + 'Step2_Iter*_PushWeights.npy'), key=iter_num)
    max_iter = max(iter_num(f) for f in sample_files) if sample_files else -1
    if max_iter < 0:
        return

    ndf = n_bins - 1
    paper_iters = [0]
    paper_chi2  = [float('nan')]

    # Prior chi2
    hists_i0 = []
    for udir in all_udirs:
        pf = glob.glob(udir + 'Step2_Iter0_*_PushWeights.npy')
        if not pf: continue
        push = np.load(pf[0])
        push = push if push.ndim == 1 else push.mean(axis=0)
        h, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
        hists_i0.append(h)
    if len(hists_i0) >= 2:
        arr0 = np.array(hists_i0)
        d0 = arr0 - arr0.mean(axis=0)
        c0 = (d0.T @ d0) / len(arr0) + np.eye(n_bins) * 1e-6 * np.diag((d0.T @ d0) / len(arr0)).mean()
        try:
            dp = nom_hist - arr0.mean(axis=0)
            paper_chi2[0] = float(dp @ np.linalg.inv(c0) @ dp)
        except np.linalg.LinAlgError:
            pass

    for it in range(max_iter + 1):
        hists_it = []
        for udir in all_udirs:
            pf = glob.glob(udir + f'Step2_Iter{it}_*_PushWeights.npy')
            if not pf: continue
            push = np.load(pf[0])
            push = push if push.ndim == 1 else push.mean(axis=0)
            h, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
            hists_it.append(h)
        if len(hists_it) < 2:
            paper_iters.append(it + 1); paper_chi2.append(float('nan')); continue
        arr = np.array(hists_it)
        mu = arr.mean(axis=0)
        di = arr - mu
        ci = (di.T @ di) / len(arr) + np.eye(n_bins) * 1e-6 * np.diag((di.T @ di) / len(arr)).mean()
        try:
            c2 = float((mu - nom_hist) @ np.linalg.inv(ci) @ (mu - nom_hist))
        except np.linalg.LinAlgError:
            c2 = float('nan')
        paper_iters.append(it + 1); paper_chi2.append(c2)

    print(f"\n  Systematic stability chi2 ({var_name}):")
    print(f"  Measures: (universe_mean - nominal)^T C^-1 (universe_mean - nominal)")
    print(f"  Interpretation: how much the systematic universes SHIFT the result vs nominal.")
    print(f"  Small (~0.05) and flat = good: systematics are stable across OmniFold iterations.")
    print(f"  This is NOT the fake-data recovery chi2 (see PlotAll.py validation for that).")
    print(f"  {'Iter':>5s} {'chi2':>10s}  note")
    for pi, c2 in zip(paper_iters, paper_chi2):
        s = f"{c2:10.2f}" if not np.isnan(c2) else "       N/A"
        note = "(prior, no unfolding)" if pi == 0 else ""
        print(f"  {pi:5d} {s}  {note}")

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    valid = [(pi, c2) for pi, c2 in zip(paper_iters, paper_chi2) if not np.isnan(c2)]
    if valid:
        vi, vc = zip(*valid)
        ax4.plot(vi, vc, 'ro-', linewidth=2, markersize=6)
    ax4.set_xlabel('OmniFold Iteration'); ax4.set_ylabel(r'$\chi^2$')
    ax4.set_title(f'Convergence: {var_name} ({flags.source})')
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/chi2_vs_iter_{flags.source}_{var_name}.png', dpi=150)
    plt.close()

    print(f"  Plots saved to {flags.plot_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
# xsec
# ═══════════════════════════════════════════════════════════════════════════════
INTEGRATED_FLUX_PER_POT = 1.0
TARGET_POT              = 6.6e20
N_TARGETS               = 1.0

def extract_xsec_single_var(var_name):
    USE_ABSOLUTE = (INTEGRATED_FLUX_PER_POT != 1.0 and N_TARGETS != 1.0)
    NORM = INTEGRATED_FLUX_PER_POT * TARGET_POT * N_TARGETS if USE_ABSOLUTE else 1.0

    bins       = BINNING[var_name]
    xlabel     = XLABEL[var_name]
    n_bins     = len(bins) - 1
    bin_widths = np.diff(bins)
    centers    = 0.5 * (bins[:-1] + bins[1:])
    ylabel     = (r'd$\sigma$/d$p$ [arb. / (MeV/c)]' if var_name == 'true_p'
                  else r'd$\sigma$/d$\cos\theta$ [arb.]')

    truth_raw  = np.load(flags.data_dir + 'mc_vals_truth_NoNorm.npy')
    mc_weights = np.load(flags.data_dir + 'mc_weights_reco.npy')
    var_idx  = 0 if var_name == 'true_p' else 1
    var_vals = truth_raw[:, var_idx]

    os.makedirs(flags.plot_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Cross-section extraction: {var_name}")
    print(f"{'='*60}")

    # Efficiency
    eff_file = flags.export_dir + f'efficiency_{var_name}.npy'
    eff = np.load(eff_file) if os.path.exists(eff_file) else np.ones(n_bins)

    # Nominal
    N_nom, _ = np.histogram(var_vals, bins=bins, weights=mc_weights)
    xsec_nom = N_nom / (eff.clip(1e-6) * bin_widths * NORM)

    # OmniFold push weights
    tilt_dir = f'weights_sbnd_fakedata_{flags.tag}/'
    push_files = sorted(glob.glob(tilt_dir + 'Step2_Iter*_PushWeights.npy'), key=iter_num)
    has_unf = bool(push_files)
    if has_unf:
        push = np.load(push_files[-1])
        push = push if push.ndim == 1 else push.mean(axis=0)
        N_unf, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
        xsec_unf = N_unf / (eff.clip(1e-6) * bin_widths * NORM)
    else:
        xsec_unf = xsec_nom

    # Data Truth
    tilt_file = flags.data_dir + f'truth_weights_sbnd_fakedata_{flags.tag}.npy'
    has_truth = os.path.exists(tilt_file)
    if has_truth:
        tilt = np.load(tilt_file)
        N_truth, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * tilt)
        xsec_truth = N_truth / (eff.clip(1e-6) * bin_widths * NORM)

    # Systematic covariance
    cov_file = getattr(flags, 'cov_dir', 'sbnd/covariance/') + f'/covariance_all_{var_name}.npz'
    has_syst = os.path.exists(cov_file)
    if has_syst:
        cd = np.load(cov_file)
        scale = eff.clip(1e-6) * bin_widths * NORM
        cov_xsec = cd['cov'] / np.outer(scale, scale)
        xsec_unc = np.sqrt(np.diag(cov_xsec))
    else:
        xsec_unc = np.zeros(n_bins)

    # ── Plot: xsec with syst band ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})
    axes[0].step(bins, np.append(xsec_nom, xsec_nom[-1]),
                 where='post', color='gray', linewidth=1.5, linestyle='--',
                 label='Nominal MC (prior)')
    if has_truth:
        axes[0].step(bins, np.append(xsec_truth, xsec_truth[-1]),
                     where='post', color='black', linewidth=2, label='Data Truth')
    if has_unf:
        axes[0].errorbar(centers, xsec_unf, yerr=xsec_unc if has_syst else None,
                         fmt='ro', markersize=5, capsize=3, linewidth=1.5,
                         label='OmniFold')
    axes[0].set_ylabel(ylabel); axes[0].legend(fontsize=10)
    axes[0].set_title(rf'SBND $\nu_e$ CC: {var_name}')
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    xsec_mean = (np.load(cov_file)['mean_hist'] / (eff.clip(1e-6) * bin_widths * NORM)
                 if has_syst else xsec_nom)
    frac_unc = xsec_unc / xsec_mean.clip(1e-30)
    if has_syst:
        axes[1].step(bins, np.append(frac_unc, frac_unc[-1]),
                     where='post', color='red', linewidth=1.5)
        for i in range(n_bins):
            axes[1].fill_between([bins[i], bins[i+1]], 0, frac_unc[i],
                                  color='red', alpha=0.2)
    axes[1].set_xlabel(xlabel); axes[1].set_ylabel('Frac. unc.')
    axes[1].set_ylim(0, max(frac_unc.max() * 1.5, 0.05) if has_syst else 0.1)
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/xsec_{var_name}.png', dpi=150)
    print(f"  Saved xsec_{var_name}.png"); plt.close()

    # ── Efficiency plot ──────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.step(bins, np.append(eff, eff[-1]), where='post', color='black', linewidth=2)
    for i in range(n_bins):
        ax2.fill_between([bins[i], bins[i+1]], 0, eff[i], color='steelblue', alpha=0.3)
    ax2.set_xlabel(xlabel); ax2.set_ylabel('Selection efficiency')
    ax2.set_ylim(0, 1); ax2.set_title(f'Efficiency: {var_name}')
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/efficiency_{var_name}.png', dpi=150)
    print(f"  Saved efficiency_{var_name}.png"); plt.close()

    # ── Ratio to truth ───────────────────────────────────────────────────────
    if has_unf and has_truth:
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ratio = xsec_unf / np.where(xsec_truth > 0, xsec_truth, 1)
        ratio_unc = xsec_unc / np.where(xsec_truth > 0, xsec_truth, 1)
        ratio_nom = xsec_nom / np.where(xsec_truth > 0, xsec_truth, 1)
        for i in range(n_bins):
            ax3.fill_between([bins[i], bins[i+1]],
                              ratio_nom[i]-0.02, ratio_nom[i]+0.02,
                              color='gray', alpha=0.3,
                              label=('Prior' if i == 0 else None))
        ax3.errorbar(centers, ratio, yerr=ratio_unc,
                     fmt='ro', markersize=5, capsize=3, linewidth=1.5,
                     label='OmniFold')
        ax3.axhline(1.0, color='black', linewidth=1)
        ax3.set_xlabel(xlabel); ax3.set_ylabel('Ratio to Data Truth')
        ax3.legend(); ax3.set_xlim(bins[0], bins[-1])
        plt.tight_layout()
        plt.savefig(f'{flags.plot_dir}/ratio_to_truth_{var_name}.png', dpi=150)
        print(f"  Saved ratio_to_truth_{var_name}.png"); plt.close()

    # Print table
    print(f"\n  {'Bin':>14s} {'Nominal':>12s} {'OmniFold':>12s} "
          f"{'Syst unc':>12s} {'Frac unc':>10s}")
    for i in range(n_bins):
        print(f"  [{bins[i]:6.0f},{bins[i+1]:6.0f}] {xsec_nom[i]:12.4e} "
              f"{xsec_unf[i]:12.4e} {xsec_unc[i]:12.4e} {frac_unc[i]:10.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════════════════
if flags.action == 'covariance':
    vars_to_run = (['true_p', 'true_costheta'] if flags.var == 'both'
                   else [flags.var])
    for v in vars_to_run:
        build_covariance_single_var(v)

elif flags.action == 'xsec':
    if not hasattr(flags, 'export_dir'):
        flags.export_dir = 'sbnd/exported_weights/'
    if not hasattr(flags, 'tag'):
        flags.tag = 'tilt_alpha0.5'
    vars_to_run = (['true_p', 'true_costheta'] if flags.var == 'both'
                   else [flags.var])
    for v in vars_to_run:
        extract_xsec_single_var(v)