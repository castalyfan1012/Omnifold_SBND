"""
MakePlots.py — SBND OmniFold validation and results plots.

Actions:
  validation   Fake-data recovery, chi2 convergence, PASS/FAIL verdict
  results      Cross-section, uncertainty budget, correlations, 2D slices
  all          Both validation and results

Tag convention (derived from --var and --alpha):
  --var true_p        --alpha 0.3  =>  tilt_p_alpha0.3
  --var true_costheta --alpha 0.3  =>  tilt_costheta_alpha0.3
  --var both          --alpha 0.3  =>  tilt_both_alpha0.3

Examples:
  python3 sbnd/MakePlots.py validation --var true_p --alpha 0.3
  python3 sbnd/MakePlots.py validation --var both --alpha 0.15
  python3 sbnd/MakePlots.py validation --var both --alpha 0.3 --extended
  python3 sbnd/MakePlots.py results --var true_p --alpha 0.3 --cov-source fds
  python3 sbnd/MakePlots.py all --var true_p --alpha 0.3 --cov-source fds
"""

import numpy as np
import glob, re, os, json, argparse
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
p_val.add_argument('--var', choices=['true_p', 'true_costheta', 'both'], required=True,
                   help="Which variable was tilted in the fake-data study. "
                        "Used for plot titles and to derive the tag. "
                        "Validation ALWAYS plots both kinematic variables.")
p_val.add_argument('--alpha', type=float, default=0.3,
                   help="Tilt strength (used to derive the tag: tilt_{var}_alpha{alpha}).")
p_val.add_argument('--tag', default=None,
                   help="Override tag (default: derived from --var and --alpha).")
p_val.add_argument('--data-dir', default='../FormattedData_SBND/')
p_val.add_argument('--weights-dir', default=None)
p_val.add_argument('--plot-dir', default='sbnd/plots_validation/')
p_val.add_argument('--cov-dir', default='sbnd/covariance/')
p_val.add_argument('--fds-cov-source', default='fds',
                   help="Combined covariance used for the fake-data band/chi2 "
                        "(stat+xsec). Falls back to analytic stat if absent.")
p_val.add_argument('--pval-thresh', type=float, default=0.05)
p_val.add_argument('--iter-rel-tol', type=float, default=0.05)
p_val.add_argument('--extended', action='store_true',
                   help="Also produce exclude-lowest-bin and merged-backward-cosθ "
                        "diagnostic plots (not shown by default).")

p_pap = sub.add_parser('results')
p_pap.add_argument('--tag', default=None,
                   help="Fake-data tag. If omitted, derived from --var and --alpha.")
p_pap.add_argument('--var', choices=['true_p', 'true_costheta', 'both'], default='true_p',
                   help="Which variable was tilted (used to derive tag if --tag is omitted). "
                        "Results are ALWAYS plotted for both kinematic variables.")
p_pap.add_argument('--alpha', type=float, default=0.3,
                   help="Tilt strength (used to derive tag if --tag is omitted).")
p_pap.add_argument('--data-dir', default='../FormattedData_SBND/')
p_pap.add_argument('--weights-base', default='sbnd')
p_pap.add_argument('--export-dir', default='sbnd/exported_weights/')
p_pap.add_argument('--plot-dir', default='sbnd/plots_xsec/')
p_pap.add_argument('--syst-dir', default='sbnd/plots_syst/')
p_pap.add_argument('--val-dir', default='sbnd/plots_validation/')
p_pap.add_argument('--cov-dir', default='sbnd/covariance/')
p_pap.add_argument('--cov-source', default='all',
                   help="Combined covariance for xsec result plots. "
                        "'all' = full budget (flux+xsec+det+stat). "
                        "'fds' = stat+xsec only (appropriate for fake-data studies).")
p_pap.add_argument('--wsvd-syst-file', default=None,
                   help="Optional JSON of Wiener-SVD per-source fractional unc for "
                        "the comparison plot. Format: {var:{source:[per-bin frac unc]}}.")
p_pap.add_argument('--ml-weights-dir', default='sbnd/weights_ml_unc/',
                   help='Directory containing replica_*/ subdirs.')

p_all = sub.add_parser('all')
p_all.add_argument('--var', choices=['true_p', 'true_costheta', 'both'], default='true_p')
p_all.add_argument('--alpha', type=float, default=0.3)
p_all.add_argument('--tag', default=None)
p_all.add_argument('--data-dir', default='../FormattedData_SBND/')
p_all.add_argument('--weights-base', default='sbnd')
p_all.add_argument('--export-dir', default='sbnd/exported_weights/')
p_all.add_argument('--plot-dir', default='sbnd/plots_xsec/')
p_all.add_argument('--syst-dir', default='sbnd/plots_syst/')
p_all.add_argument('--val-dir', default='sbnd/plots_validation/')
p_all.add_argument('--cov-dir', default='sbnd/covariance/')
p_all.add_argument('--cov-source', default='all')
p_all.add_argument('--fds-cov-source', default='fds')
p_all.add_argument('--wsvd-syst-file', default=None)
p_all.add_argument('--val-plot-dir', default='sbnd/plots_validation/')
p_all.add_argument('--ml-weights-dir', default='sbnd/weights_ml_unc/')
p_all.add_argument('--pval-thresh', type=float, default=0.05)
p_all.add_argument('--iter-rel-tol', type=float, default=0.05)
p_all.add_argument('--extended', action='store_true')

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
SRC_COLOR = {'bnb':'blue','genie':'red','extra_xsec':'orange','g4':'brown',
             'mcstat':'green','ml':'purple','stat':'black'}
SRC_LABEL = {'bnb':'BNB Flux','genie':'GENIE XSec','extra_xsec':'Extra XSec',
             'g4':'G4 Reint.','mcstat':'MC Stat','ml':'ML/NN Init','stat':'MC Stat (analytic)'}

TILT_SHORT = {'true_p': 'p', 'true_costheta': 'costheta', 'both': 'both'}


def make_fdt_tag(var, alpha):
    """Derive the fake-data tag from --var and --alpha."""
    return f'tilt_{TILT_SHORT[var]}_alpha{alpha}'


def tilt_label(var, alpha):
    """Human-readable label for plot titles: 'Tilted: p only, α=0.3'."""
    names = {'true_p': 'p only', 'true_costheta': 'cosθ only', 'both': 'p + cosθ'}
    return f'Tilted: {names[var]}, α={alpha}'


def iter_num(p):
    m = re.search(r'Iter(\d+)', p)
    return int(m.group(1)) if m else -1


def chi2_pvalue(chi2, ndf):
    """Survival function of the chi2 distribution (scipy, with a safe fallback)."""
    if ndf is None or ndf <= 0 or not np.isfinite(chi2):
        return float('nan')
    try:
        from scipy import stats
        return float(stats.chi2.sf(chi2, ndf))
    except Exception:
        import math
        k, x = float(ndf), float(chi2)
        t = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
        return float(0.5 * math.erfc(t / math.sqrt(2.0)))


def fit_annotation(chi2, ndf):
    """One-line chi2/ndf + p-value string for plot annotations (Q11)."""
    p = chi2_pvalue(chi2, ndf)
    return rf'$\chi^2$/ndf = {chi2:.2f}/{ndf} = {chi2/ndf:.2f},  p = {p:.3f}'


def cov_chi2(unf, truth, cov):
    """Full-covariance chi2 with a diagonal fallback if the matrix is singular."""
    d = np.asarray(unf) - np.asarray(truth)
    try:
        return float(d @ np.linalg.inv(cov) @ d)
    except np.linalg.LinAlgError:
        return float(np.sum(d ** 2 / np.diag(cov).clip(1e-30)))


def recommend_iteration(iters, chi2ndf, rel_tol=0.05):
    """Q5: pick a stopping iteration from the chi2/ndf-vs-iteration curve.

    Returns (global_min_iter, plateau_iter). The plateau is the first iteration
    where the relative improvement over the previous iteration drops below
    rel_tol — beyond it, further iterations mostly add variance rather than
    reduce bias, which is why we don't necessarily run to the very last one.
    """
    it = np.asarray(iters); cv = np.asarray(chi2ndf, float)
    m = it > 0
    itv, cvv = it[m], cv[m]
    if len(cvv) == 0:
        return None, None
    gmin_iter = int(itv[int(np.nanargmin(cvv))])
    plateau_iter = int(itv[-1])
    for k in range(1, len(cvv)):
        if not (np.isfinite(cvv[k]) and np.isfinite(cvv[k - 1])):
            continue
        rel = (cvv[k - 1] - cvv[k]) / max(abs(cvv[k - 1]), 1e-12)
        if rel < rel_tol:
            plateau_iter = int(itv[k]); break
    return gmin_iter, plateau_iter


def dedup_push_files(files):
    """Keep only the last-trial file per iteration number."""
    from collections import defaultdict
    by_iter = defaultdict(list)
    for f in files:
        by_iter[iter_num(f)].append(f)
    deduped = [sorted(v)[-1] for k, v in sorted(by_iter.items())]
    return deduped


def chi2_simple(obs, exp):
    mask = exp > 0
    return np.sum((obs[mask] - exp[mask])**2 / exp[mask])


def _load_cov(cov_dir, source, var_name):
    """Return (cov, mean_hist) for a combined/source covariance file, or (None, None)."""
    path = f'{cov_dir}/covariance_{source}_{var_name}.npz'
    if not os.path.exists(path):
        return None, None
    d = np.load(path)
    return d['cov'], d['mean_hist']


# ═══════════════════════════════════════════════════════════════════════════════
# validation
# ═══════════════════════════════════════════════════════════════════════════════
def do_validation():
    TILTED_VAR = flags.var
    ALPHA      = getattr(flags, 'alpha', 0.3)
    TAG        = flags.tag or make_fdt_tag(TILTED_VAR, ALPHA)
    TILT_DESC  = tilt_label(TILTED_VAR, ALPHA)
    DATA_DIR   = flags.data_dir
    WEIGHTS_DIR = flags.weights_dir or f'weights_sbnd_fakedata_{TAG}/'
    PLOT_DIR    = getattr(flags, 'val_plot_dir', None) or flags.plot_dir
    COV_DIR     = getattr(flags, 'cov_dir', 'sbnd/covariance/')
    FDS_SRC     = getattr(flags, 'fds_cov_source', 'fds')
    PVAL_THR    = getattr(flags, 'pval_thresh', 0.05)
    REL_TOL     = getattr(flags, 'iter_rel_tol', 0.05)
    EXTENDED    = getattr(flags, 'extended', False)
    os.makedirs(PLOT_DIR, exist_ok=True)

    truth_raw  = np.load(DATA_DIR + 'mc_vals_truth_NoNorm.npy')
    mc_weights = np.load(DATA_DIR + 'mc_weights_reco.npy')
    tilt_path  = DATA_DIR + f'truth_weights_sbnd_fakedata_{TAG}.npy'
    if not os.path.exists(tilt_path):
        print(f"ERROR: Truth weights not found: {tilt_path}")
        print(f"  Run: python3 sbnd/RunStudies.py make-fakedata --mode tilt "
              f"--var {TILTED_VAR} --alpha {ALPHA}")
        return
    injected_tilt = np.load(tilt_path)
    true_p, true_costheta = truth_raw[:, 0], truth_raw[:, 1]

    push_files = dedup_push_files(sorted(glob.glob(WEIGHTS_DIR + 'Step2_Iter*_PushWeights.npy'), key=iter_num))
    if not push_files:
        print(f"ERROR: No push files found in '{WEIGHTS_DIR}'")
        print(f"  OmniFold has not been trained for tag '{TAG}' yet.")
        return

    push_final = np.load(push_files[-1])
    push_mean  = push_final if push_final.ndim == 1 else push_final.mean(axis=0)
    n_iters_done = iter_num(push_files[-1]) + 1
    print(f"=== Validation: {TAG} ({n_iters_done} iterations) ===")
    print(f"  Push: mean={push_mean.mean():.4f}, std={push_mean.std():.4f}")

    def binned_mean(x, w, bins):
        out = np.zeros(len(bins) - 1)
        for i in range(len(bins) - 1):
            mask = (x >= bins[i]) & (x < bins[i + 1])
            if mask.sum() > 0: out[i] = np.average(w[mask])
        return out

    # Always validate BOTH kinematic variables regardless of what was tilted.
    var_data = {'true_p': true_p, 'true_costheta': true_costheta}

    # ── Weight-recovery plots (unchanged) ─────────────────────────────────────
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
        axes[0].set_title(f'Recovery: {var_name}  ({TILT_DESC})')
        ratio = push_binned / np.where(tilt_binned > 0, tilt_binned, 1.0)
        axes[1].plot(centers, ratio, 'k-o')
        axes[1].axhline(1.0, color='gray', linestyle='--')
        axes[1].fill_between(centers, 0.8, 1.2, alpha=0.15, color='green')
        axes[1].set_xlabel(XLABEL[var_name]); axes[1].set_ylabel('Push / Injected')
        axes[1].set_ylim(0.5, 1.5); plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/fakedata_{TAG}_recovery_{var_name}.png', dpi=150); plt.close()
        print(f"  fakedata_{TAG}_recovery_{var_name}.png")

    # ── Unfolded distributions with proper stat error + stat+xsec band (Q4/Q7) ─
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fds_status = {}
    for ax, (vn, vv) in zip(axes, var_data.items()):
        bins = BINNING[vn]; centers = 0.5 * (bins[:-1] + bins[1:]); n_bins = len(bins) - 1
        nom_h, _ = np.histogram(vv, bins=bins, weights=mc_weights)
        fd_h, _  = np.histogram(vv, bins=bins, weights=mc_weights * injected_tilt)
        unf_h, _ = np.histogram(vv, bins=bins, weights=mc_weights * push_mean)
        # proper statistical error: sqrt(sum of (w*push)^2)
        unf_w2, _ = np.histogram(vv, bins=bins, weights=(mc_weights * push_mean) ** 2)
        stat_err = np.sqrt(np.maximum(unf_w2, 0.0))

        # Band: use fds (stat+xsec) covariance if available
        cov_band, _ = _load_cov(COV_DIR, FDS_SRC, vn)
        band_src = FDS_SRC
        if cov_band is not None:
            band = np.sqrt(np.diag(cov_band))
        else:
            band = stat_err
            band_src = 'stat'

        # Verdict chi2: use STAT covariance (diagonal) — the systematic cov
        # is centered on nominal MC, not on the fake-data result, so its
        # correlated inverse inflates the chi2 artificially.
        stat_cov = np.diag(unf_w2.clip(1e-30))
        ndf = n_bins - 1
        c2 = cov_chi2(unf_h, fd_h, stat_cov)
        p = chi2_pvalue(c2, ndf)
        fds_status[vn] = (c2, ndf, p)

        ax.step(bins, np.append(nom_h, nom_h[-1]), where='post', color='gray',
                linewidth=1.5, linestyle='--', label='Nominal MC')
        ax.step(bins, np.append(fd_h, fd_h[-1]), where='post', color='black',
                linewidth=2, label='Tilted data')
        if cov_band is not None:
            for i in range(n_bins):
                ax.fill_between([bins[i], bins[i+1]], unf_h[i]-band[i], unf_h[i]+band[i],
                                color='red', alpha=0.15,
                                label=(f'{band_src} band' if i == 0 else None))
        ax.errorbar(centers, unf_h, yerr=stat_err,
                    fmt='ro', markersize=5, capsize=3, linewidth=1.5, label='OmniFold')
        ax.set_xlabel(XLABEL[vn]); ax.set_ylabel('Weighted events')
        ax.set_title(f'Unfolded: {vn}  ({TILT_DESC})')
        ax.text(0.03, 0.97, fit_annotation(c2, ndf) + f'\n[stat cov]',
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', fc='white', ec='none', alpha=0.8))
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/fakedata_{TAG}_unfolded_distributions.png', dpi=150); plt.close()
    print(f"  fakedata_{TAG}_unfolded_distributions.png")

    # ── Headline PASS/FAIL using stat covariance (Q3) ─────────────────────────
    print(f"\n  === Fake-data study verdict (unfolded vs injected truth) ===")
    print(f"      pass if p-value > {PVAL_THR:.2f}  (~2σ, stat-only covariance)")
    all_pass = True
    for vn in var_data:
        c2, ndf, p = fds_status[vn]
        ok = p > PVAL_THR
        all_pass = all_pass and ok
        print(f"      {vn:14s}: chi2/ndf = {c2:6.2f}/{ndf} = {c2/ndf:5.2f}, "
              f"p = {p:6.3f} [stat]  -> {'PASS' if ok else 'FAIL'}")
    print(f"      ==> FAKE-DATA STUDY {'PASSED' if all_pass else 'FAILED'}\n")

    # ── chi2/ndf-vs-iteration convergence (shape metric) + iteration choice ───
    colors_var = {'true_p': 'red', 'true_costheta': 'blue'}
    labels_var = {'true_p': r'$p_e$', 'true_costheta': r'$\cos\theta_e$'}
    curves = {}
    for vn, vv in var_data.items():
        bins = BINNING[vn]; ndf = len(bins) - 2
        truth_h, _ = np.histogram(vv, bins=bins, weights=mc_weights * injected_tilt)
        nom_h, _   = np.histogram(vv, bins=bins, weights=mc_weights)
        pi, pc = [0], [chi2_simple(nom_h, truth_h) / ndf]
        for f in push_files:
            it = iter_num(f)
            push = np.load(f)
            push = push if push.ndim == 1 else push.mean(axis=0)
            h, _ = np.histogram(vv, bins=bins, weights=mc_weights * push)
            pi.append(it + 1); pc.append(chi2_simple(h, truth_h) / ndf)
        curves[vn] = (pi, pc)
        print(f"\n  {vn}  (ndf={ndf})")
        print(f"  {'Iter':>6s}  {'chi2/ndf':>10s}")
        for it, c2 in zip(pi, pc):
            note = "  <- prior (no unfolding)" if it == 0 else ""
            print(f"  {it:6d}  {c2:10.4f}{note}")
        gmin, plateau = recommend_iteration(pi, pc, rel_tol=REL_TOL)
        print(f"  -> recommended iterations ({vn}): global-min={gmin}, "
              f"plateau(<{REL_TOL:.0%} improvement)={plateau}")

    # combined recommendation: take the later of the two plateau iters (both vars must be converged)
    plateaus = [recommend_iteration(pi, pc, rel_tol=REL_TOL)[1] for (pi, pc) in curves.values()]
    rec_iter = max([p for p in plateaus if p is not None], default=None)
    print(f"\n  ** Recommended stopping iteration (both variables converged): {rec_iter}")
    print(f"     (later iterations mainly add ML/statistical variance — see slide-12 discussion)\n")

    fig, ax = plt.subplots(figsize=(8, 6))
    all_vals = []
    for vn, (pi, pc) in curves.items():
        ax.plot(pi, pc, 'o-', color=colors_var[vn], linewidth=2, markersize=5, label=labels_var[vn])
        all_vals.extend(pc)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, label=r'$\chi^2$/DoF = 1')
    if rec_iter is not None:
        ax.axvline(rec_iter, color='green', linestyle='--', linewidth=1.5,
                   label=f'recommended iter = {rec_iter}')
    ax.set_xlabel('OmniFold Iteration'); ax.set_ylabel(r'$\chi^2$/DoF')
    ax.set_title(rf'Fake-data χ²/DoF convergence  ({TILT_DESC})')
    ax.legend(fontsize=11); ax.set_yscale('log')
    ax.set_ylim(min(v for v in all_vals if v > 0) * 0.5, max(all_vals) * 3); ax.set_xticks(pi)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/fakedata_{TAG}_chi2_convergence.png', dpi=150); plt.close()
    print(f"  fakedata_{TAG}_chi2_convergence.png")

    # ── Chi2 convergence excluding lowest bin(s) (--extended only) ────────
    if EXTENDED:
        print(f"\n  === Chi2 convergence excluding lowest bin(s) ===")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax_idx, (n_excl, title_sfx) in enumerate([
            (1, 'excl. lowest bin'), (2, 'excl. 2 lowest bins')
        ]):
            ax = axes[ax_idx]
            all_vals_ex = []
            pi_ex = [0]
            for vn, vv in var_data.items():
                bins = BINNING[vn]; n_bins_v = len(bins) - 1
                keep = list(range(n_excl, n_bins_v))
                ndf_ex = len(keep) - 1
                if ndf_ex < 1:
                    continue
                truth_h, _ = np.histogram(vv, bins=bins, weights=mc_weights * injected_tilt)
                nom_h, _   = np.histogram(vv, bins=bins, weights=mc_weights)

                def _chi2_keep(obs, exp, keep_idx):
                    return sum((obs[k] - exp[k])**2 / exp[k] for k in keep_idx if exp[k] > 0)

                pi_ex = [0]
                pc_ex = [_chi2_keep(nom_h, truth_h, keep) / ndf_ex]
                for f in push_files:
                    push = np.load(f)
                    push = push if push.ndim == 1 else push.mean(axis=0)
                    h, _ = np.histogram(vv, bins=bins, weights=mc_weights * push)
                    pi_ex.append(iter_num(f) + 1)
                    pc_ex.append(_chi2_keep(h, truth_h, keep) / ndf_ex)
                ax.plot(pi_ex, pc_ex, 'o-', color=colors_var[vn], linewidth=2,
                        markersize=5, label=labels_var[vn])
                all_vals_ex.extend(pc_ex)
                excl_bins_str = ', '.join(f'[{bins[k]:.0f},{bins[k+1]:.0f}]'
                                          if vn == 'true_p'
                                          else f'[{bins[k]:.1f},{bins[k+1]:.1f}]'
                                          for k in range(n_excl))
                print(f"  {vn} excl {excl_bins_str}: "
                      f"prior={pc_ex[0]:.4f}, final={pc_ex[-1]:.4f}  (ndf={ndf_ex})")
            if all_vals_ex:
                ax.axhline(1.0, color='gray', linestyle=':', linewidth=1,
                           label=r'$\chi^2$/DoF = 1')
                ax.set_xlabel('OmniFold Iteration')
                ax.set_ylabel(r'$\chi^2$/DoF')
                ax.set_title(rf'{title_sfx} ({TAG})')
                ax.legend(fontsize=10); ax.set_yscale('log')
                pos = [v for v in all_vals_ex if v > 0]
                if pos:
                    ax.set_ylim(min(pos) * 0.5, max(pos) * 3)
                ax.set_xticks(pi_ex)
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/fakedata_{TAG}_chi2_exclude_bins.png', dpi=150)
        plt.close()
        print(f"  fakedata_{TAG}_chi2_exclude_bins.png")

        # ── Chi2 convergence with merged cosθ backward bins ──────────────────────
        print(f"\n  === Chi2 with merged backward cosθ bins ===")
        merged_cos_bins = np.array([-1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        fig, ax = plt.subplots(figsize=(8, 6))
        pi_m = [0]
        for vn, vv, bins_to_use, color, label in [
            ('true_costheta', true_costheta, BINNING['true_costheta'],
             'blue', r'$\cos\theta$ (10 bins)'),
            ('true_costheta_merged', true_costheta, merged_cos_bins,
             'red', r'$\cos\theta$ (merged bwd → 6 bins)'),
        ]:
            ndf_m = len(bins_to_use) - 2
            if ndf_m < 1:
                continue
            truth_h, _ = np.histogram(vv, bins=bins_to_use,
                                      weights=mc_weights * injected_tilt)
            nom_h, _ = np.histogram(vv, bins=bins_to_use, weights=mc_weights)
            pi_m, pc_m = [0], [chi2_simple(nom_h, truth_h) / ndf_m]
            for f in push_files:
                push = np.load(f)
                push = push if push.ndim == 1 else push.mean(axis=0)
                h, _ = np.histogram(vv, bins=bins_to_use,
                                    weights=mc_weights * push)
                pi_m.append(iter_num(f) + 1)
                pc_m.append(chi2_simple(h, truth_h) / ndf_m)
            ax.plot(pi_m, pc_m, 'o-', color=color, linewidth=2, markersize=5,
                    label=f'{label} (ndf={ndf_m})')
            print(f"  {label}: prior={pc_m[0]:.4f}, final={pc_m[-1]:.4f}")
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1,
                   label=r'$\chi^2$/DoF = 1')
        ax.set_xlabel('OmniFold Iteration'); ax.set_ylabel(r'$\chi^2$/DoF')
        ax.set_title(rf'Effect of merging backward $\cos\theta$ bins ({TAG})')
        ax.legend(fontsize=10); ax.set_yscale('log')
        ax.set_xticks(pi_m)
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/fakedata_{TAG}_chi2_merged_costheta.png', dpi=150)
        plt.close()
        print(f"  fakedata_{TAG}_chi2_merged_costheta.png")

    # ── Per-bin chi2 diagnostic (with p-value annotation) ─────────────────────
    print(f"\n  === Bin-removal chi2 diagnostic ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (vn, vv) in zip(axes, var_data.items()):
        bins = BINNING[vn]; n_bins = len(bins) - 1; ndf_full = n_bins - 1
        truth_h, _ = np.histogram(vv, bins=bins, weights=mc_weights * injected_tilt)
        unf_h, _   = np.histogram(vv, bins=bins, weights=mc_weights * push_mean)
        chi2_full = chi2_simple(unf_h, truth_h)
        per_bin = np.zeros(n_bins)
        for i in range(n_bins):
            if truth_h[i] > 0: per_bin[i] = (unf_h[i] - truth_h[i])**2 / truth_h[i]
        centers = 0.5 * (bins[:-1] + bins[1:]); bar_w = np.diff(bins) * 0.6
        ax.bar(centers, per_bin, width=bar_w, color='steelblue', alpha=0.7, edgecolor='navy')
        ax.axhline(chi2_full / n_bins, color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean = {chi2_full/n_bins:.2f}')
        ax.set_xlabel(XLABEL[vn]); ax.set_ylabel(r'Per-bin $\chi^2$ contribution')
        ax.set_title(f'{vn}: {fit_annotation(chi2_full, ndf_full)}', fontsize=11)
        ax.legend()
        print(f"\n  {vn} (total={chi2_full:.2f}, chi2/ndf={chi2_full/ndf_full:.2f}, "
              f"p={chi2_pvalue(chi2_full, ndf_full):.3f}):")
        bfmt = ".0f" if vn == "true_p" else ".2f"
        for i in range(n_bins):
            frac = per_bin[i] / chi2_full if chi2_full > 0 else 0
            chi2_wo = chi2_simple(np.delete(unf_h, i), np.delete(truth_h, i))
            lo = f"{bins[i]:{bfmt}}"; hi = f"{bins[i+1]:{bfmt}}"
            print(f"    [{lo:>7s},{hi:>7s}]  chi2={per_bin[i]:8.2f}  "
                  f"w/o={chi2_wo:8.2f}  frac={frac:.1%}")
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/fakedata_{TAG}_chi2_bin_diagnostic.png', dpi=150); plt.close()
    print(f"  fakedata_{TAG}_chi2_bin_diagnostic.png")

    print(f"\n  Push weight stats per iteration:")
    for f in push_files:
        w = np.load(f); w = w if w.ndim == 1 else w.mean(axis=0)
        print(f"    Iter {iter_num(f)+1:2d}: mean={w.mean():.4f}, std={w.std():.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# paper
# ═══════════════════════════════════════════════════════════════════════════════
def do_paper():
    TAG = flags.tag or make_fdt_tag(flags.var, getattr(flags, 'alpha', 0.3))
    os.makedirs(flags.plot_dir, exist_ok=True)
    syst_dir = getattr(flags, 'syst_dir', 'sbnd/plots_syst/')
    val_dir  = getattr(flags, 'val_dir', 'sbnd/plots_validation/')
    os.makedirs(syst_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    truth_raw  = np.load(flags.data_dir + 'mc_vals_truth_NoNorm.npy')
    mc_weights = np.load(flags.data_dir + 'mc_weights_reco.npy')
    tilt_file  = flags.data_dir + f'truth_weights_sbnd_fakedata_{TAG}.npy'
    if not os.path.exists(tilt_file):
        print(f"ERROR: {tilt_file} not found"); return
    injected = np.load(tilt_file)

    tilt_dir = f'weights_sbnd_fakedata_{TAG}/'
    push_files = dedup_push_files(sorted(glob.glob(tilt_dir + 'Step2_Iter*_PushWeights.npy'), key=iter_num))
    if not push_files:
        print(f"ERROR: No push files in '{tilt_dir}'"); return

    ml_dir = getattr(flags, 'ml_weights_dir', 'sbnd/weights_ml_unc/')
    replica_dirs = sorted(glob.glob(ml_dir + 'replica_*/'))
    replica_finals = []
    for rdir in replica_dirs:
        pf = sorted(glob.glob(rdir + 'Step2_Iter*_PushWeights.npy'), key=iter_num)
        if pf:
            w = np.load(pf[-1])
            replica_finals.append(w if w.ndim == 1 else w.mean(axis=0))

    if len(replica_finals) >= 2:
        main_shape = np.load(push_files[-1]).shape
        compatible = [w for w in replica_finals if w.shape == main_shape]
        if len(compatible) >= 2:
            push_final = np.mean(compatible, axis=0)
            print(f"  Central result: mean of {len(compatible)} ML replicas "
                  f"(ML unc = σ/√{len(compatible)})")
        else:
            push_final = np.load(push_files[-1])
            push_final = push_final if push_final.ndim == 1 else push_final.mean(axis=0)
            print(f"  Central result: main fakedata run (replicas have incompatible shape)")
    else:
        push_final = np.load(push_files[-1])
        push_final = push_final if push_final.ndim == 1 else push_final.mean(axis=0)
        print(f"  Central result: main fakedata run "
              f"({len(replica_finals)} replicas found)")

    # Always plot both kinematic variables.
    vars_to_run = ['true_p', 'true_costheta']
    for vn in vars_to_run:
        _make_core_plots(vn, TAG, truth_raw, mc_weights, injected, push_files, push_final)
        _make_per_source_correlations(vn)                 # Q13
        _make_source_uncertainty_comparison(vn)           # Q14
    if len(vars_to_run) == 2:
        _make_combined_chi2(TAG, truth_raw, mc_weights, injected, push_files)
    _make_reweighting_snapshots(TAG, truth_raw, mc_weights, injected, push_files)
    _make_weight_vs_observable(TAG, truth_raw, mc_weights, injected, push_final)
    _make_weight_distributions(TAG, push_files)
    _make_weight_map_2d(TAG, truth_raw, mc_weights, push_final)
    _make_weight_change(TAG, mc_weights, push_files)
    _make_2d_xsec_slices(TAG, truth_raw, mc_weights, injected, push_final)
    _make_2d_correlation(truth_raw, mc_weights)
    print(f"\nAll paper plots saved to {flags.plot_dir}/")


def _make_core_plots(var_name, TAG, truth_raw, mc_weights, injected, push_files, push_final):
    bins = BINNING[var_name]; xlabel = XLABEL[var_name]
    n_bins = len(bins) - 1; bin_widths = np.diff(bins); centers = 0.5*(bins[:-1]+bins[1:])
    var_idx = 0 if var_name == 'true_p' else 1; var_vals = truth_raw[:, var_idx]
    cov_source = getattr(flags, 'cov_source', 'all')
    eff_file = flags.export_dir + f'efficiency_{var_name}.npy'
    eff = np.load(eff_file) if os.path.exists(eff_file) else np.ones(n_bins)
    truth_hist, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * injected)
    nom_hist, _   = np.histogram(var_vals, bins=bins, weights=mc_weights)
    unf_hist, _   = np.histogram(var_vals, bins=bins, weights=mc_weights * push_final)
    truth_xsec = truth_hist / (eff.clip(1e-6) * bin_widths)
    nom_xsec   = nom_hist   / (eff.clip(1e-6) * bin_widths)
    unf_xsec   = unf_hist   / (eff.clip(1e-6) * bin_widths)
    cov_file = f'{flags.cov_dir}/covariance_{cov_source}_{var_name}.npz'
    if os.path.exists(cov_file):
        cd = np.load(cov_file); scale = eff.clip(1e-6) * bin_widths
        cov_xsec = cd['cov'] / np.outer(scale, scale)
        xsec_unc = np.sqrt(np.diag(cov_xsec))
    else:
        print(f"  WARNING: {cov_file} not found"); xsec_unc = np.zeros(n_bins); cov_xsec = None
    print(f"\n  Paper plots: {var_name} (tag={TAG}, cov-source={cov_source})")

    # Goodness of fit: use stat covariance for chi2 (same as validation)
    gof = None
    unf_w2, _ = np.histogram(var_vals, bins=bins, weights=(mc_weights * push_final) ** 2)
    stat_diag_xsec = unf_w2 / (eff.clip(1e-6) * bin_widths) ** 2
    ndf = n_bins - 1
    d = unf_xsec - truth_xsec
    c2 = float(np.sum(d ** 2 / stat_diag_xsec.clip(1e-30)))
    p = chi2_pvalue(c2, ndf)
    gof = fit_annotation(c2, ndf)
    print(f"    Xsec vs truth (stat cov): {gof}")

    TILT_DESC_R = tilt_label(flags.var, getattr(flags, 'alpha', 0.3))

    # xsec vs truth + ratio subplot [TAG-DEPENDENT]
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})
    ax = axes[0]
    ax.step(bins, np.append(truth_xsec, truth_xsec[-1]), where='post', color='black', linewidth=2, label='Tilted data')
    ax.step(bins, np.append(nom_xsec, nom_xsec[-1]), where='post', color='gray', linewidth=1.5, linestyle='--', label='Prior')
    ax.errorbar(centers, unf_xsec, yerr=xsec_unc, fmt='o', color='red', markersize=5, capsize=3, linewidth=1.5, label='OmniFold')
    ax.set_ylabel(YLABEL_XSEC[var_name])
    ax.set_title(rf'SBND $\nu_e$ CC Inclusive  ({TILT_DESC_R})')
    ax.legend(loc='best', fontsize=10); ax.set_xlim(bins[0], bins[-1])
    ax.text(0.03, 0.97, gof, transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec='none', alpha=0.8))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    # ratio subplot
    r_unf = unf_xsec / np.where(truth_xsec > 0, truth_xsec, 1)
    r_unc = xsec_unc / np.where(truth_xsec > 0, truth_xsec, 1)
    axes[1].errorbar(centers, r_unf, yerr=r_unc, fmt='o', color='red', markersize=5, capsize=3, linewidth=1.5)
    axes[1].axhline(1.0, color='black', linewidth=1)
    axes[1].fill_between([bins[0], bins[-1]], 0.9, 1.1, color='green', alpha=0.08)
    axes[1].set_xlabel(xlabel); axes[1].set_ylabel('OmniFold / Tilted data')
    axes[1].set_ylim(0.5, 1.5); axes[1].set_xlim(bins[0], bins[-1])
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/xsec_vs_truth_{TAG}_{var_name}.png', dpi=150)
    print(f"    xsec_vs_truth_{TAG}_{var_name}.png"); plt.close()

    # uncertainty budget [TAG-INDEPENDENT]
    fig, ax = plt.subplots(figsize=(8, 6))
    _combined_ml_as_stderr = False
    if os.path.exists(cov_file):
        _cd = np.load(cov_file)
        _combined_ml_as_stderr = bool(_cd.get('ml_as_stderr', np.array(False)))
    for src in ['bnb','genie','extra_xsec','g4','mcstat','ml']:
        cf = f'{flags.cov_dir}/covariance_{src}_{var_name}.npz'
        if not os.path.exists(cf): continue
        d = np.load(cf)
        src_cov = d['cov'].copy()
        n_u = int(d['n_universes']) if 'n_universes' in d else 1
        if src == 'ml' and n_u > 1:
            # ML uncertainty is σ/√N since the central value is the replica mean
            src_cov = src_cov / n_u
            label_extra = f' (σ/√{n_u})'
        else:
            label_extra = f' ({n_u})'
        frac = np.sqrt(np.diag(src_cov)) / d['mean_hist'].clip(1e-6)
        ax.step(bins, np.append(frac, frac[-1]), where='post', color=SRC_COLOR[src],
                linewidth=1.5, label=f'{SRC_LABEL[src]}{label_extra}')
    if os.path.exists(cov_file):
        cd = np.load(cov_file); fa = np.sqrt(np.diag(cd['cov'])) / cd['mean_hist'].clip(1e-6)
        ax.step(bins, np.append(fa, fa[-1]), where='post', color='black', linewidth=2,
                label=f'Total ({cov_source})')
    ax.set_xlabel(xlabel); ax.set_ylabel('Bin Fractional Uncertainty')
    ax.set_title(r'Uncertainty budget: SBND $\nu_e$ CC'); ax.legend(fontsize=9); ax.set_xlim(bins[0], bins[-1])
    ax.set_yscale('log'); ax.set_ylim(1e-3, 0.5); plt.tight_layout()
    plt.savefig(f'{syst_dir_of()}/uncertainty_budget_{var_name}.png', dpi=150)
    print(f"    uncertainty_budget_{var_name}.png -> plots_syst/"); plt.close()

    # correlation [TAG-INDEPENDENT] -> plots_xsec
    if os.path.exists(cov_file):
        cd = np.load(cov_file); cm = cd['cov']; dg = np.sqrt(np.diag(cm))
        corr = cm / np.outer(dg.clip(1e-10), dg.clip(1e-10))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, origin='lower', aspect='auto', extent=[0,n_bins,0,n_bins], vmin=-1, vmax=1, cmap='RdBu_r')
        fmt = '.0f' if var_name == 'true_p' else '.1f'
        tl = [f'[{bins[i]:{fmt}},{bins[i+1]:{fmt}})' for i in range(n_bins)]
        ax.set_xticks(np.arange(n_bins)+0.5); ax.set_xticklabels(tl, fontsize=8, rotation=45, ha='right')
        ax.set_yticks(np.arange(n_bins)+0.5); ax.set_yticklabels(tl, fontsize=8)
        ax.set_title(f'Correlation ({var_name}, {cov_source})'); plt.colorbar(im, ax=ax); plt.tight_layout()
        plt.savefig(f'{flags.plot_dir}/correlation_{var_name}.png', dpi=150)
        print(f"    correlation_{var_name}.png"); plt.close()


def syst_dir_of():
    return getattr(flags, 'syst_dir', 'sbnd/plots_syst/')


def _make_per_source_correlations(var_name):
    """Q13: correlation matrix for each systematic source, side by side."""
    cov_dir = flags.cov_dir
    bins = BINNING[var_name]; n_bins = len(bins) - 1
    fmt = '.0f' if var_name == 'true_p' else '.1f'
    tl = [f'[{bins[i]:{fmt}},{bins[i+1]:{fmt}})' for i in range(n_bins)]
    srcs = ['bnb', 'genie', 'extra_xsec', 'g4', 'mcstat', 'ml']
    avail = [s for s in srcs
             if os.path.exists(f'{cov_dir}/covariance_{s}_{var_name}.npz')]
    if not avail:
        print(f"    (no per-source covariances for {var_name} — skip breakdown)")
        return
    ncol = 3; nrow = -(-len(avail) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.6 * nrow), squeeze=False)
    for k, s in enumerate(avail):
        ax = axes[k // ncol][k % ncol]
        d = np.load(f'{cov_dir}/covariance_{s}_{var_name}.npz')
        cm = d['cov']; dg = np.sqrt(np.diag(cm))
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.where(np.outer(dg, dg) > 1e-20,
                            cm / np.outer(dg.clip(1e-10), dg.clip(1e-10)), np.nan)
        im = ax.imshow(np.ma.masked_invalid(corr), origin='lower', aspect='auto',
                       extent=[0, n_bins, 0, n_bins], vmin=-1, vmax=1, cmap='RdBu_r')
        n_u = int(d['n_universes']) if 'n_universes' in d else '?'
        ax.set_title(f'{SRC_LABEL[s]} (n={n_u})', fontsize=11)
        ax.set_xticks(np.arange(n_bins) + 0.5); ax.set_xticklabels(tl, fontsize=6, rotation=45, ha='right')
        ax.set_yticks(np.arange(n_bins) + 0.5); ax.set_yticklabels(tl, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for k in range(len(avail), nrow * ncol):
        axes[k // ncol][k % ncol].set_visible(False)
    plt.suptitle(f'Per-source correlation matrices: {var_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/correlation_by_source_{var_name}.png', dpi=150)
    print(f"    correlation_by_source_{var_name}.png"); plt.close()


def _make_source_uncertainty_comparison(var_name):
    """Q14: per-source fractional uncertainty (OmniFold) with optional Wiener-SVD overlay.

    OmniFold per-source per-bin fractional uncertainty is read from the per-source
    covariance files. If --wsvd-syst-file (JSON) is given, matching Wiener-SVD
    numbers are overlaid so the two methods can be compared source by source.
    JSON format: {"true_p": {"bnb": [f0,f1,...], "genie": [...], ...}, "true_costheta": {...}}
    """
    cov_dir = flags.cov_dir
    bins = BINNING[var_name]; n_bins = len(bins) - 1; centers = 0.5*(bins[:-1]+bins[1:])
    srcs = ['bnb', 'genie', 'extra_xsec', 'g4', 'mcstat', 'ml']
    of = {}
    for s in srcs:
        p = f'{cov_dir}/covariance_{s}_{var_name}.npz'
        if not os.path.exists(p): continue
        d = np.load(p)
        cov_s = d['cov'].copy()
        # ML: show as σ/√N (standard error of the mean)
        if s == 'ml' and 'n_universes' in d:
            n_u = int(d['n_universes'])
            if n_u > 1:
                cov_s = cov_s / n_u
        of[s] = np.sqrt(np.diag(cov_s)) / d['mean_hist'].clip(1e-6)
    if not of:
        print(f"    (no per-source covariances for {var_name} — skip source comparison)")
        return

    # Optional Wiener-SVD numbers
    wsvd = {}
    wfile = getattr(flags, 'wsvd_syst_file', None)
    if wfile and os.path.exists(wfile):
        try:
            with open(wfile) as fh:
                wsvd = json.load(fh).get(var_name, {})
        except Exception as e:
            print(f"    (could not read --wsvd-syst-file: {e})")

    fig, ax = plt.subplots(figsize=(9, 6))
    for s in of:
        ax.step(bins, np.append(of[s], of[s][-1]), where='post',
                color=SRC_COLOR[s], linewidth=1.8, label=f'OmniFold {SRC_LABEL[s]}')
        if s in wsvd and len(wsvd[s]) == n_bins:
            ax.step(bins, np.append(np.array(wsvd[s]), wsvd[s][-1]), where='post',
                    color=SRC_COLOR[s], linewidth=1.4, linestyle='--',
                    label=f'W-SVD {SRC_LABEL[s]}')
    ax.set_yscale('log'); ax.set_ylim(1e-3, 1.0)
    ax.set_xlabel(XLABEL[var_name]); ax.set_ylabel('Fractional uncertainty')
    ttl = 'Per-source uncertainty' + (' (solid=OmniFold, dashed=W-SVD)' if wsvd else ' (OmniFold)')
    ax.set_title(f'{ttl}: {var_name}'); ax.legend(fontsize=8, ncol=2); ax.set_xlim(bins[0], bins[-1])
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/source_unc_comparison_{var_name}.png', dpi=150)
    print(f"    source_unc_comparison_{var_name}.png"); plt.close()

    # Also export OmniFold per-source totals so they can be compared to cafpyana W-SVD.
    export = {s: of[s].tolist() for s in of}
    cov_dir = getattr(flags, 'cov_dir', 'sbnd/covariance/')
    with open(f'{cov_dir}/omnifold_source_frac_unc_{var_name}.json', 'w') as fh:
        json.dump({var_name: export}, fh, indent=2)
    print(f"    omnifold_source_frac_unc_{var_name}.json -> covariance/")


def _make_combined_chi2(TAG, truth_raw, mc_weights, injected, push_files):
    colors = {'true_p': 'red', 'true_costheta': 'blue'}
    labels = {'true_p': r'$p_e$', 'true_costheta': r'$\cos\theta_e$'}
    curves = {}
    rel_tol = getattr(flags, 'iter_rel_tol', 0.05)
    for vn in ['true_p', 'true_costheta']:
        bins = BINNING[vn]; ndf = len(bins) - 2; vv = truth_raw[:, 0 if vn == 'true_p' else 1]
        th, _ = np.histogram(vv, bins=bins, weights=mc_weights * injected)
        nh, _ = np.histogram(vv, bins=bins, weights=mc_weights)
        pi, pc = [0], [chi2_simple(nh, th) / ndf]
        for f in push_files:
            push = np.load(f); push = push if push.ndim == 1 else push.mean(axis=0)
            h, _ = np.histogram(vv, bins=bins, weights=mc_weights * push)
            pi.append(iter_num(f) + 1); pc.append(chi2_simple(h, th) / ndf)
        curves[vn] = (pi, pc)
        gmin, plateau = recommend_iteration(pi, pc, rel_tol=rel_tol)
        print(f"  chi2/ndf ({vn}, ndf={ndf}): prior={pc[0]:.4f}, final={pc[-1]:.4f}, "
              f"p(final)={chi2_pvalue(pc[-1]*ndf, ndf):.3f}, "
              f"rec-iter(min={gmin}, plateau={plateau})")
    fig, ax = plt.subplots(figsize=(8, 6)); av = []
    plateaus = []
    for vn, (pi, pc) in curves.items():
        ax.plot(pi, pc, 'o-', color=colors[vn], linewidth=2, markersize=5, label=labels[vn]); av.extend(pc)
        plateaus.append(recommend_iteration(pi, pc, rel_tol=rel_tol)[1])
    rec_iter = max([p for p in plateaus if p is not None], default=None)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, label=r'$\chi^2$/DoF = 1')
    if rec_iter is not None:
        ax.axvline(rec_iter, color='green', linestyle='--', linewidth=1.5,
                   label=f'recommended iter = {rec_iter}')
    ax.set_xlabel('OmniFold Iteration'); ax.set_ylabel(r'$\chi^2$/DoF')
    ax.set_title(rf'$\chi^2$ convergence ({TAG})'); ax.legend(fontsize=12); ax.set_yscale('log')
    ax.set_ylim(min(v for v in av if v > 0)*0.5, max(av)*3); ax.set_xticks(pi); plt.tight_layout()
    val_dir = getattr(flags, 'val_dir', 'sbnd/plots_validation/')
    plt.savefig(f'{val_dir}/chi2_convergence_{TAG}.png', dpi=150)
    print(f"    chi2_convergence_{TAG}.png -> plots_validation/"); plt.close()


def _make_reweighting_snapshots(TAG, truth_raw, mc_weights, injected, push_files):
    n_iters = len(push_files)
    snap = sorted(set(s for s in [1, 3, 5, n_iters] if s <= n_iters))
    cols = plt.cm.viridis(np.linspace(0.2, 0.9, len(snap)))
    for vn in ['true_p', 'true_costheta']:
        bins = BINNING[vn]; vv = truth_raw[:, 0 if vn == 'true_p' else 1]
        centers = 0.5*(bins[:-1]+bins[1:])
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        prior_h, _ = np.histogram(vv, bins=bins, weights=mc_weights)
        truth_h, _ = np.histogram(vv, bins=bins, weights=mc_weights * injected)
        axes[0].step(bins, np.append(prior_h, prior_h[-1]), where='post', color='gray', linewidth=2, linestyle='--', label='Prior')
        axes[0].step(bins, np.append(truth_h, truth_h[-1]), where='post', color='black', linewidth=2.5, label='Tilted data')
        for ci, pit in enumerate(snap):
            push = np.load(push_files[pit-1]); push = push if push.ndim == 1 else push.mean(axis=0)
            h, _ = np.histogram(vv, bins=bins, weights=mc_weights * push)
            axes[0].errorbar(centers, h, yerr=np.sqrt(np.maximum(h, 0)), fmt='o', color=cols[ci],
                             markersize=4, capsize=2, linewidth=1.2, label=f'Iter {pit}')
            axes[1].plot(centers, h / np.where(truth_h > 0, truth_h, 1), 'o-', color=cols[ci], markersize=4)
        axes[1].plot(centers, prior_h / np.where(truth_h > 0, truth_h, 1), 's--', color='gray', markersize=4)
        axes[0].set_ylabel('Weighted events'); axes[0].set_title(rf'Reweighting snapshots: {vn} ({TAG})')
        axes[0].legend(fontsize=9, ncol=2); axes[1].axhline(1.0, color='black', linewidth=1)
        axes[1].set_xlabel(XLABEL[vn]); axes[1].set_ylabel('Ratio to Truth'); axes[1].set_ylim(0.7, 1.3)
        plt.tight_layout()
        plt.savefig(f'{flags.plot_dir}/reweighting_snapshots_{TAG}_{vn}.png', dpi=150)
        print(f"    reweighting_snapshots_{TAG}_{vn}.png"); plt.close()


def _make_weight_vs_observable(TAG, truth_raw, mc_weights, injected, push_final):
    for vn in ['true_p', 'true_costheta']:
        bins = BINNING[vn]; vi = 0 if vn == 'true_p' else 1; vv = truth_raw[:, vi]
        centers = 0.5*(bins[:-1]+bins[1:])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        n_plot = min(len(vv), 5000); idx = np.random.default_rng(42).choice(len(vv), n_plot, replace=False)
        sc = axes[0].scatter(vv[idx], push_final[idx], c=push_final[idx], cmap='RdBu_r', vmin=0.5, vmax=1.5, s=8, alpha=0.4, edgecolors='none')
        axes[0].axhline(1.0, color='black', linewidth=1, linestyle='--')
        axes[0].set_xlabel(XLABEL[vn]); axes[0].set_ylabel('Push weight')
        axes[0].set_title(f'Per-event weights: {vn} ({TAG})')
        axes[0].set_ylim(0, max(3.0, np.percentile(push_final, 99.5)*1.2)); plt.colorbar(sc, ax=axes[0], label='Weight')
        bm, bs, bmed, tm_ = [np.zeros(len(bins)-1) for _ in range(4)]
        for i in range(len(bins)-1):
            m = (vv >= bins[i]) & (vv < bins[i+1])
            if m.sum() > 0:
                w = push_final[m]; bm[i] = w.mean(); bs[i] = w.std(); bmed[i] = np.median(w)
                tm_[i] = injected[m].mean()
        axes[1].errorbar(centers, bm, yerr=bs, fmt='ro-', capsize=4, linewidth=1.5, markersize=6, label=r'Mean $\pm$ std')
        axes[1].plot(centers, bmed, 'bs--', markersize=5, linewidth=1, label='Median')
        axes[1].plot(centers, tm_, 'g^-', markersize=5, linewidth=1, label='Injected tilt')
        axes[1].axhline(1.0, color='black', linewidth=1, linestyle='--')
        axes[1].set_xlabel(XLABEL[vn]); axes[1].set_ylabel('Push weight')
        axes[1].set_title(f'Weight profile: {vn} ({TAG})'); axes[1].legend(); axes[1].set_ylim(0, 3)
        plt.tight_layout()
        plt.savefig(f'{flags.plot_dir}/weights_vs_observable_{TAG}_{vn}.png', dpi=150)
        print(f"    weights_vs_observable_{TAG}_{vn}.png"); plt.close()


def _make_weight_distributions(TAG, push_files):
    n_iters = len(push_files)
    snap = sorted(set(s for s in [1, 3, 5, n_iters] if s <= n_iters))
    cols = plt.cm.viridis(np.linspace(0.2, 0.9, len(snap)))
    fig, ax = plt.subplots(figsize=(8, 5))
    for ci, pit in enumerate(snap):
        push = np.load(push_files[pit-1]); push = push if push.ndim == 1 else push.mean(axis=0)
        ax.hist(push, bins=100, range=(0.5, 2.0), density=True, alpha=0.5, color=cols[ci],
                label=f'Iter {pit}', zorder=2+ci)
    ax.set_xlabel('Push weight'); ax.set_ylabel('Density')
    ax.set_title(rf'Push weight distributions ({TAG})')
    h, l = ax.get_legend_handles_labels(); ax.legend(h[::-1], l[::-1])
    ax.axvline(1.0, color='black', linewidth=1, linestyle='--', zorder=1); plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/weight_distributions_{TAG}.png', dpi=150)
    print(f"    weight_distributions_{TAG}.png"); plt.close()


def _make_weight_map_2d(TAG, truth_raw, mc_weights, push_final):
    true_p, true_cos = truth_raw[:, 0], truth_raw[:, 1]
    p_bins = BINNING['true_p']; cos_bins = BINNING['true_costheta']
    stat = np.full((len(p_bins)-1, len(cos_bins)-1), np.nan)
    for ip in range(len(p_bins)-1):
        for ic in range(len(cos_bins)-1):
            m = ((true_p >= p_bins[ip]) & (true_p < p_bins[ip+1]) & (true_cos >= cos_bins[ic]) & (true_cos < cos_bins[ic+1]))
            if m.sum() > 0: stat[ip, ic] = push_final[m].mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(cos_bins, p_bins, stat, cmap='RdBu_r', vmin=0.8, vmax=1.2)
    ax.set_xlabel(r'True $\cos\theta_e$'); ax.set_ylabel(r'True electron momentum [MeV/c]')
    ax.set_title(f'Mean push weight 2D ({TAG})'); plt.colorbar(im, ax=ax, label='Mean weight')
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/weight_map_2d_{TAG}.png', dpi=150)
    print(f"    weight_map_2d_{TAG}.png"); plt.close()


def _make_weight_change(TAG, mc_weights, push_files):
    N = 5
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(push_files) >= 2:
        w0 = np.load(push_files[0]); w0 = w0 if w0.ndim == 1 else w0.mean(axis=0)
        w1 = np.load(push_files[1]); w1 = w1 if w1.ndim == 1 else w1.mean(axis=0)
        ax.hist(w1 - w0, bins=100, range=(-0.15, 0.15), density=True,
                alpha=0.5, color='red', label='Iters 1→2')
    if len(push_files) >= 5:
        ww = []
        for f in push_files[0:5]:
            w = np.load(f); ww.append(w if w.ndim == 1 else w.mean(axis=0))
        if len(ww) >= 2:
            avg = np.diff(np.array(ww), axis=0).mean(axis=0)
            ax.hist(avg, bins=100, range=(-0.15, 0.15), density=True,
                    alpha=0.5, color='blue', label='After 5 iters')
    n_final = min(len(push_files), 10)
    if n_final >= 6:
        start = max(0, n_final - N)
        ww = []
        for f in push_files[start:n_final]:
            w = np.load(f); ww.append(w if w.ndim == 1 else w.mean(axis=0))
        if len(ww) >= 2:
            avg = np.diff(np.array(ww), axis=0).mean(axis=0)
            ax.hist(avg, bins=100, range=(-0.15, 0.15), density=True,
                    alpha=0.5, color='orange', label=f'After {n_final} iters')
    ax.set_xlabel('Per-event weight change'); ax.set_ylabel('Density')
    ax.set_title(rf'Weight convergence ({TAG})'); ax.legend(); plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/weight_change_{TAG}.png', dpi=150)
    print(f"    weight_change_{TAG}.png"); plt.close()


def _make_2d_xsec_slices(TAG, truth_raw, mc_w, tilt, push):
    true_p, true_cos = truth_raw[:, 0], truth_raw[:, 1]
    cos_slices = [(-1, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.0)]
    p_bins = np.array([0, 200, 400, 600, 800, 1200, 2000])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for si, (clo, chi) in enumerate(cos_slices):
        ax = axes[si//3][si%3]; m = (true_cos >= clo) & (true_cos < chi)
        if m.sum() == 0: ax.set_visible(False); continue
        bw = np.diff(p_bins); cw = chi - clo; cen = 0.5*(p_bins[:-1]+p_bins[1:])
        th, _ = np.histogram(true_p[m], bins=p_bins, weights=mc_w[m]*tilt[m])
        nh, _ = np.histogram(true_p[m], bins=p_bins, weights=mc_w[m])
        uh, _ = np.histogram(true_p[m], bins=p_bins, weights=mc_w[m]*push[m])
        uh_w2, _ = np.histogram(true_p[m], bins=p_bins, weights=(mc_w[m]*push[m])**2)
        uh_err = np.sqrt(np.maximum(uh_w2, 0))
        scale = bw * cw
        # stat-only chi2/p for this slice (Q11 on 2D)
        good = uh_w2 > 0
        c2 = float(np.sum((uh[good]-th[good])**2 / uh_w2[good]))
        ndf = max(int(good.sum()) - 1, 1)
        ax.step(p_bins, np.append(th/scale, (th/scale)[-1]), where='post',
                color='black', linewidth=2, label='Tilted data')
        ax.errorbar(cen, uh/scale, yerr=uh_err/scale, fmt='o', color='red',
                    markersize=4, capsize=2, linewidth=1.2, label='OmniFold')
        ax.step(p_bins, np.append(nh/scale, (nh/scale)[-1]), where='post',
                color='gray', linewidth=1, linestyle='--', label='Prior')
        ax.set_title(f'{clo:.1f} < cos$\\theta$ < {chi:.1f}', fontsize=11)
        ax.text(0.03, 0.97, fit_annotation(c2, ndf), transform=ax.transAxes,
                va='top', ha='left', fontsize=7,
                bbox=dict(boxstyle='round', fc='white', ec='none', alpha=0.75))
        ax.set_xlabel('p [MeV/c]', fontsize=10)
        if si == 0: ax.legend(fontsize=8)
    plt.suptitle(rf'SBND $\nu_e$ CC: $d^2\sigma / dp\, d\cos\theta$ ({TAG}) — stat-only $\chi^2$', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/xsec_2d_slices_{TAG}.png', dpi=150)
    print(f"    xsec_2d_slices_{TAG}.png"); plt.close()

    # ── Complementary: dσ/dcosθ in slices of momentum ────────────────────────
    p_slices = [(200, 400), (400, 600), (600, 800), (800, 1000), (1000, 1400), (1400, 2000)]
    cos_bins_2d = np.linspace(-1, 1, 11)
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for si, (plo, phi) in enumerate(p_slices):
        ax = axes2[si//3][si%3]; m = (true_p >= plo) & (true_p < phi)
        if m.sum() == 0: ax.set_visible(False); continue
        bw = np.diff(cos_bins_2d); pw = phi - plo
        cen = 0.5*(cos_bins_2d[:-1]+cos_bins_2d[1:])
        th, _ = np.histogram(true_cos[m], bins=cos_bins_2d, weights=mc_w[m]*tilt[m])
        nh, _ = np.histogram(true_cos[m], bins=cos_bins_2d, weights=mc_w[m])
        uh, _ = np.histogram(true_cos[m], bins=cos_bins_2d, weights=mc_w[m]*push[m])
        uh_w2, _ = np.histogram(true_cos[m], bins=cos_bins_2d, weights=(mc_w[m]*push[m])**2)
        uh_err = np.sqrt(np.maximum(uh_w2, 0))
        scale = bw * pw
        good = uh_w2 > 0
        c2 = float(np.sum((uh[good]-th[good])**2 / uh_w2[good]))
        ndf = max(int(good.sum()) - 1, 1)
        ax.step(cos_bins_2d, np.append(th/scale, (th/scale)[-1]), where='post',
                color='black', linewidth=2, label='Tilted data')
        ax.errorbar(cen, uh/scale, yerr=uh_err/scale, fmt='o', color='red',
                    markersize=4, capsize=2, linewidth=1.2, label='OmniFold')
        ax.step(cos_bins_2d, np.append(nh/scale, (nh/scale)[-1]), where='post',
                color='gray', linewidth=1, linestyle='--', label='Prior')
        ax.set_title(f'{plo:.0f} < p < {phi:.0f} MeV/c', fontsize=11)
        ax.text(0.03, 0.97, fit_annotation(c2, ndf), transform=ax.transAxes,
                va='top', ha='left', fontsize=7,
                bbox=dict(boxstyle='round', fc='white', ec='none', alpha=0.75))
        ax.set_xlabel(r'$\cos\theta_e$', fontsize=10)
        if si == 0: ax.legend(fontsize=8)
    plt.suptitle(rf'SBND $\nu_e$ CC: $d^2\sigma / d\cos\theta\, dp$ ({TAG}) — stat-only $\chi^2$', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/xsec_2d_slices_costheta_{TAG}.png', dpi=150)
    print(f"    xsec_2d_slices_costheta_{TAG}.png"); plt.close()


def _make_2d_correlation(truth_raw, mc_w):
    true_p, true_cos = truth_raw[:, 0], truth_raw[:, 1]
    p_bins = np.array([0, 200, 400, 600, 800, 1000, 1400, 2000])
    cos_bins = np.array([-1.0, -0.5, 0.0, 0.5, 0.75, 1.0])
    n_p, n_c = len(p_bins)-1, len(cos_bins)-1
    n_2d = n_p * n_c
    all_flat = []
    for src in ['bnb', 'genie', 'extra_xsec', 'g4', 'mcstat']:
        pat = f'{flags.weights_base}/weights_{src}/{src}_univ*/Step2_Iter*_PushWeights.npy'
        files = sorted(glob.glob(pat)); ufiles = {}
        for f in files:
            m = re.search(r'univ(\d+)', f)
            if m:
                uid = int(m.group(1)); it = iter_num(f)
                if uid not in ufiles or it > ufiles[uid][0]: ufiles[uid] = (it, f)
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
    dg = np.sqrt(np.diag(cov_2d))
    valid = dg > 1e-10
    corr = np.zeros((n_2d, n_2d))
    for i in range(n_2d):
        for j in range(n_2d):
            corr[i, j] = cov_2d[i, j] / (dg[i] * dg[j]) if (valid[i] and valid[j]) else np.nan
    bl = []
    for ic in range(n_c):
        for ip in range(n_p):
            cl = f'c[{cos_bins[ic]:.1f},{cos_bins[ic+1]:.1f}]'
            pl = f'p[{p_bins[ip]:.0f},{p_bins[ip+1]:.0f}]'
            bl.append(f'{cl}\n{pl}')
    fig, ax = plt.subplots(figsize=(14, 12))
    corr_masked = np.ma.masked_invalid(corr)
    im = ax.pcolormesh(np.arange(n_2d+1), np.arange(n_2d+1), corr_masked,
                        vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(np.arange(n_2d)+0.5); ax.set_xticklabels(bl, fontsize=5, rotation=90)
    ax.set_yticks(np.arange(n_2d)+0.5); ax.set_yticklabels(bl, fontsize=5)
    for ic in range(1, n_c):
        pos = ic * n_p
        ax.axhline(pos, color='black', linewidth=0.5, alpha=0.5)
        ax.axvline(pos, color='black', linewidth=0.5, alpha=0.5)
    ax.set_title(r'Correlation: $(p, \cos\theta)$ 2D bins'
                 f' ({n_p}×{n_c} = {n_2d} bins, {len(all_flat)} universes)')
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/correlation_2d_p_costheta.png', dpi=150)
    print(f"    correlation_2d_p_costheta.png ({n_2d} bins from {len(all_flat)} universes)")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════════════════
if flags.action in ('validation', 'all'):
    if flags.action == 'all':
        flags.plot_dir_save = flags.plot_dir
        flags.plot_dir = getattr(flags, 'val_plot_dir', 'sbnd/plots_validation/')
    do_validation()
    if flags.action == 'all':
        flags.plot_dir = flags.plot_dir_save

if flags.action in ('results', 'all'):
    if not hasattr(flags, 'weights_base'): flags.weights_base = 'sbnd'
    if not hasattr(flags, 'export_dir'): flags.export_dir = 'sbnd/exported_weights/'
    if not hasattr(flags, 'cov_dir'): flags.cov_dir = 'sbnd/covariance/'
    if not hasattr(flags, 'cov_source'): flags.cov_source = 'all'
    do_paper()