"""
BuildResults.py — Build covariance matrices and extract cross-sections.

Actions:
  covariance   Build covariance matrices from systematic universes
  xsec         Extract differential cross-sections with uncertainty bands

Tag convention (derived from --tilted-var and --alpha in xsec action):
  --tilted-var true_p --alpha 0.3  =>  tilt_p_alpha0.3

Source groups for covariance:
  all  = bnb + genie + extra_xsec + g4 + mcstat (+ ml)
  fds  = mcstat + genie + extra_xsec  (stat+xsec, for fake-data studies)
  stat = analytic diagonal MC stat only

Examples:
  python3 sbnd/BuildResults.py covariance --source all
  python3 sbnd/BuildResults.py covariance --source fds
  python3 sbnd/BuildResults.py covariance --source stat
  python3 sbnd/BuildResults.py xsec --tilted-var true_p --alpha 0.3 --cov-source fds
"""

import numpy as np
import glob, re, os, argparse
import matplotlib.pyplot as plt

# Predefined source groups (Q7/Q12). A fake-data study should use 'fds' = stat+xsec.
GROUP_MEMBERS = {
    'all':  ['bnb', 'genie', 'extra_xsec', 'g4', 'mcstat'],
    'fds':  ['mcstat', 'genie', 'extra_xsec'],   # stat + xsec
    'xsec': ['genie', 'extra_xsec'],
    'flux': ['bnb'],
}
SINGLE_SOURCES = ['bnb', 'genie', 'extra_xsec', 'g4', 'mcstat', 'ml', 'stat']

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest='action', required=True)

# ── covariance ────────────────────────────────────────────────────────────────
p_cov = sub.add_parser('covariance')
p_cov.add_argument('--source',
                   choices=SINGLE_SOURCES + list(GROUP_MEMBERS.keys()),
                   default='all')
p_cov.add_argument('--var', choices=['true_p', 'true_costheta', 'both'],
                    default='both')
p_cov.add_argument('--data-dir', default='../FormattedData_SBND/')
p_cov.add_argument('--weights-base', default='sbnd')
p_cov.add_argument('--ml-weights-dir', default='sbnd/weights_ml_unc/')
p_cov.add_argument('--plot-dir', default='sbnd/plots_syst/')
p_cov.add_argument('--cov-dir', default='sbnd/covariance/',
                   help='Directory to save .npz covariance files (separate from plots)')
p_cov.add_argument('--freeze-ml', action='store_true',
                   help='When --source all, skip recomputing ML covariance if the .npz already '
                        'exists. Use this after a completed ML replica run to avoid re-training.')
p_cov.add_argument('--ml-label', default=None,
                   help='Optional tag for the ML covariance file, e.g. "50rep_15iter". '
                        'Saves as covariance_ml_{label}_{var}.npz in addition to the standard '
                        'covariance_ml_{var}.npz so past runs are never overwritten.')
p_cov.add_argument('--ml-as-stderr', action='store_true',
                   help='Scale ML covariance by 1/N_replicas so it represents the standard '
                        'error of the mean (appropriate when the final result averages all '
                        'replicas). Without this flag, ML covariance represents the spread '
                        'of a single replica (standard deviation).')

# ── xsec ──────────────────────────────────────────────────────────────────────
p_xs = sub.add_parser('xsec')
p_xs.add_argument('--var', choices=['true_p', 'true_costheta', 'both'],
                   default='both')
p_xs.add_argument('--data-dir', default='../FormattedData_SBND/')
p_xs.add_argument('--weights-base', default='sbnd')
p_xs.add_argument('--export-dir', default='sbnd/exported_weights/')
p_xs.add_argument('--plot-dir', default='sbnd/plots_xsec/')
p_xs.add_argument('--cov-dir', default='sbnd/covariance/')
p_xs.add_argument('--cov-source', default='all',
                  help="Which combined covariance to use: 'all', 'fds' (stat+xsec), etc. "
                       "For fake-data studies use 'fds'.")
p_xs.add_argument('--tag', default=None,
                  help="Override tag. If omitted, derived from --tilted-var and --alpha.")
p_xs.add_argument('--tilted-var', choices=['true_p', 'true_costheta', 'both'],
                  default='true_p', help='Which variable was tilted (for tag derivation).')
p_xs.add_argument('--alpha', type=float, default=0.3,
                  help='Tilt strength (for tag derivation).')

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
TILT_SHORT = {'true_p': 'p', 'true_costheta': 'costheta', 'both': 'both'}


def make_fdt_tag(var, alpha):
    return f'tilt_{TILT_SHORT[var]}_alpha{alpha}'

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
    p = chi2_pvalue(chi2, ndf)
    return rf'$\chi^2$/ndf = {chi2:.2f}/{ndf} = {chi2/ndf:.2f},  p = {p:.3f}'


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

    # ── Analytic MC-statistical covariance (diagonal, Σw²) — always saved ──────
    # Provides an honest statistical component even without bootstrap 'mcstat'
    # universes, and is used as a fallback in the 'fds' group.
    stat_var = np.zeros(n_bins)
    for i in range(n_bins):
        m = (var_vals >= bins[i]) & (var_vals < bins[i + 1])
        stat_var[i] = np.sum(mc_weights[m] ** 2)
    stat_cov = np.diag(stat_var)
    stat_frac = np.zeros_like(stat_cov)
    for i in range(n_bins):
        if nom_hist[i] > 0:
            stat_frac[i, i] = stat_var[i] / (nom_hist[i] ** 2)
    np.savez(f'{cov_dir}/covariance_stat_{var_name}.npz',
             cov=stat_cov, frac_cov=stat_frac, bins=bins,
             mean_hist=nom_hist, nom_hist=nom_hist,
             all_hists=nom_hist[np.newaxis, :], n_universes=np.array(1))
    print(f"  Analytic MC-stat covariance saved: covariance_stat_{var_name}.npz "
          f"(mean frac stat unc = {np.sqrt(np.diag(stat_frac)).mean():.4f})")

    if flags.source == 'stat':
        print(f"  --source stat: only the analytic MC-stat covariance was requested.")
        return

    # ── Determine sources ─────────────────────────────────────────────────────
    def _universe_exists(s):
        return bool(glob.glob(
            f'{flags.weights_base}/weights_{s}/{s}_univ*/Step2_Iter*_PushWeights.npy'))

    ml_dirs = []
    add_analytic_stat = False   # for 'fds' when bootstrap mcstat is unavailable

    if flags.source in GROUP_MEMBERS:
        requested = GROUP_MEMBERS[flags.source]
        sources = [s for s in requested if _universe_exists(s)]
        missing = [s for s in requested if not _universe_exists(s)]
        if missing:
            print(f"  NOTE: no universes found for {missing} (skipped for '{flags.source}')")
        if flags.source == 'all':
            ml_dirs = sorted(glob.glob(flags.ml_weights_dir + 'replica_*/'))
            if ml_dirs:
                sources.append('ml')
        if flags.source == 'fds' and 'mcstat' not in sources:
            add_analytic_stat = True
            print(f"  'fds' group: bootstrap 'mcstat' unavailable -> using analytic "
                  f"MC-stat covariance for the statistical component.")
    elif flags.source == 'ml':
        sources = ['ml']
        ml_dirs = sorted(glob.glob(flags.ml_weights_dir + 'replica_*/'))
    else:
        sources = [flags.source]

    # ── Collect per-source ────────────────────────────────────────────────────
    all_hists = []
    hists_by_source = {}
    all_univ_dirs_by_source = {}

    for src in sources:
        src_hists = []

        if src == 'ml':
            print(f"ml: {len(ml_dirs)} replicas found")
            all_univ_dirs_by_source['ml'] = ml_dirs
            stale_ml = 0
            for rdir in ml_dirs:
                pf = sorted(glob.glob(rdir + 'Step2_Iter*_PushWeights.npy'), key=iter_num)
                if not pf:
                    continue
                push = np.load(pf[-1])
                push = push if push.ndim == 1 else push.mean(axis=0)
                if push.shape[0] != mc_weights.shape[0]:
                    stale_ml += 1; continue
                h, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
                all_hists.append(h)
                src_hists.append(h)
            if stale_ml:
                print(f"  WARNING: {stale_ml} ML replicas skipped (stale — trained on "
                      f"{push.shape[0]} events, current sample has {mc_weights.shape[0]}). "
                      f"Rerun: python3 sbnd/RunStudies.py run-ml-unc --var <var> --alpha <alpha>")
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
            stale_syst = 0
            for uid in sorted(univ_files.keys()):
                push = np.load(univ_files[uid][1])
                push = push if push.ndim == 1 else push.mean(axis=0)
                if push.shape[0] != mc_weights.shape[0]:
                    stale_syst += 1; continue
                h, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
                all_hists.append(h)
                src_hists.append(h)
            if stale_syst:
                print(f"  WARNING: {stale_syst}/{len(univ_files)} {src} universes skipped "
                      f"(stale — trained on old sample). "
                      f"Rerun: python3 sbnd/RunStudies.py run-syst --source {src}")

        hists_by_source[src] = np.array(src_hists) if src_hists else np.array([])

    all_hists = np.array(all_hists)
    n_univ = len(all_hists)
    print(f"Total universes ({var_name}): {n_univ}")
    if n_univ == 0 and not add_analytic_stat:
        print("ERROR: No universe results found."); return

    # ── Per-source covariance computation ────────────────────────────────────
    freeze_ml    = getattr(flags, 'freeze_ml', False)
    ml_as_stderr = getattr(flags, 'ml_as_stderr', False)
    ml_label     = getattr(flags, 'ml_label', None)

    cov_per_source = {}   # src -> (cov_matrix, mean_hist, n_universes)

    for src, src_arr in hists_by_source.items():
        if len(src_arr) < 2:
            continue

        standard_path = f'{cov_dir}/covariance_{src}_{var_name}.npz'
        if src == 'ml' and freeze_ml and os.path.exists(standard_path):
            print(f"  --freeze-ml: loading existing {standard_path}")
            cached = np.load(standard_path)
            cov_per_source[src] = (cached['cov'], cached['mean_hist'],
                                   int(cached['n_universes']))
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

        payload = dict(cov=sc, frac_cov=sf, bins=bins,
                       mean_hist=sm, nom_hist=nom_hist, all_hists=src_arr,
                       n_universes=np.array(len(src_arr)))

        # Always persist per-source files (needed by the uncertainty budget /
        # per-source correlation breakdowns in MakePlots).
        np.savez(standard_path, **payload)
        print(f"  Per-source saved: covariance_{src}_{var_name}.npz "
              f"({len(src_arr)} universes)")

        if src == 'ml':
            label = ml_label or f"{len(src_arr)}rep"
            snapshot_path = f'{cov_dir}/covariance_ml_{label}_{var_name}.npz'
            np.savez(snapshot_path, **payload)
            print(f"  ML snapshot saved: covariance_ml_{label}_{var_name}.npz")

        cov_per_source[src] = (sc, sm, len(src_arr))

    # ── Build total covariance as sum of per-source ───────────────────────────
    cov = np.zeros((n_bins, n_bins))
    for src, (sc, sm, n_u) in cov_per_source.items():
        if src == 'ml' and ml_as_stderr:
            cov += sc / n_u
            print(f"  ML covariance scaled by 1/{n_u} (--ml-as-stderr): "
                  f"√diag goes from {np.sqrt(np.diag(sc)).mean():.1f} "
                  f"to {np.sqrt(np.diag(sc/n_u)).mean():.1f}")
        else:
            cov += sc

    if add_analytic_stat:
        cov += stat_cov
        cov_per_source['stat'] = (stat_cov, nom_hist, 1)
        print(f"  Added analytic MC-stat covariance to the total (fds).")

    if n_univ > 0:
        mean_hist = all_hists.mean(axis=0)
    else:
        mean_hist = nom_hist   # pure analytic-stat total

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
             mean_hist=mean_hist, nom_hist=nom_hist,
             all_hists=all_hists if n_univ > 0 else mean_hist[np.newaxis, :],
             ml_as_stderr=np.array(ml_as_stderr),
             sources=np.array(list(cov_per_source.keys())))
    print(f"  Combined saved: covariance_{flags.source}_{var_name}.npz "
          f"(sources: {list(cov_per_source.keys())})")

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'Bin center':>10s} {'Nominal':>10s} {'Mean':>10s} "
          f"{'Abs unc':>10s} {'Frac unc':>10s}")
    for i in range(n_bins):
        print(f"{centers[i]:10.1f} {nom_hist[i]:10.1f} {mean_hist[i]:10.1f} "
              f"{diag_unc[i]:10.2f} {frac_unc[i]:10.4f}")

    # ── Plot 1: Covariance matrices ───────────────────────────────────────────
    # ── Universe spread and stability (terminal only, no plot) ───────────────
    if n_univ == 0:
        print(f"  (No universe spread / chi2-vs-iter for analytic-only total.)")
        return

    # ── Plot 4: Chi2 vs iteration (systematic stability diagnostic) ───────────
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

    hists_i0 = []
    stale_count = 0
    for udir in all_udirs:
        pf = glob.glob(udir + 'Step2_Iter0_*_PushWeights.npy')
        if not pf: continue
        push = np.load(pf[0])
        push = push if push.ndim == 1 else push.mean(axis=0)
        if push.shape[0] != mc_weights.shape[0]:
            stale_count += 1; continue
        h, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
        hists_i0.append(h)
    if stale_count:
        print(f"  WARNING: {stale_count} universe dirs skipped in stability chi2 "
              f"(stale weights — rerun run-syst)")
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
            if push.shape[0] != mc_weights.shape[0]:
                continue
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
    print(f"  {'Iter':>5s} {'chi2':>10s} {'p-value':>9s}  note")
    for pi, c2 in zip(paper_iters, paper_chi2):
        if np.isnan(c2):
            s, ps = "       N/A", "      N/A"
        else:
            s  = f"{c2:10.2f}"
            ps = f"{chi2_pvalue(c2, ndf):9.3f}"
        note = "(prior, no unfolding)" if pi == 0 else ""
        print(f"  {pi:5d} {s} {ps}  {note}")


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

    xsec_tag = flags.tag or make_fdt_tag(flags.tilted_var, flags.alpha)

    print(f"\n{'='*60}")
    print(f"Cross-section extraction: {var_name}  (cov-source={flags.cov_source})")
    print(f"{'='*60}")

    eff_file = flags.export_dir + f'efficiency_{var_name}.npy'
    eff = np.load(eff_file) if os.path.exists(eff_file) else np.ones(n_bins)

    N_nom, _ = np.histogram(var_vals, bins=bins, weights=mc_weights)
    xsec_nom = N_nom / (eff.clip(1e-6) * bin_widths * NORM)

    tilt_dir = f'weights_sbnd_fakedata_{xsec_tag}/'
    push_files = sorted(glob.glob(tilt_dir + 'Step2_Iter*_PushWeights.npy'), key=iter_num)
    has_unf = bool(push_files)
    if has_unf:
        push = np.load(push_files[-1])
        push = push if push.ndim == 1 else push.mean(axis=0)
        if push.shape[0] != mc_weights.shape[0]:
            print(f"  ERROR: push weights have {push.shape[0]} events but current sample "
                  f"has {mc_weights.shape[0]}. Retrain the fake-data OmniFold run.")
            has_unf = False
            xsec_unf = xsec_nom
        else:
            N_unf, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
            xsec_unf = N_unf / (eff.clip(1e-6) * bin_widths * NORM)
    else:
        xsec_unf = xsec_nom

    tilt_file = flags.data_dir + f'truth_weights_sbnd_fakedata_{xsec_tag}.npy'
    has_truth = os.path.exists(tilt_file)
    if has_truth:
        tilt = np.load(tilt_file)
        N_truth, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * tilt)
        xsec_truth = N_truth / (eff.clip(1e-6) * bin_widths * NORM)

    cov_file = getattr(flags, 'cov_dir', 'sbnd/covariance/') + \
        f'/covariance_{flags.cov_source}_{var_name}.npz'
    has_syst = os.path.exists(cov_file)
    if has_syst:
        cd = np.load(cov_file)
        scale = eff.clip(1e-6) * bin_widths * NORM
        cov_xsec = cd['cov'] / np.outer(scale, scale)
        xsec_unc = np.sqrt(np.diag(cov_xsec))
    else:
        print(f"  WARNING: {cov_file} not found — build it with "
              f"'covariance --source {flags.cov_source}'")
        xsec_unc = np.zeros(n_bins); cov_xsec = None

    # ── Goodness of fit (stat cov for chi2 — see MakePlots validation for same reason) ─
    if has_unf and has_truth:
        d = xsec_unf - xsec_truth
        # Use diagonal stat covariance for the chi2 (the correlated systematic cov
        # is centered on nominal, not on the fake-data result)
        unf_w2, _ = np.histogram(var_vals, bins=bins, weights=(mc_weights * push) ** 2)
        stat_diag_xsec = unf_w2 / (eff.clip(1e-6) * bin_widths * NORM) ** 2
        c2 = float(np.sum(d ** 2 / stat_diag_xsec.clip(1e-30)))
        ndf = n_bins - 1
        p = chi2_pvalue(c2, ndf)
        gof = fit_annotation(c2, ndf)
        print(f"  Xsec vs tilted truth (stat cov): {gof}")
    else:
        gof = None

    # ── Plot: xsec + frac unc subplot ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})
    axes[0].step(bins, np.append(xsec_nom, xsec_nom[-1]),
                 where='post', color='gray', linewidth=1.5, linestyle='--',
                 label='Nominal MC (prior)')
    if has_truth:
        axes[0].step(bins, np.append(xsec_truth, xsec_truth[-1]),
                     where='post', color='black', linewidth=2, label='Tilted data')
    if has_unf:
        axes[0].errorbar(centers, xsec_unf, yerr=xsec_unc if has_syst else None,
                         fmt='ro', markersize=5, capsize=3, linewidth=1.5,
                         label='OmniFold')
    axes[0].set_ylabel(ylabel); axes[0].legend(loc='best', fontsize=10)
    axes[0].set_title(rf'SBND $\nu_e$ CC: {var_name}')
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    if gof:
        axes[0].text(0.03, 0.97, gof, transform=axes[0].transAxes,
                     va='top', ha='left', fontsize=9,
                     bbox=dict(boxstyle='round', fc='white', ec='none', alpha=0.8))

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

    # ── Print cross-section table ─────────────────────────────────────────────
    bfmt = '.0f' if var_name == 'true_p' else '.2f'
    print(f"\n  {'Bin':>14s} {'Nominal':>12s} {'OmniFold':>12s} "
          f"{'Syst unc':>12s} {'Frac unc':>10s}")
    for i in range(n_bins):
        lo = f"{bins[i]:{bfmt}}"; hi = f"{bins[i+1]:{bfmt}}"
        print(f"  [{lo:>6s},{hi:>6s}] {xsec_nom[i]:12.4e} "
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
    if not hasattr(flags, 'cov_source'):
        flags.cov_source = 'all'
    vars_to_run = (['true_p', 'true_costheta'] if flags.var == 'both'
                   else [flags.var])
    for v in vars_to_run:
        extract_xsec_single_var(v)