"""
RunStudies.py — Run all SBND OmniFold studies.

Consolidates: MakeFakeDataWeights_SBND, CheckSBNDClosure, RunSystematicUniverses.

Actions:
  check-closure   Check closure-test push/pull weight stats (+ statistical metric)
  make-fakedata   Create fake-data weights (tilt on p OR cosθ, or BNB universe)
  run-syst        Run OmniFold for each systematic universe
  run-ml-unc      Run N single-network replicas for ML uncertainty

CHANGE LOG (this revision)
--------------------------
* Q3  check-closure now reports a *statistical* pass/fail in addition to the raw
      bias: the unfolded truth spectrum is compared to the nominal truth with a
      per-bin MC-stat covariance, giving chi2/ndf and a p-value. "Passed" now
      means p > 0.05 (i.e. consistent within ~2σ), the same spirit as the
      Wiener-SVD 2σ criterion, rather than an ad-hoc |bias|<1%.
* Q8  make-fakedata gains --tilt-var {true_p,true_costheta}. A cosθ tilt lets us
      inject a known distortion in the angular variable and check that OmniFold
      tracks it (complementary to the momentum tilt). Momentum keeps the old tag
      'tilt_alpha{a}' for backward compatibility; cosθ uses 'tilt_costheta_alpha{a}'.

Usage:
    python3 sbnd/RunStudies.py check-closure
    python3 sbnd/RunStudies.py make-fakedata --mode tilt --tilt-var true_p       --alpha 0.5
    python3 sbnd/RunStudies.py make-fakedata --mode tilt --tilt-var true_costheta --alpha 0.3
    python3 sbnd/RunStudies.py run-syst --source bnb --start 0 --end 100
    python3 sbnd/RunStudies.py run-ml-unc --n-replicas 10 --tag tilt_alpha0.5
"""

import numpy as np
import os, sys, glob, re, argparse, subprocess
import matplotlib.pyplot as plt

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
# short tag suffix per tilt variable
TILT_SHORT = {'true_p': 'p', 'true_costheta': 'costheta', 'both': 'both'}


def make_fdt_tag(var, alpha):
    """Derive the fake-data tag from --var and --alpha.

    true_p        -> tilt_p_alpha0.3
    true_costheta -> tilt_costheta_alpha0.3
    both          -> tilt_both_alpha0.3
    """
    return f'tilt_{TILT_SHORT[var]}_alpha{alpha}'


def tilt_label(var, alpha):
    """Human-readable label for plot titles."""
    names = {'true_p': 'p only', 'true_costheta': 'cosθ only', 'both': 'p + cosθ'}
    return f'Tilted: {names[var]}, α={alpha}'


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
        # Wilson–Hilferty approximation
        t = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
        return float(0.5 * math.erfc(t / math.sqrt(2.0)))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest='action', required=True)

# ── check-closure ─────────────────────────────────────────────────────────────
p_cl = sub.add_parser('check-closure')
p_cl.add_argument('--closure-dir', default='weights_sbnd_closure/')
p_cl.add_argument('--data-dir', default='../FormattedData_SBND/',
                  help='Needed for the binned statistical closure metric.')
p_cl.add_argument('--plot-dir', default='sbnd/plots_validation/')
p_cl.add_argument('--pval-thresh', type=float, default=0.05,
                  help='Closure passes if the binned chi2 p-value exceeds this '
                       '(0.05 ~ 2σ, matching the Wiener-SVD convention).')
p_cl.add_argument('--var', choices=['true_p', 'true_costheta', 'both'], default='both',
                  help='Which variable(s) to check: true_p, true_costheta, or both.')
p_cl.add_argument('--show-trials', action='store_true',
                  help='Plot individual NTRIAL weight distributions to show run-to-run spread.')

# ── make-fakedata ─────────────────────────────────────────────────────────────
p_fd = sub.add_parser('make-fakedata')
p_fd.add_argument('--mode', choices=['tilt', 'universe'], default='tilt')
p_fd.add_argument('--var', choices=['true_p', 'true_costheta', 'both'], default='true_p',
                  help="Which variable(s) to tilt. "
                       "'true_p': tilt momentum only. "
                       "'true_costheta': tilt angle only. "
                       "'both': simultaneous tilt on p AND cosθ.")
p_fd.add_argument('--alpha', type=float, default=0.5)
p_fd.add_argument('--universe-file', type=str, default=None)
p_fd.add_argument('--universe-idx', type=int, default=0)
p_fd.add_argument('--data-dir', default='../FormattedData_SBND/')
p_fd.add_argument('--plot-dir', default='sbnd/plots_validation/')

# ── run-syst ──────────────────────────────────────────────────────────────────
p_sy = sub.add_parser('run-syst')
p_sy.add_argument('--source', choices=['bnb', 'genie', 'extra_xsec', 'g4', 'mcstat'], required=True)
p_sy.add_argument('--start', type=int, default=0)
p_sy.add_argument('--end', type=int, default=100)
p_sy.add_argument('--ntrial', type=int, default=3,
                  help='NTRIAL per syst universe (default 3). Use ≥3 to reduce ML noise '
                       'contamination in per-source covariances. ML-unc replicas always '
                       'use NTRIAL=1.')
p_sy.add_argument('--data-dir', default='../FormattedData_SBND/')
p_sy.add_argument('--universe-file', type=str, default=None)

# ── run-ml-unc ────────────────────────────────────────────────────────────────
p_ml = sub.add_parser('run-ml-unc')
p_ml.add_argument('--n-replicas', type=int, default=10)
p_ml.add_argument('--start', type=int, default=None)
p_ml.add_argument('--end', type=int, default=None)
p_ml.add_argument('--data-dir', default='../FormattedData_SBND/')
p_ml.add_argument('--tag', type=str, default=None,
                  help='Fake data tag (override). If omitted, derived from --var and --alpha.')
p_ml.add_argument('--var', choices=['true_p', 'true_costheta', 'both'], default='true_p',
                  help='Which variable was tilted (used to derive tag if --tag is omitted).')
p_ml.add_argument('--alpha', type=float, default=0.3,
                  help='Tilt strength (used to derive tag if --tag is omitted).')
p_ml.add_argument('--weights-base', default='sbnd/weights_ml_unc/')
p_ml.add_argument('--niter', type=int, default=10)
p_ml.add_argument('--epochs', type=int, default=100)

flags = parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# check-closure
# ═══════════════════════════════════════════════════════════════════════════════
def do_check_closure():
    d = flags.closure_dir
    PLOT_DIR = flags.plot_dir
    os.makedirs(PLOT_DIR, exist_ok=True)

    def itn(p):
        m = re.search(r'Iter(\d+)', p)
        return int(m.group(1)) if m else -1

    push_files = sorted(glob.glob(os.path.join(d, 'Step2_Iter*_PushWeights.npy')), key=itn)
    pull_files = sorted(glob.glob(os.path.join(d, 'Step1_Iter*_PullWeights.npy')), key=itn)
    if not push_files or not pull_files:
        print(f"ERROR: No weight files in {d}. Run closure test first.")
        return

    # Final-iteration averaged weights (shape: N_events or N_events x NTRIAL)
    push_raw = np.load(push_files[-1])
    pull_raw = np.load(pull_files[-1])

    # If NTRIAL > 1, the file shape is (N_events, NTRIAL); average gives the combined weight
    push_trials = push_raw if push_raw.ndim == 2 else push_raw[:, np.newaxis]
    pull_trials = pull_raw if pull_raw.ndim == 2 else pull_raw[:, np.newaxis]
    n_trials_push = push_trials.shape[1]
    n_trials_pull = pull_trials.shape[1]

    push = push_trials.mean(axis=1)  # per-event average across trials
    pull = pull_trials.mean(axis=1)

    push_bias = push.mean() - 1.0
    pull_bias = pull.mean() - 1.0

    print(f"=== Closure Test Check ({d}) ===")
    print(f"  NTRIAL detected — push: {n_trials_push}, pull: {n_trials_pull}")
    print(f"  Push (averaged): mean={push.mean():.4f}, std={push.std():.4f}, "
          f"range=[{push.min():.4f}, {push.max():.4f}]")
    print(f"  Pull (averaged): mean={pull.mean():.4f}, std={pull.std():.4f}")
    print(f"  Closure bias (push): {push_bias:+.4f} ({push_bias*100:+.2f}%)")
    print(f"  Closure bias (pull): {pull_bias:+.4f} ({pull_bias*100:+.2f}%)")

    # Per-trial bias breakdown — shows run-to-run spread that causes re-run variability
    if n_trials_push > 1:
        trial_biases = push_trials.mean(axis=0) - 1.0
        print(f"  Per-trial push biases: " +
              ", ".join(f"{b*100:+.2f}%" for b in trial_biases))
        print(f"  Trial bias spread (std): {trial_biases.std()*100:.2f}% "
              f"<- this is the run-to-run variability you observe")
        print(f"  Expected run-to-run variability: ±{trial_biases.std()*100/np.sqrt(n_trials_push):.2f}% "
              f"(±1 std of the mean over {n_trials_push} trials)")

    # ── Q3: STATISTICAL closure metric (binned, covariance-based) ─────────────
    # A closure test should reproduce the *nominal truth* spectrum. Instead of an
    # ad-hoc |bias|<1%, we ask whether the unfolded spectrum is consistent with the
    # nominal truth within the MC statistical uncertainty. This mirrors the
    # Wiener-SVD "within 2σ" convention: pass if p-value > pval_thresh (~0.05).
    stat_status = None
    try:
        truth_raw  = np.load(flags.data_dir + 'mc_vals_truth_NoNorm.npy')
        mc_weights = np.load(flags.data_dir + 'mc_weights_reco.npy')
        # Guard: weights were trained on a different sample size than the current inputs.
        # This happens when FormatData was rerun (e.g. new selection / POT scale) but
        # the closure OmniFold job was not.  Catch it early with a clear message.
        if push.shape[0] != mc_weights.shape[0]:
            print(f"\n  ERROR: shape mismatch — closure weights have {push.shape[0]} events "
                  f"but current mc_weights_reco.npy has {mc_weights.shape[0]} events.")
            print(f"  The closure OmniFold job was trained on the OLD sample. "
                  f"Retrain it first:")
            print(f"    bash sbnd/runOmnifold_sbnd.sh   # (or your closure training script)")
            print(f"  Then re-run:  python3 sbnd/RunStudies.py check-closure")
            print(f"\n  Skipping statistical closure metric.")
            stat_status = None
            raise StopIteration
        all_vars = {'true_p': truth_raw[:, 0], 'true_costheta': truth_raw[:, 1]}
        if flags.var == 'both':
            var_map = all_vars
        else:
            var_map = {flags.var: all_vars[flags.var]}
        print(f"\n  --- Statistical closure metric (unfolded vs nominal truth) ---")
        print(f"      pass if p-value > {flags.pval_thresh:.2f} (~2σ, per bin covariance = MC stat)")
        all_pass = True
        for vn, vv in var_map.items():
            bins = BINNING[vn]; n_bins = len(bins) - 1
            nom_h,  _ = np.histogram(vv, bins=bins, weights=mc_weights)
            unf_h,  _ = np.histogram(vv, bins=bins, weights=mc_weights * push)
            # MC statistical variance of the unfolded spectrum: sum of (w*push)^2
            var_h,  _ = np.histogram(vv, bins=bins, weights=(mc_weights * push) ** 2)
            good = var_h > 0
            d_bin = (unf_h - nom_h)[good]
            chi2 = float(np.sum(d_bin ** 2 / var_h[good]))
            # total is preserved by OmniFold -> one constraint removed
            ndf = max(int(good.sum()) - 1, 1)
            pval = chi2_pvalue(chi2, ndf)
            ok = pval > flags.pval_thresh
            all_pass = all_pass and ok
            # largest per-bin pull, for context
            pulls = d_bin / np.sqrt(var_h[good])
            print(f"      {vn:14s}: chi2/ndf = {chi2:7.2f}/{ndf} = {chi2/ndf:6.2f}, "
                  f"p = {pval:6.3f}, max|pull| = {np.abs(pulls).max():4.2f}  "
                  f"-> {'PASS' if ok else 'FAIL'}")
        stat_status = ("PASSED (statistical: unfolded ≈ nominal truth within ~2σ)"
                       if all_pass else
                       "FAILED (statistical: unfolded deviates from nominal truth > 2σ)")
    except StopIteration:
        pass  # message already printed by the shape-mismatch guard above
    except FileNotFoundError:
        print(f"\n  (Statistical closure metric skipped — {flags.data_dir} inputs not found.)")

    # ── Legacy tiered verdict on the raw per-event bias (kept for continuity) ──
    abs_bias = abs(push_bias)
    if push.std() >= 0.2:
        status = "FAIL — std too large, network not converging"
    elif abs_bias < 0.01:
        status = "raw-bias verdict: |bias| < 1% (excellent)"
    elif abs_bias < 0.03:
        status = (f"raw-bias verdict: |bias|={abs_bias*100:.1f}% < 3%. "
                  f"Acceptable for fake-data studies (subdominant vs 5-30% systematics). "
                  f"Run-to-run variability from finite NTRIAL={n_trials_push} is normal.")
    elif abs_bias < 0.05:
        status = (f"raw-bias verdict: |bias|={abs_bias*100:.1f}% (3-5%). "
                  f"Marginal — increase NTRIAL to 5-7 before publishing.")
    else:
        status = (f"raw-bias verdict: |bias|={abs_bias*100:.1f}% >= 5%. "
                  f"Investigate reco-truth correlation and increase NTRIAL.")
    print(f"\n  {status}")
    if stat_status is not None:
        print(f"  ==> CLOSURE {stat_status}")

    # ── Plot 1: weight distributions (pull left, push right) ──────────────────
    XLIM = (0.85, 1.15)
    BINS = np.linspace(XLIM[0], XLIM[1], 81)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(pull, bins=BINS, color="darkorange", alpha=0.8)
    axes[0].axvline(1.0, color="red", linestyle="--", linewidth=2)
    axes[0].set_xlim(XLIM)
    axes[0].set_xlabel("Pull weight"); axes[0].set_ylabel("Events")
    axes[0].set_title(f"Closure pull weights — Step 1 (reco space)\n"
                      f"mean={pull.mean():.4f}, std={pull.std():.4f}, "
                      f"bias={pull_bias*100:+.2f}%")

    if getattr(flags, 'show_trials', False) and n_trials_push > 1:
        trial_cols = plt.cm.Blues(np.linspace(0.4, 0.9, n_trials_push))
        for t in range(n_trials_push):
            axes[1].hist(push_trials[:, t], bins=BINS, density=True,
                         alpha=0.3, color=trial_cols[t],
                         label=f'Trial {t+1} (bias={push_trials[:,t].mean()-1:+.3f})')
        axes[1].hist(push, bins=BINS, histtype='step', color='navy',
                     linewidth=2, density=True, label=f'Average (bias={push_bias*100:+.2f}%)')
        axes[1].axvline(1.0, color="red", linestyle="--", linewidth=2)
        axes[1].set_xlim(XLIM)
        axes[1].set_xlabel("Push weight"); axes[1].set_ylabel("Density")
        axes[1].set_title(f"Closure push weights — Step 2 (truth space)\n"
                          f"mean={push.mean():.4f}, {n_trials_push} trials shown individually")
        axes[1].legend(fontsize=8)
    else:
        axes[1].hist(push, bins=BINS, color="steelblue", alpha=0.8)
        axes[1].axvline(1.0, color="red", linestyle="--", linewidth=2)
        axes[1].set_xlim(XLIM)
        axes[1].set_xlabel("Push weight"); axes[1].set_ylabel("Events")
        axes[1].set_title(f"Closure push weights — Step 2 (truth space)\n"
                          f"mean={push.mean():.4f}, std={push.std():.4f}, "
                          f"bias={push_bias*100:+.2f}%")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/closure_weight_distributions.png", dpi=150)
    plt.close()
    print(f"  Plot saved: {PLOT_DIR}/closure_weight_distributions.png")

    # ── Plot 2: per-iteration convergence (pull left, push right) ─────────────
    if len(push_files) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for ax, files, label, color, name in [
            (axes[0], pull_files, "Pull weight mean ± std", "darkorange", "pull"),
            (axes[1], push_files, "Push weight mean ± std", "steelblue",  "push"),
        ]:
            iters, means, stds = [], [], []
            for fi in sorted(files, key=itn):
                w = np.load(fi)
                w = w.mean(axis=1) if w.ndim == 2 else w
                iters.append(itn(fi) + 1)
                means.append(w.mean()); stds.append(w.std())
            ax.errorbar(iters, means, yerr=stds, fmt="o-", color=color,
                        capsize=3, linewidth=2)
            ax.axhline(1.0, color="red", linestyle="--")
            ax.fill_between(iters,
                            [m - 0.01 for m in means],
                            [m + 0.01 for m in means],
                            alpha=0.15, color=color, label="±1% band")
            ax.set_xlabel("OmniFold Iteration"); ax.set_ylabel(label)
            ax.set_title(f"Closure {name} convergence\nbias at final iter: {means[-1]-1:+.4f}")
            ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/closure_convergence.png", dpi=150)
        plt.close()
        print(f"  Plot saved: {PLOT_DIR}/closure_convergence.png")

    # ── Plot 3: per-trial bias bar chart (if NTRIAL > 1) ──────────────────────
    if n_trials_push > 1:
        fig, ax = plt.subplots(figsize=(max(6, n_trials_push * 1.2), 4))
        trial_biases_pct = (push_trials.mean(axis=0) - 1.0) * 100
        colors_t = ['steelblue' if b >= 0 else 'tomato' for b in trial_biases_pct]
        ax.bar(range(1, n_trials_push + 1), trial_biases_pct, color=colors_t, alpha=0.8)
        ax.axhline(0, color='black', linewidth=1)
        ax.axhline(trial_biases_pct.mean(), color='navy', linewidth=2,
                   linestyle='--', label=f'Mean bias = {trial_biases_pct.mean():+.2f}%')
        ax.fill_between([-0.5, n_trials_push + 0.5], -1, 1,
                        alpha=0.1, color='green', label='±1% band')
        ax.fill_between([-0.5, n_trials_push + 0.5], -3, 3,
                        alpha=0.07, color='orange', label='±3% band')
        ax.set_xlabel("Trial index"); ax.set_ylabel("Bias (%)")
        ax.set_title(f"Per-trial closure bias (push, final iteration)\n"
                     f"spread std={trial_biases_pct.std():.2f}% — "
                     f"this is the run-to-run variability")
        ax.set_xlim(0.5, n_trials_push + 0.5)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/closure_trial_biases.png", dpi=150)
        plt.close()
        print(f"  Plot saved: {PLOT_DIR}/closure_trial_biases.png")
        print(f"  Tip: run with --show-trials to overlay individual trial distributions.")


# ═══════════════════════════════════════════════════════════════════════════════
# make-fakedata
# ═══════════════════════════════════════════════════════════════════════════════
def _tilt_transform(var_name, values):
    """Return a standardised coordinate z(x) used to build the tilt.

    true_p        : log-standardised (momentum spans orders of magnitude)
    true_costheta : linear-standardised (bounded variable, no log)
    """
    if var_name == 'true_p':
        x = np.log(values.clip(1.0, None))
    else:  # true_costheta (or any bounded/linear variable)
        x = values.astype(np.float64)
    return (x - np.mean(x)) / np.std(x)


def do_make_fakedata():
    OUT = flags.data_dir
    mc_weights_reco = np.load(OUT + 'mc_weights_reco.npy')
    truth_raw       = np.load(OUT + 'mc_vals_truth_NoNorm.npy')
    n = len(mc_weights_reco)
    true_p        = truth_raw[:, 0]
    true_costheta = truth_raw[:, 1]
    var_map = {'true_p': true_p, 'true_costheta': true_costheta}

    print(f"Loaded {n:,} events from {OUT}")

    PLOT_DIR = flags.plot_dir
    os.makedirs(PLOT_DIR, exist_ok=True)

    if flags.mode == 'tilt':
        ALPHA = flags.alpha
        VAR   = flags.var
        tag   = make_fdt_tag(VAR, ALPHA)

        # ── Build tilt weight(s) ───────────────────────────────────────────────
        if VAR == 'both':
            z_p   = _tilt_transform('true_p',        true_p)
            z_cos = _tilt_transform('true_costheta',  true_costheta)
            tilt  = (1.0 + ALPHA * z_p) * (1.0 + ALPHA * z_cos)
        else:
            z     = _tilt_transform(VAR, var_map[VAR])
            tilt  = 1.0 + ALPHA * z
        tilt = np.clip(tilt, 0.0, None)
        tilt = tilt * (mc_weights_reco.sum() / (mc_weights_reco * tilt).sum())
        data_weights = mc_weights_reco * tilt

        np.save(OUT + f'data_weights_sbnd_fakedata_{tag}.npy', data_weights)
        np.save(OUT + f'truth_weights_sbnd_fakedata_{tag}.npy', tilt)

        title = tilt_label(VAR, ALPHA)
        print(f"=== Synthetic Tilt Mode ===")
        print(f"  {title}")
        print(f"  tag = {tag}")
        print(f"  Tilt range: [{tilt.min():.3f}, {tilt.max():.3f}]")
        print(f"  Saved: {OUT}data_weights_sbnd_fakedata_{tag}.npy")
        print(f"         {OUT}truth_weights_sbnd_fakedata_{tag}.npy")
        print(f"\n  Next: nohup bash sbnd/runOmnifold_sbnd_fakedata.sh --var {VAR} --alpha {ALPHA}")

        # ── Injection summary plot ─────────────────────────────────────────────
        tilted_set = {'true_p', 'true_costheta'} if VAR == 'both' else {VAR}
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'Fake-data injection  ({title})', fontsize=13, fontweight='bold')

        bins_p   = np.linspace(0, np.percentile(true_p, 99), 25)
        bins_cos = np.linspace(-1, 1, 20)

        p_label   = 'true_p  ← TILTED'   if 'true_p'        in tilted_set else 'true_p  (projected)'
        cos_label = 'cosθ  ← TILTED'     if 'true_costheta' in tilted_set else 'cosθ  (projected)'

        axes[0].hist(true_p, bins=bins_p, weights=mc_weights_reco,
                     alpha=0.55, color='steelblue', label='Nominal MC')
        axes[0].hist(true_p, bins=bins_p, weights=data_weights,
                     alpha=0.55, color='tomato', label='Fake Data')
        axes[0].set_xlabel('true_p [MeV/c]'); axes[0].set_ylabel('Weighted events')
        axes[0].set_title(p_label); axes[0].legend()

        axes[1].hist(true_costheta, bins=bins_cos, weights=mc_weights_reco,
                     alpha=0.55, color='steelblue', label='Nominal MC')
        axes[1].hist(true_costheta, bins=bins_cos, weights=data_weights,
                     alpha=0.55, color='tomato', label='Fake Data')
        axes[1].set_xlabel(r'true $\cos\theta_e$'); axes[1].set_ylabel('Weighted events')
        axes[1].set_title(cos_label); axes[1].legend()

        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/fakedata_injected_{tag}.png', dpi=150)
        print(f"  Plot: {PLOT_DIR}/fakedata_injected_{tag}.png")

    elif flags.mode == 'universe':
        print("=== BNB Universe Mode ===")
        if flags.universe_file is None:
            print("ERROR: --universe-file required.")
            sys.exit(1)
        uni_all = np.load(flags.universe_file)
        if uni_all.ndim == 1:
            uni_all = uni_all[:, np.newaxis]
        assert uni_all.shape[0] == n, \
            f"Shape mismatch: {uni_all.shape[0]} vs {n}"
        idx = flags.universe_idx
        uni_vals = np.clip(uni_all[:, idx], 0.0, 10.0)
        data_weights = mc_weights_reco * uni_vals
        data_weights = data_weights * (mc_weights_reco.sum() / data_weights.sum())
        tag = f'bnb_univ{idx}'
        np.save(OUT + f'data_weights_sbnd_fakedata_{tag}.npy', data_weights)
        np.save(OUT + f'truth_weights_sbnd_fakedata_{tag}.npy', uni_vals)
        print(f"  Universe {idx}: mean={uni_vals.mean():.4f}, std={uni_vals.std():.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# run-syst
# ═══════════════════════════════════════════════════════════════════════════════
def do_run_syst():
    if flags.universe_file is None:
        flags.universe_file = f'sbnd/exported_weights/{flags.source}_universe_weights.npy'
    uni_all = np.load(flags.universe_file)
    mc_weights = np.load(flags.data_dir + 'mc_weights_reco.npy')

    for idx in range(flags.start, flags.end):
        tag = f'{flags.source}_univ{idx}'
        weights_dir = f'sbnd/weights_{flags.source}/{tag}/'
        out_file = flags.data_dir + f'data_weights_sbnd_syst_{tag}.npy'

        uni_vals = np.clip(uni_all[:, idx], 0.0, 10.0)
        data_weights = mc_weights * uni_vals
        data_weights = data_weights * (mc_weights.sum() / data_weights.sum())
        np.save(out_file, data_weights)

        config = {
            'FILE_MC_RECO': 'mc_vals_reco.npy', 'FILE_MC_GEN': 'mc_vals_truth.npy',
            'FILE_MC_FLAG_RECO': 'mc_pass_reco.npy', 'FILE_MC_FLAG_GEN': 'mc_pass_truth.npy',
            'FILE_DATA_RECO': 'mc_vals_reco.npy', 'FILE_DATA_FLAG_RECO': 'mc_pass_reco.npy',
            'FILE_DATA_WEIGHT': f'data_weights_sbnd_syst_{tag}.npy',
            'FILE_MC_RECO_WEIGHT': 'mc_weights_reco.npy',
            'FILE_MC_GEN_WEIGHT': 'mc_weights_truth.npy',
            'NITER': 5, 'NTRIAL': flags.ntrial, 'LR': 1e-3, 'BATCH_SIZE': 512,
            'EPOCHS': 50, 'NAME': f'sbnd_syst_{tag}', 'NPATIENCE': 7,
        }
        config_path = f'sbnd/config_syst_{tag}.json'
        with open(config_path, 'w') as f:
            f.write('{\n')
            for i, (k, v) in enumerate(config.items()):
                comma = ',' if i < len(config) - 1 else ''
                if isinstance(v, str):
                    f.write(f"'{k}':'{v}'{comma}\n")
                else:
                    f.write(f"'{k}': {v}{comma}\n")
            f.write('}\n')
        os.makedirs(weights_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Running universe {idx} ({flags.source})")
        print(f"{'='*60}")
        subprocess.run([
            sys.executable, 'run_sbnd.py',
            '--config', config_path, '--file_path', flags.data_dir,
            '--weights_folder', weights_dir, '--no_eff', '--verbose'
        ])
        os.remove(config_path)

    print("\nAll universes complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# run-ml-unc
# ═══════════════════════════════════════════════════════════════════════════════
def do_run_ml_unc():
    # Derive the fake-data tag from --var/--alpha if --tag not given explicitly.
    fdt_tag = flags.tag or make_fdt_tag(flags.var, flags.alpha)

    if flags.start is not None and flags.end is not None:
        replica_range = range(flags.start, flags.end)
    else:
        replica_range = range(flags.n_replicas)

    os.makedirs(flags.weights_base, exist_ok=True)

    for idx in replica_range:
        tag = f'replica_{idx}'
        weights_dir = f'{flags.weights_base}/{tag}/'
        os.makedirs(weights_dir, exist_ok=True)

        data_weight_file = f'data_weights_sbnd_fakedata_{fdt_tag}.npy'

        config = {
            'FILE_MC_RECO': 'mc_vals_reco.npy', 'FILE_MC_GEN': 'mc_vals_truth.npy',
            'FILE_MC_FLAG_RECO': 'mc_pass_reco.npy', 'FILE_MC_FLAG_GEN': 'mc_pass_truth.npy',
            'FILE_DATA_RECO': 'mc_vals_reco.npy', 'FILE_DATA_FLAG_RECO': 'mc_pass_reco.npy',
            'FILE_DATA_WEIGHT': data_weight_file,
            'FILE_MC_RECO_WEIGHT': 'mc_weights_reco.npy',
            'FILE_MC_GEN_WEIGHT': 'mc_weights_truth.npy',
            'NITER': flags.niter, 'NTRIAL': 1,
            'LR': 1e-3, 'BATCH_SIZE': 512, 'EPOCHS': flags.epochs,
            'NAME': f'sbnd_ml_unc_{tag}', 'NPATIENCE': 10,
        }
        config_path = f'sbnd/config_ml_unc_{tag}.json'
        with open(config_path, 'w') as f:
            f.write('{\n')
            for i, (k, v) in enumerate(config.items()):
                comma = ',' if i < len(config) - 1 else ''
                if isinstance(v, str):
                    f.write(f"'{k}':'{v}'{comma}\n")
                else:
                    f.write(f"'{k}': {v}{comma}\n")
            f.write('}\n')

        print(f"\n{'='*60}")
        print(f"ML uncertainty replica {idx} (NTRIAL=1, NITER={flags.niter})")
        print(f"{'='*60}")

        env = os.environ.copy()
        env['TF_RANDOM_SEED'] = str(42 + idx * 137)
        env['PYTHONHASHSEED'] = str(idx)

        subprocess.run([
            sys.executable, 'run_sbnd.py',
            '--config', config_path, '--file_path', flags.data_dir,
            '--weights_folder', weights_dir, '--no_eff', '--verbose'
        ], env=env)
        if os.path.exists(config_path):
            os.remove(config_path)

    print(f"\nAll {len(list(replica_range))} ML replicas complete.")
    print(f"Run: python3 sbnd/BuildResults.py covariance --source ml --var true_p")


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════════════════
{'check-closure': do_check_closure,
 'make-fakedata': do_make_fakedata,
 'run-syst':      do_run_syst,
 'run-ml-unc':    do_run_ml_unc}[flags.action]()