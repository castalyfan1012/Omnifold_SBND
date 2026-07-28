"""
RunStudies.py — Run all SBND OmniFold studies.

Consolidates: MakeFakeDataWeights_SBND, CheckSBNDClosure, RunSystematicUniverses.

Actions:
  check-closure   Check closure-test push/pull weight stats
  make-fakedata   Create fake-data weights (tilt or BNB universe)
  run-syst        Run OmniFold for each systematic universe
  run-ml-unc      Run N single-network replicas for ML uncertainty

Usage:
    python3 sbnd/RunStudies.py check-closure
    python3 sbnd/RunStudies.py make-fakedata --mode tilt --alpha 0.5
    python3 sbnd/RunStudies.py run-syst --source bnb --start 0 --end 100
    python3 sbnd/RunStudies.py run-ml-unc --n-replicas 10 --tag tilt_alpha0.5
"""

import numpy as np
import os, sys, glob, re, argparse, subprocess
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest='action', required=True)

# ── check-closure ─────────────────────────────────────────────────────────────
p_cl = sub.add_parser('check-closure')
p_cl.add_argument('--closure-dir', default='weights_sbnd_closure/')
p_cl.add_argument('--plot-dir', default='sbnd/plots_validation/')

# ── make-fakedata ─────────────────────────────────────────────────────────────
p_fd = sub.add_parser('make-fakedata')
p_fd.add_argument('--mode', choices=['tilt', 'universe'], default='tilt')
p_fd.add_argument('--alpha', type=float, default=0.5)
p_fd.add_argument('--universe-file', type=str, default=None)
p_fd.add_argument('--universe-idx', type=int, default=0)
p_fd.add_argument('--data-dir', default='../FormattedData_SBND/')
p_fd.add_argument('--plot-dir', default='sbnd/plots_validation/')

# ── run-syst ──────────────────────────────────────────────────────────────────
p_sy = sub.add_parser('run-syst')
p_sy.add_argument('--source', choices=['bnb', 'genie', 'mcstat'], required=True)
p_sy.add_argument('--start', type=int, default=0)
p_sy.add_argument('--end', type=int, default=100)
p_sy.add_argument('--data-dir', default='../FormattedData_SBND/')
p_sy.add_argument('--universe-file', type=str, default=None)

# ── run-ml-unc ────────────────────────────────────────────────────────────────
p_ml = sub.add_parser('run-ml-unc')
p_ml.add_argument('--n-replicas', type=int, default=10)
p_ml.add_argument('--start', type=int, default=None)
p_ml.add_argument('--end', type=int, default=None)
p_ml.add_argument('--data-dir', default='../FormattedData_SBND/')
p_ml.add_argument('--tag', type=str, default=None,
                  help='Fake data tag. None = nominal MC as data (closure-like).')
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
    push = np.load(push_files[-1])
    pull = np.load(pull_files[-1])
    print(f"=== Closure Test Check ({d}) ===")
    print(f"  Push: mean={push.mean():.4f}, std={push.std():.4f}, "
          f"range=[{push.min():.4f}, {push.max():.4f}]")
    print(f"  Pull: mean={pull.mean():.4f}, std={pull.std():.4f}")
    passed = abs(push.mean() - 1.0) < 0.05 and push.std() < 0.2
    print("  CLOSURE TEST PASSED" if passed else
          "  WARNING: closure test looks off — investigate before proceeding")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(push, bins=80, color="steelblue", alpha=0.8)
    axes[0].axvline(1.0, color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Push weight"); axes[0].set_ylabel("Events")
    axes[0].set_title(f"Closure push weights (mean={push.mean():.4f}, std={push.std():.4f})")
    axes[1].hist(pull, bins=80, color="darkorange", alpha=0.8)
    axes[1].axvline(1.0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Pull weight"); axes[1].set_ylabel("Events")
    axes[1].set_title(f"Closure pull weights (mean={pull.mean():.4f}, std={pull.std():.4f})")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/closure_weight_distributions.png", dpi=150)
    plt.close()
    print(f"  Plot saved: {PLOT_DIR}/closure_weight_distributions.png")

    # Per-iteration push weight convergence
    if len(push_files) > 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        iters, means, stds = [], [], []
        for fi in push_files:
            w = np.load(fi)
            iters.append(itn(fi) + 1)
            means.append(w.mean()); stds.append(w.std())
        ax.errorbar(iters, means, yerr=stds, fmt="bo-", capsize=3, linewidth=2)
        ax.axhline(1.0, color="red", linestyle="--")
        ax.set_xlabel("OmniFold Iteration"); ax.set_ylabel("Push weight mean ± std")
        ax.set_title("Closure test convergence")
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/closure_convergence.png", dpi=150)
        plt.close()
        print(f"  Plot saved: {PLOT_DIR}/closure_convergence.png")


# ═══════════════════════════════════════════════════════════════════════════════
# make-fakedata
# ═══════════════════════════════════════════════════════════════════════════════
def do_make_fakedata():
    OUT = flags.data_dir
    mc_weights_reco = np.load(OUT + 'mc_weights_reco.npy')
    truth_raw       = np.load(OUT + 'mc_vals_truth_NoNorm.npy')
    n = len(mc_weights_reco)
    true_p        = truth_raw[:, 0]
    true_costheta = truth_raw[:, 1]

    print(f"Loaded {n:,} events from {OUT}")

    PLOT_DIR = flags.plot_dir
    os.makedirs(PLOT_DIR, exist_ok=True)

    if flags.mode == 'tilt':
        ALPHA = flags.alpha
        print(f"=== Synthetic Tilt Mode (alpha={ALPHA}, variable=true_p) ===")
        log_p      = np.log(true_p.clip(1.0, None))
        log_p_mean = np.mean(log_p)
        log_p_std  = np.std(log_p)
        tilt = 1.0 + ALPHA * (log_p - log_p_mean) / log_p_std
        tilt = np.clip(tilt, 0.0, None)
        tilt = tilt * (mc_weights_reco.sum() / (mc_weights_reco * tilt).sum())
        data_weights = mc_weights_reco * tilt
        tag = f'tilt_alpha{ALPHA}'
        np.save(OUT + f'data_weights_sbnd_fakedata_{tag}.npy', data_weights)
        np.save(OUT + f'truth_weights_sbnd_fakedata_{tag}.npy', tilt)
        print(f"  Tilt range: [{tilt.min():.3f}, {tilt.max():.3f}]")
        print(f"  Saved to {OUT}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        bins_p = np.linspace(0, np.percentile(true_p, 99), 25)
        axes[0].hist(true_p, bins=bins_p, weights=mc_weights_reco, alpha=0.6, label='Nominal MC')
        axes[0].hist(true_p, bins=bins_p, weights=data_weights, alpha=0.6, label='Fake Data')
        axes[0].set_xlabel('true_p [MeV/c]'); axes[0].legend()
        axes[0].set_title('Injected distortion: true_p')
        bins_cos = np.linspace(-1, 1, 20)
        axes[1].hist(true_costheta, bins=bins_cos, weights=mc_weights_reco, alpha=0.6, label='Nominal MC')
        axes[1].hist(true_costheta, bins=bins_cos, weights=data_weights, alpha=0.6, label='Fake Data')
        axes[1].set_xlabel(r'true $\cos\theta$'); axes[1].legend()
        axes[1].set_title(r'Projected: true $\cos\theta$')
        axes[2].scatter(true_p[::5], tilt[::5], s=3, alpha=0.3)
        axes[2].axhline(1.0, color='r', linestyle='--')
        axes[2].set_xlabel('true_p [MeV/c]'); axes[2].set_ylabel('Tilt weight')
        axes[2].set_title(f'Tilt function (alpha={ALPHA})')
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
            'NITER': 5, 'NTRIAL': 1, 'LR': 1e-3, 'BATCH_SIZE': 512,
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
            sys.executable, 't2k.py',
            '--config', config_path, '--file_path', flags.data_dir,
            '--weights_folder', weights_dir, '--no_eff', '--verbose'
        ])
        os.remove(config_path)

    print("\nAll universes complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# run-ml-unc
# ═══════════════════════════════════════════════════════════════════════════════
def do_run_ml_unc():
    if flags.start is not None and flags.end is not None:
        replica_range = range(flags.start, flags.end)
    else:
        replica_range = range(flags.n_replicas)

    os.makedirs(flags.weights_base, exist_ok=True)

    for idx in replica_range:
        tag = f'replica_{idx}'
        weights_dir = f'{flags.weights_base}/{tag}/'
        os.makedirs(weights_dir, exist_ok=True)

        data_weight_file = (f'data_weights_sbnd_fakedata_{flags.tag}.npy'
                            if flags.tag else 'mc_weights_reco.npy')

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
            sys.executable, 't2k.py',
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