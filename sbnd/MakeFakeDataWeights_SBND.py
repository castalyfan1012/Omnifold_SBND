"""
MakeFakeDataWeights_SBND.py

Creates fake data weights for SBND OmniFold validation.

Two modes:
  --mode tilt       Synthetic log(true_ke) tilt (default, self-contained)
  --mode universe   BNB universe weights (requires pre-saved aligned weights)

Also includes a --check-closure flag to verify a closure test before
proceeding to fake data generation.

Usage:
    python3 MakeFakeDataWeights_SBND.py                        # tilt mode
    python3 MakeFakeDataWeights_SBND.py --mode universe        # BNB universe
    python3 MakeFakeDataWeights_SBND.py --check-closure        # just check closure
"""

import numpy as np
import os
import glob
import re
import argparse
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = '../FormattedData_SBND/'
PLOT_DIR   = 'plots_sbnd_fakedata/'

os.makedirs(PLOT_DIR, exist_ok=True)

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['tilt', 'universe'], default='tilt',
                    help='Fake data mode: synthetic tilt or BNB universe')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Tilt strength for synthetic mode (default: 0.5)')
parser.add_argument('--universe-file', type=str, default=None,
                    help='Path to pre-saved aligned universe weights .npy '
                         '(shape must match mc_weights_reco.npy)')
parser.add_argument('--universe-idx', type=int, default=0,
                    help='Which universe column to use (default: 0)')
parser.add_argument('--check-closure', action='store_true',
                    help='Only check closure test weights, then exit')
parser.add_argument('--closure-dir', type=str, default='weights_sbnd_closure/',
                    help='Directory containing closure test weight files')
flags = parser.parse_args()


# ── Closure check ─────────────────────────────────────────────────────────────
def check_closure(weights_dir):
    """Verify closure test weights are ~1.0 before proceeding."""

    def iter_num(p):
        m = re.search(r'Iter(\d+)', p)
        return int(m.group(1)) if m else -1

    push_files = sorted(glob.glob(os.path.join(weights_dir, 'Step2_Iter*_PushWeights.npy')),
                        key=iter_num)
    pull_files = sorted(glob.glob(os.path.join(weights_dir, 'Step1_Iter*_PullWeights.npy')),
                        key=iter_num)

    if not push_files or not pull_files:
        print(f"ERROR: No weight files found in {weights_dir}")
        print("  Run the closure test first.")
        return False

    push = np.load(push_files[-1])
    pull = np.load(pull_files[-1])

    print(f"=== Closure Test Check ({weights_dir}) ===")
    print(f"  Push file: {os.path.basename(push_files[-1])}")
    print(f"  Push weights: mean={push.mean():.4f}, std={push.std():.4f}, "
          f"range=[{push.min():.4f}, {push.max():.4f}]")
    print(f"  Pull weights: mean={pull.mean():.4f}, std={pull.std():.4f}, "
          f"range=[{pull.min():.4f}, {pull.max():.4f}]")

    passed = abs(push.mean() - 1.0) < 0.05 and push.std() < 0.2
    if passed:
        print("  CLOSURE TEST PASSED")
    else:
        print("  WARNING: closure test looks off — investigate before fake data")
    return passed


if flags.check_closure:
    check_closure(flags.closure_dir)
    exit(0)


# ── Load saved arrays from FormatData_SBND.py ────────────────────────────────
mc_weights_reco = np.load(OUTPUT_DIR + 'mc_weights_reco.npy')
truth_raw       = np.load(OUTPUT_DIR + 'mc_vals_truth_NoNorm.npy')
n_events = len(mc_weights_reco)

print(f"Loaded {n_events:,} events from {OUTPUT_DIR}")
print(f"  mc_weights_reco: mean={mc_weights_reco.mean():.4f}")

# Check closure first
print()
if os.path.exists(flags.closure_dir):
    check_closure(flags.closure_dir)
else:
    print(f"  Closure dir {flags.closure_dir} not found — skipping check")
print()

# Truth variables (columns from FormatData_SBND.py: true_ke, true_costheta, true_p)
true_ke       = truth_raw[:, 0]
true_costheta = truth_raw[:, 1]
true_p        = truth_raw[:, 2]


# ── Mode: synthetic tilt ──────────────────────────────────────────────────────
if flags.mode == 'tilt':
    ALPHA = flags.alpha
    print(f"=== Synthetic Tilt Mode (alpha={ALPHA}) ===")

    # Tilt in log(true_ke) — same approach as the validated T2K test
    # Avoids extreme weights from the 4-order-of-magnitude KE range
    log_ke      = np.log(true_ke.clip(1.0, None))  # clip to avoid log(0)
    log_ke_mean = np.mean(log_ke)
    log_ke_std  = np.std(log_ke)

    tilt = 1.0 + ALPHA * (log_ke - log_ke_mean) / log_ke_std
    tilt = np.clip(tilt, 0.0, None)
    tilt = tilt * (mc_weights_reco.sum() / (mc_weights_reco * tilt).sum())

    data_weights = mc_weights_reco * tilt

    tag = f'tilt_alpha{ALPHA}'
    out_data = OUTPUT_DIR + f'data_weights_sbnd_fakedata_{tag}.npy'
    out_tilt = OUTPUT_DIR + f'truth_weights_sbnd_fakedata_{tag}.npy'
    np.save(out_data, data_weights)
    np.save(out_tilt, tilt)

    print(f"  Tilt range: [{tilt.min():.3f}, {tilt.max():.3f}]")
    print(f"  Data weights: mean={data_weights.mean():.4f}, std={data_weights.std():.4f}")
    print(f"  Saved: {out_data}")
    print(f"  Saved: {out_tilt}")

    # Diagnostic plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    bins_ke = np.linspace(0, true_ke.max() * 1.05, 25)
    axes[0].hist(true_ke, bins=bins_ke, weights=mc_weights_reco, alpha=0.6, label='Nominal MC')
    axes[0].hist(true_ke, bins=bins_ke, weights=data_weights,    alpha=0.6, label='Fake Data (tilt)')
    axes[0].set_xlabel('true_ke [MeV]')
    axes[0].set_ylabel('Weighted events')
    axes[0].legend()
    axes[0].set_title('Injected distortion: true electron KE')

    axes[1].scatter(true_ke[::3], tilt[::3], s=3, alpha=0.4)
    axes[1].axhline(1.0, color='r', linestyle='--')
    axes[1].set_xlabel('true_ke [MeV]')
    axes[1].set_ylabel('Tilt weight')
    axes[1].set_title(f'Tilt function (alpha={ALPHA})')

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/sbnd_fakedata_injected_{tag}.png', dpi=150)
    print(f"  Plot: {PLOT_DIR}/sbnd_fakedata_injected_{tag}.png")


# ── Mode: BNB universe ───────────────────────────────────────────────────────
elif flags.mode == 'universe':
    print(f"=== BNB Universe Mode ===")

    if flags.universe_file is None:
        print("ERROR: --universe-file required for universe mode.")
        print("")
        print("To generate it, add this cell to your systematics notebook")
        print("(nuecc_systematics.ipynb) AFTER cell 3 (wgt_aligned is built):")
        print("")
        print("  # ── Save aligned BNB universe weights for OmniFold ──────────")
        print("  import numpy as np")
        print("  # Filter to same events as FormatData_SBND.py:")
        print("  #   is_sig & sel_vertex_distance & valid reco features")
        print("  sig_sel = sel_topo[sel_topo['is_sig'] & sel_topo['sel_vertex_distance']].copy()")
        print("  reco_raw = sig_sel[['reco_ke','reco_costheta','reco_p']].values")
        print("  valid = ~np.isnan(reco_raw).any(axis=1)")
        print("  sig_sel = sig_sel[valid]")
        print("  # Align weights to these events")
        print("  nwl = wgt_aligned.index.nlevels")
        print("  sidx = (sig_sel.index.droplevel(list(range(nwl, sig_sel.index.nlevels)))")
        print("          if sig_sel.index.nlevels > nwl else sig_sel.index)")
        print("  bnb_vals = wgt_aligned[bnb_cols].reindex(sidx).values")
        print("  bnb_vals = np.where(np.isfinite(bnb_vals), bnb_vals, 1.0)")
        print("  np.save('bnb_universe_weights_selected_signal.npy', bnb_vals)")
        print(f"  print(f'Saved: {{bnb_vals.shape}}')")
        print("")
        print("Then run:")
        print("  python3 MakeFakeDataWeights_SBND.py --mode universe \\")
        print("      --universe-file /path/to/bnb_universe_weights_selected_signal.npy")
        exit(1)

    uni_all = np.load(flags.universe_file)   # shape: (N_events, N_universes)
    if uni_all.ndim == 1:
        uni_all = uni_all[:, np.newaxis]

    assert uni_all.shape[0] == n_events, \
        f"Shape mismatch: universe file has {uni_all.shape[0]} events, " \
        f"expected {n_events}. Re-export from notebook with matching filter."

    idx = flags.universe_idx
    uni_vals = np.clip(uni_all[:, idx], 0.0, 10.0)

    data_weights = mc_weights_reco * uni_vals
    data_weights = data_weights * (mc_weights_reco.sum() / data_weights.sum())

    tag = f'bnb_univ{idx}'
    out_data = OUTPUT_DIR + f'data_weights_sbnd_fakedata_{tag}.npy'
    out_uni  = OUTPUT_DIR + f'truth_weights_sbnd_fakedata_{tag}.npy'
    np.save(out_data, data_weights)
    np.save(out_uni,  uni_vals)

    print(f"  Universe {idx}: mean={uni_vals.mean():.4f}, std={uni_vals.std():.4f}")
    print(f"  Data weights: mean={data_weights.mean():.4f}, std={data_weights.std():.4f}")
    print(f"  Saved: {out_data}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins_ke = np.linspace(0, true_ke.max() * 1.05, 25)
    axes[0].hist(true_ke, bins=bins_ke, weights=mc_weights_reco, alpha=0.6, label='Nominal MC')
    axes[0].hist(true_ke, bins=bins_ke, weights=data_weights,    alpha=0.6, label=f'Fake Data (BNB univ {idx})')
    axes[0].set_xlabel('true_ke [MeV]')
    axes[0].set_ylabel('Weighted events')
    axes[0].legend()
    axes[0].set_title('Injected distortion: true electron KE')

    axes[1].scatter(true_ke[::3], uni_vals[::3], s=3, alpha=0.4)
    axes[1].axhline(1.0, color='r', linestyle='--')
    axes[1].set_xlabel('true_ke [MeV]')
    axes[1].set_ylabel('Universe weight')
    axes[1].set_title(f'BNB universe {idx} weights')

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/sbnd_fakedata_injected_{tag}.png', dpi=150)
    print(f"  Plot: {PLOT_DIR}/sbnd_fakedata_injected_{tag}.png")