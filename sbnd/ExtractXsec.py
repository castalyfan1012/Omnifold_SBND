"""
ExtractCrossSection.py

Converts OmniFold push weights into differential cross-sections
with systematic uncertainty bands.

Primary variables: true_p (col 0), true_costheta (col 1)
  — must match FormatData_SBND.py

Usage:
    python3 sbnd/ExtractCrossSection.py --var true_p
    python3 sbnd/ExtractCrossSection.py --var true_costheta
    python3 sbnd/ExtractCrossSection.py --var both
"""

import numpy as np
import glob, re, os, argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--var', choices=['true_p', 'true_costheta', 'both'], default='both')
parser.add_argument('--data-dir',     type=str, default='../FormattedData_SBND/')
parser.add_argument('--weights-base', type=str, default='sbnd')
parser.add_argument('--export-dir',   type=str, default='sbnd/exported_weights/')
parser.add_argument('--plot-dir',     type=str, default='sbnd/plots_xsec/')
parser.add_argument('--tag',          type=str, default='tilt_alpha0.5',
                    help='Fake data tag used to find push weights')
flags = parser.parse_args()

os.makedirs(flags.plot_dir, exist_ok=True)

# ── Binning ───────────────────────────────────────────────────────────────────
BINNING = {
    'true_p':        np.array([0, 200, 400, 600, 800, 1000, 1400, 2000]),
    'true_costheta': np.linspace(-1, 1, 11),
}
XLABEL = {
    'true_p':        r'True electron momentum [MeV/c]',
    'true_costheta': r'True $\cos\theta_e$',
}
YLABEL = {
    'true_p':        r'd$\sigma$/d$p$ [arb. units / (MeV/c)]',
    'true_costheta': r'd$\sigma$/d$\cos\theta$ [arb. units]',
}

# ── Normalization ─────────────────────────────────────────────────────────────
# Set INTEGRATED_FLUX_PER_POT and N_TARGETS for absolute normalization.
# Until then, the cross-section is in arbitrary (shape-only) units.
INTEGRATED_FLUX_PER_POT = 1.0   # nue / cm2 / POT — get from your BNB flux files
TARGET_POT              = 6.6e20
N_TARGETS               = 1.0   # argon nucleons in fiducial volume

USE_ABSOLUTE = (INTEGRATED_FLUX_PER_POT != 1.0 and N_TARGETS != 1.0)
NORM = INTEGRATED_FLUX_PER_POT * TARGET_POT * N_TARGETS if USE_ABSOLUTE else 1.0

if not USE_ABSOLUTE:
    print("NOTE: INTEGRATED_FLUX_PER_POT / N_TARGETS not set.")
    print("      Producing shape-only (arbitrary units) cross-section.\n")

# ── Load ──────────────────────────────────────────────────────────────────────
truth_raw  = np.load(flags.data_dir + 'mc_vals_truth_NoNorm.npy')
mc_weights = np.load(flags.data_dir + 'mc_weights_reco.npy')
# Columns: [0]=true_p, [1]=true_costheta  (set by FormatData_SBND.py)

def iter_num(p):
    m = re.search(r'Iter(\d+)', p)
    return int(m.group(1)) if m else -1


def extract_xsec(var_name):
    bins       = BINNING[var_name]
    xlabel     = XLABEL[var_name]
    ylabel     = YLABEL[var_name]
    n_bins     = len(bins) - 1
    bin_widths = np.diff(bins)
    centers    = 0.5 * (bins[:-1] + bins[1:])
    var_idx    = 0 if var_name == 'true_p' else 1
    var_vals   = truth_raw[:, var_idx]

    print(f"\n{'='*60}")
    print(f"Cross-section extraction: {var_name}")
    print(f"{'='*60}")

    # ── Efficiency ────────────────────────────────────────────────────────────
    eff_file = flags.export_dir + f'efficiency_{var_name}.npy'
    if not os.path.exists(eff_file):
        print(f"  WARNING: {eff_file} not found — using flat efficiency=1")
        eff = np.ones(n_bins)
    else:
        eff = np.load(eff_file)
    print(f"  Efficiency: {[f'{e:.3f}' for e in eff]}")

    # ── Nominal histogram (push weights = 1, no unfolding) ───────────────────
    N_nom, _ = np.histogram(var_vals, bins=bins, weights=mc_weights)
    xsec_nom = N_nom / (eff.clip(1e-6) * bin_widths * NORM)

    # ── Load OmniFold push weights (final iteration) ──────────────────────────
    tilt_dir   = f'weights_sbnd_fakedata_{flags.tag}/'
    push_files = sorted(glob.glob(tilt_dir + 'Step2_Iter*_PushWeights.npy'), key=iter_num)
    if push_files:
        push = np.load(push_files[-1])
        push = push if push.ndim == 1 else push.mean(axis=0)
        N_unf, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * push)
        xsec_unf  = N_unf / (eff.clip(1e-6) * bin_widths * NORM)
        has_unf   = True
        print(f"  Push file: {push_files[-1]}")
    else:
        print(f"  No push files in {tilt_dir} — showing nominal only")
        xsec_unf = xsec_nom
        has_unf  = False

    # ── Load injected truth (fake data answer key) ────────────────────────────
    tilt_file = flags.data_dir + f'truth_weights_sbnd_fakedata_{flags.tag}.npy'
    if os.path.exists(tilt_file):
        tilt = np.load(tilt_file)
        N_truth, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * tilt)
        xsec_truth  = N_truth / (eff.clip(1e-6) * bin_widths * NORM)
        has_truth   = True
    else:
        has_truth = False

    # ── Load systematic covariance ────────────────────────────────────────────
    cov_file = f'{flags.weights_base}/plots_systematics/covariance_all_{var_name}.npz'
    if os.path.exists(cov_file):
        cov_data   = np.load(cov_file)
        cov_events = cov_data['cov']
        mean_events = cov_data['mean_hist']
        scale      = eff.clip(1e-6) * bin_widths * NORM
        cov_xsec   = cov_events / np.outer(scale, scale)
        xsec_unc   = np.sqrt(np.diag(cov_xsec))
        xsec_mean  = mean_events / (eff.clip(1e-6) * bin_widths * NORM)
        has_syst   = True
    else:
        print(f"  No covariance file at {cov_file} — no systematic band")
        xsec_unc  = np.zeros(n_bins)
        xsec_mean = xsec_nom
        has_syst  = False

    frac_unc = xsec_unc / xsec_mean.clip(1e-30)

    # ── Print table ───────────────────────────────────────────────────────────
    unit = 'cm2/(MeV/c)/nucleon' if (USE_ABSOLUTE and var_name=='true_p') else \
           'cm2/nucleon' if USE_ABSOLUTE else 'arb. units'
    print(f"\n  {'Bin':>14s} {'Nominal':>12s} {'OmniFold':>12s} "
          f"{'Syst unc':>12s} {'Frac unc':>10s}  [{unit}]")
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        unf_val = xsec_unf[i] if has_unf else float('nan')
        print(f"  [{lo:6.0f},{hi:6.0f}] {xsec_nom[i]:12.4e} {unf_val:12.4e} "
              f"{xsec_unc[i]:12.4e} {frac_unc[i]:10.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # Plot 1: Differential cross-section with systematic band + OmniFold result
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})

    # Nominal MC
    axes[0].step(bins, np.append(xsec_nom, xsec_nom[-1]),
                 where='post', color='gray', linewidth=1.5,
                 linestyle='--', label='Nominal MC (prior)')

    # Data Truth (injected)
    if has_truth:
        axes[0].step(bins, np.append(xsec_truth, xsec_truth[-1]),
                     where='post', color='black', linewidth=2, label='Data Truth (injected)')

    # OmniFold unfolded with systematic band
    if has_unf:
        axes[0].errorbar(centers, xsec_unf, yerr=xsec_unc if has_syst else None,
                         fmt='ro', markersize=5, capsize=3, linewidth=1.5,
                         label='OmniFold unfolded')

    axes[0].set_ylabel(ylabel if USE_ABSOLUTE else ylabel.replace('cm$^2$', 'arb.').replace('/nucleon', ''))
    axes[0].legend(fontsize=10)
    axes[0].set_title(rf'SBND $\nu_e$ CC Inclusive: {var_name}')
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    # Fractional systematic uncertainty
    if has_syst:
        axes[1].step(bins, np.append(frac_unc, frac_unc[-1]),
                     where='post', color='red', linewidth=1.5)
        for i in range(n_bins):
            axes[1].fill_between([bins[i], bins[i+1]], 0, frac_unc[i],
                                  color='red', alpha=0.2)
    else:
        axes[1].axhline(0, color='gray', linestyle='--')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('Frac. syst. unc.')
    axes[1].set_ylim(0, max(frac_unc.max() * 1.5, 0.05) if has_syst else 0.1)

    plt.tight_layout()
    outname = f'{flags.plot_dir}/xsec_{var_name}.png'
    plt.savefig(outname, dpi=150)
    print(f"  Saved {outname}")
    plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # Plot 2: Efficiency per bin
    # ══════════════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.step(bins, np.append(eff, eff[-1]), where='post', color='black', linewidth=2)
    for i in range(n_bins):
        ax2.fill_between([bins[i], bins[i+1]], 0, eff[i], color='steelblue', alpha=0.3)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Selection efficiency')
    ax2.set_ylim(0, 1)
    ax2.set_title(f'Selection efficiency: {var_name}')
    plt.tight_layout()
    outname2 = f'{flags.plot_dir}/efficiency_{var_name}.png'
    plt.savefig(outname2, dpi=150)
    print(f"  Saved {outname2}")
    plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # Plot 3: Ratio of OmniFold / Data Truth (recovery check)
    # ══════════════════════════════════════════════════════════════════════════
    if has_unf and has_truth:
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ratio     = xsec_unf / np.where(xsec_truth > 0, xsec_truth, 1)
        ratio_nom = xsec_nom  / np.where(xsec_truth > 0, xsec_truth, 1)
        ratio_unc = xsec_unc  / np.where(xsec_truth > 0, xsec_truth, 1)

        # Prior ratio band
        for i in range(n_bins):
            ax3.fill_between([bins[i], bins[i+1]],
                              ratio_nom[i] - 0.02, ratio_nom[i] + 0.02,
                              color='gray', alpha=0.3,
                              label=('Prior' if i == 0 else None))
        ax3.step(bins, np.append(ratio_nom, ratio_nom[-1]),
                 where='post', color='gray', linewidth=1.5, linestyle='--')

        # OmniFold ratio
        ax3.errorbar(centers, ratio, yerr=ratio_unc,
                     fmt='ro', markersize=5, capsize=3, linewidth=1.5,
                     label='OmniFold')
        ax3.axhline(1.0, color='black', linewidth=1)
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel('Ratio to Data Truth')
        ax3.set_title(f'Recovery check: {var_name}')
        ax3.legend()
        ax3.set_xlim(bins[0], bins[-1])
        plt.tight_layout()
        outname3 = f'{flags.plot_dir}/ratio_to_truth_{var_name}.png'
        plt.savefig(outname3, dpi=150)
        print(f"  Saved {outname3}")
        plt.close()

    return xsec_nom, xsec_mean, xsec_unc, eff


# ── Run ───────────────────────────────────────────────────────────────────────
vars_to_run = ['true_p', 'true_costheta'] if flags.var == 'both' else [flags.var]
for v in vars_to_run:
    extract_xsec(v)

print(f"\nAll plots saved to {flags.plot_dir}/")
if not USE_ABSOLUTE:
    print("NOTE: Set INTEGRATED_FLUX_PER_POT and N_TARGETS for absolute normalization.")