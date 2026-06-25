"""
ExtractCrossSection.py

Converts OmniFold push weights into differential cross-sections
with systematic uncertainty bands.

Produces the "unfolded xsec test on MC" plots for presentation.

Usage:
    python3 sbnd/ExtractCrossSection.py --var true_ke
    python3 sbnd/ExtractCrossSection.py --var true_costheta
    python3 sbnd/ExtractCrossSection.py --var both
"""

import numpy as np
import glob, re, os, argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--var', choices=['true_ke', 'true_costheta', 'both'], default='both')
parser.add_argument('--data-dir', type=str, default='../FormattedData_SBND/')
parser.add_argument('--weights-base', type=str, default='sbnd')
parser.add_argument('--export-dir', type=str, default='sbnd/exported_weights/')
parser.add_argument('--plot-dir', type=str, default='sbnd/plots_xsec/')
flags = parser.parse_args()

os.makedirs(flags.plot_dir, exist_ok=True)

# ── Binning and labels ────────────────────────────────────────────────────────
BINNING = {
    'true_ke':       np.array([0, 200, 400, 600, 800, 1000, 1400, 2000]),
    'true_costheta': np.linspace(-1, 1, 11),
}
XLABEL = {
    'true_ke':       r'True electron KE [MeV]',
    'true_costheta': r'True $\cos\theta_e$',
}
YLABEL = {
    'true_ke':       r'd$\sigma$/dKE [cm$^2$/MeV/nucleon]',
    'true_costheta': r'd$\sigma$/d$\cos\theta$ [cm$^2$/nucleon]',
}

# ── Normalization constants ───────────────────────────────────────────────────
# POT
pot_scale_arr = np.load(flags.export_dir + 'pot_scale.npy')
pot_scale = float(pot_scale_arr[0])

# Integrated flux: nue per cm2 per POT at SBND
# PLACEHOLDER — replace with your actual integrated nue flux value
# from the BNB flux prediction at the SBND location
# Typical value: ~few × 10^-10 nue/cm2/POT for BNB at 110m
INTEGRATED_FLUX_PER_POT = 1.0   # SET THIS from your flux files
TARGET_POT = 6.6e20             # target POT for normalization

# N_targets: number of argon nucleons in fiducial volume
# You computed this in your analysis: LAr density × fiducial volume × N_A / A_Ar
# PLACEHOLDER — replace with your actual value
N_TARGETS = 1.0                 # SET THIS from your fiducial volume calculation

# If normalization constants aren't set, produce shape-only xsec
USE_ABSOLUTE = (INTEGRATED_FLUX_PER_POT != 1.0 and N_TARGETS != 1.0)
if not USE_ABSOLUTE:
    print("WARNING: INTEGRATED_FLUX_PER_POT and/or N_TARGETS not set.")
    print("         Producing shape-only cross-section (arbitrary units).")
    print("         Update these constants for absolute normalization.\n")

FLUX = INTEGRATED_FLUX_PER_POT * TARGET_POT
NORM = FLUX * N_TARGETS if USE_ABSOLUTE else 1.0

# ── Load data ─────────────────────────────────────────────────────────────────
truth_raw  = np.load(flags.data_dir + 'mc_vals_truth_NoNorm.npy')
mc_weights = np.load(flags.data_dir + 'mc_weights_reco.npy')

# ── Helper ────────────────────────────────────────────────────────────────────
def iter_num(p):
    m = re.search(r'Iter(\d+)', p)
    return int(m.group(1)) if m else -1


def extract_xsec(var_name):
    bins      = BINNING[var_name]
    xlabel    = XLABEL[var_name]
    ylabel    = YLABEL[var_name]
    n_bins    = len(bins) - 1
    bin_widths = np.diff(bins)
    centers   = 0.5 * (bins[:-1] + bins[1:])

    var_idx  = 0 if var_name == 'true_ke' else 1
    var_vals = truth_raw[:, var_idx]

    # Load efficiency
    eff_file = flags.export_dir + f'efficiency_{var_name}.npy'
    if not os.path.exists(eff_file):
        print(f"ERROR: {eff_file} not found. Run the notebook export cell first.")
        return
    eff = np.load(eff_file)
    print(f"\n{'='*60}")
    print(f"Cross-section extraction: {var_name}")
    print(f"{'='*60}")
    print(f"Efficiency: {[f'{e:.3f}' for e in eff]}")

    # ── Nominal unfolded xsec (push weights = 1) ─────────────────────────────
    N_nom, _ = np.histogram(var_vals, bins=bins, weights=mc_weights)
    xsec_nom = N_nom / (eff.clip(1e-6) * bin_widths * NORM)

    # ── Load covariance from systematic propagation ───────────────────────────
    cov_file = f'{flags.weights_base}/plots_systematics/covariance_all_{var_name}.npz'
    if os.path.exists(cov_file):
        cov_data  = np.load(cov_file)
        cov_events = cov_data['cov']       # covariance on event counts
        mean_events = cov_data['mean_hist']

        # Propagate covariance through efficiency and bin width division
        # xsec_i = N_i / (eff_i * dX_i * NORM)
        # Var(xsec_i, xsec_j) = Cov(N_i, N_j) / (eff_i*dX_i*NORM * eff_j*dX_j*NORM)
        scale = eff.clip(1e-6) * bin_widths * NORM
        cov_xsec = cov_events / np.outer(scale, scale)
        xsec_unc = np.sqrt(np.diag(cov_xsec))

        xsec_mean = mean_events / (eff.clip(1e-6) * bin_widths * NORM)
        has_syst = True
    else:
        print(f"  Covariance file not found: {cov_file}")
        print(f"  Plotting without systematic uncertainty.")
        xsec_unc  = np.zeros(n_bins)
        xsec_mean = xsec_nom
        has_syst  = False

    frac_unc = xsec_unc / xsec_mean.clip(1e-30)

    # ── Print table ───────────────────────────────────────────────────────────
    unit_label = 'cm2/MeV/nucleon' if USE_ABSOLUTE else 'arb. units'
    print(f"\n  {'Bin':>12s} {'xsec_nom':>12s} {'xsec_mean':>12s} "
          f"{'abs unc':>12s} {'frac unc':>10s}  [{unit_label}]")
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        print(f"  [{lo:5.0f},{hi:5.0f}] {xsec_nom[i]:12.4e} {xsec_mean[i]:12.4e} "
              f"{xsec_unc[i]:12.4e} {frac_unc[i]:10.4f}")

    # ── Plot 1: Differential cross-section with systematic band ───────────────
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})

    # Top: xsec
    axes[0].step(bins, np.append(xsec_nom, xsec_nom[-1]),
                 where='post', color='blue', linewidth=2, label='Nominal MC (truth)')

    if has_syst:
        axes[0].step(bins, np.append(xsec_mean, xsec_mean[-1]),
                     where='post', color='red', linewidth=2,
                     label='Mean (300 universes)')
        upper = xsec_mean + xsec_unc
        lower = xsec_mean - xsec_unc
        for i in range(n_bins):
            axes[0].fill_between([bins[i], bins[i+1]], lower[i], upper[i],
                                  color='red', alpha=0.2,
                                  label=(r'$\pm 1\sigma$ syst' if i == 0 else None))

    axes[0].set_ylabel(ylabel if USE_ABSOLUTE else r'd$\sigma$/dX [arb. units]')
    axes[0].legend(fontsize=9)
    axes[0].set_title(f'SBND $\\nu_e$ CC Inclusive: {var_name}')
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    # Bottom: fractional uncertainty
    if has_syst:
        axes[1].step(bins, np.append(frac_unc, frac_unc[-1]),
                     where='post', color='red', linewidth=1.5)
        for i in range(n_bins):
            axes[1].fill_between([bins[i], bins[i+1]], 0, frac_unc[i],
                                  color='red', alpha=0.2)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('Fractional uncertainty')
    axes[1].set_ylim(0, max(frac_unc.max() * 1.5, 0.05) if has_syst else 0.05)

    plt.tight_layout()
    outname = f'{flags.plot_dir}/xsec_{var_name}.png'
    plt.savefig(outname, dpi=150)
    print(f"  Saved {outname}")

    # ── Plot 2: Efficiency ────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.step(bins, np.append(eff, eff[-1]), where='post', color='black', linewidth=2)
    for i in range(n_bins):
        ax2.fill_between([bins[i], bins[i+1]], 0, eff[i], color='gray', alpha=0.2)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Selection efficiency')
    ax2.set_ylim(0, 1)
    ax2.set_title(f'Efficiency: {var_name}')
    plt.tight_layout()
    outname2 = f'{flags.plot_dir}/efficiency_{var_name}.png'
    plt.savefig(outname2, dpi=150)
    print(f"  Saved {outname2}")

    return xsec_nom, xsec_mean, xsec_unc, eff


# ── Run ───────────────────────────────────────────────────────────────────────
vars_to_run = ['true_ke', 'true_costheta'] if flags.var == 'both' else [flags.var]
for v in vars_to_run:
    extract_xsec(v)

print(f"\nAll plots saved to {flags.plot_dir}/")
if not USE_ABSOLUTE:
    print("\nNOTE: Cross-sections are in arbitrary units.")
    print("To get absolute normalization, set INTEGRATED_FLUX_PER_POT")
    print("and N_TARGETS in this script from your analysis framework.")