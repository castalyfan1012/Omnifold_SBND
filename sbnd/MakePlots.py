"""
MakePlots.py — All SBND OmniFold plots in one script.

Actions:
  validation   Fake-data recovery, chi2 convergence, unfolded distributions
               -> sbnd/plots_validation/  (all filenames include the tag)
  paper        Publication-style plots: xsec, ratio, chi2, unc budget,
               correlation, 2D slices, reweighting snapshots, etc.
               -> sbnd/plots_xsec/  (tag-dependent plots include the tag,
                  tag-independent plots like uncertainty_budget do not)
  all          Both validation and paper

All fake-data-dependent plots embed the tag in the filename so different
tilts never overwrite each other.

Usage:
    python3 sbnd/MakePlots.py validation --tag tilt_alpha0.5
    python3 sbnd/MakePlots.py validation --tag tilt_alpha0.3
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
p_val.add_argument('--plot-dir', default='sbnd/plots_validation/')

p_pap = sub.add_parser('paper')
p_pap.add_argument('--var', choices=['true_p', 'true_costheta', 'both'], default='both')
p_pap.add_argument('--tag', default='tilt_alpha0.5')
p_pap.add_argument('--data-dir', default='../FormattedData_SBND/')
p_pap.add_argument('--weights-base', default='sbnd')
p_pap.add_argument('--export-dir', default='sbnd/exported_weights/')
p_pap.add_argument('--plot-dir', default='sbnd/plots_xsec/')
p_pap.add_argument('--cov-dir', default='sbnd/covariance/')

p_all = sub.add_parser('all')
p_all.add_argument('--var', choices=['true_p', 'true_costheta', 'both'], default='both')
p_all.add_argument('--tag', default='tilt_alpha0.5')
p_all.add_argument('--data-dir', default='../FormattedData_SBND/')
p_all.add_argument('--weights-base', default='sbnd')
p_all.add_argument('--export-dir', default='sbnd/exported_weights/')
p_all.add_argument('--plot-dir', default='sbnd/plots_xsec/')
p_all.add_argument('--cov-dir', default='sbnd/covariance/')
p_all.add_argument('--val-plot-dir', default='sbnd/plots_validation/')

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

def chi2_simple(obs, exp):
    mask = exp > 0
    return np.sum((obs[mask] - exp[mask])**2 / exp[mask])


def do_validation():
    TAG = flags.tag
    DATA_DIR    = flags.data_dir
    WEIGHTS_DIR = flags.weights_dir or f'weights_sbnd_fakedata_{TAG}/'
    PLOT_DIR    = getattr(flags, 'val_plot_dir', None) or flags.plot_dir
    os.makedirs(PLOT_DIR, exist_ok=True)

    truth_raw  = np.load(DATA_DIR + 'mc_vals_truth_NoNorm.npy')
    mc_weights = np.load(DATA_DIR + 'mc_weights_reco.npy')
    tilt_path  = DATA_DIR + f'truth_weights_sbnd_fakedata_{TAG}.npy'
    if not os.path.exists(tilt_path):
        print(f"ERROR: Truth weights not found: {tilt_path}")
        print(f"  Run: python3 sbnd/RunStudies.py make-fakedata --mode tilt --alpha <value>")
        return
    injected_tilt = np.load(tilt_path)
    true_p, true_costheta = truth_raw[:, 0], truth_raw[:, 1]

    push_files = sorted(glob.glob(WEIGHTS_DIR + 'Step2_Iter*_PushWeights.npy'), key=iter_num)
    if not push_files:
        print(f"ERROR: No push files found in '{WEIGHTS_DIR}'")
        print(f"  OmniFold has not been trained for tag '{TAG}' yet.")
        print(f"  To fix:")
        print(f"    1. python3 sbnd/RunStudies.py make-fakedata --mode tilt --alpha <value>")
        print(f"    2. bash sbnd/runOmnifold_sbnd_fakedata.sh {TAG}")
        print(f"    3. python3 sbnd/MakePlots.py validation --tag {TAG}")
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

    var_data = {'true_p': true_p, 'true_costheta': true_costheta}

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
        axes[1].set_ylim(0.5, 1.5); plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/fakedata_{TAG}_recovery_{var_name}.png', dpi=150); plt.close()
        print(f"  fakedata_{TAG}_recovery_{var_name}.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (vn, vv) in zip(axes, var_data.items()):
        bins = BINNING[vn]; centers = 0.5 * (bins[:-1] + bins[1:])
        nom_h, _ = np.histogram(vv, bins=bins, weights=mc_weights)
        fd_h, _  = np.histogram(vv, bins=bins, weights=mc_weights * injected_tilt)
        unf_h, _ = np.histogram(vv, bins=bins, weights=mc_weights * push_mean)
        ax.step(bins, np.append(nom_h, nom_h[-1]), where='post', color='gray',
                linewidth=1.5, linestyle='--', label='Nominal MC')
        ax.step(bins, np.append(fd_h, fd_h[-1]), where='post', color='black',
                linewidth=2, label='Fake Data')
        ax.errorbar(centers, unf_h, yerr=np.sqrt(np.maximum(unf_h, 0)),
                    fmt='ro', markersize=5, capsize=3, linewidth=1.5, label='OmniFold')
        ax.set_xlabel(XLABEL[vn]); ax.set_ylabel('Weighted events')
        ax.legend(fontsize=9); ax.set_title(f'Unfolded: {vn} ({TAG})')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/fakedata_{TAG}_unfolded_distributions.png', dpi=150); plt.close()
    print(f"  fakedata_{TAG}_unfolded_distributions.png")

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
        print(f"  {vn} (ndf={ndf}): prior={pc[0]:.4f}, final={pc[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    all_vals = []
    for vn, (pi, pc) in curves.items():
        ax.plot(pi, pc, 'o-', color=colors_var[vn], linewidth=2, markersize=5, label=labels_var[vn])
        all_vals.extend(pc)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, label=r'$\chi^2$/DoF = 1')
    ax.set_xlabel('OmniFold Iteration'); ax.set_ylabel(r'$\chi^2$/DoF')
    ax.set_title(rf'Fake-data recovery: {TAG}'); ax.legend(fontsize=12); ax.set_yscale('log')
    ax.set_ylim(min(v for v in all_vals if v > 0) * 0.5, max(all_vals) * 3); ax.set_xticks(pi)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/fakedata_{TAG}_chi2_convergence.png', dpi=150); plt.close()
    print(f"  fakedata_{TAG}_chi2_convergence.png")

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
        ax.set_title(f'{vn}: total $\\chi^2$={chi2_full:.2f}, ndf={ndf_full}'); ax.legend()
        print(f"\n  {vn} (total={chi2_full:.2f}, chi2/ndf={chi2_full/ndf_full:.2f}):")
        for i in range(n_bins):
            frac = per_bin[i] / chi2_full if chi2_full > 0 else 0
            chi2_wo = chi2_simple(np.delete(unf_h, i), np.delete(truth_h, i))
            print(f"    [{bins[i]:6.0f},{bins[i+1]:6.0f}]  chi2={per_bin[i]:8.2f}  "
                  f"w/o={chi2_wo:8.2f}  frac={frac:.1%}")
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/fakedata_{TAG}_chi2_bin_diagnostic.png', dpi=150); plt.close()
    print(f"  fakedata_{TAG}_chi2_bin_diagnostic.png")

    print(f"\n  Push weight stats per iteration:")
    for f in push_files:
        w = np.load(f); w = w if w.ndim == 1 else w.mean(axis=0)
        print(f"    Iter {iter_num(f)+1:2d}: mean={w.mean():.4f}, std={w.std():.4f}")


def do_paper():
    TAG = flags.tag
    os.makedirs(flags.plot_dir, exist_ok=True)
    truth_raw  = np.load(flags.data_dir + 'mc_vals_truth_NoNorm.npy')
    mc_weights = np.load(flags.data_dir + 'mc_weights_reco.npy')
    tilt_file  = flags.data_dir + f'truth_weights_sbnd_fakedata_{TAG}.npy'
    if not os.path.exists(tilt_file):
        print(f"ERROR: {tilt_file} not found"); return
    injected = np.load(tilt_file)
    tilt_dir = f'weights_sbnd_fakedata_{TAG}/'
    push_files = sorted(glob.glob(tilt_dir + 'Step2_Iter*_PushWeights.npy'), key=iter_num)
    if not push_files:
        print(f"ERROR: No push files in '{tilt_dir}'")
        print(f"  Train first: bash sbnd/runOmnifold_sbnd_fakedata.sh {TAG}"); return
    push_final = np.load(push_files[-1])
    push_final = push_final if push_final.ndim == 1 else push_final.mean(axis=0)
    vars_to_run = (['true_p', 'true_costheta'] if flags.var == 'both' else [flags.var])
    for vn in vars_to_run:
        _make_core_plots(vn, TAG, truth_raw, mc_weights, injected, push_files, push_final)
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
    eff_file = flags.export_dir + f'efficiency_{var_name}.npy'
    eff = np.load(eff_file) if os.path.exists(eff_file) else np.ones(n_bins)
    truth_hist, _ = np.histogram(var_vals, bins=bins, weights=mc_weights * injected)
    nom_hist, _   = np.histogram(var_vals, bins=bins, weights=mc_weights)
    unf_hist, _   = np.histogram(var_vals, bins=bins, weights=mc_weights * push_final)
    truth_xsec = truth_hist / (eff.clip(1e-6) * bin_widths)
    nom_xsec   = nom_hist   / (eff.clip(1e-6) * bin_widths)
    unf_xsec   = unf_hist   / (eff.clip(1e-6) * bin_widths)
    cov_file = f'{flags.cov_dir}/covariance_all_{var_name}.npz'
    if os.path.exists(cov_file):
        cd = np.load(cov_file); scale = eff.clip(1e-6) * bin_widths
        cov_xsec = cd['cov'] / np.outer(scale, scale)
        xsec_unc = np.sqrt(np.diag(cov_xsec))
    else:
        print(f"  WARNING: {cov_file} not found"); xsec_unc = np.zeros(n_bins); cov_xsec = None
    print(f"\n  Paper plots: {var_name} (tag={TAG})")

    # xsec vs truth [TAG-DEPENDENT]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.step(bins, np.append(truth_xsec, truth_xsec[-1]), where='post', color='black', linewidth=2, label='Data Truth')
    ax.step(bins, np.append(nom_xsec, nom_xsec[-1]), where='post', color='gray', linewidth=1.5, linestyle='--')
    for i in range(n_bins):
        ax.fill_between([bins[i], bins[i+1]], nom_xsec[i]*0.95, nom_xsec[i]*1.05,
                         color='gray', alpha=0.2, label=('Prior' if i == 0 else None))
    ax.errorbar(centers, unf_xsec, yerr=xsec_unc, fmt='o', color='red', markersize=5, capsize=3, linewidth=1.5, label='OmniFold')
    ax.set_xlabel(xlabel); ax.set_ylabel(YLABEL_XSEC[var_name])
    ax.set_title(rf'SBND $\nu_e$ CC Inclusive ({TAG})'); ax.legend(); ax.set_xlim(bins[0], bins[-1])
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2)); plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/xsec_vs_truth_{TAG}_{var_name}.png', dpi=150)
    print(f"    xsec_vs_truth_{TAG}_{var_name}.png"); plt.close()

    # ratio [TAG-DEPENDENT]
    fig, ax = plt.subplots(figsize=(8, 6))
    r_unf = unf_xsec / np.where(truth_xsec > 0, truth_xsec, 1)
    r_unc = xsec_unc / np.where(truth_xsec > 0, truth_xsec, 1)
    r_pr  = nom_xsec / np.where(truth_xsec > 0, truth_xsec, 1)
    for i in range(n_bins):
        ax.fill_between([bins[i], bins[i+1]], r_pr[i]-0.02, r_pr[i]+0.02,
                         color='gray', alpha=0.3, label=('Prior' if i == 0 else None))
    ax.errorbar(centers, r_unf, yerr=r_unc, fmt='o', color='red', markersize=5, capsize=3, linewidth=1.5, label='OmniFold')
    ax.axhline(1.0, color='black', linewidth=1); ax.set_xlabel(xlabel); ax.set_ylabel('Ratio to Data Truth')
    ax.set_title(rf'Ratio ({TAG})'); ax.legend(); ax.set_xlim(bins[0], bins[-1]); plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/ratio_to_truth_{TAG}_{var_name}.png', dpi=150)
    print(f"    ratio_to_truth_{TAG}_{var_name}.png"); plt.close()

    if cov_xsec is not None:
        d = unf_xsec - truth_xsec
        try: c2 = float(d @ np.linalg.inv(cov_xsec) @ d)
        except np.linalg.LinAlgError: c2 = float(np.sum(d**2 / np.diag(cov_xsec).clip(1e-30)))
        print(f"    Xsec chi2 ({var_name}): {c2:.2f} / {n_bins} = {c2/n_bins:.2f}")

    # uncertainty budget [TAG-INDEPENDENT]
    fig, ax = plt.subplots(figsize=(8, 6))
    csrc = {'bnb':'blue','genie':'red','mcstat':'green','ml':'purple'}
    lsrc = {'bnb':'BNB Flux','genie':'GENIE XSec','mcstat':'MC Stat','ml':'ML/NN Init'}
    for src in ['bnb','genie','mcstat','ml']:
        cf = f'{flags.cov_dir}/covariance_{src}_{var_name}.npz'
        if not os.path.exists(cf): continue
        d = np.load(cf); frac = np.sqrt(np.diag(d['cov'])) / d['mean_hist'].clip(1e-6)
        n_u = int(d['n_universes']) if 'n_universes' in d else '?'
        ax.step(bins, np.append(frac, frac[-1]), where='post', color=csrc[src], linewidth=1.5, label=f'{lsrc[src]} ({n_u})')
    if os.path.exists(cov_file):
        cd = np.load(cov_file); fa = np.sqrt(np.diag(cd['cov'])) / cd['mean_hist'].clip(1e-6)
        ax.step(bins, np.append(fa, fa[-1]), where='post', color='black', linewidth=2, label='Total')
    ax.set_xlabel(xlabel); ax.set_ylabel('Bin Fractional Uncertainty')
    ax.set_title(r'Uncertainty budget: SBND $\nu_e$ CC'); ax.legend(); ax.set_xlim(bins[0], bins[-1])
    ax.set_yscale('log'); ax.set_ylim(1e-3, 0.5); plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/uncertainty_budget_{var_name}.png', dpi=150)
    print(f"    uncertainty_budget_{var_name}.png"); plt.close()

    # correlation [TAG-INDEPENDENT]
    if os.path.exists(cov_file):
        cd = np.load(cov_file); cm = cd['cov']; dg = np.sqrt(np.diag(cm))
        corr = cm / np.outer(dg.clip(1e-10), dg.clip(1e-10))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, origin='lower', aspect='auto', extent=[0,n_bins,0,n_bins], vmin=-1, vmax=1, cmap='RdBu_r')
        fmt = '.0f' if var_name == 'true_p' else '.1f'
        tl = [f'[{bins[i]:{fmt}},{bins[i+1]:{fmt}})' for i in range(n_bins)]
        ax.set_xticks(np.arange(n_bins)+0.5); ax.set_xticklabels(tl, fontsize=8, rotation=45, ha='right')
        ax.set_yticks(np.arange(n_bins)+0.5); ax.set_yticklabels(tl, fontsize=8)
        ax.set_title(f'Correlation ({var_name})'); plt.colorbar(im, ax=ax); plt.tight_layout()
        plt.savefig(f'{flags.plot_dir}/correlation_{var_name}.png', dpi=150)
        print(f"    correlation_{var_name}.png"); plt.close()


def _make_combined_chi2(TAG, truth_raw, mc_weights, injected, push_files):
    colors = {'true_p': 'red', 'true_costheta': 'blue'}
    labels = {'true_p': r'$p_e$', 'true_costheta': r'$\cos\theta_e$'}
    curves = {}
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
        print(f"  chi2/ndf ({vn}, ndf={ndf}): prior={pc[0]:.4f}, final={pc[-1]:.4f}")
    fig, ax = plt.subplots(figsize=(8, 6)); av = []
    for vn, (pi, pc) in curves.items():
        ax.plot(pi, pc, 'o-', color=colors[vn], linewidth=2, markersize=5, label=labels[vn]); av.extend(pc)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, label=r'$\chi^2$/DoF = 1')
    ax.set_xlabel('OmniFold Iteration'); ax.set_ylabel(r'$\chi^2$/DoF')
    ax.set_title(rf'$\chi^2$ convergence ({TAG})'); ax.legend(fontsize=12); ax.set_yscale('log')
    ax.set_ylim(min(v for v in av if v > 0)*0.5, max(av)*3); ax.set_xticks(pi); plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/chi2_convergence_{TAG}.png', dpi=150)
    print(f"    chi2_convergence_{TAG}.png"); plt.close()


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
        axes[0].step(bins, np.append(truth_h, truth_h[-1]), where='post', color='black', linewidth=2.5, label='Data Truth')
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
    N = 5; checkpoints = sorted(set([5, 10, len(push_files)]))
    fig, ax = plt.subplots(figsize=(7, 5)); cpc = ['blue','orange','green']; ci = 0
    for ti in checkpoints:
        if ti > len(push_files): continue
        start = max(0, ti - N)
        ww = [np.load(f) if np.load(f).ndim == 1 else np.load(f).mean(axis=0) for f in push_files[start:ti]]
        if len(ww) < 2: continue
        avg = np.diff(np.array(ww), axis=0).mean(axis=0)
        ax.hist(avg, bins=100, range=(-0.15, 0.15), density=True, alpha=0.6, color=cpc[ci], label=f'After {ti} iters'); ci += 1
    ax.set_xlabel(f'Avg weight change (last {N} iters)'); ax.set_ylabel('Fraction')
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
        ax.step(p_bins, np.append(th/(bw*cw), (th/(bw*cw))[-1]), where='post', color='black', linewidth=2, label='Data Truth')
        ax.errorbar(cen, uh/(bw*cw), fmt='o', color='red', markersize=4, capsize=2, label='OmniFold')
        ax.step(p_bins, np.append(nh/(bw*cw), (nh/(bw*cw))[-1]), where='post', color='gray', linewidth=1, linestyle='--', label='Prior')
        ax.set_title(f'{clo:.1f} < cos$\\theta$ < {chi:.1f}', fontsize=11); ax.set_xlabel('p [MeV/c]', fontsize=10)
        if si == 0: ax.legend(fontsize=8)
    plt.suptitle(rf'SBND $\nu_e$ CC: $d^2\sigma / dp\, d\cos\theta$ ({TAG})', fontsize=14); plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/xsec_2d_slices_{TAG}.png', dpi=150)
    print(f"    xsec_2d_slices_{TAG}.png"); plt.close()


def _make_2d_correlation(truth_raw, mc_w):
    true_p, true_cos = truth_raw[:, 0], truth_raw[:, 1]
    p_bins = np.array([0, 500, 1000, 2000]); cos_bins = np.array([-1, 0, 0.5, 0.75, 1.0])
    n_p, n_c = len(p_bins)-1, len(cos_bins)-1; n_2d = n_p * n_c; all_flat = []
    for src in ['bnb', 'genie', 'mcstat']:
        pat = f'{flags.weights_base}/weights_{src}/{src}_univ*/Step2_Iter*_PushWeights.npy'
        files = sorted(glob.glob(pat)); ufiles = {}
        for f in files:
            m = re.search(r'univ(\d+)', f)
            if m:
                uid = int(m.group(1)); it = iter_num(f)
                if uid not in ufiles or it > ufiles[uid][0]: ufiles[uid] = (it, f)
        for uid in sorted(ufiles.keys()):
            push = np.load(ufiles[uid][1]); push = push if push.ndim == 1 else push.mean(axis=0)
            h, _, _ = np.histogram2d(true_cos, true_p, bins=[cos_bins, p_bins], weights=mc_w * push)
            all_flat.append(h.flatten())
    if len(all_flat) < 2: print("  Not enough universes for 2D correlation"); return
    all_flat = np.array(all_flat); diff = all_flat - all_flat.mean(axis=0)
    cov_2d = (diff.T @ diff) / len(all_flat); dg = np.sqrt(np.diag(cov_2d))
    corr = cov_2d / np.outer(dg.clip(1e-10), dg.clip(1e-10))
    bl = [f'c[{cos_bins[ic]:.1f},{cos_bins[ic+1]:.1f}]\np[{p_bins[ip]:.0f},{p_bins[ip+1]:.0f}]'
          for ic in range(n_c) for ip in range(n_p)]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.pcolormesh(np.arange(n_2d+1), np.arange(n_2d+1), corr, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(np.arange(n_2d)+0.5); ax.set_xticklabels(bl, fontsize=7, rotation=90)
    ax.set_yticks(np.arange(n_2d)+0.5); ax.set_yticklabels(bl, fontsize=7)
    ax.set_title(r'Correlation: $(p, \cos\theta)$ 2D bins'); plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(f'{flags.plot_dir}/correlation_2d_p_costheta.png', dpi=150)
    print(f"    correlation_2d_p_costheta.png"); plt.close()


if flags.action in ('validation', 'all'):
    if flags.action == 'all':
        flags.plot_dir_save = flags.plot_dir
        flags.plot_dir = getattr(flags, 'val_plot_dir', 'sbnd/plots_validation/')
    do_validation()
    if flags.action == 'all':
        flags.plot_dir = flags.plot_dir_save

if flags.action in ('paper', 'all'):
    if not hasattr(flags, 'weights_base'): flags.weights_base = 'sbnd'
    if not hasattr(flags, 'export_dir'): flags.export_dir = 'sbnd/exported_weights/'
    if not hasattr(flags, 'cov_dir'): flags.cov_dir = 'sbnd/covariance/'
    do_paper()