"""
FormatData_SBND.py

Converts the selected nueCC signal events into the .npy inputs OmniFold needs,
computes the per-bin selection efficiency, and (NEW) flags low-efficiency bins.

CHANGE LOG (this revision)
--------------------------
* Q9 (low-momentum / efficiency): a per-variable reliability MASK is now saved
  next to each efficiency file. Any analysis bin whose efficiency falls below
  EFF_THRESHOLD is flagged as unreliable so downstream scripts (and the reader)
  can optionally drop or de-weight it. This makes the "minimum efficiency
  threshold" discussion concrete and reproducible instead of eyeballed.
"""
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

SEL_FILE    = '/home/castalyf/cafpyana/analysis_village/nueCC/nuecc_dfs/selected_nuecc_qual.pkl'
OUTPUT_DIR  = '../FormattedData_SBND/'
FINAL_STAGE = 'sel_vertex_distance'
RECO_VARS   = ['reco_ke', 'reco_costheta', 'reco_p']
TRUTH_VARS  = ['true_ke', 'true_costheta', 'true_p']

# Bins whose selection efficiency is below this fraction are flagged unreliable.
# The [0,200] MeV/c momentum bin (~9% efficiency, slide 13) is the motivating case.
EFF_THRESHOLD = 0.10

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading sel_topo ...")
sel_topo = pd.read_pickle(SEL_FILE)
if sel_topo.index.duplicated().any():
    sel_topo = sel_topo[~sel_topo.index.duplicated(keep='first')]
pot_scale = float(sel_topo['pot_scale'].iloc[0])
print(f"  sel_topo shape : {sel_topo.shape}")
print(f"  pot_scale      : {pot_scale:.4f}")

# ── Filter to SELECTED SIGNAL events only ─────────────────────────────────────
# OmniFold requires len(mc_reco) == len(mc_gen) because RunStep1 multiplies
# weights_push[pass_gen] (shape N_gen) * weights_mc_reco (shape N_reco).
# The cleanest solution: keep only signal events passing the final selection,
# so mc_reco and mc_gen are 1-to-1 matched at the same N events.
# pass_reco = pass_gen = all True → --no_eff handles this correctly.

selected_signal = sel_topo[
    sel_topo['is_sig'] & sel_topo[FINAL_STAGE]
].copy()

print(f"\nSelected signal events ({FINAL_STAGE} & is_sig): {len(selected_signal):,}")

# Drop NaNs in reco features (shower not reconstructed for some signal events)
reco_raw  = selected_signal[RECO_VARS].values.astype(np.float32)
valid     = ~np.isnan(reco_raw).any(axis=1)
if (~valid).sum() > 0:
    print(f"  Dropping {(~valid).sum()} events with NaN reco features")
    selected_signal = selected_signal[valid]
    reco_raw = reco_raw[valid]

truth_raw = selected_signal[TRUTH_VARS].values.astype(np.float32)
n = len(selected_signal)
print(f"  Final N (reco & truth): {n:,}")

# Check truth NaNs
for name, arr in [('reco', reco_raw), ('truth', truth_raw)]:
    nans = np.isnan(arr).sum()
    if nans > 0:
        print(f"  WARNING: {nans} NaNs in {name} — filling with column mean")
        col_means = np.nanmean(arr, axis=0)
        inds = np.where(np.isnan(arr))
        arr[inds] = col_means[inds[1]]

# pass_reco and pass_gen are all True — both arrays are already filtered
# to the selected signal set. --no_eff in the run script handles this.
pass_reco = np.ones(n, dtype=bool)
pass_gen  = np.ones(n, dtype=bool)
weights   = np.full(n, pot_scale, dtype=np.float32)

# Normalize
scaler_reco  = StandardScaler()
scaler_truth = StandardScaler()
reco_norm  = scaler_reco.fit_transform(reco_raw).astype(np.float32)
truth_norm = scaler_truth.fit_transform(truth_raw).astype(np.float32)

np.save(OUTPUT_DIR + 'mc_vals_reco.npy',        reco_norm)
np.save(OUTPUT_DIR + 'mc_vals_truth.npy',        truth_norm)
np.save(OUTPUT_DIR + 'mc_vals_truth_NoNorm.npy', truth_raw)
np.save(OUTPUT_DIR + 'mc_pass_reco.npy',         pass_reco)
np.save(OUTPUT_DIR + 'mc_pass_truth.npy',         pass_gen)
np.save(OUTPUT_DIR + 'mc_weights_reco.npy',       weights)
np.save(OUTPUT_DIR + 'mc_weights_truth.npy',      weights)

print(f"\nSaved to {OUTPUT_DIR}")
print(f"  mc_vals_reco.npy       {reco_norm.shape}")
print(f"  mc_vals_truth.npy      {truth_norm.shape}")
print(f"  mc_pass_reco/gen       all True ({n:,} events)")
print(f"  mc_weights             mean={weights.mean():.4f}")

# Quick sanity: print variable ranges (un-normalized)
print(f"\nVariable ranges (un-normalized):")
for i, v in enumerate(RECO_VARS):
    print(f"  reco  {v:20s}: {np.nanmin(reco_raw[:,i]):.2f} -- {np.nanmax(reco_raw[:,i]):.2f}")
for i, v in enumerate(TRUTH_VARS):
    print(f"  truth {v:20s}: {np.nanmin(truth_raw[:,i]):.2f} -- {np.nanmax(truth_raw[:,i]):.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Per-bin efficiency diagnostic (+ reliability mask)
# ═══════════════════════════════════════════════════════════════════════════════
# Efficiency = N_selected_signal / N_all_signal_in_sample
# Denominator is all is_sig events in sel_topo (passed through quality cuts but
# not necessarily the final vertex-distance cut). This is a PARTIAL efficiency
# measuring the vertex-distance cut acceptance on top of earlier stages.
# For the full efficiency (including reco+quality), the pre-selection evtdf
# would be needed.
print(f"\n{'='*60}")
print(f"Per-bin efficiency diagnostic (threshold = {EFF_THRESHOLD:.0%})")
print(f"{'='*60}")

all_signal = sel_topo[sel_topo['is_sig']].copy()
print(f"  All signal in sample (is_sig):           {len(all_signal):,}")
print(f"  Selected signal (is_sig & {FINAL_STAGE}): {len(selected_signal):,}")

BINNING_EFF = {
    'true_p':        np.array([0, 200, 400, 600, 800, 1000, 1400, 2000]),
    'true_costheta': np.linspace(-1, 1, 11),
}

for var_name, bins in BINNING_EFF.items():
    n_bins = len(bins) - 1

    # Denominator: all signal events (before final cut, but with valid truth)
    gen_vals = all_signal[var_name].values.astype(np.float32)
    gen_valid = ~np.isnan(gen_vals)
    gen_vals_clean = gen_vals[gen_valid]

    # Numerator: selected signal (the events that made it into OmniFold)
    sel_vals = selected_signal[var_name].values.astype(np.float32)

    N_gen, _ = np.histogram(gen_vals_clean, bins=bins)
    N_sel, _ = np.histogram(sel_vals, bins=bins)
    eff = np.where(N_gen > 0, N_sel / N_gen, 0.0)

    np.save(OUTPUT_DIR + f'efficiency_{var_name}.npy', eff)

    # NEW: reliability mask — True where efficiency is trustworthy.
    reliable = eff >= EFF_THRESHOLD
    np.save(OUTPUT_DIR + f'efficiency_mask_{var_name}.npy', reliable)

    print(f"\n  {var_name}:")
    fmt = '.0f' if var_name == 'true_p' else '.2f'
    print(f"  {'Bin':>20s}  {'N_gen':>8s}  {'N_sel':>8s}  {'Eff':>8s}  {'Reliable':>9s}")
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        flag = 'yes' if reliable[i] else 'NO (<thr)'
        print(f"  [{lo:{fmt}},{hi:{fmt}})  {N_gen[i]:8d}  {N_sel[i]:8d}  "
              f"{eff[i]:8.4f}  {flag:>9s}")
    print(f"  Saved: {OUTPUT_DIR}efficiency_{var_name}.npy")
    print(f"  Saved: {OUTPUT_DIR}efficiency_mask_{var_name}.npy  "
          f"({reliable.sum()}/{n_bins} bins reliable)")

    low = [i for i in range(n_bins) if not reliable[i]]
    if low:
        low_str = ", ".join(f"[{bins[i]:{fmt}},{bins[i+1]:{fmt}}) eff={eff[i]:.3f}"
                            for i in low)
        print(f"  ** LOW-EFFICIENCY BINS FLAGGED (< {EFF_THRESHOLD:.0%}): {low_str}")

print(f"\nNote: This efficiency is PARTIAL — it measures the {FINAL_STAGE}")
print(f"cut acceptance on top of earlier selection stages. To get the full")
print(f"efficiency, rerun with the pre-selection evtdf as input.")
print(f"\nReliability masks let BuildResults/MakePlots optionally exclude bins")
print(f"below {EFF_THRESHOLD:.0%} efficiency (see --drop-unreliable there).")