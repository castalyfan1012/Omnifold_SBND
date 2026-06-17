"""
FormatData_SBND.py
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