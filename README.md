# Omnifold_SBND

ML-based unbinned unfolding for the SBND $\nu_e$ CC inclusive cross section measurement.
Based on [OmniFold](https://arxiv.org/abs/1911.09107) and [Huang et al. (2025)](https://arxiv.org/abs/2504.06857).

---

## Setup

```bash
source setup.sh --install   # first time only 
source setup.sh             # every session 
```

After sourcing, `python3` resolves to the venv python even on EAF where conda
normally overrides PATH.

---

## When the input data changes

If `FormatData_SBND.py` produces a different N_events (updated selection pkl,
new production), **all weight files become invalid** and you get shape mismatches.

Fix:
1. Re-run `FormatData_SBND.py`
2. Re-export universe weights from notebook (all sources must match new N_events)
3. Clean stale weights and retrain:

```bash
rm -rf weights_sbnd_closure/ weights_sbnd_fakedata_*/
rm -rf sbnd/weights_bnb/ sbnd/weights_genie/ sbnd/weights_extra_xsec/
rm -rf sbnd/weights_g4/ sbnd/weights_mcstat/ sbnd/weights_ml_unc/
rm -rf sbnd/covariance/*.npz
```

---

## Full pipeline

### Step 1 — Format data
```bash
python3 sbnd/FormatData_SBND.py
```
Produces `mc_vals_*.npy`, `mc_weights_*.npy`, `efficiency_*.npy` in
`../FormattedData_SBND/`.

### Step 2 — Closure test
```bash
nohup bash sbnd/runOmnifold_sbnd_closure.sh > closure.log 2>&1 &
python3 sbnd/RunStudies.py check-closure
```
Target: push bias < 1%.

### Step 3 — Fake-data test (default $\alpha$ = 0.3)
```bash
python3 sbnd/RunStudies.py make-fakedata --mode tilt --alpha 0.3
nohup bash sbnd/runOmnifold_sbnd_fakedata.sh tilt_alpha0.3 > fd03.log 2>&1 &
```

### Step 4 — Validation plots
```bash
python3 sbnd/MakePlots.py validation --tag tilt_alpha0.3
```

### Step 5 — Export universe weights (in notebook)
In `nue_syst.ipynb` [link here](https://github.com/castalyfan1012/cafpyana/blob/feature/cfan_nue_ana/analysis_village/nueCC/nue_syst.ipynb), re-run export cells for all sources.
All output arrays must have shape `(N_events, N_universes)`.

| Source | N_univ | Notebook key |
|---|---|---|
| BNB flux | 100 | `flux` |
| GENIE xsec | 100 | `genie` |
| extra_xsec | 100 | `extra_xsec` |
| G4 reinteraction | 300 | `g4` (3 particle types combined) |
| MCstat | 100 | Poisson bootstrap |

### Step 6 — Systematic universes (run in parallel)
```bash
nohup python3 sbnd/RunStudies.py run-syst --source bnb        --start 0 --end 100 > syst_bnb.log    2>&1 &
nohup python3 sbnd/RunStudies.py run-syst --source genie      --start 0 --end 100 > syst_genie.log  2>&1 &
nohup python3 sbnd/RunStudies.py run-syst --source extra_xsec --start 0 --end 100 > syst_extra.log  2>&1 &
nohup python3 sbnd/RunStudies.py run-syst --source g4         --start 0 --end 300 > syst_g4.log     2>&1 &
nohup python3 sbnd/RunStudies.py run-syst --source mcstat     --start 0 --end 100 > syst_mc.log     2>&1 &
```

### Step 7 — ML replicas
```bash
nohup python3 sbnd/RunStudies.py run-ml-unc \
    --n-replicas 50 --tag tilt_alpha0.3 --niter 10 --epochs 100 > ml_unc.log 2>&1 &
```

### Step 8 — Build covariance
```bash
python3 sbnd/BuildResults.py covariance --source all --var both \
    --ml-as-stderr --ml-label 50rep_10iter
```
`--ml-as-stderr` divides the ML covariance by N_replicas, giving the correct
standard error when the central result averages all replicas (see Step 9).

To protect completed ML runs from being recomputed:
```bash
python3 sbnd/BuildResults.py covariance --source all --var both \
    --ml-as-stderr --freeze-ml
```

### Step 9 — Make plots for cross section analysis
```bash
python3 sbnd/BuildResults.py xsec --var both --tag tilt_alpha0.3

python3 sbnd/MakePlots.py paper --var both --tag tilt_alpha0.3
```

`MakePlots.py paper` auto-detects ML replicas in `sbnd/weights_ml_unc/replica_*/`.
If ≥ 2 replicas with matching shape are found, their averaged final push weights
are used as the central result and the iteration convergence plots still come from
the main 10-iteration fakedata run. No extra flags needed.

---

## Plot directory layout

| Directory | Contents |
|---|---|
| `sbnd/plots_validation/` | Closure, fake-data recovery, chi2 convergence, chi2-excluding-bins, bin diagnostic |
| `sbnd/plots_syst/` | Covariance/fractional-covariance matrices, universe spread spaghetti, uncertainty budget |
| `sbnd/plots_xsec/` | Cross-section vs truth, ratios, correlations, 2D slices, reweighting snapshots, weight maps |

### Full plot inventory

**`plots_validation/`**
- `closure_weight_distributions.png` — pull/push weight histograms, should peak at 1.0
- `closure_convergence.png` — push/pull mean ± std vs iteration
- `fakedata_tilt_alpha0.3_recovery_{var}.png` — OmniFold vs injected weight profile
- `fakedata_tilt_alpha0.3_unfolded_distributions.png` — nominal / fake-data / OmniFold
- `fakedata_tilt_alpha0.3_chi2_convergence.png` — chi2/ndf vs iteration (both vars)
- `fakedata_tilt_alpha0.3_chi2_exclude_bins.png` — same, excluding 1 or 2 lowest bins
- `fakedata_tilt_alpha0.3_chi2_bin_diagnostic.png` — per-bin chi2 bar chart
- `chi2_convergence_tilt_alpha0.3.png` — paper-style chi2 convergence (from `paper` action)

**`plots_syst/`**
- `cov_matrix_all_{var}.png` — total covariance + fractional covariance matrices
- `unfolded_with_unc_all_{var}.png` — unfolded spectrum with total systematic band
- `universe_spread_all_{var}.png` — spaghetti of all universe histograms
- `syst_chi2_vs_iter_all_{var}.png` — systematic stability chi2 vs OmniFold iteration
- `uncertainty_budget_{var}.png` — fractional uncertainty per bin, per source

**`plots_xsec/`**
- `xsec_vs_truth_tilt_alpha0.3_{var}.png` — main result: OmniFold xsec vs data truth
- `ratio_to_truth_tilt_alpha0.3_{var}.png` — ratio panel
- `correlation_{var}.png` — bin-to-bin correlation matrix
- `correlation_2d_p_costheta.png` — 2D (p, cosθ) correlation matrix
- `reweighting_snapshots_tilt_alpha0.3_{var}.png` — how unfolded dist evolves iter by iter
- `weights_vs_observable_tilt_alpha0.3_{var}.png` — per-event weights vs observable
- `weight_distributions_tilt_alpha0.3.png` — push weight distribution at each snapshot
- `weight_map_2d_tilt_alpha0.3.png` — mean push weight in (p, cosθ) 2D space
- `xsec_2d_slices_tilt_alpha0.3.png` — dσ/dp in slices of cosθ (double-differential)

---

## Systematic sources

| Source | N_univ | Notes |
|---|---|---|
| `bnb` | 100 | BNB flux uncertainties |
| `genie` | 100 | GENIE cross-section knobs |
| `extra_xsec` | 100 | MINERvA 2p2h, nuenumu ratio, NOvA NonResPion |
| `g4` | 300 | Hadronic reinteractions (piminus + piplus + proton × 100) |
| `mcstat` | 100 | Poisson bootstrap of MC |
| `ml` | 50 | NN initialization (NTRIAL=1, different seeds) |

## ML uncertainty: std vs stderr

The 50 ML replicas with NTRIAL=1 produce a spread σ. The correct uncertainty
contribution when averaging all 50 replicas as the central result is σ/√50.
`--ml-as-stderr` applies this scaling in the combined covariance automatically.
The per-source `covariance_ml_*.npz` always stores the raw σ for diagnostics.
The uncertainty budget plot reads `ml_as_stderr` from the combined covariance
and applies the same scaling there, so the budget is always self-consistent.


## Configuration reference

| Parameter | Main run | Syst universe | ML replica |
|---|---|---|---|
| `NITER` | 10 | 5 | 10 |
| `NTRIAL` | 3 | 1 | 1 |
| `EPOCHS` | 100 | 50 | 100 |
| `NPATIENCE` | 10 | 7 | 10 |

## Closure test thresholds

| Push bias | Status |
|---|---|
| < 1% | Excellent |
| 1–3% | Acceptable |
| 3–5% | Marginal — increase NTRIAL |
| > 5% | Fail |


## Credits

Based on [rhuang1/OmnifoldT2K](https://github.com/rhuang1/OmnifoldT2K).
SBND adaptation by Castaly Fan with guidance from Roger Huang.