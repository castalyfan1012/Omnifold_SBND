# Omnifold_SBND

ML-based unbinned unfolding using [OmniFold](https://arxiv.org/abs/1911.09107) for the SBND nueCC inclusive cross-section measurement. Based on [Huang et al. (2025)](https://arxiv.org/abs/2504.06857).

## Repository structure

```
Omnifold_SBND/
├── setup.sh                    # Environment setup (EAF + GPVM)
├── omnifold.py                 # Core OmniFold engine
├── utils.py                    # Data loading and plotting utilities
├── t2k.py                      # Training driver script
├── t2k/                        # T2K validation files
└── sbnd/
    ├── FormatData_SBND.py      # Format cafpyana output for OmniFold
    ├── RunStudies.py           # Closure check, fake data, syst universes, ML unc
    ├── BuildResults.py         # Covariance matrices + cross-section extraction
    ├── MakePlots.py            # All plots: validation, paper-style, diagnostics
    ├── runOmnifold_sbnd_fakedata.sh   # Parameterised training script
    ├── runOmnifold_sbnd_closure.sh
    ├── config_omnifold_sbnd_*.json
    └── exported_weights/       # Universe weights exported from notebook
```

## Output directories

| Directory | Contents | Overwrite-safe? |
|---|---|---|
| `../FormattedData_SBND/` | Formatted numpy arrays | N/A |
| `weights_sbnd_fakedata_{tag}/` | OmniFold weights per tag | Yes — separate per tag |
| `sbnd/weights_ml_unc/replica_*/` | ML replica weights | Yes — use `--freeze-ml` |
| `sbnd/plots_validation/` | Closure + fake-data plots | Yes — tag in filename |
| `sbnd/plots_syst/` | Covariance diagnostic plots | Tag-independent |
| `sbnd/plots_xsec/` | Paper-style plots | Tag in filename where relevant |
| `sbnd/covariance/` | Covariance .npz files | Use `--ml-label` for snapshots |

## What produces what (important — read this)

| Stage | Weights used | Plots produced | What it tests |
|---|---|---|---|
| **Closure** (`RunStudies check-closure`) | `weights_sbnd_closure/` | `sbnd/plots_validation/closure_*.png` | OmniFold returns w=1 when data=MC |
| **Fake data** (`MakePlots validation`) | `weights_sbnd_fakedata_{tag}/` | `sbnd/plots_validation/fakedata_{tag}_*.png` | Can OmniFold recover a known distortion? |
| **Syst universes** (`BuildResults covariance`) | `sbnd/weights_bnb/` etc. | `sbnd/plots_syst/` + `sbnd/covariance/*.npz` | Systematic uncertainty propagation |
| **ML replicas** (`BuildResults covariance --source ml`) | `sbnd/weights_ml_unc/replica_*/` | Feeds into `covariance_ml_*.npz` | NN initialization sensitivity |
| **Paper plots** (`MakePlots paper`) | fake-data weights + covariance | `sbnd/plots_xsec/` | Combined picture: xsec + systematics |

The **ML replicas** are NOT more iterations. Each of the 50 replicas is an independent OmniFold run (10 iterations each) with a different random seed. The *spread* across 50 final results = ML uncertainty. This goes into the uncertainty budget alongside BNB/GENIE/MCstat.

---

## Setup

```bash
source setup.sh --install   # first time only
source setup.sh             # every subsequent session
```

---

## Full SBND pipeline

### Step 1 — Format data
```bash
python3 sbnd/FormatData_SBND.py
```

### Step 2 — Closure test
```bash
nohup bash sbnd/runOmnifold_sbnd_closure.sh > omnifold_sbnd_closure.log 2>&1 &
python3 sbnd/RunStudies.py check-closure
```

### Step 3 — Fake data test

The new shell script accepts a tag and optional NITER:
```bash
# Generate fake data weights
python3 sbnd/RunStudies.py make-fakedata --mode tilt --alpha 0.5

# Train OmniFold (default: NITER=10)
nohup bash sbnd/runOmnifold_sbnd_fakedata.sh tilt_alpha0.5 > fakedata_0.5.log 2>&1 &

# Or with more iterations:
nohup bash sbnd/runOmnifold_sbnd_fakedata.sh tilt_alpha0.5 20 > fakedata_0.5_n20.log 2>&1 &

# Make validation plots
python3 sbnd/MakePlots.py validation --tag tilt_alpha0.5
```

**To try a different tilt:**
```bash
python3 sbnd/RunStudies.py make-fakedata --mode tilt --alpha 0.3
nohup bash sbnd/runOmnifold_sbnd_fakedata.sh tilt_alpha0.3 > fakedata_0.3.log 2>&1 &
python3 sbnd/MakePlots.py validation --tag tilt_alpha0.3
```

All plots include the tag in the filename — `tilt_alpha0.5` and `tilt_alpha0.3` never overwrite each other.

### Step 4 — Systematic uncertainty propagation

#### 4a. Export universe weights from notebook
Export BNB, GENIE, MCstat weights from `nuecc_systematics.ipynb` to `sbnd/exported_weights/`.

#### 4b. Run systematic universes
```bash
nohup python3 sbnd/RunStudies.py run-syst --source bnb   --start 0 --end 100 > syst_bnb.log   2>&1 &
nohup python3 sbnd/RunStudies.py run-syst --source genie --start 0 --end 100 > syst_genie.log 2>&1 &
nohup python3 sbnd/RunStudies.py run-syst --source mcstat \
    --universe-file sbnd/exported_weights/mcstat_universe_weights.npy \
    --start 0 --end 100 > syst_mcstat.log 2>&1 &
```

#### 4c. Build covariance matrices
```bash
python3 sbnd/BuildResults.py covariance --source all --var both
```

### Step 5 — Cross-section extraction
```bash
python3 sbnd/BuildResults.py xsec --var both
```

### Step 6 — Paper plots
```bash
python3 sbnd/MakePlots.py paper --var both --tag tilt_alpha0.5
# Or all at once:
python3 sbnd/MakePlots.py all --var both --tag tilt_alpha0.5
```

### Step 7 — ML/NN-initialization uncertainty

```bash
# Run 50 replicas (already done)
nohup python3 sbnd/RunStudies.py run-ml-unc \
    --n-replicas 50 --tag tilt_alpha0.5 --niter 10 --epochs 100 > ml_unc.log 2>&1 &

# Build covariance with ML snapshot label (so it's never overwritten)
python3 sbnd/BuildResults.py covariance --source all --var both --ml-label 50rep_10iter

# Future covariance rebuilds won't recompute ML:
python3 sbnd/BuildResults.py covariance --source all --var both --freeze-ml

# Replot paper figures
python3 sbnd/MakePlots.py paper --var both --tag tilt_alpha0.5
```

---

## Key plot differences

| Plot | Location | Error bars show | Depends on tag? |
|---|---|---|---|
| `fakedata_{tag}_unfolded_distributions.png` | `plots_validation/` | sqrt(N) statistical only | Yes |
| `xsec_vs_truth_{tag}_{var}.png` | `plots_xsec/` | Systematic unc from covariance (BNB+GENIE+MCstat+ML) | Yes |
| `uncertainty_budget_{var}.png` | `plots_xsec/` | Per-source fractional uncertainty | No (from covariance only) |
| `correlation_{var}.png` | `plots_xsec/` | Bin-to-bin correlation | No |

The xsec error bars are larger than the fakedata bars because they include all systematic sources.

---

## Tilt reference

```
weight_i = 1 + alpha * (log(p_i) - mean(log(p))) / std(log(p))
```

| alpha | Effect |
|---|---|
| 0.0 | No distortion (closure) |
| 0.3 | Mild (~30% peak-to-peak) |
| 0.5 | Default moderate |
| 1.0 | Strong; bottom bins approach zero |

---

## Plot reference

| Directory | Script | Key plots |
|---|---|---|
| `sbnd/plots_validation/` | `RunStudies.py check-closure` | `closure_weight_distributions.png`, `closure_convergence.png` |
| `sbnd/plots_validation/` | `MakePlots.py validation` | `fakedata_{tag}_recovery_{var}.png`, `fakedata_{tag}_unfolded_distributions.png`, `fakedata_{tag}_chi2_convergence.png`, `fakedata_{tag}_chi2_bin_diagnostic.png` |
| `sbnd/plots_syst/` | `BuildResults.py covariance` | `cov_matrix_{src}_{var}.png`, `unfolded_with_unc_{src}_{var}.png`, `universe_spread_{src}_{var}.png`, `syst_chi2_vs_iter_{src}_{var}.png` |
| `sbnd/plots_xsec/` | `BuildResults.py xsec` | `xsec_{var}.png`, `efficiency_{var}.png` |
| `sbnd/plots_xsec/` | `MakePlots.py paper` | `xsec_vs_truth_{tag}_{var}.png`, `ratio_to_truth_{tag}_{var}.png`, `chi2_convergence_{tag}.png`, `uncertainty_budget_{var}.png`, `correlation_{var}.png`, `reweighting_snapshots_{tag}_{var}.png`, `weight_distributions_{tag}.png`, `weight_map_2d_{tag}.png`, `xsec_2d_slices_{tag}.png`, `correlation_2d_p_costheta.png` |

## Configuration reference

| Parameter | SBND main | SBND syst | SBND ML unc |
|---|---|---|---|
| `NITER` | 10 (try 20 for true_p) | 5 | 10-15 |
| `NTRIAL` | 3 | 1 | 1 |
| `EPOCHS` | 100 | 50 | 100-150 |

## Credits

Based on OmniFold from [rhuang1/OmnifoldT2K](https://github.com/rhuang1/OmnifoldT2K). SBND adaptation by Castaly Fan with guidance from Roger Huang.