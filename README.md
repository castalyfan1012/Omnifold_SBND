# Omnifold_SBND

Machine-learning-based unbinned unfolding using the [OmniFold](https://arxiv.org/abs/1911.09107) method for the SBND νeCC inclusive cross-section measurement, with initial validation on a T2K public dataset. Based on the methodology described in [Huang et al. (2025)](https://arxiv.org/abs/2504.06857).

## Repository structure

```
Omnifold_SBND/
├── setup.sh                    # Environment setup (EAF + GPVM)
├── omnifold.py                 # Core OmniFold engine
├── utils.py                    # Data loading and plotting utilities
├── t2k.py                      # Training driver script
│
├── t2k/                        # T2K validation files
│   └── ...
│
└── sbnd/
    ├── FormatData_SBND.py      # (unchanged) Format cafpyana output for OmniFold
    ├── RunStudies.py           # Closure check, fake data, syst universes, ML unc
    ├── BuildResults.py         # Covariance matrices + cross-section extraction
    ├── MakePlots.py            # All plots: validation, paper-style, diagnostics
    ├── config_omnifold_sbnd_closure.json
    ├── config_omnifold_sbnd_fakedata_tilt.json
    ├── runOmnifold_sbnd_closure.sh
    ├── runOmnifold_sbnd_fakedata_tilt.sh
    └── exported_weights/       # Universe weights exported from notebook
```

## Output directories

| Directory | Contents |
|---|---|
| `../FormattedData_SBND/` | Formatted numpy arrays (FormatData_SBND.py) |
| `plots_sbnd_closure/` | Closure test weight plots (RunStudies.py check-closure) |
| `plots_sbnd_fakedata_{tag}/` | Fake-data validation plots (MakePlots.py validation) |
| `sbnd/covariance/` | Covariance .npz files (BuildResults.py covariance) |
| `sbnd/plots_systematics/` | Covariance diagnostic plots (BuildResults.py covariance) |
| `sbnd/plots_xsec/` | All paper-style and diagnostic plots (MakePlots.py paper) |

## Setup

```bash
source setup.sh --install   # first time only
source setup.sh             # every subsequent session
```

All commands below assume `source setup.sh` has been run and you are in the repo root.

---

## Full SBND pipeline

### Step 1 — Format data

```bash
python3 sbnd/FormatData_SBND.py
```

Reads `selected_nuecc_qual.pkl` from the νeCC analysis pipeline. Output: `../FormattedData_SBND/`.

---

### Step 2 — Closure test

```bash
nohup bash sbnd/runOmnifold_sbnd_closure.sh > omnifold_sbnd_closure.log 2>&1 &
python3 sbnd/RunStudies.py check-closure
```

Output plots: `plots_sbnd_closure/closure_weight_distributions.png`, `closure_convergence.png`.
**Result:** Push mean≈1.0, std<0.1 → PASSED.

---

### Step 3 — Fake data test

**Default tilt (alpha=0.5):**
```bash
python3 sbnd/RunStudies.py make-fakedata --mode tilt --alpha 0.5
nohup bash sbnd/runOmnifold_sbnd_fakedata_tilt.sh > omnifold_sbnd_fakedata_tilt.log 2>&1 &
python3 sbnd/MakePlots.py validation --tag tilt_alpha0.5
```

**To change the tilt strength:** edit `--alpha`. Larger alpha = stronger distortion, harder for OmniFold to recover.
- `--alpha 0.3` — mild tilt (~30% variation across p range)
- `--alpha 0.5` — default, moderate
- `--alpha 1.0` — strong; tests OmniFold limits

After changing alpha you need to: (1) re-run `make-fakedata`, (2) re-run the OmniFold shell script, (3) re-run `MakePlots.py validation`.

**To use a BNB universe as fake data instead:**
```bash
python3 sbnd/RunStudies.py make-fakedata --mode universe \
    --universe-file sbnd/exported_weights/bnb_universe_weights.npy \
    --universe-idx 0
```

Output plots: `plots_sbnd_fakedata_{tag}/recovery_{var}.png`, `unfolded_distributions.png`, `chi2_vs_iterations.png`.

---

### Step 4 — Systematic uncertainty propagation

#### 4a. Export universe weights from notebook

Export BNB, GENIE, and MCstat universe weights from `nuecc_systematics.ipynb` to `sbnd/exported_weights/`.

#### 4b. Run systematic universes

```bash
nohup python3 sbnd/RunStudies.py run-syst --source bnb   --start 0 --end 100 > syst_bnb.log   2>&1 &
nohup python3 sbnd/RunStudies.py run-syst --source genie --start 0 --end 100 > syst_genie.log 2>&1 &
nohup python3 sbnd/RunStudies.py run-syst --source mcstat \
    --universe-file sbnd/exported_weights/mcstat_universe_weights.npy \
    --start 0 --end 100 > syst_mcstat.log 2>&1 &
```

#### 4c. Build covariance matrices

`--source all` saves both the combined `covariance_all_*.npz` and per-source breakdowns
(`covariance_bnb_*.npz`, etc.) to `sbnd/covariance/` — both are needed for the uncertainty budget plot.

```bash
python3 sbnd/BuildResults.py covariance --source all --var true_p
python3 sbnd/BuildResults.py covariance --source all --var true_costheta
```

Output `.npz` files go to `sbnd/covariance/`. Diagnostic plots go to `sbnd/plots_systematics/`.

---

### Step 5 — Cross-section extraction

```bash
python3 sbnd/BuildResults.py xsec --var both
```

Output plots: `sbnd/plots_xsec/xsec_{var}.png`, `efficiency_{var}.png`, `ratio_to_truth_{var}.png`.

---

### Step 6 — All paper and diagnostic plots

```bash
python3 sbnd/MakePlots.py paper --var both --tag tilt_alpha0.5
```

Or validation + paper together:
```bash
python3 sbnd/MakePlots.py all --var both --tag tilt_alpha0.5
```

Output: `sbnd/plots_xsec/` — see plot reference table below.

---

### Step 7 — ML/NN-initialization uncertainty (optional)

```bash
nohup python3 sbnd/RunStudies.py run-ml-unc \
    --n-replicas 10 --tag tilt_alpha0.5 > ml_unc.log 2>&1 &

# After replicas finish:
python3 sbnd/BuildResults.py covariance --source ml --var true_p
python3 sbnd/BuildResults.py covariance --source ml --var true_costheta
python3 sbnd/MakePlots.py paper --var both   # reruns to include ML band
```

---

## How to increase statistics / events

The event count (currently ~6,400) is set by the selection in `FormatData_SBND.py`:
```python
FINAL_STAGE = 'sel_vertex_distance'   # loosen this to include more events
```
Loosening the cut (e.g. using an earlier cut stage) increases N but also lowers purity. OmniFold handles lower-purity samples via the efficiency correction — but you need to re-export efficiency from the notebook after changing the cut.

**Do we need more iterations with more events?** Not necessarily — more events makes each iteration more stable (less NN variance), so you may need *fewer*. The chi2/ndf convergence plot tells you: if it plateaus well before the final iteration, you have enough.

---

## How to change the tilt

The tilt is a per-event reweighting function applied to `true_p`:
```
weight_i = 1 + alpha * (log(p_i) - mean(log(p))) / std(log(p))
```
- Controlled by `--alpha` in `RunStudies.py make-fakedata`
- `alpha=0` → no distortion (closure test)
- `alpha=0.5` → ~50% variation peak-to-peak across the p range
- `alpha=1.0` → strong; some bins get weights near 0 or 2

To try a completely different distortion shape, edit `do_make_fakedata()` in `RunStudies.py`.

---

## Plot reference

| Output directory | Script | Plots |
|---|---|---|
| `plots_sbnd_closure/` | `RunStudies.py check-closure` | `closure_weight_distributions.png`, `closure_convergence.png` |
| `plots_sbnd_fakedata_{tag}/` | `MakePlots.py validation` | `recovery_{var}.png`, `unfolded_distributions.png`, `chi2_vs_iterations.png` |
| `sbnd/covariance/` | `BuildResults.py covariance` | `covariance_{source}_{var}.npz` (data files, not plots) |
| `sbnd/plots_systematics/` | `BuildResults.py covariance` | `cov_matrix_{source}_{var}.png`, `unfolded_with_unc_{source}_{var}.png`, `universe_spread_{source}_{var}.png`, `chi2_vs_iter_{source}_{var}.png` |
| `sbnd/plots_xsec/` | `BuildResults.py xsec` | `xsec_{var}.png`, `efficiency_{var}.png`, `ratio_to_truth_{var}.png` |
| `sbnd/plots_xsec/` | `MakePlots.py paper` | `xsec_vs_truth_{var}.png`, `ratio_to_truth_{var}.png`, `chi2_convergence_{var}.png`, `chi2_convergence_combined.png`, `uncertainty_budget_{var}.png`, `correlation_{var}.png`, `reweighting_snapshots_{var}.png`, `weights_vs_observable_{var}.png`, `weight_distributions.png`, `weight_map_2d.png`, `weight_change_distribution.png`, `xsec_2d_slices.png`, `correlation_2d_p_costheta.png` |

## Configuration reference

| Parameter | T2K | SBND | SBND syst |
|---|---|---|---|
| `NITER` | 15 | 10 | 5 |
| `NTRIAL` | 3 | 3 | 1 |
| `LR` | 1e-4 | 1e-3 | 1e-3 |
| `BATCH_SIZE` | 4096 | 512 | 512 |
| `EPOCHS` | 50 | 100 | 50 |
| `NPATIENCE` | 7 | 10 | 7 |

## Credits

Based on OmniFold code from [rhuang1/OmnifoldT2K](https://github.com/rhuang1/OmnifoldT2K). SBND adaptation by Castaly Fan with guidance from Roger Huang and Afroditi Papadopoulou.