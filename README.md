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
│   ├── FormatData.py
│   ├── MakeFakeDataWeights.py
│   ├── ValidateFakeData.py
│   ├── GetOmnifoldWeights.py
│   ├── config_omnifold.json
│   ├── config_omnifold_fakedata.json
│   ├── runOmnifold.sh
│   └── runOmnifold_fakedata.sh
│
├── sbnd/                       # SBND analysis files
│   ├── FormatData_SBND.py
│   ├── MakeFakeDataWeights_SBND.py
│   ├── ValidateFakeData_SBND.py
│   ├── RunSystematicUniverses.py
│   ├── BuildCovarianceMatrix.py
│   ├── ExtractXsec.py
│   ├── PaperStylePlots.py
│   ├── config_omnifold_sbnd_closure.json
│   ├── config_omnifold_sbnd_fakedata_tilt.json
│   ├── runOmnifold_sbnd_closure.sh
│   ├── runOmnifold_sbnd_fakedata_tilt.sh
│   └── exported_weights/       # Universe weights exported from notebook
│
└── .gitignore
```

## Setup

Works on both FNAL EAF (AlmaLinux 9) and GPVM (SL7).

```bash
# First time:
source setup.sh --install

# Every subsequent session:
source setup.sh
```

All commands below assume you have run `source setup.sh` and are in the repo root.

---

## Quick reference: full SBND pipeline

```bash
# 1. Format data
python3 sbnd/FormatData_SBND.py

# 2. Closure test
nohup bash sbnd/runOmnifold_sbnd_closure.sh > omnifold_sbnd_closure.log 2>&1 &
python3 sbnd/MakeFakeDataWeights_SBND.py --check-closure

# 3. Fake data test
python3 sbnd/MakeFakeDataWeights_SBND.py --mode tilt --alpha 0.5
nohup bash sbnd/runOmnifold_sbnd_fakedata_tilt.sh > omnifold_sbnd_fakedata_tilt.log 2>&1 &
python3 sbnd/ValidateFakeData_SBND.py --tag tilt_alpha0.5

# 4. Systematic universes (export weights from notebook first)
nohup python3 sbnd/RunSystematicUniverses.py --source bnb --start 0 --end 100 > syst_bnb.log 2>&1 &
nohup python3 sbnd/RunSystematicUniverses.py --source genie --start 0 --end 100 > syst_genie.log 2>&1 &
nohup python3 sbnd/RunSystematicUniverses.py --source mcstat \
    --universe-file sbnd/exported_weights/mcstat_universe_weights.npy \
    --start 0 --end 100 > syst_mcstat.log 2>&1 &

# 5. Covariance matrices
python3 sbnd/BuildCovarianceMatrix.py --source all --var true_ke
python3 sbnd/BuildCovarianceMatrix.py --source all --var true_costheta

# 6. Cross-section extraction
python3 sbnd/ExtractXsec.py --var both

# 7. Paper-style presentation plots
python3 sbnd/PaperStylePlots.py --var both
```

---

## T2K validation

Uses the public T2K ND280 νμCC0π training sample (~407k events):
https://zenodo.org/doi/10.5281/zenodo.15183090

### 1. Format data

```bash
python3 t2k/FormatData.py
```

### 2. Closure test

```bash
nohup bash t2k/runOmnifold.sh > omnifold_t2k_closure.log 2>&1 &
```

**Result:** Loss flat ~0.0487. Push/pull weights ≈ 1.0.

### 3. Known-reweighting test

```bash
python3 t2k/MakeFakeDataWeights.py
nohup bash t2k/runOmnifold_fakedata.sh > omnifold_t2k_fakedata.log 2>&1 &
python3 t2k/ValidateFakeData.py
```

**Result:** Push mean=0.9755 vs injected 0.9677 (0.8% agreement). Confirmed by Roger Huang.

---

## SBND analysis

### Variables and dataset

- **Reco features:** `reco_ke`, `reco_costheta` (electron kinematics from SPINE/DLP)
- **Truth features:** `true_ke`, `true_costheta`
- **Selection:** `sel_vertex_distance` (final cut stage)
- **Events:** 6,399 selected signal events

### Step 1: Format SBND data

Requires `selected_nuecc_qual.pkl` from the νeCC analysis pipeline.

```bash
python3 sbnd/FormatData_SBND.py
```

### Step 2: Closure test

```bash
nohup bash sbnd/runOmnifold_sbnd_closure.sh > omnifold_sbnd_closure.log 2>&1 &
python3 sbnd/MakeFakeDataWeights_SBND.py --check-closure
```

**Result:** Push mean=1.0091, std=0.0070. PASSED.

### Step 3: Fake data test (synthetic tilt)

```bash
python3 sbnd/MakeFakeDataWeights_SBND.py --mode tilt --alpha 0.5
nohup bash sbnd/runOmnifold_sbnd_fakedata_tilt.sh > omnifold_sbnd_fakedata_tilt.log 2>&1 &
python3 sbnd/ValidateFakeData_SBND.py --tag tilt_alpha0.5
```

**Result:** Push mean=1.0036 vs injected 1.0000 (0.4%). Chi2/ndf: 655→2.2 (true_ke), 334→0.25 (true_cosθ). Monotonic convergence, no overfitting.

### Step 4: Systematic uncertainty propagation

#### 4a. Export universe weights from notebook

Add export cells to `nuecc_systematics.ipynb` (after cell 3) to save aligned BNB, GENIE, and MCstat universe weights to `sbnd/exported_weights/`.

#### 4b. Run 300 universes

```bash
source setup.sh
nohup python3 sbnd/RunSystematicUniverses.py --source bnb --start 0 --end 100 > syst_bnb.log 2>&1 &
nohup python3 sbnd/RunSystematicUniverses.py --source genie --start 0 --end 100 > syst_genie.log 2>&1 &
nohup python3 sbnd/RunSystematicUniverses.py --source mcstat \
    --universe-file sbnd/exported_weights/mcstat_universe_weights.npy \
    --start 0 --end 100 > syst_mcstat.log 2>&1 &
```

#### 4c. Build covariance matrices

```bash
python3 sbnd/BuildCovarianceMatrix.py --source bnb --var true_ke
python3 sbnd/BuildCovarianceMatrix.py --source genie --var true_ke
python3 sbnd/BuildCovarianceMatrix.py --source all --var true_ke
python3 sbnd/BuildCovarianceMatrix.py --source bnb --var true_costheta
python3 sbnd/BuildCovarianceMatrix.py --source genie --var true_costheta
python3 sbnd/BuildCovarianceMatrix.py --source all --var true_costheta
```

**Result (combined, true_ke):** 1.7–4.0% fractional uncertainty. Chi2/ndf ≈ 0.007, flat across iterations.

### Step 5: Cross-section extraction

Export per-bin efficiency from the notebook first, then:

```bash
python3 sbnd/ExtractXsec.py --var both
```

Produces dσ/dKE and dσ/dcosθ with systematic bands. Currently shape-only; set `INTEGRATED_FLUX_PER_POT` and `N_TARGETS` in the script for absolute normalization.

### Step 6: Paper-style presentation plots

```bash
python3 sbnd/PaperStylePlots.py --var both
```

Produces plots matching [Huang et al. (2025)](https://arxiv.org/abs/2504.06857):
- `xsec_vs_truth_{var}.png` — Unfolded xsec vs Data Truth (Fig 9 style)
- `ratio_to_truth_{var}.png` — Ratio to Data Truth (Fig 10 style)
- `chi2_convergence_{var}.png` — Chi2/DoF vs iterations (Fig 4 style)
- `uncertainty_budget_{var}.png` — Fractional uncertainty breakdown (Fig 8 style)
- `correlation_{var}.png` — Correlation matrix (Appendix A style)

---

## Configuration reference

| Parameter | T2K | SBND | SBND syst |
|-----------|-----|------|-----------|
| `NITER` | 15 | 10 | 5 |
| `NTRIAL` | 3 | 3 | 1 |
| `LR` | 1e-4 | 1e-3 | 1e-3 |
| `BATCH_SIZE` | 4096 | 512 | 512 |
| `EPOCHS` | 50 | 100 | 50 |
| `NPATIENCE` | 7 | 10 | 7 |

## Core files

- **`t2k.py`**: Training driver. Use `--no_eff` to exclude unmatched truth events from Step 2.
- **`omnifold.py`**: OmniFold implementation. Based on [AlephOmniFold](https://github.com/ViniciusMikuni/AlephOmniFold).
- **`utils.py`**: Data loader and plotting utilities.

## Credits

Based on OmniFold code from [rhuang1/OmnifoldT2K](https://github.com/rhuang1/OmnifoldT2K). SBND adaptation by Castaly Fan with guidance from Roger Huang and Afroditi Papadopoulou.