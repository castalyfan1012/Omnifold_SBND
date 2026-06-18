# Omnifold_SBND

Machine-learning-based unbinned unfolding using the [OmniFold](https://arxiv.org/abs/1911.09107) method for the SBND νeCC inclusive cross-section measurement, with initial validation on a T2K public dataset.

## Overview

OmniFold trains neural network classifiers to iteratively reweight MC events so they match data, operating in the full multi-dimensional feature space without binning. Each iteration has two steps:

- **Step 1 (Pull):** Classifier learns reco-level differences between data and MC → produces reco-level reweighting
- **Step 2 (Push):** Classifier propagates the reco-level correction back to truth level → produces unfolded truth-level weights

The final **push weights** are the unfolded result: applying them to MC truth events gives the unfolded distribution in any observable.

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
# First time — creates venv and installs all dependencies:
source setup.sh --install

# Every subsequent session:
source setup.sh
```

---

## T2K validation

Uses the public T2K ND280 νμCC0π training sample (~407k events):
https://zenodo.org/doi/10.5281/zenodo.15183090

All commands are run from the repo root directory (`Omnifold_SBND/`).

### 1. Format data

Download and extract the T2K dataset, then convert ROOT files to numpy arrays:

```bash
source setup.sh
python3 t2k/FormatData.py
```

Output: `../FormattedData/` containing `mc_vals_reco.npy`, `mc_vals_truth.npy`, and weight/flag arrays.

### 2. Closure test

Data = MC (no distortion). Verifies the pipeline runs end-to-end.

```bash
nohup bash t2k/runOmnifold.sh > omnifold_t2k_closure.log 2>&1 &
tail -f omnifold_t2k_closure.log
```

**Pass criteria:** Loss flat at ~0.0487. Push/pull weights ≈ 1.0 with std < 0.02.

### 3. Known-reweighting test

Inject a synthetic truth-level distortion (log(pmu) tilt), rerun, verify recovery:

```bash
python3 t2k/MakeFakeDataWeights.py
nohup bash t2k/runOmnifold_fakedata.sh > omnifold_t2k_fakedata.log 2>&1 &
# After completion:
python3 t2k/ValidateFakeData.py
```

**Result:** Push mean=0.9755 vs injected 0.9677 (0.8% agreement). ✅ Confirmed by Roger Huang.

---

## SBND analysis

Adapted for the SBND νeCC inclusive cross-section measurement using SPINE/DLP reconstruction.

- **Reco variables:** `reco_ke`, `reco_costheta`, `reco_p` (electron kinematics from SPINE)
- **Truth variables:** `true_ke`, `true_costheta`, `true_p`
- **Selection:** `sel_vertex_distance` (final cut stage)
- **Events:** 6,399 selected signal events (`is_sig` & passes all cuts & valid reco)

All commands are run from the repo root directory (`Omnifold_SBND/`).

### 1. Format SBND data

Requires `selected_nuecc_qual.pkl` from the νeCC analysis pipeline (`cafpyana`).

```bash
source setup.sh
python3 sbnd/FormatData_SBND.py
```

Output: `../FormattedData_SBND/` with 7 numpy arrays matching the OmniFold input format.

### 2. Closure test

```bash
nohup bash sbnd/runOmnifold_sbnd_closure.sh > omnifold_sbnd_closure.log 2>&1 &
tail -f omnifold_sbnd_closure.log

# After completion — check:
python3 sbnd/MakeFakeDataWeights_SBND.py --check-closure
```

**Pass criteria:** Loss flat at ~1.006. Push weights mean ≈ 1.0, std < 0.02.

**Result:** Push mean=1.0091, std=0.0070. ✅ PASSED.

### 3. Fake data test (synthetic tilt)

```bash
# Generate fake data weights (log(true_ke) tilt, alpha=0.5):
python3 sbnd/MakeFakeDataWeights_SBND.py --mode tilt --alpha 0.5

# Run OmniFold:
nohup bash sbnd/runOmnifold_sbnd_fakedata_tilt.sh > omnifold_sbnd_fakedata_tilt.log 2>&1 &
tail -f omnifold_sbnd_fakedata_tilt.log

# After completion — validate:
python3 sbnd/ValidateFakeData_SBND.py --tag tilt_alpha0.5
```

**Result:** Loss decreased 1.006→0.958. Push mean=1.0036 vs injected 1.0000 (0.4%). Recovery within ±20% across true_ke bins, ±10% in true cosθ. ✅ PASSED.

### 4. Systematic uncertainty propagation

Export aligned universe weights from the systematics notebook (`nuecc_systematics.ipynb`), then run OmniFold once per universe.

#### 4a. Export weights (in the notebook, after cell 3)

Add the export cell (see `sbnd/exported_weights/` README or the notebook itself) to save `bnb_universe_weights_6399.npy` and `genie_universe_weights_6399.npy` aligned to the 6,399 OmniFold events.

#### 4b. Run 100 universes per source

```bash
# BNB flux universes (~3-4 hours):
nohup python3 sbnd/RunSystematicUniverses.py --source bnb --start 0 --end 100 \
    > syst_bnb.log 2>&1 &

# GENIE cross-section universes (~3-4 hours, can run in parallel):
nohup python3 sbnd/RunSystematicUniverses.py --source genie --start 0 --end 100 \
    > syst_genie.log 2>&1 &
```

Weight files are saved to `sbnd/weights_bnb/{bnb_univ0..99}/` and `sbnd/weights_genie/{genie_univ0..99}/`.

#### 4c. Build covariance matrices

```bash
python3 sbnd/BuildCovarianceMatrix.py --source bnb --var true_ke
python3 sbnd/BuildCovarianceMatrix.py --source genie --var true_ke
python3 sbnd/BuildCovarianceMatrix.py --source all --var true_ke
python3 sbnd/BuildCovarianceMatrix.py --source bnb --var true_costheta
python3 sbnd/BuildCovarianceMatrix.py --source genie --var true_costheta
python3 sbnd/BuildCovarianceMatrix.py --source all --var true_costheta
```

Plots saved to `sbnd/plots_systematics/`.

**Result (true_ke, combined BNB+GENIE):**

| Bin [MeV] | Nominal | Mean | Frac unc |
|-----------|---------|------|----------|
| 0–200 | 90 | 91 | 3.8% |
| 200–400 | 1307 | 1312 | 2.9% |
| 400–600 | 1554 | 1555 | 2.1% |
| 600–800 | 1297 | 1297 | 1.7% |
| 800–1000 | 1153 | 1153 | 1.4% |
| 1000–1400 | 1708 | 1710 | 1.6% |
| 1400–2000 | 1329 | 1331 | 2.2% |

---

## Configuration reference

| Parameter | T2K | SBND | SBND syst | Description |
|-----------|-----|------|-----------|-------------|
| `NITER` | 15 | 10 | 5 | OmniFold iterations |
| `NTRIAL` | 3 | 3 | 1 | Classifiers per iteration |
| `LR` | 1e-4 | 1e-3 | 1e-3 | Initial learning rate |
| `BATCH_SIZE` | 4096 | 512 | 512 | Training batch size |
| `EPOCHS` | 50 | 100 | 50 | Max epochs per classifier |
| `NPATIENCE` | 7 | 10 | 7 | Early stopping patience |

SBND uses smaller batch size / higher LR for the smaller dataset (6.4k vs 407k events). Systematic universes use fewer iterations and single trial since 200 universes provide averaging.

## Core files

- **`t2k.py`**: Training driver. Loads config, data, sets up Multifold, runs unfolding. Use `--no_eff` to exclude unmatched truth events from Step 2.
- **`omnifold.py`**: OmniFold implementation with weighted binary cross-entropy, iterative pull/push reweighting, multi-trial averaging. Based on [AlephOmniFold](https://github.com/ViniciusMikuni/AlephOmniFold).
- **`utils.py`**: Data loader (`DataLoader`), plotting utilities, style configuration.

## Credits

Based on OmniFold code from [rhuang1/OmnifoldT2K](https://github.com/rhuang1/OmnifoldT2K). SBND adaptation by Castaly Fan with guidance from Roger Huang.