# OmniFold for T2K and SBND
 
Machine-learning-based unbinned unfolding using the OmniFold method, validated on a T2K public dataset and adapted for the SBND νeCC inclusive cross-section measurement.
 
## Overview
 
OmniFold trains neural network classifiers to iteratively reweight MC events so they match data, operating in the full multi-dimensional feature space without binning. Each iteration has two steps:
- **Step 1 (Pull):** Classifier learns reco-level differences between data and MC → produces reco-level reweighting
- **Step 2 (Push):** Classifier propagates reco-level reweighting back to truth level → produces unfolded truth-level weights
The final **push weights** are the unfolded result: applying them to MC truth events gives the unfolded distribution in any observable.
 
## Repository Structure
 
```
OmnifoldT2K/
├── setup.sh                    # Environment setup (EAF + GPVM)
├── omnifold.py                 # Core OmniFold engine
├── utils.py                    # Data loading and plotting utilities
├── t2k.py                      # Training driver script
│
├── t2k/                        # T2K validation files
│   ├── FormatData.py           # ROOT → numpy conversion
│   ├── MakeFakeDataWeights.py  # Synthetic tilt for fake data test
│   ├── ValidateFakeData.py     # Recovery validation plots
│   ├── config_omnifold.json    # Closure test config
│   ├── config_omnifold_fakedata.json
│   ├── runOmnifold.sh
│   └── runOmnifold_fakedata.sh
│
├── sbnd/                       # SBND adaptation files
│   ├── FormatData_SBND.py      # sel_topo DataFrame → numpy
│   ├── MakeFakeDataWeights_SBND.py  # Fake data (tilt or BNB universe)
│   ├── ValidateFakeData_SBND.py     # Recovery validation plots
│   ├── config_omnifold_sbnd_closure.json
│   ├── config_omnifold_sbnd_fakedata_tilt.json
│   ├── runOmnifold_sbnd_closure.sh
│   └── runOmnifold_sbnd_fakedata_tilt.sh
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
 
This creates `venv_omnifold/` with Python 3.9+, TensorFlow CPU 2.15, numpy, scikit-learn, pandas, matplotlib, pyyaml, and tqdm.
 
---
 
## T2K Validation
 
Uses the public T2K ND280 νμCC0π training sample (~407k events):
https://zenodo.org/doi/10.5281/zenodo.15183090
 
### Step 1: Format data
 
Download and extract the T2K dataset, then convert ROOT files to numpy arrays:
 
```bash
python3 t2k/FormatData.py
```
 
Output: `../FormattedData/` containing `mc_vals_reco.npy`, `mc_vals_truth.npy`, weight and flag arrays.
 
### Step 2: Closure test
 
Data = MC (no distortion). Verifies the pipeline runs correctly.
 
```bash
nohup bash t2k/runOmnifold.sh > omnifold_closure.log 2>&1 &
tail -f omnifold_closure.log
```
 
**Pass criteria:** Loss flat at ~0.0487 from epoch 1. Push/pull weights ≈ 1.0 with std < 0.02.
 
### Step 3: Known-reweighting test
 
Inject a synthetic truth-level distortion (log(pmu) tilt), rerun, verify recovery:
 
```bash
python3 t2k/MakeFakeDataWeights.py
nohup bash t2k/runOmnifold_fakedata.sh > omnifold_fakedata.log 2>&1 &
# After completion:
python3 t2k/ValidateFakeData.py
```
 
**Pass criteria:** Push weights match injected tilt within ~1% on mean/std. Per-bin ratio ≈ 1.0.
 
**Results:** Push mean=0.9755 vs injected 0.9677 (0.8% agreement). Validated by Roger Huang.
 
---
 
## SBND Adaptation
 
Adapted for the SBND νeCC inclusive cross-section measurement using SPINE/DLP reconstruction.
 
**Variables:**
- Reco: `reco_ke`, `reco_costheta`, `reco_p` (electron kinematics from SPINE)
- Truth: `true_ke`, `true_costheta`, `true_p`
- Selection: `sel_vertex_distance` (final cut stage)
- Events: 6,399 selected signal events (is_sig & passes all cuts & valid reco)
### Step 1: Format SBND data
 
Requires `selected_nuecc_qual.pkl` from the νeCC analysis pipeline (`cafpyana`):
 
```bash
python3 sbnd/FormatData_SBND.py
```
 
Output: `../FormattedData_SBND/` with 7 numpy arrays matching the OmniFold input format.
 
### Step 2: SBND closure test
 
```bash
nohup bash sbnd/runOmnifold_sbnd_closure.sh > omnifold_sbnd_closure.log 2>&1 &
# After completion — check closure:
python3 sbnd/MakeFakeDataWeights_SBND.py --check-closure
```
 
**Pass criteria:** Loss flat at ~1.006. Push weights mean ≈ 1.0, std < 0.02.
 
**Results:** Push mean=1.0091, std=0.0070. PASSED.
 
### Step 3: SBND fake data test (synthetic tilt)
 
```bash
# Generate fake data weights (log(true_ke) tilt, alpha=0.5):
python3 sbnd/MakeFakeDataWeights_SBND.py --mode tilt --alpha 0.5
 
# Run OmniFold:
nohup bash sbnd/runOmnifold_sbnd_fakedata_tilt.sh > omnifold_sbnd_fakedata_tilt.log 2>&1 &
 
# After completion — validate:
python3 sbnd/ValidateFakeData_SBND.py --tag tilt_alpha0.5
```
 
**Pass criteria:** Loss decreases (not flat). Push weights recover injected tilt. Per-bin ratio ≈ 1.0 within statistical uncertainty.
 
**Results:** Loss dropped from 1.006 → 0.958. Push mean=1.0036 vs injected 1.0000 (0.4%). Recovery within ±20% across true_ke bins. cosθ recovery within ±10%.
 
### Step 4: SBND fake data test (BNB universe) — optional
 
Uses actual BNB flux systematic universe weights for a physics-motivated fake data test. Requires exporting aligned weights from the systematics notebook:
 
```bash
# After exporting from nuecc_systematics.ipynb:
python3 sbnd/MakeFakeDataWeights_SBND.py --mode universe \
    --universe-file /path/to/bnb_universe_weights_selected_signal.npy
 
# Then run OmniFold with a corresponding config and validate.
```
 
---
 
## Configuration Reference
 
Key parameters in `config_omnifold_*.json`:
 
| Parameter | T2K | SBND | Description |
|-----------|-----|------|-------------|
| `NITER` | 15 | 10 | OmniFold iterations |
| `NTRIAL` | 3 | 3 | Classifiers per iteration (averaged) |
| `LR` | 1e-4 | 1e-3 | Initial learning rate |
| `BATCH_SIZE` | 4096 | 512 | Training batch size |
| `EPOCHS` | 50 | 100 | Max epochs per classifier |
| `NPATIENCE` | 7 | 10 | Early stopping patience |
 
SBND uses smaller batch size and higher LR to compensate for the smaller dataset (6.4k vs 407k events).
 
---
 
## Core Files
 
- **`t2k.py`**: Training driver. Loads config, data, sets up Multifold, runs unfolding. Use `--no_eff` to exclude unmatched truth events from Step 2.
- **`omnifold.py`**: OmniFold implementation with weighted binary cross-entropy loss, iterative pull/push reweighting, multi-trial averaging. Based on [AlephOmniFold](https://github.com/ViniciusMikuni/AlephOmniFold).
- **`utils.py`**: Data loader (`DataLoader`), plotting utilities, style configuration.
## Credits
 
Based on OmniFold code from [rhuang1/OmnifoldT2K](https://github.com/rhuang1/OmnifoldT2K). SBND adaptation by Castaly Fan with guidance from Roger Huang.