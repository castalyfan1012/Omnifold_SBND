# Omnifold_SBND

ML-based unbinned unfolding for the SBND νe CC inclusive cross-section measurement.
Based on [OmniFold](https://arxiv.org/abs/1911.09107) and [Huang et al. (2025)](https://arxiv.org/abs/2504.06857).

---

## Setup

```bash
source setup.sh --install   # first time only
source setup.sh             # every session
```

---

## Tag convention

Fake-data tags are derived from `--var` (which variable was tilted) and `--alpha` (tilt strength):

| `--var` | `--alpha` | Tag | Description |
|---|---|---|---|
| `true_p` | 0.3 | `tilt_p_alpha0.3` | Tilt momentum only |
| `true_costheta` | 0.3 | `tilt_costheta_alpha0.3` | Tilt angle only |
| `both` | 0.15 | `tilt_both_alpha0.15` | Tilt both simultaneously |

---

## Full pipeline

### Step 0 — Format inputs

```bash
python3 sbnd/FormatData_SBND.py
```

Outputs to `../FormattedData_SBND/`. Prints per-bin efficiency and flags bins below 10%.

> **If the event count changes** (new selection, new production), all downstream weight files become invalid. Delete them and rerun from Step 1:
> ```bash
> rm -rf weights_sbnd_closure/ weights_sbnd_fakedata_*/
> rm -rf sbnd/weights_*/  sbnd/covariance/*.npz
> ```

### Step 1 — Closure test

```bash
nohup bash sbnd/runOmnifold_sbnd_closure.sh > closure.log 2>&1 &
# once finished:
python3 sbnd/RunStudies.py check-closure
```

### Step 2 — Fake-data studies

```bash
# Tilt p only (α=0.3)
python3 sbnd/RunStudies.py make-fakedata --mode tilt --var true_p --alpha 0.3
nohup bash sbnd/runOmnifold_sbnd_fakedata.sh --var true_p --alpha 0.3 > fdt_p_03.log 2>&1 &

# Tilt both p and cosθ (α=0.15)
python3 sbnd/RunStudies.py make-fakedata --mode tilt --var both --alpha 0.15
nohup bash sbnd/runOmnifold_sbnd_fakedata.sh --var both --alpha 0.15 > fdt_both_015.log 2>&1 &
```

### Step 3 — Validation plots

```bash
python3 sbnd/MakePlots.py validation --var true_p --alpha 0.3
python3 sbnd/MakePlots.py validation --var both --alpha 0.15

# Extended diagnostics (exclude-bins, merged-cosθ):
python3 sbnd/MakePlots.py validation --var both --alpha 0.3 --extended
```

### Step 4 — Export universe weights

In [`nue_syst.ipynb`](https://github.com/castalyfan1012/cafpyana/blob/feature/cfan_nue_ana/analysis_village/nueCC/nue_syst.ipynb), re-run export cells. All arrays must have shape `(N_events, N_universes)`.

| Source | N_univ | Notes |
|---|---|---|
| `bnb` | 100 | BNB flux |
| `genie` | 100 | GENIE cross-section knobs |
| `extra_xsec` | 100 | MINERvA 2p2h, nuenumu ratio, NOvA NonResPion |
| `g4` | 100 | Hadronic reinteractions |
| `mcstat` | 100 | Poisson bootstrap |

### Step 5 — Systematic universes

```bash
for SRC in bnb genie extra_xsec g4 mcstat; do
    nohup python3 sbnd/RunStudies.py run-syst --source $SRC --start 0 --end 100 \
        > syst_${SRC}.log 2>&1 &
done
```

### Step 6 — ML replicas

```bash
nohup python3 sbnd/RunStudies.py run-ml-unc --n-replicas 50 --var true_p --alpha 0.3 \
    > ml_unc_p.log 2>&1 &
nohup python3 sbnd/RunStudies.py run-ml-unc --n-replicas 50 --var both --alpha 0.15 \
    > ml_unc_both.log 2>&1 &
```

### Step 7 — Build covariance matrices

```bash
# Per-source (needed for breakdown plots)
for SRC in bnb genie extra_xsec g4 mcstat; do
    python3 sbnd/BuildResults.py covariance --source $SRC
done
python3 sbnd/BuildResults.py covariance --source ml

# Combined: full budget
python3 sbnd/BuildResults.py covariance --source all

# Combined: stat+xsec only (for fake-data study plots)
python3 sbnd/BuildResults.py covariance --source fds
```

### Step 8 — Results plots

```bash
python3 sbnd/MakePlots.py results --var true_p --alpha 0.3 --cov-source fds
python3 sbnd/MakePlots.py results --var both --alpha 0.15 --cov-source fds
```

### Step 9 — Cross-section extraction

```bash
python3 sbnd/BuildResults.py xsec --tilted-var true_p --alpha 0.3 --cov-source fds
python3 sbnd/BuildResults.py xsec --tilted-var both --alpha 0.15 --cov-source fds
```

---

## Covariance source groups

| `--source` | Includes | Use for |
|---|---|---|
| `all` | bnb + genie + extra_xsec + g4 + mcstat + ml | Full error budget |
| `fds` | mcstat + genie + extra_xsec | Fake-data study (no flux/detector) |
| `stat` | Analytic diagonal Σw² | Quick cross-check |

---

## Plot output directories

| Directory | Contents |
|---|---|
| `sbnd/plots_validation/` | Fake-data recovery, χ² convergence, PASS/FAIL verdict |
| `sbnd/plots_syst/` | Uncertainty budget, covariance matrices |
| `sbnd/plots_xsec/` | Cross-section, correlations, 2D slices, weight diagnostics |

---

## Configuration

| Parameter | Main / ML replica | Syst universe |
|---|---|---|
| `NITER` | 10 | 5 |
| `NTRIAL` | 3 / 1 | 1 |
| `EPOCHS` | 100 | 50 |
| `NPATIENCE` | 10 | 7 |

---

## Credits

Based on [rhuang1/OmnifoldT2K](https://github.com/rhuang1/OmnifoldT2K).
SBND adaptation by Castaly Fan with guidance from Roger Huang.