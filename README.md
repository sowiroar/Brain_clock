# Multisite MEG Brain-Age Clock

A self-contained, reproducible machine learning framework for brain-age prediction using resting-state magnetoencephalography (MEG) functional connectivity.

The model predicts chronological age from **3,321 unique off-diagonal AAL82 CMI (Cross-Mutual Information) connectivity features** using support vector regression (SVR) with a radial basis function (RBF) kernel. The framework implements strict data leakage controls, repeated nested cross-validation, training-only ensemble selection, site-aware minimax optimization, and independent held-out evaluation.

---

## Repository Structure

The repository is organized following standard scientific reproducible research conventions into `code/`, `data/`, and `results/`:

```
Brain_clock/
├── code/
│   ├── Brain_clock.py               # Main pipeline: nested CV, selection, held-out evaluation
│   └── Sensitivity_ComBat.py        # Diagnostic script: ComBat scanner harmonization stability
├── data/
│   ├── train_database.csv           # Multisite training dataset (N = 825)
│   └── test_database.csv            # Independent held-out dataset (N = 144)
├── results/
│   ├── figures/                     # Publication figures (.png and .svg)
│   ├── reference_results/           # Benchmark frozen model predictions and metadata
│   ├── results_full_nested/         # Complete 50-fold nested CV logs, tables, and provenance
│   └── sensitivity_combat_results.csv # Empirical results of ComBat sensitivity analysis
├── AUDIT_NOTES.md                   # Independent audit notes on data leakage controls
├── requirements.txt                 # Pinned dependencies (compatible with Python 3.10)
├── SHA256SUMS.csv                   # SHA-256 integrity hashes of repository files
└── README.md                        # Documentation and replication guide
```

---

## Environment Setup and Installation

### Prerequisites

- **Python**: Recommended `3.10.x` (e.g., 3.10.9 or 3.10.20).
- Compatible with Windows, Linux, and macOS.

### 1. Using `uv` (Recommended for fast setup)

```bash
# Create Python 3.10 virtual environment
uv venv --python 3.10 .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1
# On Linux/macOS:
# source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Using standard `venv` and `pip`

```bash
python -m venv .venv

# Activate environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
# .venv\Scripts\activate.bat
# Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> [!NOTE]
> `neuroHarmonize` requires `numpy < 2` (tested with `numpy==1.23.5`), `nibabel==5.3.3`, and `neurocombat==0.2.12`. All compatible versions are pinned in `requirements.txt`.

---

## Execution and Usage

Scripts can be invoked directly from the root of the repository without requiring manual path configurations.

### 1. Fast Smoke-Test Pipeline Verification

To verify that the complete machine learning pipeline, ComBat harmonization, site residualization, ensemble selection, and held-out evaluation execute cleanly without awaiting hours of compute:

```bash
python code/Brain_clock.py --smoke-test --out results/test_run
```

### 2. Full Nested Cross-Validation and Independent Held-Out Evaluation

To run the complete scientific analysis (50 outer folds with repeated 3-fold inner grid optimization across 54 SVR hyperparameter configurations):

```bash
python code/Brain_clock.py --out results/full_run
```

Optional CLI parameters:
- `--train <path>`: Path to training CSV (default: `data/train_database.csv`).
- `--test <path>`: Path to test CSV (default: `data/test_database.csv`).
- `--out <path>`: Output directory (default: `results`).
- `--n-jobs <int>`: Parallel worker processes (default: `2`).

### 3. ComBat Scanner Harmonization Sensitivity Analysis

To verify that scanner harmonization preserves biological age associations:

```bash
python code/Sensitivity_ComBat.py
```

Outputs will be saved directly to `results/sensitivity_combat_results.csv`.

---

## Methodological Design and Leakage Controls

### Preprocessing and Harmonization Pipeline

For every training partition in inner CV, outer CV, and final held-out fit:

1. **Scanner Hardware Assignment**: Scanner is defined *a priori* by hardware:
   - `Elekta`: CamCAN and BCBL.
   - `CTF` (`omega_other`): OMEGA.
2. **ComBat Empirical Bayes Harmonization**:
   - Location and scale harmonization is fitted **only on the training partition**.
   - Sex is retained as a preserved biological covariate.
   - **Chronological age is intentionally excluded** from ComBat covariates because age is the prediction target. Preserving age during harmonization on test/held-out folds would require feeding true test chronological age, causing target leakage.
   - The frozen ComBat transformation parameters are transferred unchanged to validation and held-out sets.
3. **Feature-wise Site Residualization**:
   - Fitted exclusively on the training partition.
   - Nuisance design includes age so that only acquisition site variance is removed while preserving age effects.
4. **Model Architecture**:
   - Nonlinear RBF-Kernel Support Vector Regression (`SVR(kernel='rbf')`).
   - Hyperparameter grid: $C \in \{100, 300, 1000\}$, $\gamma \in \{0.1, 0.2, 0.3, 0.5, 0.75, 1.0\}$, $\epsilon \in \{0.01, 0.1, 0.5\}$.
   - No data augmentation is used in the frozen ensemble.

### Hyperparameter Selection: Minimax Worst-Cell Criterion

- Evaluated across **10 repeated 3-fold inner CV splits**, stratified jointly by acquisition site and within-site age quartile.
- Out-of-fold (OOF) predictions are computed for each participant per repetition.
- MAE is computed across every site $\times$ age-quartile cell.
- The selection metric is the **maximum (worst-cell) MAE**, averaged across repetitions. This guarantees that model selection does not favor configurations that excel in large subgroups at the expense of underrepresented ages or acquisition sites.
- **One-Standard-Error (1-SE) Rule**: Candidates within 1 SE of the minimax minimum are retained. Predictions are ensembled using normalized weights inversely proportional to their inner-OOF mean site-specific MAE.

### Complete Partition Independence

- **Held-out Isolation**: `data/test_database.csv` is loaded strictly *after* full training selection has completed.
- **Audit Verification**: Full details on partition overlap assertions, hash checks, and leakage controls are documented in [`AUDIT_NOTES.md`](file:///c:/Users/eguen/Documents/Githubs-Redlat/Brain_clock/AUDIT_NOTES.md).

---

## Benchmark Reference Performance

### Independent Held-Out Sample ($N = 144$)

Performance on unseen participants from representative sites (BCBL and OMEGA):

| Sample | N | Pearson $r$ | MAE (years) | RMSE (years) | $R^2$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Pooled (OMEGA + BCBL)** | **144** | **0.801** | **9.41** | **11.91** | **0.583** |
| BCBL | 91 | 0.652 | 10.47 | 13.43 | 0.326 |
| OMEGA | 53 | 0.731 | 7.59 | 8.71 | 0.385 |

*Confidence intervals and exact statistics are stored in [`results/reference_results/heldout_raw_performance.csv`](file:///c:/Users/eguen/Documents/Githubs-Redlat/Brain_clock/results/reference_results/heldout_raw_performance.csv).*

### Repeated Nested Cross-Validation ($N = 825$)

Performance across 10 complete outer 5-fold cross-validation repetitions:

| Sample | N | Pearson $r$ | MAE (years) | RMSE (years) | $R^2$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Full Training Sample** | **825** | **0.941 ± 0.003** | **5.28 ± 0.11** | **7.40 ± 0.15** | **0.877 ± 0.005** |

*Complete breakdowns per site and repeat are stored in [`results/results_full_nested/cross_validation/metrics_mean_sd_across_repeats.csv`](file:///c:/Users/eguen/Documents/Githubs-Redlat/Brain_clock/results/results_full_nested/cross_validation/metrics_mean_sd_across_repeats.csv).*

---

## Provenance and Integrity

- Integrity verification for all data matrices and code files was recorded in `SHA256SUMS.csv`.
- Each execution writes machine-readable run parameters and hash fingerprints to `artifact_provenance.json`.
- ODQ (Off-Diagonal Quality) summary metrics are retained strictly for descriptive quality control.
