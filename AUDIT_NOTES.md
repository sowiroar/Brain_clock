# Audit Notes

## Leakage controls

- The held-out CSV is loaded only after the full-training hyperparameter
  selection has completed.
- ComBat scanner harmonization is re-fitted inside every inner and outer fold.
- Site residualization is re-fitted after ComBat inside every fold.
- Validation and held-out transformations use fitted training parameters
  unchanged.
- There is no selection cache and no reuse of fold artifacts.
- Participant keys and exact connectivity-matrix hashes are checked within and
  across train/test.
- File, model-input, script, and selection-table SHA-256 hashes are recorded.

## Variables used by the clock

- Predictors: 3,321 unique off-diagonal AAL82 CMI connections.
- Target: chronological age.
- ComBat batch: scanner (`Elekta` versus `CTF`).
- ComBat preserved covariate: sex.
- Site residualizer: site offsets with age retained in its nuisance model.
- ODQ is retained only for descriptive quality control and is not a predictor
  or exclusion criterion.

## Frozen selected ensemble

The completed training-only selection retained six RBF-SVR candidates. Exact
parameters and weights are in
`reference_results/selected_models_and_weights.csv`.

