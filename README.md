# MEG Brain Clock

The analysis is self-contained: the Python script imports no project-specific
module and uses the two CSV files stored beside it.

## Files

- `Brain_clock.py`: complete clock, nested CV, training-only
  model selection, held-out evaluation, checks, and provenance hashes.
- `train_database.csv`: training sample (`N=825`).
- `test_database.csv`: independent held-out sample (`N=144`).
- `reference_results/`: predictions, metrics, selected models, and
  preprocessing metadata from the frozen final candidate.
- `results_full_nested/`: complete fresh execution of all 50 outer folds,
  including every inner selection table, OOF predictions, CV summaries, final
  selection, held-out outputs, and provenance.
- `figures/`: raw predicted-age performance in repeated nested CV and the
  independent held-out sample.
- `Sensitivity_ComBat.py`: training-only diagnostic of whether scanner
  harmonization changes age-connectivity associations.
- `sensitivity_combat_results.csv`: results of the ComBat sensitivity analysis.
- `requirements.txt`: versions used for the reference run.
- `SHA256SUMS.csv`: integrity hashes for every packaged file.

## Final preprocessing and model

For every inner or outer fold, and for the final train-to-held-out fit:

1. Scanner is defined a priori as `Elekta` for CamCAN and BCBL and `CTF` for
   OMEGA (internally labelled `omega_other`).
2. ComBat location/scale harmonization is fitted on the current training
   partition only. Sex is preserved; chronological age is not included in the
   ComBat covariates.
3. The frozen ComBat transform is applied to the corresponding validation or
   held-out partition.
4. Feature-wise site residualization is fitted on the same training partition,
   including age in the nuisance fit so that only the site coefficients are
   subtracted.
5. RBF-SVR models are fitted to the resulting 3,321 off-diagonal AAL82 CMI
   features. No data augmentation is used.

Chronological age is deliberately excluded from the ComBat covariates because
it is the prediction target. Preserving age during harmonization would require
supplying each validation or held-out participant's true age when applying the
transform, making the transformed features target-dependent. A training-only
sensitivity analysis (`Sensitivity_ComBat.py`) checks whether this decision
materially attenuates age-related connectivity information. The partial
edge-age association maps before and after the final sex-preserving ComBat
transformation were nearly identical (`r = 0.996`); all 1,109 edges with an
original absolute age association of at least 0.10 retained their direction.
Within each scanner, the original and harmonized age-association maps were
identical to numerical precision (`r = 1.000`). Full results are stored in
`sensitivity_combat_results.csv`.

Hyperparameters are evaluated using 10 repeated three-fold inner CV splits,
stratified by site and within-site age quartile. Within each repeat, the three
validation folds are combined so that every participant has one out-of-fold
prediction. MAE is then calculated separately in every site-by-age-quartile
cell. The score for that candidate and repeat is the largest of these cell-wise
MAEs, and the selection criterion is its mean across the ten repeats. Thus,
model selection targets performance in the worst-served site and age range,
rather than allowing strong performance in a larger or easier subgroup to
conceal poor performance elsewhere.

Candidates whose mean worst-cell MAE lies within one estimated standard error
of the minimum are retained. Their predictions are averaged using weights
inversely proportional to mean site-specific out-of-fold MAE. The one-standard-
error rule is a model-selection heuristic, not a test of statistical
equivalence.

The outer performance estimate uses five-fold CV repeated ten times. All
preprocessing and ensemble selection are repeated inside each outer-training
partition.

## Run

From this directory:

```powershell
python .\Brain_clock.py
```

The complete nested analysis is computationally expensive and intentionally
does not use cached selection tables. Outputs are written to `results/`.

Alternative paths can be supplied explicitly:

```powershell
python .\Brain_clock.py --train train_database.csv --test test_database.csv --out results
```

## Reference held-out performance

All values below are raw independent predictions:

| Sample | N | Pearson r | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|---:|
| OMEGA + BCBL | 144 | 0.80123 | 9.4131 | 11.9112 | 0.58332 |
| BCBL | 91 | 0.65159 | 10.4732 | 13.4273 | 0.32553 |
| OMEGA | 53 | 0.73121 | 7.5929 | 8.7131 | 0.38450 |

Exact confidence intervals and p-values are stored in
`reference_results/heldout_raw_performance.csv`.

## Repeated nested cross-validation performance

Values are the mean and standard deviation across the ten complete outer-CV
repetitions. Within each repetition, every training participant receives one
strictly out-of-fold prediction.

| Sample | N | Pearson r | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|---:|
| Train sample | 825 | 0.94128 +/- 0.00271 | 5.2757 +/- 0.1143 | 7.3998 +/- 0.1517 | 0.87657 +/- 0.00506 |

The complete machine-readable table, including site-specific results, is
`results_full_nested/cross_validation/metrics_mean_sd_across_repeats.csv`.

## Interpretation boundaries

- Raw held-out metrics are the evidence of predictive performance.
- The script also estimates additive age-bias correction from nested OOF
  training predictions. Because applying that correction uses chronological age
  after prediction, corrected accuracy is descriptive and is not independent
  predictive evidence.
- The held-out set contains new participants from represented sites, not a new
  acquisition site.
- ODQ is retained as descriptive quality control. It is neither a predictor nor
  an exclusion criterion because no a priori threshold was specified.
