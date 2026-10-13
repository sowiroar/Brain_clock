"""Final self-contained multisite MEG brain-age clock.

The clock predicts chronological age from the 3,321 unique off-diagonal
AAL82 CMI connectivity features using support vector regression (SVR) with a
nonlinear radial-basis-function kernel. It implements repeated nested
cross-validation, training-only hyperparameter selection, final model fitting,
and independent held-out evaluation in one script.

The outer validation uses 5 folds repeated 10 times. Within every outer fit,
SVR hyperparameters are evaluated by repeated 3-fold inner cross-validation.
ComBat scanner harmonization is first estimated using only the current fitting
partition, preserving sex but not chronological age, and is transferred
unchanged to validation. Scanner is defined a priori from acquisition hardware:
CamCAN and BCBL are Elekta; OMEGA is the other scanner. Feature-wise site
residualization is then estimated using only the same fitting partition while
retaining its fitted age term. The same training-only sequence is used for the
complete training database and independent held-out sample.

Model selection uses a site-aware MAE criterion. Candidates whose repeated
inner-CV worst site-by-age-stratum MAE lies within one estimated standard error
of the minimum are retained. Their predictions are averaged with weights
inversely proportional to their repeated inner out-of-fold mean site-specific
MAE. Eligibility and weights are re-estimated inside every outer-training
partition, so the ensemble itself is fully nested.

This final configuration does not use data augmentation. The independent test
database is accessed only after training-only model selection is complete.

Input validation, participant and matrix-overlap checks, provenance hashes,
feature preparation, fitting, and performance metrics are implemented here;
no project-specific Python module is imported. Raw held-out metrics provide
the independent predictive-performance estimates. An additive age-bias
correction is learned exclusively from nested out-of-fold training predictions;
because its application uses held-out chronological age after prediction,
corrected metrics are reported descriptively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from neuroHarmonize import harmonizationApply, harmonizationLearn
from scipy.stats import norm as normal_distribution
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.svm import SVR


# Frozen final configuration. Data augmentation is deliberately disabled.
USE_AUGMENTATION = "no"
AUGMENTATION_AGE_RANGE = (30.0, 40.0)
N_AUGMENTED_ROWS = 120
N_AUGMENTATION_MODELS = 20

RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_OUTER_REPEATS = 10
N_INNER_SPLITS = 3
N_INNER_REPEATS = 10
N_JOBS = 2
N_ROI = 82
N_BOOTSTRAP = 10_000
GRID = {
    "C": (100.0, 300.0, 1000.0),
    "gamma": (0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
    "epsilon": (0.01, 0.1, 0.5),
}

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_TRAIN = (REPO_DIR / "data" / "train_database.csv") if (REPO_DIR / "data" / "train_database.csv").exists() else (SCRIPT_DIR / "train_database.csv")
DEFAULT_TEST = (REPO_DIR / "data" / "test_database.csv") if (REPO_DIR / "data" / "test_database.csv").exists() else (SCRIPT_DIR / "test_database.csv")
DEFAULT_OUT = REPO_DIR / "results"


def feature_columns() -> list[str]:
    """Return the 3,321 unique off-diagonal AAL82 connectivity columns."""
    return [
        f"ROIS_{first}_vs_ROI_{second}"
        for first in range(N_ROI)
        for second in range(first + 1, N_ROI)
    ]


def normalize_text(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower()


def validate_frame(frame: pd.DataFrame, name: str, features: list[str]) -> None:
    required = ["subject_ID", "site", "Age", "Sex", *features]
    missing = [column for column in required if column not in frame]
    if missing:
        raise RuntimeError(f"{name} is missing columns: {missing[:10]}")
    for column in ("subject_ID", "site"):
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise RuntimeError(f"{name} contains missing or blank {column} values")
    keys = pd.DataFrame(
        {
            "site": normalize_text(frame["site"]),
            "subject": normalize_text(frame["subject_ID"]),
        }
    )
    if keys.duplicated().any():
        duplicates = keys.loc[keys.duplicated(keep=False)].drop_duplicates()
        raise RuntimeError(
            f"{name} contains duplicated site-participant keys: "
            f"{duplicates.head().to_dict('records')}"
        )
    ages = pd.to_numeric(frame["Age"], errors="coerce").to_numpy(float)
    connectivity = frame[features].to_numpy(float)
    if not np.isfinite(ages).all():
        raise RuntimeError(f"{name} contains missing or infinite ages")
    if not np.isfinite(connectivity).all():
        raise RuntimeError(f"{name} contains missing or infinite connectivity values")
    row_hashes = pd.Series(matrix_row_hashes(frame, features), dtype=str)
    duplicated_matrices = row_hashes.duplicated(keep=False)
    if duplicated_matrices.any():
        rows = np.flatnonzero(duplicated_matrices.to_numpy())[:10].tolist()
        raise RuntimeError(
            f"{name} contains exact duplicate connectivity matrices at rows {rows}"
        )


def _update_text_values(hasher: object, values: pd.Series) -> None:
    for value in values.astype(str):
        encoded = value.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "little"))
        hasher.update(encoded)


def model_input_fingerprint(frame: pd.DataFrame, features: list[str]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"brain-clock-combat-scanner-model-input-v1")
    _update_text_values(hasher, pd.Series(features, dtype=str))
    _update_text_values(hasher, normalize_text(frame["site"]))
    _update_text_values(hasher, normalize_text(frame["subject_ID"]))
    ages = np.ascontiguousarray(
        pd.to_numeric(frame["Age"], errors="raise").to_numpy(dtype="<f8")
    )
    hasher.update(ages.tobytes(order="C"))
    sex = np.ascontiguousarray(
        pd.to_numeric(frame["Sex"], errors="raise").to_numpy(dtype="<f8")
    )
    hasher.update(sex.tobytes(order="C"))
    values = np.ascontiguousarray(frame[features].to_numpy(dtype="<f8"))
    for start in range(0, len(values), 32):
        hasher.update(values[start : start + 32].tobytes(order="C"))
    return hasher.hexdigest()


def matrix_row_hashes(frame: pd.DataFrame, features: list[str]) -> list[str]:
    values = np.ascontiguousarray(frame[features].to_numpy(dtype="<f8"))
    return [hashlib.sha256(row.tobytes(order="C")).hexdigest() for row in values]


def check_no_overlap(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> None:
    train_keys = set(
        zip(
            normalize_text(train["site"]),
            normalize_text(train["subject_ID"]),
            strict=True,
        )
    )
    test_keys = set(
        zip(
            normalize_text(test["site"]),
            normalize_text(test["subject_ID"]),
            strict=True,
        )
    )
    duplicated_keys = train_keys.intersection(test_keys)
    if duplicated_keys:
        raise RuntimeError(
            f"Train/test participant overlap detected: {sorted(duplicated_keys)[:5]}"
        )
    train_hashes = matrix_row_hashes(train, features)
    test_hashes = matrix_row_hashes(test, features)
    duplicated_hashes = set(train_hashes).intersection(test_hashes)
    if duplicated_hashes:
        raise RuntimeError(
            "Exact train/test connectivity-matrix overlap detected under possibly "
            "different IDs"
        )


def odq_descriptive_summary(frame: pd.DataFrame, dataset: str) -> list[dict]:
    """Summarize ODQ without using it as a predictor or exclusion criterion."""
    if "ODQ" not in frame.columns:
        return [
            {
                "dataset": dataset,
                "site": "pooled",
                "n_total": len(frame),
                "n_available": 0,
                "n_missing": len(frame),
                "minimum": np.nan,
                "q1": np.nan,
                "median": np.nan,
                "q3": np.nan,
                "maximum": np.nan,
                "n_below_80": 0,
            }
        ]
    values = pd.to_numeric(frame["ODQ"], errors="coerce")
    sites = normalize_text(frame["site"])
    rows: list[dict] = []
    groups = [("pooled", np.ones(len(frame), dtype=bool))]
    groups.extend(
        (site, sites.eq(site).to_numpy()) for site in sorted(sites.unique())
    )
    for site, mask in groups:
        current = values.loc[mask]
        available = current.dropna()
        rows.append(
            {
                "dataset": dataset,
                "site": site,
                "n_total": int(mask.sum()),
                "n_available": int(available.size),
                "n_missing": int(current.isna().sum()),
                "minimum": float(available.min()) if available.size else np.nan,
                "q1": float(available.quantile(0.25)) if available.size else np.nan,
                "median": float(available.median()) if available.size else np.nan,
                "q3": float(available.quantile(0.75)) if available.size else np.nan,
                "maximum": float(available.max()) if available.size else np.nan,
                "n_below_80": int(available.lt(80.0).sum()),
            }
        )
    return rows


def augment_gap_real_parents(
    ages: np.ndarray, features: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate GAP120 rows using only real rows supplied to this function."""
    rng = np.random.default_rng(seed)
    low, high = AUGMENTATION_AGE_RANGE
    synthetic_ages: list[float] = []
    synthetic_features: list[np.ndarray] = []
    attempts = 0
    max_attempts = N_AUGMENTED_ROWS * 20
    while len(synthetic_ages) < N_AUGMENTED_ROWS and attempts < max_attempts:
        first, second = rng.choice(len(ages), size=2, replace=False)
        age_first = ages[first]
        age_second = ages[second]
        if age_first == age_second:
            attempts += 1
            continue
        alpha_low = (low - age_first) / (age_second - age_first)
        alpha_high = (high - age_first) / (age_second - age_first)
        lower = max(0.0, min(alpha_low, alpha_high))
        upper = min(1.0, max(alpha_low, alpha_high))
        if lower >= upper:
            attempts += 1
            continue
        alpha = rng.uniform(lower, upper)
        synthetic_ages.append(age_first + alpha * (age_second - age_first))
        synthetic_features.append(
            features[first] + alpha * (features[second] - features[first])
        )
        attempts += 1
    if len(synthetic_ages) != N_AUGMENTED_ROWS:
        raise RuntimeError(
            f"Generated {len(synthetic_ages)} GAP rows; expected {N_AUGMENTED_ROWS}"
        )
    return (
        np.concatenate([ages, np.asarray(synthetic_ages)]),
        np.vstack([features, np.stack(synthetic_features)]),
    )


def residualize_site_train_test(
    train_features: np.ndarray,
    target_features: np.ndarray,
    train_ages: np.ndarray,
    train_sites: pd.Series,
    target_sites: pd.Series,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Remove fit-estimated site offsets while retaining the fitted age term."""
    normalized_train_sites = normalize_text(train_sites)
    normalized_target_sites = normalize_text(target_sites)
    levels = sorted(normalized_train_sites.unique().tolist())
    unknown = sorted(set(normalized_target_sites.unique()) - set(levels))
    if unknown:
        raise RuntimeError(f"Sites absent from the training partition: {unknown}")
    train_site_columns = [
        normalized_train_sites.eq(level).astype(float).to_numpy()
        for level in levels[1:]
    ]
    target_site_columns = [
        normalized_target_sites.eq(level).astype(float).to_numpy()
        for level in levels[1:]
    ]
    train_design = np.column_stack(
        [np.ones(len(train_sites)), train_ages - np.mean(train_ages), *train_site_columns]
    )
    target_site_design = (
        np.column_stack(target_site_columns)
        if target_site_columns
        else np.zeros((len(target_sites), 0), dtype=float)
    )
    coefficients = np.linalg.pinv(train_design) @ train_features
    site_coefficients = coefficients[2:, :]
    train_component = train_design[:, 2:] @ site_coefficients
    target_component = target_site_design @ site_coefficients
    return (
        train_features - train_component,
        target_features - target_component,
        {
            "site_levels": levels,
            "subtracted_site_columns": levels[1:],
            "age_in_nuisance_fit": True,
            "parameters_learned_from": "current training partition only",
        },
    )


def scanner_covariates(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the prespecified scanner batch and the preserved sex covariate."""
    site = normalize_text(frame["site"])
    expected = {"camcan", "omega", "bcbl"}
    unknown = sorted(set(site.unique()) - expected)
    if unknown:
        raise RuntimeError(f"Cannot assign scanner for unknown sites: {unknown}")
    sex = pd.to_numeric(frame["Sex"], errors="coerce").to_numpy(float)
    if not np.isfinite(sex).all():
        raise RuntimeError("Sex contains missing or non-numeric values")
    return pd.DataFrame(
        {
            "SITE": np.where(site.eq("omega"), "omega_other", "elekta"),
            "sex": sex,
        }
    )


def combat_scanner_train_test(
    train_features: np.ndarray,
    target_features: np.ndarray,
    train: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit ComBat by scanner on train and apply the frozen transform to target."""
    train_covariates = scanner_covariates(train)
    target_covariates = scanner_covariates(target)
    train_scanners = set(train_covariates["SITE"].astype(str))
    unknown = sorted(set(target_covariates["SITE"].astype(str)) - train_scanners)
    if unknown:
        raise RuntimeError(
            f"Scanners absent from the current training partition: {unknown}"
        )
    model, harmonized_train = harmonizationLearn(
        train_features,
        train_covariates,
        eb=True,
        seed=RANDOM_STATE,
    )
    harmonized_target = harmonizationApply(
        target_features,
        target_covariates,
        model,
    )
    if not np.isfinite(harmonized_train).all() or not np.isfinite(harmonized_target).all():
        raise RuntimeError("ComBat produced missing or infinite feature values")
    return (
        harmonized_train,
        harmonized_target,
        {
            "scanner_levels": sorted(train_scanners),
            "batch_definition": {
                "elekta": ["camcan", "bcbl"],
                "omega_other": ["omega"],
            },
            "preserved_covariates": ["sex"],
            "age_preserved": False,
            "parameters_learned_from": "current training partition only",
        },
    )


def prepare_features(
    train: pd.DataFrame, target: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    train_age = pd.to_numeric(train["Age"], errors="raise").to_numpy(float)
    target_age = pd.to_numeric(target["Age"], errors="raise").to_numpy(float)
    combat_train, combat_target, combat_metadata = combat_scanner_train_test(
        train[features].to_numpy(float), target[features].to_numpy(float), train, target
    )
    train_values, target_values, site_metadata = residualize_site_train_test(
        combat_train,
        combat_target,
        train_age,
        train["site"],
        target["site"],
    )
    metadata = {
        "order": ["combat_scanner", "site_residualization"],
        "combat_scanner": combat_metadata,
        "site_residualization": site_metadata,
    }
    return train_age, target_age, train_values, target_values, metadata


def performance(age: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    correlation = np.nan
    if len(age) >= 2 and np.std(age) > 0 and np.std(prediction) > 0:
        correlation = float(pearsonr(age, prediction).statistic)
    return {
        "r": correlation,
        "mae": float(mean_absolute_error(age, prediction)),
        "rmse": float(np.sqrt(np.mean((prediction - age) ** 2))),
        "r2": float(r2_score(age, prediction)),
        "bias_pred_minus_age": float(np.mean(prediction - age)),
    }


def pearson_fisher_ci(correlation: float, n: int) -> tuple[float, float]:
    if not np.isfinite(correlation) or n <= 3:
        return np.nan, np.nan
    clipped = float(np.clip(correlation, -1 + 1e-12, 1 - 1e-12))
    z_value = np.arctanh(clipped)
    margin = float(normal_distribution.ppf(0.975) / np.sqrt(n - 3))
    return float(np.tanh(z_value - margin)), float(np.tanh(z_value + margin))


def mae_bootstrap_ci(
    age: np.ndarray, prediction: np.ndarray, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    errors = np.abs(np.asarray(prediction) - np.asarray(age))
    values = np.empty(N_BOOTSTRAP, dtype=float)
    for start in range(0, N_BOOTSTRAP, 1000):
        stop = min(start + 1000, N_BOOTSTRAP)
        indices = rng.integers(0, len(errors), size=(stop - start, len(errors)))
        values[start:stop] = errors[indices].mean(axis=1)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def heldout_metrics(
    frame: pd.DataFrame,
    age: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    site = normalize_text(frame["site"]).to_numpy()
    subsets = [("pooled", np.ones(len(frame), dtype=bool))]
    subsets.extend((level, site == level) for level in sorted(np.unique(site)))
    rows = []
    for subset_index, (label, mask) in enumerate(subsets):
        for variant_index, (variant, prediction) in enumerate(predictions.items()):
            metrics = performance(age[mask], prediction[mask])
            result = pearsonr(age[mask], prediction[mask])
            r_low, r_high = pearson_fisher_ci(metrics["r"], int(mask.sum()))
            mae_low, mae_high = mae_bootstrap_ci(
                age[mask], prediction[mask], RANDOM_STATE + subset_index * 100 + variant_index
            )
            rows.append(
                {
                    "subset": label,
                    "variant": variant,
                    "n": int(mask.sum()),
                    **metrics,
                    "r_p_value": float(result.pvalue),
                    "r_ci95_low": r_low,
                    "r_ci95_high": r_high,
                    "mae_ci95_low": mae_low,
                    "mae_ci95_high": mae_high,
                }
            )
    return pd.DataFrame(rows)


def fit_additive_age_bias_correction(
    age: np.ndarray, prediction_oof: np.ndarray
) -> tuple[float, float, np.ndarray]:
    bag = prediction_oof - age
    slope, intercept = np.polyfit(age, bag, deg=1)
    corrected = prediction_oof - (slope * age + intercept)
    return float(slope), float(intercept), corrected


def raw_metrics_by_site_age_band(
    frame: pd.DataFrame, age: np.ndarray, prediction: np.ndarray
) -> pd.DataFrame:
    bands = pd.cut(
        age,
        bins=[-np.inf, 30.0, 41.0, 60.0, np.inf],
        labels=["<30", "30-40", "41-59", "60+"],
        right=False,
    )
    sites = normalize_text(frame["site"]).to_numpy()
    rows = []
    for site_name in ["pooled", *sorted(np.unique(sites))]:
        site_mask = np.ones(len(frame), dtype=bool) if site_name == "pooled" else sites == site_name
        for band in bands.categories:
            mask = site_mask & (np.asarray(bands.astype(str)) == str(band))
            if not mask.any():
                continue
            rows.append(
                {
                    "site": site_name,
                    "age_band": str(band),
                    "n": int(mask.sum()),
                    **performance(age[mask], prediction[mask]),
                }
            )
    return pd.DataFrame(rows)


def load_training(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path, low_memory=False)
    features = feature_columns()
    validate_frame(frame, "training database", features)
    frame["site"] = normalize_text(frame["site"])
    return frame, features


def load_test(path: Path, features: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    validate_frame(frame, "held-out database", features)
    frame["site"] = normalize_text(frame["site"])
    return frame


class StandaloneCore:
    """Namespace preserving the internal call interface without dependencies."""

    GAP_RANGE = AUGMENTATION_AGE_RANGE
    N_GAP_SYNTHETIC = N_AUGMENTED_ROWS
    normalize_text = staticmethod(normalize_text)
    model_input_fingerprint = staticmethod(model_input_fingerprint)
    check_no_overlap = staticmethod(check_no_overlap)
    augment_gap_real_parents = staticmethod(augment_gap_real_parents)
    prepare_features = staticmethod(prepare_features)
    performance = staticmethod(performance)
    heldout_metrics = staticmethod(heldout_metrics)
    fit_additive_age_bias_correction = staticmethod(fit_additive_age_bias_correction)
    raw_metrics_by_site_age_band = staticmethod(raw_metrics_by_site_age_band)
    load_training = staticmethod(load_training)
    load_test = staticmethod(load_test)


def load_core() -> StandaloneCore:
    return StandaloneCore()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def site_age_strata(core, data: pd.DataFrame, required_splits: int) -> np.ndarray:
    site = core.normalize_text(data["site"])
    age = pd.to_numeric(data["Age"], errors="raise")
    labels = pd.Series(index=data.index, dtype=object)
    for site_name, indices in data.groupby(site).groups.items():
        bins = pd.qcut(age.loc[indices], q=4, labels=False, duplicates="drop")
        labels.loc[indices] = site_name + "_ageq" + bins.astype(str)
    counts = labels.value_counts()
    if (counts < required_splits).any():
        raise RuntimeError(
            f"Site-age strata too small for {required_splits}-fold CV: "
            f"{counts[counts < required_splits].to_dict()}"
        )
    return labels.to_numpy()


def candidates() -> list[dict[str, float]]:
    return [
        {"C": float(c), "gamma": float(g), "epsilon": float(e)}
        for c, g, e in product(GRID["C"], GRID["gamma"], GRID["epsilon"])
    ]


def key(params: dict[str, float]) -> str:
    return f"{params['C']:g}|{params['gamma']:g}|{params['epsilon']:g}"


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.einsum("ij,ij->i", left, left)[:, None]
    right_norm = np.einsum("ij,ij->i", right, right)[None, :]
    distances = left_norm + right_norm - 2.0 * (left @ right.T)
    np.maximum(distances, 0.0, out=distances)
    return distances


def predict_precomputed(
    kernel_fit: np.ndarray,
    age_fit: np.ndarray,
    kernel_target: np.ndarray,
    c_value: float,
    epsilon: float,
) -> np.ndarray:
    model = SVR(
        kernel="precomputed", C=c_value, epsilon=epsilon, cache_size=1000
    )
    model.fit(kernel_fit, age_fit)
    return model.predict(kernel_target)


def fold_predictions(
    core,
    age_fit: np.ndarray,
    x_fit: np.ndarray,
    x_target: np.ndarray,
    parameter_list: list[dict[str, float]],
    use_augmentation: bool,
    seed_base: int,
) -> dict[str, np.ndarray]:
    """Fit every candidate and optionally average 20 training-only GAP120 fits."""
    output = {
        key(item): np.zeros(len(x_target), dtype=float) for item in parameter_list
    }
    seeds = (
        [RANDOM_STATE + seed_base * 1000 + index for index in range(N_AUGMENTATION_MODELS)]
        if use_augmentation
        else [None]
    )
    for seed in seeds:
        if seed is None:
            model_age, model_x = age_fit, x_fit
        else:
            model_age, model_x = core.augment_gap_real_parents(
                age_fit, x_fit, int(seed)
            )
        distance_fit = squared_distances(model_x, model_x)
        distance_target = squared_distances(x_target, model_x)
        for gamma in sorted({item["gamma"] for item in parameter_list}):
            selected = [item for item in parameter_list if item["gamma"] == gamma]
            kernel_fit = np.exp(-gamma * distance_fit)
            kernel_target = np.exp(-gamma * distance_target)
            predictions = Parallel(n_jobs=N_JOBS, prefer="threads")(
                delayed(predict_precomputed)(
                    kernel_fit,
                    model_age,
                    kernel_target,
                    item["C"],
                    item["epsilon"],
                )
                for item in selected
            )
            for item, prediction in zip(selected, predictions, strict=True):
                output[key(item)] += prediction
    return {candidate_key: value / len(seeds) for candidate_key, value in output.items()}


def grouped_mae(
    age: np.ndarray, prediction: np.ndarray, labels: np.ndarray
) -> list[float]:
    return [
        float(mean_absolute_error(age[labels == level], prediction[labels == level]))
        for level in np.unique(labels)
    ]


def compute_selection(
    core,
    training: pd.DataFrame,
    features: list[str],
    selection_seed: int,
    use_augmentation: bool,
) -> pd.DataFrame:
    """Repeated inner CV using only the supplied training partition."""
    parameter_list = candidates()
    by_key = {key(item): item for item in parameter_list}
    age = pd.to_numeric(training["Age"], errors="raise").to_numpy(float)
    site = core.normalize_text(training["site"]).to_numpy()
    strata = site_age_strata(core, training, N_INNER_SPLITS)
    repeat_rows: list[dict[str, float | int]] = []

    for inner_repeat in range(N_INNER_REPEATS):
        splitter = StratifiedKFold(
            n_splits=N_INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + selection_seed * 100 + inner_repeat,
        )
        oof = {
            candidate_key: np.full(len(training), np.nan, dtype=float)
            for candidate_key in by_key
        }
        for inner_fold, (fit_index, validation_index) in enumerate(
            splitter.split(np.arange(len(training)), strata), start=1
        ):
            fit = training.iloc[fit_index].copy()
            validation = training.iloc[validation_index].copy()
            age_fit, _, x_fit, x_validation, _ = core.prepare_features(
                fit, validation, features
            )
            predictions = fold_predictions(
                core,
                age_fit,
                x_fit,
                x_validation,
                parameter_list,
                use_augmentation,
                seed_base=(
                    selection_seed * N_INNER_REPEATS * N_INNER_SPLITS
                    + inner_repeat * N_INNER_SPLITS
                    + inner_fold
                ),
            )
            for candidate_key, prediction in predictions.items():
                oof[candidate_key][validation_index] = prediction
        for candidate_key, prediction in oof.items():
            if np.isnan(prediction).any():
                raise RuntimeError(f"Incomplete inner OOF predictions: {candidate_key}")
            params = by_key[candidate_key]
            site_maes = grouped_mae(age, prediction, site)
            site_age_maes = grouped_mae(age, prediction, strata)
            repeat_rows.append(
                {
                    "inner_repeat": inner_repeat + 1,
                    **params,
                    "global_mae": float(mean_absolute_error(age, prediction)),
                    "mean_site_mae": float(np.mean(site_maes)),
                    "max_site_age_mae": float(np.max(site_age_maes)),
                }
            )
        print(
            f"selection {selection_seed}: inner repeat "
            f"{inner_repeat + 1}/{N_INNER_REPEATS}",
            flush=True,
        )

    results = pd.DataFrame(repeat_rows)
    summary = (
        results.groupby(["C", "gamma", "epsilon"])[
            ["global_mae", "mean_site_mae", "max_site_age_mae"]
        ]
        .agg(["mean", "std", "count"])
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary["max_site_age_mae_se_repeats"] = (
        summary["max_site_age_mae_std"].fillna(0.0)
        / np.sqrt(summary["max_site_age_mae_count"])
    )
    best = summary.loc[summary["max_site_age_mae_mean"].idxmin()]
    threshold = float(
        best["max_site_age_mae_mean"] + best["max_site_age_mae_se_repeats"]
    )
    summary["within_one_se_minimax"] = (
        summary["max_site_age_mae_mean"] <= threshold
    )
    summary["one_se_threshold"] = threshold
    return summary.sort_values(
        ["within_one_se_minimax", "max_site_age_mae_mean"],
        ascending=[False, True],
    ).reset_index(drop=True)


def ensemble_spec(table: pd.DataFrame) -> tuple[list[dict[str, float]], np.ndarray]:
    eligible = table.loc[table["within_one_se_minimax"].astype(bool)].copy()
    if eligible.empty:
        raise RuntimeError("No candidate satisfies the one-SE rule")
    eligible = eligible.sort_values(["C", "gamma", "epsilon"])
    inverse_error = 1.0 / eligible["mean_site_mae_mean"].to_numpy(float)
    weights = inverse_error / inverse_error.sum()
    params = [
        {"C": float(row.C), "gamma": float(row.gamma), "epsilon": float(row.epsilon)}
        for row in eligible.itertuples(index=False)
    ]
    return params, weights


def fit_ensemble(
    core,
    age_fit: np.ndarray,
    x_fit: np.ndarray,
    x_target: np.ndarray,
    params: list[dict[str, float]],
    weights: np.ndarray,
    use_augmentation: bool,
    seed_base: int,
) -> tuple[np.ndarray, np.ndarray]:
    prediction_by_model: list[np.ndarray] = []
    for item in params:
        seeds = (
            [
                RANDOM_STATE
                + seed_base * 1000
                + index
                for index in range(N_AUGMENTATION_MODELS)
            ]
            if use_augmentation
            else [None]
        )
        seed_predictions = []
        for seed in seeds:
            if seed is None:
                model_age, model_x = age_fit, x_fit
            else:
                model_age, model_x = core.augment_gap_real_parents(
                    age_fit, x_fit, int(seed)
                )
            model = SVR(kernel="rbf", cache_size=1000, **item)
            model.fit(model_x, model_age)
            seed_predictions.append(model.predict(x_target))
        prediction_by_model.append(np.mean(seed_predictions, axis=0))
    matrix = np.vstack(prediction_by_model)
    return weights @ matrix, matrix


def compute_and_save_selection_table(
    core,
    fit: pd.DataFrame,
    features: list[str],
    selection_seed: int,
    destination: Path,
    use_augmentation: bool,
) -> pd.DataFrame:
    table = compute_selection(
        core, fit, features, selection_seed, use_augmentation
    )
    table.to_csv(destination, index=False)
    return table


def nested_cv(
    core,
    train: pd.DataFrame,
    features: list[str],
    out: Path,
    use_augmentation: bool,
) -> np.ndarray:
    out.mkdir(parents=True, exist_ok=True)
    table_dir = out / "inner_selection_tables"
    table_dir.mkdir(exist_ok=True)
    age = pd.to_numeric(train["Age"], errors="raise").to_numpy(float)
    site = core.normalize_text(train["site"]).to_numpy()
    strata = site_age_strata(core, train, N_OUTER_SPLITS)
    splitter = RepeatedStratifiedKFold(
        n_splits=N_OUTER_SPLITS,
        n_repeats=N_OUTER_REPEATS,
        random_state=RANDOM_STATE,
    )
    predictions = np.full((N_OUTER_REPEATS, len(train)), np.nan, dtype=float)
    fold_rows: list[dict[str, float | int]] = []
    weight_rows: list[dict[str, float | int]] = []

    for outer_index, (fit_index, validation_index) in enumerate(
        splitter.split(np.arange(len(train)), strata), start=1
    ):
        repeat = (outer_index - 1) // N_OUTER_SPLITS
        fold = (outer_index - 1) % N_OUTER_SPLITS + 1
        fit = train.iloc[fit_index].copy()
        validation = train.iloc[validation_index].copy()
        destination = table_dir / f"outer_{outer_index:02d}.csv"
        table = compute_and_save_selection_table(
            core,
            fit,
            features,
            outer_index,
            destination,
            use_augmentation,
        )
        params, weights = ensemble_spec(table)
        age_fit, age_validation, x_fit, x_validation, _ = core.prepare_features(
            fit, validation, features
        )
        prediction, _ = fit_ensemble(
            core,
            age_fit,
            x_fit,
            x_validation,
            params,
            weights,
            use_augmentation,
            seed_base=100_000 + outer_index,
        )
        predictions[repeat, validation_index] = prediction
        metric = core.performance(age_validation, prediction)
        fold_rows.append(
            {
                "repeat": repeat + 1,
                "fold": fold,
                "n_fit": len(fit),
                "n_validation": len(validation),
                "n_eligible_models": len(params),
                **metric,
            }
        )
        for item, weight in zip(params, weights, strict=True):
            weight_rows.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    **item,
                    "ensemble_weight": float(weight),
                }
            )
        print(
            f"outer {outer_index}/{N_OUTER_SPLITS * N_OUTER_REPEATS}: "
            f"{len(params)} eligible models; r={metric['r']:.3f}, "
            f"MAE={metric['mae']:.2f}",
            flush=True,
        )

    if np.isnan(predictions).any():
        raise RuntimeError("Nested OOF prediction matrix is incomplete")
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for repeat in range(N_OUTER_REPEATS):
        current = predictions[repeat]
        for index in range(len(train)):
            prediction_rows.append(
                {
                    "repeat": repeat + 1,
                    "subject_ID": train.iloc[index]["subject_ID"],
                    "site": site[index],
                    "age": age[index],
                    "predicted_age_raw": current[index],
                    "bag_raw": current[index] - age[index],
                }
            )
        subsets = [("pooled", np.ones(len(train), dtype=bool))]
        subsets.extend(
            (level, site == level) for level in sorted(np.unique(site))
        )
        for label, mask in subsets:
            metric_rows.append(
                {
                    "repeat": repeat + 1,
                    "subset": label,
                    "n": int(mask.sum()),
                    **core.performance(age[mask], current[mask]),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby(["subset", "n"], as_index=False)
        .agg(
            r_mean=("r", "mean"),
            r_sd=("r", "std"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
        )
        .sort_values("subset")
    )
    pd.DataFrame(fold_rows).to_csv(out / "outer_fold_metrics.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(out / "outer_ensemble_weights.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        out / "nested_oof_predictions_by_repeat.csv", index=False
    )
    metrics.to_csv(out / "metrics_by_repeat.csv", index=False)
    summary.to_csv(out / "metrics_mean_sd_across_repeats.csv", index=False)
    np.save(out / "nested_oof_predictions.npy", predictions)
    print("\nNESTED CV RAW PERFORMANCE")
    print(summary.to_string(index=False))
    return predictions


def full_selection(
    core,
    train: pd.DataFrame,
    features: list[str],
    out: Path,
    use_augmentation: bool,
) -> tuple[list[dict[str, float]], np.ndarray, Path]:
    out.mkdir(parents=True, exist_ok=True)
    table_path = out / "full_training_selection_table.csv"
    table = compute_and_save_selection_table(
        core, train, features, 0, table_path, use_augmentation
    )
    params, weights = ensemble_spec(table)
    rows = []
    for item, weight in zip(params, weights, strict=True):
        rows.append({**item, "ensemble_weight": float(weight)})
    pd.DataFrame(rows).to_csv(out / "eligible_models_and_weights.csv", index=False)
    return params, weights, table_path


def run_heldout(
    core,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    nested_predictions: np.ndarray,
    params: list[dict[str, float]],
    weights: np.ndarray,
    out: Path,
    use_augmentation: bool,
) -> None:
    core.check_no_overlap(train, test, features)
    age_train, age_test, x_train, x_test, residualization = core.prepare_features(
        train, test, features
    )
    prediction, prediction_matrix = fit_ensemble(
        core,
        age_train,
        x_train,
        x_test,
        params,
        weights,
        use_augmentation,
        seed_base=900_000,
    )
    oof_mean = nested_predictions.mean(axis=0)
    slope, intercept, oof_corrected = core.fit_additive_age_bias_correction(
        age_train, oof_mean
    )
    prediction_corrected = prediction - (slope * age_test + intercept)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "predictions_by_eligible_model.npy", prediction_matrix)
    pd.DataFrame(
        {
            "subject_ID": train["subject_ID"],
            "site": core.normalize_text(train["site"]),
            "age": age_train,
            "predicted_age_oof_raw": oof_mean,
            "bag_oof_raw": oof_mean - age_train,
            "predicted_age_oof_corrected": oof_corrected,
            "bag_oof_corrected": oof_corrected - age_train,
        }
    ).to_csv(out / "training_age_bias_oof_predictions.csv", index=False)
    pd.DataFrame(
        {
            "subject_ID": test["subject_ID"],
            "site": core.normalize_text(test["site"]),
            "age": age_test,
            "predicted_age_raw": prediction,
            "bag_raw": prediction - age_test,
            "predicted_age_corrected": prediction_corrected,
            "bag_corrected": prediction_corrected - age_test,
        }
    ).to_csv(out / "heldout_predictions.csv", index=False)
    metrics = core.heldout_metrics(
        test,
        age_test,
        {"raw": prediction, "age_bias_corrected": prediction_corrected},
    )
    training_median = float(np.median(age_train))
    metrics["training_median_age"] = training_median
    metrics["baseline_mae"] = [
        float(
            np.mean(
                np.abs(
                    age_test[
                        np.ones(len(test), dtype=bool)
                        if subset == "pooled"
                        else core.normalize_text(test["site"]).eq(subset).to_numpy()
                    ]
                    - training_median
                )
            )
        )
        for subset in metrics["subset"]
    ]
    metrics["mae_improvement_vs_baseline_pct"] = (
        100.0 * (metrics["baseline_mae"] - metrics["mae"])
        / metrics["baseline_mae"]
    )
    raw_metrics = metrics.loc[metrics["variant"].eq("raw")].copy()
    corrected_metrics = metrics.loc[
        metrics["variant"].eq("age_bias_corrected")
    ].copy()
    # Keep the conventional filename predictive-only. Corrected results are
    # separated because chronological age is used after prediction.
    raw_metrics.to_csv(out / "heldout_performance.csv", index=False)
    raw_metrics.to_csv(out / "heldout_performance_raw.csv", index=False)
    corrected_metrics.to_csv(
        out / "heldout_performance_age_bias_corrected_descriptive.csv",
        index=False,
    )
    core.raw_metrics_by_site_age_band(test, age_test, prediction).to_csv(
        out / "heldout_performance_raw_by_site_age_band.csv", index=False
    )
    metadata = {
        "augmentation": {
            "enabled": use_augmentation,
            "age_range": list(AUGMENTATION_AGE_RANGE),
            "synthetic_rows_per_model": N_AUGMENTED_ROWS if use_augmentation else 0,
            "models_per_candidate": N_AUGMENTATION_MODELS if use_augmentation else 1,
            "parents": "real rows from the current fit partition only",
        },
        "ensemble_rule": (
            "all candidates within one estimated SE of the repeated-inner-CV "
            "minimum worst site-by-age MAE"
        ),
        "weight_rule": "normalized inverse repeated inner-OOF mean-site MAE",
        "n_models": len(params),
        "site_residualization": residualization,
        "age_bias_correction": {
            "model": "bag_oof_raw = intercept + slope * chronological_age",
            "slope": slope,
            "intercept": intercept,
            "fit_sample": "nested OOF predictions from training only",
            "heldout_true_age_used_after_prediction": True,
            "interpretation": "descriptive; raw metrics are predictive evidence",
        },
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nINDEPENDENT HELD-OUT RAW PREDICTIVE PERFORMANCE")
    print(raw_metrics.to_string(index=False))
    print("\nAGE-BIAS-CORRECTED DESCRIPTIVE METRICS")
    print(corrected_metrics.to_string(index=False))


def main() -> None:
    global N_OUTER_SPLITS, N_OUTER_REPEATS, N_INNER_SPLITS, N_INNER_REPEATS, N_JOBS, GRID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast minimal verification with reduced folds and grid",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS,
        help=f"Number of parallel worker processes (default: {N_JOBS})",
    )
    args = parser.parse_args()

    if args.n_jobs is not None:
        N_JOBS = args.n_jobs
    if args.smoke_test:
        print("[SMOKE TEST] Running fast minimal pipeline verification...")
        N_OUTER_SPLITS = 2
        N_OUTER_REPEATS = 1
        N_INNER_SPLITS = 2
        N_INNER_REPEATS = 1
        GRID = {
            "C": (100.0,),
            "gamma": (0.1,),
            "epsilon": (0.1,),
        }

    core = load_core()
    use_augmentation = False
    train, features = core.load_training(args.train)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    nested_predictions = nested_cv(
        core,
        train,
        features,
        out / "cross_validation",
        use_augmentation,
    )
    params, weights, selection_table = full_selection(
        core,
        train,
        features,
        out / "full_training_selection",
        use_augmentation,
    )

    # The held-out file is deliberately loaded only after training-only selection.
    test = core.load_test(args.test, features)
    odq_rows = odq_descriptive_summary(train, "training")
    odq_rows.extend(odq_descriptive_summary(test, "heldout"))
    pd.DataFrame(odq_rows).to_csv(
        out / "odq_descriptive_summary.csv", index=False
    )
    run_heldout(
        core,
        train,
        test,
        features,
        nested_predictions,
        params,
        weights,
        out / "heldout",
        use_augmentation,
    )
    provenance = {
        "training_file_sha256": file_sha256(args.train),
        "heldout_file_sha256": file_sha256(args.test),
        "training_model_input_sha256": core.model_input_fingerprint(train, features),
        "heldout_model_input_sha256": core.model_input_fingerprint(test, features),
        "script_sha256": file_sha256(Path(__file__)),
        "selection_table_sha256": file_sha256(selection_table),
        "configuration": {
            "augmentation": {"enabled": False},
            "outer_folds": N_OUTER_SPLITS,
            "outer_repeats": N_OUTER_REPEATS,
            "inner_folds": N_INNER_SPLITS,
            "inner_repeats": N_INNER_REPEATS,
            "grid": {name: list(values) for name, values in GRID.items()},
            "selection": "one-SE minimax site-by-age",
            "weights": "inverse repeated-inner-OOF mean-site MAE",
            "scanner_harmonization": {
                "method": "ComBat empirical Bayes location and scale",
                "batch": "acquisition scanner: CamCAN+BCBL=Elekta; OMEGA=other",
                "preserved_covariates": ["sex"],
                "age_preserved": False,
                "fit_scope": "current fitting partition only",
            },
            "site_residualization": "fit partition only",
            "selection_cache": "disabled; every inner grid is recomputed",
            "odq_policy": (
                "descriptive quality-control measure only; not a predictor and "
                "not used for exclusion because no a priori cutoff was specified"
            ),
        },
    }
    (out / "artifact_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
