"""Training-only sensitivity analysis for scanner ComBat harmonization.

This script tests whether the final clock's scanner harmonization 
changes age-connectivity associations in the training sample. It never reads
the held-out test set and does not write harmonized connectivity matrices.

Two diagnostic transformations are compared with the original features:

1. ComBat preserving sex only, as used by the final predictive clock.
2. ComBat preserving sex and chronological age, used here only as a
   non-predictive diagnostic reference.

Chronological age must not be supplied when transforming held-out observations
for brain-age prediction, because it is the prediction target. Consequently,
the sex+age variant below is not a candidate predictive pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationLearn
from scipy.stats import pearsonr


N_ROI = 82
RANDOM_STATE = 42
STRONG_EDGE_THRESHOLD = 0.10
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN = SCRIPT_DIR / "train_database.csv"
DEFAULT_OUT = SCRIPT_DIR / "sensitivity_combat_results.csv"


def feature_columns() -> list[str]:
    """Return the 3,321 unique off-diagonal AAL82 connectivity columns."""
    return [
        f"ROIS_{first}_vs_ROI_{second}"
        for first in range(N_ROI)
        for second in range(first + 1, N_ROI)
    ]


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_site(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower()


def scanner_labels(frame: pd.DataFrame) -> np.ndarray:
    site = normalize_site(frame["site"])
    expected = {"camcan", "omega", "bcbl"}
    unknown = sorted(set(site.unique()) - expected)
    if unknown:
        raise RuntimeError(f"Unknown sites: {unknown}")
    return np.where(site.eq("omega"), "omega_other", "elekta")


def partial_age_correlations(
    features: np.ndarray,
    age: np.ndarray,
    nuisance: np.ndarray,
) -> np.ndarray:
    """Return feature-wise partial correlations with age."""
    design = np.asarray(nuisance, dtype=float)
    feature_residuals = features - design @ np.linalg.lstsq(
        design, features, rcond=None
    )[0]
    age_residuals = age - design @ np.linalg.lstsq(design, age, rcond=None)[0]
    denominator = np.sqrt(
        np.sum(feature_residuals**2, axis=0) * np.sum(age_residuals**2)
    )
    correlations = np.full(features.shape[1], np.nan, dtype=float)
    valid = denominator > 0
    correlations[valid] = (
        age_residuals @ feature_residuals[:, valid]
    ) / denominator[valid]
    return correlations


def comparison_row(
    scope: str,
    comparison: str,
    original: np.ndarray,
    transformed: np.ndarray,
    n_subjects: int,
    input_sha256: str,
) -> dict[str, object]:
    valid = np.isfinite(original) & np.isfinite(transformed)
    strong = valid & (np.abs(original) >= STRONG_EDGE_THRESHOLD)
    return {
        "section": "age_association_preservation",
        "scope": scope,
        "comparison": comparison,
        "n_subjects": n_subjects,
        "n_features": int(valid.sum()),
        "age_mean": np.nan,
        "age_sd": np.nan,
        "age_min": np.nan,
        "age_max": np.nan,
        "scanner_age_smd": np.nan,
        "map_pearson_r": float(pearsonr(original[valid], transformed[valid]).statistic),
        "median_abs_age_r_original": float(np.median(np.abs(original[valid]))),
        "median_abs_age_r_transformed": float(
            np.median(np.abs(transformed[valid]))
        ),
        "sign_retention_all": float(
            np.mean(np.sign(original[valid]) == np.sign(transformed[valid]))
        ),
        "strong_edge_abs_r_threshold": STRONG_EDGE_THRESHOLD,
        "n_strong_edges": int(strong.sum()),
        "sign_retention_strong_edges": float(
            np.mean(np.sign(original[strong]) == np.sign(transformed[strong]))
        ),
        "train_sha256": input_sha256,
        "neuroHarmonize_version": version("neuroHarmonize"),
    }


def distribution_row(
    scanner: str,
    age: np.ndarray,
    scanner_age_smd: float,
    n_features: int,
    input_sha256: str,
) -> dict[str, object]:
    return {
        "section": "scanner_age_distribution",
        "scope": scanner,
        "comparison": "not_applicable",
        "n_subjects": len(age),
        "n_features": n_features,
        "age_mean": float(np.mean(age)),
        "age_sd": float(np.std(age, ddof=1)),
        "age_min": float(np.min(age)),
        "age_max": float(np.max(age)),
        "scanner_age_smd": scanner_age_smd,
        "map_pearson_r": np.nan,
        "median_abs_age_r_original": np.nan,
        "median_abs_age_r_transformed": np.nan,
        "sign_retention_all": np.nan,
        "strong_edge_abs_r_threshold": STRONG_EDGE_THRESHOLD,
        "n_strong_edges": np.nan,
        "sign_retention_strong_edges": np.nan,
        "train_sha256": input_sha256,
        "neuroHarmonize_version": version("neuroHarmonize"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    features = feature_columns()
    required = ["Age", "Sex", "site", *features]
    frame = pd.read_csv(args.train, usecols=required, low_memory=False)
    age = pd.to_numeric(frame["Age"], errors="raise").to_numpy(float)
    sex = pd.to_numeric(frame["Sex"], errors="raise").to_numpy(float)
    values = frame[features].to_numpy(float)
    scanner = scanner_labels(frame)

    if not np.isfinite(age).all() or not np.isfinite(sex).all():
        raise RuntimeError("Age or sex contains missing/non-finite values")
    if not np.isfinite(values).all():
        raise RuntimeError("Connectivity contains missing/non-finite values")

    sex_covariates = pd.DataFrame({"SITE": scanner, "sex": sex})
    age_covariates = pd.DataFrame(
        {"SITE": scanner, "sex": sex, "age": age}
    )
    _, combat_sex = harmonizationLearn(
        values, sex_covariates, eb=True, seed=RANDOM_STATE
    )
    _, combat_sex_age = harmonizationLearn(
        values, age_covariates, eb=True, seed=RANDOM_STATE
    )

    input_sha256 = sha256_file(args.train)
    rows: list[dict[str, object]] = []

    elekta_age = age[scanner == "elekta"]
    other_age = age[scanner == "omega_other"]
    pooled_sd = np.sqrt(
        (
            (len(elekta_age) - 1) * np.var(elekta_age, ddof=1)
            + (len(other_age) - 1) * np.var(other_age, ddof=1)
        )
        / (len(elekta_age) + len(other_age) - 2)
    )
    scanner_age_smd = float((np.mean(elekta_age) - np.mean(other_age)) / pooled_sd)
    for level in ("elekta", "omega_other"):
        mask = scanner == level
        rows.append(
            distribution_row(
                level,
                age[mask],
                scanner_age_smd,
                len(features),
                input_sha256,
            )
        )

    pooled_nuisance = np.column_stack(
        [np.ones(len(frame)), sex, (scanner == "omega_other").astype(float)]
    )
    pooled_original = partial_age_correlations(values, age, pooled_nuisance)
    pooled_sex = partial_age_correlations(combat_sex, age, pooled_nuisance)
    pooled_sex_age = partial_age_correlations(
        combat_sex_age, age, pooled_nuisance
    )
    rows.extend(
        [
            comparison_row(
                "pooled_adjusted_for_sex_and_scanner",
                "original_vs_combat_preserve_sex",
                pooled_original,
                pooled_sex,
                len(frame),
                input_sha256,
            ),
            comparison_row(
                "pooled_adjusted_for_sex_and_scanner",
                "original_vs_combat_preserve_sex_and_age_diagnostic_only",
                pooled_original,
                pooled_sex_age,
                len(frame),
                input_sha256,
            ),
            comparison_row(
                "pooled_adjusted_for_sex_and_scanner",
                "combat_preserve_sex_vs_combat_preserve_sex_and_age",
                pooled_sex,
                pooled_sex_age,
                len(frame),
                input_sha256,
            ),
        ]
    )

    for level in ("elekta", "omega_other"):
        mask = scanner == level
        nuisance = np.column_stack([np.ones(mask.sum()), sex[mask]])
        original = partial_age_correlations(values[mask], age[mask], nuisance)
        sex_only = partial_age_correlations(combat_sex[mask], age[mask], nuisance)
        sex_age = partial_age_correlations(
            combat_sex_age[mask], age[mask], nuisance
        )
        rows.extend(
            [
                comparison_row(
                    f"within_{level}_adjusted_for_sex",
                    "original_vs_combat_preserve_sex",
                    original,
                    sex_only,
                    int(mask.sum()),
                    input_sha256,
                ),
                comparison_row(
                    f"within_{level}_adjusted_for_sex",
                    "original_vs_combat_preserve_sex_and_age_diagnostic_only",
                    original,
                    sex_age,
                    int(mask.sum()),
                    input_sha256,
                ),
            ]
        )

    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False, float_format="%.12g")

    print(f"Training-only sensitivity complete: {args.out}")
    print(
        result.loc[
            result["section"].eq("age_association_preservation"),
            [
                "scope",
                "comparison",
                "map_pearson_r",
                "sign_retention_all",
                "n_strong_edges",
                "sign_retention_strong_edges",
            ],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
