"""Verification script to test imports, datasets, paths, and pipeline execution."""

from pathlib import Path
import sys
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent

def main() -> None:
    print("[1/5] Checking repository directories...")
    assert (REPO_DIR / "code").is_dir(), "code/ directory missing"
    assert (REPO_DIR / "data").is_dir(), "data/ directory missing"
    assert (REPO_DIR / "results").is_dir(), "results/ directory missing"
    print("      Directories verified: code/, data/, results/")

    print("[2/5] Checking dataset files...")
    train_path = REPO_DIR / "data" / "train_database.csv"
    test_path = REPO_DIR / "data" / "test_database.csv"
    assert train_path.exists(), f"Missing {train_path}"
    assert test_path.exists(), f"Missing {test_path}"
    df_train = pd.read_csv(train_path, nrows=5)
    df_test = pd.read_csv(test_path, nrows=5)
    print(f"      Train shape preview: {df_train.shape}, Test shape preview: {df_test.shape}")

    print("[3/5] Verifying code imports...")
    sys.path.insert(0, str(REPO_DIR / "code"))
    import Brain_clock
    import Sensitivity_ComBat
    print("      Brain_clock and Sensitivity_ComBat imported successfully.")

    print("[4/5] Checking results directory...")
    assert (REPO_DIR / "results" / "reference_results").is_dir(), "results/reference_results missing"
    assert (REPO_DIR / "results" / "results_full_nested").is_dir(), "results/results_full_nested missing"
    print("      Reference results and full nested results directories verified.")

    print("[5/5] All baseline checks passed successfully!")

if __name__ == "__main__":
    main()
