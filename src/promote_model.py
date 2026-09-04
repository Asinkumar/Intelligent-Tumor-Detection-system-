from __future__ import annotations

import shutil
from datetime import datetime

import joblib
import mlflow
import pandas as pd

from src.data import PROJECT_ROOT


# =======================================================
# PATHS
# =======================================================

CURRENT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_best_model.pkl"
)

CURRENT_THRESHOLD_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_prediction_threshold.txt"
)

CANDIDATE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "retrained_candidate_model.pkl"
)

BACKUP_DIR = (
    PROJECT_ROOT
    / "models"
    / "backups"
)

BACKUP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROMOTION_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "promotion"
)

PROMOTION_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROMOTION_HISTORY = (
    PROMOTION_REPORT_DIR
    / "model_promotion_history.csv"
)

REGISTERED_MODEL_NAME = (
    "TumorDecisionSupportSVM"
)


# =======================================================
# BACKUP CURRENT MODEL
# =======================================================

def backup_current_model() -> str:

    if not CURRENT_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Production model not found: "
            f"{CURRENT_MODEL_PATH}"
        )

    timestamp = (
        datetime.now()
        .strftime("%Y%m%d_%H%M%S")
    )

    backup_path = (
        BACKUP_DIR
        / f"final_best_model_{timestamp}.pkl"
    )

    shutil.copy2(
        CURRENT_MODEL_PATH,
        backup_path,
    )

    if CURRENT_THRESHOLD_PATH.exists():

        threshold_backup = (
            BACKUP_DIR
            / (
                "final_prediction_threshold_"
                f"{timestamp}.txt"
            )
        )

        shutil.copy2(
            CURRENT_THRESHOLD_PATH,
            threshold_backup,
        )

    print(
        f"Production model backup created:"
    )

    print(
        backup_path
    )

    return str(
        backup_path
    )


# =======================================================
# PROMOTION HISTORY
# =======================================================

def log_promotion(
    backup_path: str,
    mlflow_run_id: str | None,
) -> None:

    record = pd.DataFrame(
        [
            {
                "Timestamp": (
                    datetime.now()
                    .isoformat(
                        timespec="seconds"
                    )
                ),
                "Previous_Model_Backup": (
                    backup_path
                ),
                "Promoted_Model": (
                    str(
                        CURRENT_MODEL_PATH
                    )
                ),
                "MLflow_Run_ID": (
                    mlflow_run_id
                    or "Not Provided"
                ),
                "Status": (
                    "PROMOTED"
                ),
            }
        ]
    )

    if PROMOTION_HISTORY.exists():

        record.to_csv(
            PROMOTION_HISTORY,
            mode="a",
            header=False,
            index=False,
        )

    else:

        record.to_csv(
            PROMOTION_HISTORY,
            index=False,
        )


# =======================================================
# PROMOTE CANDIDATE
# =======================================================

def promote_candidate(
    mlflow_run_id: str | None = None,
    test_mode: bool = False,
) -> None:

    print("=" * 70)
    print("MODEL PROMOTION")
    print("=" * 70)

    if test_mode:

        print(
            "\nTEST MODE ENABLED"
        )

        print(
            "Production model will NOT be changed."
        )

        return

    if not CANDIDATE_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Candidate model not found: "
            f"{CANDIDATE_MODEL_PATH}"
        )

    # Verify candidate can be loaded
    joblib.load(
        CANDIDATE_MODEL_PATH
    )

    print(
        "\nCandidate model validation: PASSED"
    )

    backup_path = (
        backup_current_model()
    )

    shutil.copy2(
        CANDIDATE_MODEL_PATH,
        CURRENT_MODEL_PATH,
    )

    print(
        "\nCandidate copied to production model path:"
    )

    print(
        CURRENT_MODEL_PATH
    )

    log_promotion(
        backup_path=backup_path,
        mlflow_run_id=mlflow_run_id,
    )

    print(
        "\nPromotion history updated:"
    )

    print(
        PROMOTION_HISTORY
    )

    print("\n" + "=" * 70)
    print("MODEL PROMOTED SUCCESSFULLY")
    print("=" * 70)


def main() -> None:

    # Direct execution is intentionally safe.
    # It does NOT promote automatically.
    print("=" * 70)
    print("MODEL PROMOTION MODULE")
    print("=" * 70)

    print(
        "\nThis module is intended to be called "
        "only after a candidate passes model evaluation."
    )

    print(
        "Direct execution will not modify production."
    )


if __name__ == "__main__":
    main()