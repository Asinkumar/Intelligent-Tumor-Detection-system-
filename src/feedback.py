from datetime import datetime

import pandas as pd

from src.audit import log_event
from src.data import PROJECT_ROOT


PREDICTION_DIR = PROJECT_ROOT / "reports" / "predictions"
FEEDBACK_DIR = PROJECT_ROOT / "reports" / "feedback"

FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

FEEDBACK_FILE = FEEDBACK_DIR / "doctor_feedback.csv"


def get_latest_prediction():

    prediction_files = list(
        PREDICTION_DIR.glob("*_prediction.csv")
    )

    if not prediction_files:
        return None

    latest_file = max(
        prediction_files,
        key=lambda path: path.stat().st_mtime,
    )

    return latest_file


def collect_feedback():

    print("=" * 70)
    print("DOCTOR FEEDBACK SYSTEM")
    print("=" * 70)

    # -------------------------------------------------------
    # Get latest prediction
    # -------------------------------------------------------

    prediction_file = get_latest_prediction()

    if prediction_file is None:

        print("\nNo prediction records found.")
        print("Run python -m src.predict first.")

        return

    # -------------------------------------------------------
    # Load prediction details
    # -------------------------------------------------------

    prediction_data = pd.read_csv(
        prediction_file
    )

    prediction = prediction_data.iloc[0]

    case_id = prediction["Case_ID"]

    predicted_diagnosis = prediction[
        "Predicted_Diagnosis"
    ]

    malignant_probability = float(
        prediction[
            "Malignant_Probability"
        ]
    )

    risk_level = prediction[
        "Risk_Level"
    ]

    print(
        f"\nCase ID: "
        f"{case_id}"
    )

    print(
        f"Model Prediction: "
        f"{predicted_diagnosis}"
    )

    print(
        f"Malignant Probability: "
        f"{malignant_probability:.2%}"
    )

    print(
        f"Risk Level: "
        f"{risk_level}"
    )

    # -------------------------------------------------------
    # Collect reviewer details
    # -------------------------------------------------------

    doctor_name = input(
        "\nDoctor / Reviewer Name: "
    ).strip()

    # Prevent empty reviewer name
    while not doctor_name:

        print(
            "Reviewer name cannot be empty."
        )

        doctor_name = input(
            "Doctor / Reviewer Name: "
        ).strip()

    # -------------------------------------------------------
    # Collect final clinical decision
    # -------------------------------------------------------

    final_diagnosis = input(
        "Final Clinical Decision "
        "(Benign/Malignant/Uncertain): "
    ).strip().title()

    while final_diagnosis not in [
        "Benign",
        "Malignant",
        "Uncertain",
    ]:

        print(
            "Please enter Benign, "
            "Malignant, or Uncertain."
        )

        final_diagnosis = input(
            "Final Clinical Decision: "
        ).strip().title()

    # -------------------------------------------------------
    # Collect comments
    # -------------------------------------------------------

    clinical_comments = input(
        "Clinical Comments: "
    ).strip()

    if not clinical_comments:

        clinical_comments = (
            "No additional comments provided."
        )

    # -------------------------------------------------------
    # Compare model prediction with clinical decision
    # -------------------------------------------------------

    if final_diagnosis == "Uncertain":

        agreement = "Not Applicable"

    elif final_diagnosis == predicted_diagnosis:

        agreement = "Agree"

    else:

        agreement = "Disagree"

    # -------------------------------------------------------
    # Create feedback record
    # -------------------------------------------------------

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    feedback_record = pd.DataFrame(
        [
            {
                "Timestamp": timestamp,
                "Case_ID": case_id,
                "Doctor_Name": doctor_name,
                "Model_Prediction": predicted_diagnosis,
                "Malignant_Probability": malignant_probability,
                "Risk_Level": risk_level,
                "Final_Clinical_Decision": final_diagnosis,
                "Model_Clinical_Agreement": agreement,
                "Clinical_Comments": clinical_comments,
            }
        ]
    )

    # -------------------------------------------------------
    # Save feedback history
    # -------------------------------------------------------

    if FEEDBACK_FILE.exists():

        feedback_record.to_csv(
            FEEDBACK_FILE,
            mode="a",
            header=False,
            index=False,
        )

    else:

        feedback_record.to_csv(
            FEEDBACK_FILE,
            index=False,
        )

    # -------------------------------------------------------
    # Log feedback in audit trail
    # -------------------------------------------------------

    log_event(
        case_id,
        "DOCTOR_FEEDBACK_RECEIVED",
        (
            f"Reviewer={doctor_name}; "
            f"FinalDecision={final_diagnosis}; "
            f"Agreement={agreement}"
        ),
    )

    # -------------------------------------------------------
    # Display confirmation
    # -------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FEEDBACK SAVED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Case ID: "
        f"{case_id}"
    )

    print(
        f"Reviewer: "
        f"{doctor_name}"
    )

    print(
        f"Final Clinical Decision: "
        f"{final_diagnosis}"
    )

    print(
        f"Model / Clinical Agreement: "
        f"{agreement}"
    )

    print(
        f"\nFeedback history saved to: "
        f"{FEEDBACK_FILE}"
    )


if __name__ == "__main__":
    collect_feedback()