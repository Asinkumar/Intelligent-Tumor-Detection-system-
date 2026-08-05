from src.feedback import collect_feedback
from src.predict import main as run_prediction
from src.report_generator import main as generate_report


def main():

    print("=" * 70)
    print("INTELLIGENT TUMOR DECISION SUPPORT WORKFLOW")
    print("=" * 70)

    # Step 1: Generate prediction and Case ID
    print("\nSTEP 1: RUNNING RISK PREDICTION")
    print("-" * 70)

    run_prediction()

    # Step 2: Generate clinical report for latest prediction
    print("\n\nSTEP 2: GENERATING CLINICAL REPORT")
    print("-" * 70)

    generate_report()

    # Step 3: Ask whether reviewer feedback should be collected
    print("\n\nSTEP 3: CLINICAL REVIEW")
    print("-" * 70)

    feedback_choice = input(
        "\nWould you like to add reviewer feedback now? (yes/no): "
    ).strip().lower()

    if feedback_choice in ["yes", "y"]:

        print()
        collect_feedback()

    else:

        print(
            "\nReviewer feedback skipped. "
            "It can be added later using:"
        )

        print(
            "python -m src.feedback"
        )

    # Workflow completed
    print("\n")
    print("=" * 70)
    print("WORKFLOW COMPLETED")
    print("=" * 70)

    print(
        "\nPrediction, clinical report, and audit records "
        "have been processed successfully."
    )


if __name__ == "__main__":
    main()