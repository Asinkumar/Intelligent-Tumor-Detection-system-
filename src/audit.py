from datetime import datetime

import pandas as pd

from src.data import PROJECT_ROOT


AUDIT_DIR = PROJECT_ROOT / "reports" / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_FILE = AUDIT_DIR / "audit_trail.csv"


def log_event(case_id, event_type, details=""):

    event_record = pd.DataFrame(
        [
            {
                "Timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "Case_ID": case_id,
                "Event_Type": event_type,
                "Details": details,
            }
        ]
    )

    if AUDIT_FILE.exists():

        event_record.to_csv(
            AUDIT_FILE,
            mode="a",
            header=False,
            index=False,
        )

    else:

        event_record.to_csv(
            AUDIT_FILE,
            index=False,
        )

    print(
        f"Audit event logged: "
        f"{event_type} -> {case_id}"
    )


def view_audit_history(case_id=None):

    if not AUDIT_FILE.exists():

        print(
            "No audit history found."
        )

        return

    audit_data = pd.read_csv(
        AUDIT_FILE
    )

    if case_id:

        audit_data = audit_data[
            audit_data["Case_ID"]
            == case_id
        ]

    if audit_data.empty:

        print(
            "No audit records found."
        )

        return

    print("\n")
    print("=" * 70)
    print("AUDIT TRAIL")
    print("=" * 70)

    print(
        audit_data.to_string(
            index=False
        )
    )


if __name__ == "__main__":

    print("=" * 70)
    print("AUDIT TRAIL SYSTEM")
    print("=" * 70)

    view_audit_history()