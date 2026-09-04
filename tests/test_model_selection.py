from __future__ import annotations

from src.retrain_model import candidate_is_better


def test_candidate_with_higher_recall_is_selected() -> None:

    current_metrics = {
        "accuracy": 0.98,
        "precision": 0.98,
        "recall": 0.95,
        "f1": 0.96,
        "roc_auc": 0.99,
    }

    candidate_metrics = {
        "accuracy": 0.97,
        "precision": 0.96,
        "recall": 0.97,
        "f1": 0.96,
        "roc_auc": 0.99,
    }

    assert (
        candidate_is_better(
            current_metrics=current_metrics,
            candidate_metrics=candidate_metrics,
        )
        is True
    )


def test_candidate_with_lower_recall_is_rejected() -> None:

    current_metrics = {
        "accuracy": 0.98,
        "precision": 0.98,
        "recall": 0.98,
        "f1": 0.97,
        "roc_auc": 0.99,
    }

    candidate_metrics = {
        "accuracy": 0.99,
        "precision": 1.00,
        "recall": 0.95,
        "f1": 0.97,
        "roc_auc": 0.995,
    }

    assert (
        candidate_is_better(
            current_metrics=current_metrics,
            candidate_metrics=candidate_metrics,
        )
        is False
    )


def test_f1_breaks_recall_tie() -> None:

    current_metrics = {
        "accuracy": 0.97,
        "precision": 0.95,
        "recall": 0.97,
        "f1": 0.95,
        "roc_auc": 0.99,
    }

    candidate_metrics = {
        "accuracy": 0.98,
        "precision": 0.96,
        "recall": 0.97,
        "f1": 0.97,
        "roc_auc": 0.99,
    }

    assert (
        candidate_is_better(
            current_metrics=current_metrics,
            candidate_metrics=candidate_metrics,
        )
        is True
    )


def test_roc_auc_breaks_recall_and_f1_tie() -> None:

    current_metrics = {
        "accuracy": 0.98,
        "precision": 0.97,
        "recall": 0.97,
        "f1": 0.97,
        "roc_auc": 0.990,
    }

    candidate_metrics = {
        "accuracy": 0.98,
        "precision": 0.97,
        "recall": 0.97,
        "f1": 0.97,
        "roc_auc": 0.995,
    }

    assert (
        candidate_is_better(
            current_metrics=current_metrics,
            candidate_metrics=candidate_metrics,
        )
        is True
    )


def test_equal_candidate_is_not_selected() -> None:

    current_metrics = {
        "accuracy": 0.98,
        "precision": 0.97,
        "recall": 0.97,
        "f1": 0.97,
        "roc_auc": 0.995,
    }

    candidate_metrics = current_metrics.copy()

    assert (
        candidate_is_better(
            current_metrics=current_metrics,
            candidate_metrics=candidate_metrics,
        )
        is False
    )