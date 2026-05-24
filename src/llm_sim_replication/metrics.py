from __future__ import annotations

from collections.abc import Sequence


def binary_metrics(labels: Sequence[bool], predictions: Sequence[bool]) -> dict[str, float]:
    """Accuracy, precision, recall, and F1.

    For Table 2 replication, True means "correct/valid triple". This matches
    the paper's yes/no prompt and the reported metric pattern.
    """
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length.")
    total = max(len(labels), 1)
    tp = sum(label and pred for label, pred in zip(labels, predictions))
    tn = sum((not label) and (not pred) for label, pred in zip(labels, predictions))
    fp = sum((not label) and pred for label, pred in zip(labels, predictions))
    fn = sum(label and (not pred) for label, pred in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "positive_class": "correct/valid triple",
    }
