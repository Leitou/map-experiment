from typing import Tuple, Any

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report


def calculate_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, Any]:
    correct = np.count_nonzero(y_test == y_pred)
    f1 = f1_score(y_test, y_pred)
    cm_fed = confusion_matrix(y_test, y_pred)  # could also extract via tn, fp, fn, tp = confusion_matrix().ravel()
    return correct / len(y_pred), f1, cm_fed


def print_experiment_scores(y_test: np.ndarray, y_pred: np.ndarray, federated=True):
    if federated:
        print("\n\nResults Federated Model:")
    else:
        print("\n\nResults Centralized Model:")

    accuracy, f1, cm_fed = calculate_metrics(y_test, y_pred)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Infected"])
          if len(np.unique(y_pred)) > 1
          else "only single class predicted, no report generated")
    print(f"Details:\nConfusion matrix \n[(TN, FP),\n(FN, TP)]:\n{cm_fed}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%, F1 score: {f1 * 100:.2f}%")
