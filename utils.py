import csv
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, classification_report


def print_experiment_scores(y_test, y_pred, correct, federated=True):
    if federated:
        print("\n\nResults Federated Model:")
    else:
        print("\n\nResults Centralized Model:")
    f1 = f1_score(y_test, y_pred)
    cm_fed = confusion_matrix(y_test, y_pred)  # could also extract via tn, fp, fn, tp = confusion_matrix().ravel()

    print(classification_report(y_test, y_pred, target_names=["Normal", "Infected"])
                                if len(np.unique(y_pred)) > 1
                                else "only single class predicted, no report generated")
    print(f"Details:\nConfusion matrix \n[(TN, FP),\n(FN, TP)]:\n{cm_fed}")
    print(f"Test Accuracy: {correct * 100 / len(y_pred):.2f}%, F1 score: {f1 * 100:.2f}%")
