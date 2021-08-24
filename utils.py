from typing import Tuple, Any, List, Dict, Union

import numpy as np
from math import floor
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from custom_types import RaspberryPi, Attack


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


# Assumption we test at most on what we train (attack types)
def select_federation_composition(participants_per_arch: List, normals: List[Tuple[Attack, int]], attacks: List[Attack],
                                  val_percentage: float, attack_frac: float, nnorm_test: int, natt_test_samples: int) \
        -> Tuple[List[Tuple[Any, Dict[Attack, Union[int, float]], Dict[Attack, Union[int, float]]]], List[
            Tuple[Any, Dict[Attack, int]]]]:
    # populate train and test_devices for
    train_devices, test_devices = [], []
    for i, num_p in enumerate(participants_per_arch):
        for p in range(num_p):

            # add all normal monitorings for the training + validation + testing per participant
            train_d, val_d, test_d = {}, {}, {}
            for normal in normals:
                train_d[normal[0]] = normal[1]
                val_d[normal[0]] = floor(normal[1] * val_percentage)
                test_d[normal[0]] = nnorm_test

            # add all attacks for training + validation per participant
            for attack in attacks:
                # TODO: add here choice whether attack is in-/excluded per device? random or determ.
                train_d[attack] = floor(normals[0][1] * attack_frac)
                val_d[attack] = floor(normals[0][1] * attack_frac * val_percentage)
                # if p == 0:  # add test set only once if we have this type normal
                #     test_d[attack] = natt_test_samples

            train_devices.append((list(RaspberryPi)[i], train_d, val_d))


            # now populate the test dictionary with all selected attacks
            for attack in attacks:
                test_dd = dict(test_d)
                test_dd[attack] = natt_test_samples
                test_devices.append((list(RaspberryPi)[i], test_dd))

    return train_devices, test_devices
