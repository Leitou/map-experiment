from sys import exit

import numpy as np
import torch
from tabulate import tabulate

from copy import deepcopy
from custom_types import Behavior, ModelArchitecture, Scaler
from data_handler import DataHandler
from devices import Participant, Server
from utils import select_federation_composition, get_sampling_per_device, calculate_metrics, \
    get_confusion_matrix_vals_in_percent

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    print(f'GPU available: {torch.cuda.is_available()}')
    print("Starting demo experiment: Federated vs Centralized Binary Classification\n"
          "Training on a range of attacks and testing for each attack how well the joint model performs.\n")

    # define collective experiment config:
    # TODO: take care not to exceed available data too much
    participants_per_arch = [1, 1, 0, 1]
    normals = [(Behavior.NORMAL, 1000)]
    # attacks = [val for val in Behavior if val not in [Behavior.NORMAL, Behavior.NORMAL_V2]]
    attacks = [Behavior.DELAY, Behavior.DISORDER, Behavior.FREEZE]
    val_percentage = 0.1
    train_attack_frac = 1 / len(attacks) if len(normals) == 1 else 2 / len(attacks)  # enforce balancing per device
    num_att_test_samples = 100

    train_devices, test_devices = select_federation_composition(participants_per_arch, normals, attacks, val_percentage,
                                                                train_attack_frac, num_att_test_samples)
    print("Training devices:", len(train_devices))
    print(train_devices)
    print("Testing devices:", len(test_devices))
    print(test_devices)

    incl_test = False
    incl_train = True
    incl_val = False
    print("Number of samples used per device type:", "\nincl. test samples - ", incl_test, "\nincl. val samples -",
          incl_val, "\nincl. train samples -", incl_train)
    sample_requirements = get_sampling_per_device(train_devices, test_devices, incl_train, incl_val, incl_test)
    print(tabulate(sample_requirements, headers=["device"] + [val.value for val in Behavior] + ["Normal/Attack"],
                   tablefmt="pretty"))

    print("Train Federation")
    train_sets, test_sets = DataHandler.get_all_clients_data(train_devices, test_devices)
    train_sets_fed, test_sets_fed = deepcopy(train_sets), deepcopy(test_sets)
    train_sets_fed, test_sets_fed = DataHandler.scale(train_sets_fed, test_sets_fed, scaling=Scaler.STANDARD_SCALER)

    participants = [Participant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                    x_train, y_train, x_valid, y_valid in train_sets_fed]
    server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
    server.train_global_model(aggregation_rounds=5)
    print()

    print("Train Centralized")
    x_train_all = np.concatenate(tuple(x_train for x_train, y_train, x_valid, y_valid in train_sets))
    y_train_all = np.concatenate(tuple(y_train for x_train, y_train, x_valid, y_valid in train_sets))
    x_valid_all = np.concatenate(tuple(x_valid for x_train, y_train, x_valid, y_valid in train_sets))
    y_valid_all = np.concatenate(tuple(y_valid for x_train, y_train, x_valid, y_valid in train_sets))
    train_set_cen = [(x_train_all, y_train_all, x_valid_all, y_valid_all)]
    train_set_cen, test_sets_cen = DataHandler.scale(train_set_cen, test_sets, central=True)
    central_participants = [Participant(train_set_cen[0][0], train_set_cen[0][1],
                                        train_set_cen[0][2], train_set_cen[0][3], batch_size_valid=1)]
    central_server = Server(central_participants, ModelArchitecture.MLP_MONO_CLASS)
    central_server.train_global_model(aggregation_rounds=5)
    # print_experiment_scores(y_test.flatten(), y_predicted_central.flatten().numpy(), federated=False)

    results, central_results = [], []
    for i, (tfed, tcen) in enumerate(zip(test_sets_fed, test_sets_cen)):
        y_predicted = server.predict_using_global_model(tfed[0])
        y_predicted_central = central_server.predict_using_global_model(tcen[0])
        behavior = list(Behavior)[i % len(Behavior)].value
        normal = normals[0][0].value if len(normals) == 1 else "normal/normal_v2"
        # federated results
        acc, _, conf_mat = calculate_metrics(tfed[1].flatten(), y_predicted.flatten().numpy())
        (tn, fp, fn, tp) = get_confusion_matrix_vals_in_percent(acc, conf_mat, behavior)
        results.append(
            [test_devices[i][0], normal, behavior, f'{acc * 100:.2f}%', f'{tn * 100:.2f}%', f'{fp * 100:.2f}%',
             f'{fn * 100:.2f}%', f'{tp * 100:.2f}%'])

        # centralized results
        acc, _, conf_mat = calculate_metrics(tcen[1].flatten(), y_predicted_central.flatten().numpy())
        (tn, fp, fn, tp) = get_confusion_matrix_vals_in_percent(acc, conf_mat, behavior)
        central_results.append(
            [test_devices[i][0], normal, behavior, f'{acc * 100:.2f}%', f'{tn * 100:.2f}%', f'{fp * 100:.2f}%',
             f'{fn * 100:.2f}%', f'{tp * 100:.2f}%'])

    print("Federated Results")
    print(tabulate(results, headers=['Device', 'Normal', 'Attack', 'Accuracy', 'tn', 'fp', 'fn', 'tp'],
                   tablefmt="pretty"))
    print("Centralized Results")
    print(tabulate(central_results, headers=['Device', 'Normal', 'Attack', 'Accuracy',  'tn', 'fp', 'fn', 'tp'], tablefmt="pretty"))
