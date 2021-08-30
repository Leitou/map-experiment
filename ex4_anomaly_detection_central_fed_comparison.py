from sys import exit

import numpy as np
import torch
from tabulate import tabulate

from copy import deepcopy
from custom_types import Behavior, ModelArchitecture
from data_handler import DataHandler
from devices import AutoEncoderParticipant, Server
from utils import select_federation_composition, get_sampling_per_device, calculate_metrics


if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    print(f'GPU available: {torch.cuda.is_available()}')
    print("Starting demo experiment: Federated vs Centralized Anomaly Detection\n"
          "Training on a range of attacks and testing for each attack how well the joint model performs.\n")

    # define collective experiment config:
    # TODO: take care not to exceed available data too much
    participants_per_arch = [1, 1, 0, 1]
    normals = [(Behavior.NORMAL, 6000)]
    attacks = [val for val in Behavior if val not in [Behavior.NORMAL, Behavior.NORMAL_V2]]
    #attacks = [Behavior.DELAY, Behavior.DISORDER, Behavior.FREEZE]
    val_percentage = 0.1
    train_attack_frac = 1 / len(attacks)  # enforce balancing per device
    nnorm_test_samples = 0
    natt_test_samples = 250

    train_devices, test_devices = select_federation_composition(participants_per_arch, normals, attacks, val_percentage,
                                                                train_attack_frac, nnorm_test_samples,
                                                                natt_test_samples, is_anomaly=True)
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
    train_sets_fed, test_sets_fed = DataHandler.scale(train_sets_fed, test_sets_fed)

    participants = [AutoEncoderParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                    x_train, y_train, x_valid, y_valid in train_sets_fed]
    server = Server(participants, ModelArchitecture.AUTO_ENCODER)
    server.train_global_model(aggregation_rounds=5)
    print()

    print("Train Centralized")
    x_train_all = np.concatenate(tuple(x_train for x_train, y_train, x_valid, y_valid in train_sets))
    y_train_all = np.concatenate(tuple(y_train for x_train, y_train, x_valid, y_valid in train_sets))
    x_valid_all = np.concatenate(tuple(x_valid for x_train, y_train, x_valid, y_valid in train_sets))
    y_valid_all = np.concatenate(tuple(y_valid for x_train, y_train, x_valid, y_valid in train_sets))
    train_set_cen = [(x_train_all, y_train_all, x_valid_all, y_valid_all)]
    train_set_cen, test_sets_cen = DataHandler.scale(train_set_cen, test_sets, central=True)
    central_participant = [AutoEncoderParticipant(train_set_cen[0][0], train_set_cen[0][1],
                                        train_set_cen[0][2], train_set_cen[0][3], batch_size_valid=1)]
    central_server = Server(central_participant, ModelArchitecture.AUTO_ENCODER)
    central_server.train_global_model(aggregation_rounds=5)
    # print_experiment_scores(y_test.flatten(), y_predicted_central.flatten().numpy(), federated=False)

    results, central_results = [], []
    for i, (tfed, tcen) in enumerate(zip(test_sets_fed, test_sets_cen)):
        y_predicted = server.predict_using_global_model(tfed[0])
        y_predicted_central = central_server.predict_using_global_model(tcen[0])
        attack = list(set(test_devices[i][1].keys()) - {Behavior.NORMAL, Behavior.NORMAL_V2})[0].value
        normal = list(
            set(test_devices[i][1].keys()) - {Behavior.DELAY, Behavior.DISORDER, Behavior.FREEZE, Behavior.HOP,
                                              Behavior.MIMIC,
                                              Behavior.NOISE, Behavior.REPEAT, Behavior.SPOOF})[0].value
        # federated results
        acc, f1, _ = calculate_metrics(tfed[1].flatten(), y_predicted.flatten().numpy())
        results.append([test_devices[i][0], normal, attack, f'{acc * 100:.2f}%', f'{f1 * 100:.2f}%'])
        # centralized results
        acc, f1, _ = calculate_metrics(tcen[1].flatten(), y_predicted_central.flatten().numpy())
        central_results.append([test_devices[i][0], normal, attack, f'{acc * 100:.2f}%', f'{f1 * 100:.2f}%'])

    print("Federated Results")
    print(tabulate(results, headers=['Device', 'Normal', 'Attack', 'Accuracy', 'F1 score'], tablefmt="pretty"))
    print("Centralized Results")
    print(tabulate(central_results, headers=['Device', 'Normal', 'Attack', 'Accuracy', 'F1 score'], tablefmt="pretty"))
