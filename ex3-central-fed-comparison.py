from sys import exit

import numpy as np
import torch
from tabulate import tabulate

from custom_types import Attack, ModelArchitecture
from data_handler import DataHandler
from devices import Participant, Server
from utils import select_federation_composition, calculate_metrics

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    print(f'GPU available: {torch.cuda.is_available()}')
    print("Starting demo experiment: Federated vs Centralized Anomaly Detection\n"
          "Training on all attacks and testing for each attack how well the joint model performs.")

    # define collective experiment config:
    # TODO: take care not to exceed available data too much
    participants_per_arch = [1, 1, 1, 1]
    normals = [(Attack.NORMAL, 2000)]  # Adapt this value if number of participants is adapted extraordinarily
    attacks = [val for val in Attack if val not in [Attack.NORMAL, Attack.NORMAL_V2]]
    val_percentage = 0.1
    attack_frac = 1 / 5
    nnorm_test = 500
    natt_test_samples = 100

    train_devices, test_devices = select_federation_composition(participants_per_arch, normals, attacks, val_percentage,
                                                                attack_frac, nnorm_test, natt_test_samples)
    print(train_devices)
    print(test_devices)

    print("Train Federation")
    train_sets, test_sets = DataHandler.get_all_clients_data(train_devices, test_devices)
    train_sets, test_sets = DataHandler.scale(train_sets, test_sets)

    participants = [Participant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                    x_train, y_train, x_valid, y_valid in train_sets]
    server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
    server.train_global_model(aggregation_rounds=5)

    # # no local inference
    # x_test, y_test = test_sets[0]
    # for x, y in test_sets[1:]:
    #     x_test = np.concatenate((x_test, x))
    #     y_test = np.concatenate((y_test, y))

    # y_predicted = server.predict_using_global_model(x_test)
    # print_experiment_scores(y_test.flatten(), y_predicted.flatten().numpy(), federated=True)

    print("Train Centralized")
    x_train_all = np.concatenate(tuple(x_train for x_train, y_train, x_valid, y_valid in train_sets))
    y_train_all = np.concatenate(tuple(y_train for x_train, y_train, x_valid, y_valid in train_sets))
    x_valid_all = np.concatenate(tuple(x_valid for x_train, y_train, x_valid, y_valid in train_sets))
    y_valid_all = np.concatenate(tuple(y_valid for x_train, y_train, x_valid, y_valid in train_sets))
    central_participants = [Participant(x_train_all, y_train_all,
                                        x_valid_all, y_valid_all, batch_size_valid=1)]
    central_server = Server(central_participants, ModelArchitecture.MLP_MONO_CLASS)
    central_server.train_global_model(aggregation_rounds=5)
    # print_experiment_scores(y_test.flatten(), y_predicted_central.flatten().numpy(), federated=False)

    results, central_results = [], []
    for i, (x_test, y_test) in enumerate(test_sets):
        y_predicted = server.predict_using_global_model(x_test)
        y_predicted_central = central_server.predict_using_global_model(x_test)
        attack = list(set(test_devices[i][1].keys()) - {Attack.NORMAL, Attack.NORMAL_V2})[0].value
        normal = list(
            set(test_devices[i][1].keys()) - {Attack.DELAY, Attack.DISORDER, Attack.FREEZE, Attack.HOP,
                                              Attack.MIMIC,
                                              Attack.NOISE, Attack.REPEAT, Attack.SPOOF})[0].value
        # federated results
        acc, f1, _ = calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
        results.append([test_devices[i][0], normal, attack, f'{acc * 100:.2f}%', f'{f1 * 100:.2f}%'])
        # centralized results
        acc, f1, _ = calculate_metrics(y_test.flatten(), y_predicted_central.flatten().numpy())
        central_results.append([test_devices[i][0], normal, attack, f'{acc * 100:.2f}%', f'{f1 * 100:.2f}%'])

    print("Federated Results")
    print(tabulate(results, headers=['Device', 'Normal', 'Attack', 'Accuracy', 'F1 score'], tablefmt="pretty"))
    print("Centralized Results")
    print(tabulate(central_results, headers=['Device', 'Normal', 'Attack', 'Accuracy', 'F1 score'], tablefmt="pretty"))
