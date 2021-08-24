import numpy as np
import torch
from math import floor
from custom_types import Attack, RaspberryPi, ModelArchitecture
from devices import Participant, Server
from data_handler import DataHandler
from utils import print_experiment_scores

from sys import exit

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    print(f'GPU available: {torch.cuda.is_available()}')
    print("Starting demo experiment: Federated vs Centralized Anomaly Detection")

    # define collective experiment config:
    participants_per_arch = [1,1,1,1]
    normals = [(Attack.NORMAL, 2000)] # Adapt this value if number of participants is adapted extraordinarily
    attacks = [val for val in Attack if val not in [Attack.NORMAL, Attack.NORMAL_V2]]
    val_percentage = 0.1
    attack_frac = 1 / 20 # heuristic as to how much malicious samples can max be expected per participant
    nnorm_test = 500
    natt_test = 100

    print(attacks)
    # populate train and test_devices for
    train_devices, test_devices = [], []
    for i, num_p in enumerate(participants_per_arch):
        for p in range(num_p):

            # populate the train and val dicts
            train_d, val_d, test_d = {}, {}, {}
            for normal in normals:
                train_d[normal[0]] = normal[1]
                val_d[normal[0]] = floor(normal[1] * val_percentage)
                if p == 0:  # add test set only once if we have this type normal
                   test_d[normal[0]] = nnorm_test

            for attack in attacks:
                # TODO: add here choice whether attack is in-/excluded per device? random or determ.
                train_d[attack] = floor(normal[1] * attack_frac)
                val_d[attack] = floor(normal[1] * attack_frac * val_percentage)
                if p == 0:  # add test set only once if we have this type normal
                   test_d[attack] = natt_test

            train_devices.append((list(RaspberryPi)[i], train_d, val_d))
        test_devices.append((list(RaspberryPi)[i], test_d))

    train_sets, test_sets = DataHandler.get_all_clients_data(train_devices, test_devices)
    train_sets, test_sets = DataHandler.scale(train_sets, test_sets)

    participants = [Participant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                    x_train, y_train, x_valid, y_valid in train_sets]
    server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
    server.train_global_model(aggregation_rounds=5)
    # no local inference
    x_test, y_test = test_sets[0]
    for x, y in test_sets[1:]:
        x_test = np.concatenate((x_test, x))
        y_test = np.concatenate((y_test, y))

    y_predicted = server.predict_using_global_model(x_test)
    print_experiment_scores(y_test.flatten(), y_predicted.flatten().numpy(), federated=True)

    print("------------------------------ CENTRALIZED BASELINE -----------------------")
    x_train_all = np.concatenate(tuple(x_train for x_train, y_train, x_valid, y_valid in train_sets))
    y_train_all = np.concatenate(tuple(y_train for x_train, y_train, x_valid, y_valid in train_sets))
    x_valid_all = np.concatenate(tuple(x_valid for x_train, y_train, x_valid, y_valid in train_sets))
    y_valid_all = np.concatenate(tuple(y_valid for x_train, y_train, x_valid, y_valid in train_sets))
    central_participants = [Participant(x_train_all, y_train_all,
                                                   x_valid_all, y_valid_all, batch_size_valid=1)]
    central_server = Server(central_participants, ModelArchitecture.MLP_MONO_CLASS)
    central_server.train_global_model(aggregation_rounds=5)

    y_predicted_central = central_server.predict_using_global_model(x_test)
    print_experiment_scores(y_test.flatten(), y_predicted_central.flatten().numpy(), federated=False)
