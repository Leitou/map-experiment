import numpy as np
import torch
from tabulate import tabulate

from custom_types import Behavior, RaspberryPi, ModelArchitecture, Scaler
from devices import Server, Participant
from data_handler import DataHandler
from utils import calculate_metrics

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    print("Use case federated Binary Classification\n"
          "Is the federated model able to detect attacks as anomalies,\nie. recognize the difference from attacks"
          " to normal samples? Which attacks are hardest to detect?\n")

    normals = [Behavior.NORMAL, Behavior.NORMAL_V2]
    train_devices = []
    test_devices = []
    for device in RaspberryPi:
        for normal in normals:
            train_devices.append((device, {normal: 1000}, {normal: 100}))
            train_devices.append((device, {normal: 1000, Behavior.DELAY: 100, Behavior.DISORDER: 100, Behavior.FREEZE: 100,
                                           Behavior.HOP: 100, Behavior.MIMIC: 100, Behavior.NOISE: 100, Behavior.REPEAT: 100,
                                           Behavior.SPOOF: 100},
                                  {normal: 100, Behavior.DELAY: 10, Behavior.DISORDER: 10, Behavior.FREEZE: 10,
                                   Behavior.HOP: 10, Behavior.MIMIC: 10, Behavior.NOISE: 10, Behavior.REPEAT: 10,
                                   Behavior.SPOOF: 10}))
            test_devices += [(device, {normal: 250, Behavior.DELAY: 100}),
                             (device, {normal: 250, Behavior.DISORDER: 100}),
                             (device, {normal: 250, Behavior.FREEZE: 100}),
                             (device, {normal: 250, Behavior.HOP: 100}),
                             (device, {normal: 250, Behavior.MIMIC: 100}),
                             (device, {normal: 250, Behavior.NOISE: 100}),
                             (device, {normal: 250, Behavior.REPEAT: 100}),
                             (device, {normal: 250, Behavior.SPOOF: 100})]

    train_sets, test_sets = DataHandler.get_all_clients_data(
        train_devices,
        test_devices)

    train_sets, test_sets = DataHandler.scale(train_sets, test_sets, scaling=Scaler.MINMAX_SCALER)

    participants = [Participant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                    x_train, y_train, x_valid, y_valid in train_sets]
    server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
    server.train_global_model(aggregation_rounds=5)

    results = []
    for i, (x_test, y_test) in enumerate(test_sets):
        y_predicted = server.predict_using_global_model(x_test)
        attack = list(set(test_devices[i][1].keys()) - {Behavior.NORMAL, Behavior.NORMAL_V2})[0].value
        normal = list(
            set(test_devices[i][1].keys()) - {Behavior.DELAY, Behavior.DISORDER, Behavior.FREEZE, Behavior.HOP, Behavior.MIMIC,
                                              Behavior.NOISE, Behavior.REPEAT, Behavior.SPOOF})[0].value
        acc, f1, _ = calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
        results.append([test_devices[i][0], normal, attack, f'{acc * 100:.2f}%', f'{f1 * 100:.2f}%'])

    print(tabulate(results, headers=['Device', 'Normal', 'Attack', 'Accuracy', 'F1 score'], tablefmt="pretty"))
