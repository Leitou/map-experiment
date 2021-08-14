import numpy as np
import torch
from tabulate import tabulate

from custom_types import Attack, RaspberryPi, ModelArchitecture
from devices import Server, Participant
from sampling import DataSampler
from utils import calculate_metrics

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    print("Use case federated Binary Classification\n"
          "Is the federated model able to detect attacks as anomalies,\nie. recognize the difference from attacks"
          " to normal samples? Which attacks are hardest to detect?\n")

    normals = [Attack.NORMAL, Attack.NORMAL_V2]
    train_devices = []
    test_devices = []
    for device in RaspberryPi:
        for normal in normals:
            train_devices.append((device, {normal: 1000}, {normal: 100}))
            train_devices.append((device, {normal: 1000, Attack.DELAY: 100, Attack.DISORDER: 100, Attack.FREEZE: 100,
                                           Attack.HOP: 100, Attack.MIMIC: 100, Attack.NOISE: 100, Attack.REPEAT: 100,
                                           Attack.SPOOF: 100},
                                  {normal: 100, Attack.DELAY: 10, Attack.DISORDER: 10, Attack.FREEZE: 10,
                                   Attack.HOP: 10, Attack.MIMIC: 10, Attack.NOISE: 10, Attack.REPEAT: 10,
                                   Attack.SPOOF: 10}))
            test_devices += [(device, {normal: 250, Attack.DELAY: 100}),
                             (device, {normal: 250, Attack.DISORDER: 100}),
                             (device, {normal: 250, Attack.FREEZE: 100}),
                             (device, {normal: 250, Attack.HOP: 100}),
                             (device, {normal: 250, Attack.MIMIC: 100}),
                             (device, {normal: 250, Attack.NOISE: 100}),
                             (device, {normal: 250, Attack.REPEAT: 100}),
                             (device, {normal: 250, Attack.SPOOF: 100})]

    train_sets, test_sets = DataSampler.get_all_clients_data(
        train_devices,
        test_devices)

    train_sets, test_sets = DataSampler.scale(train_sets, test_sets)

    participants = [Participant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                    x_train, y_train, x_valid, y_valid in train_sets]
    server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
    server.train_global_model(aggregation_rounds=5)

    results = []
    for i, (x_test, y_test) in enumerate(test_sets):
        y_predicted = server.predict_using_global_model(x_test)
        attack = list(set(test_devices[i][1].keys()) - {Attack.NORMAL, Attack.NORMAL_V2})[0].value
        normal = list(
            set(test_devices[i][1].keys()) - {Attack.DELAY, Attack.DISORDER, Attack.FREEZE, Attack.HOP, Attack.MIMIC,
                                              Attack.NOISE, Attack.REPEAT, Attack.SPOOF})[0].value
        acc, f1, _ = calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
        results.append([test_devices[i][0], normal, attack, f'{acc * 100:.2f}%', f'{f1 * 100:.2f}%'])

    print(tabulate(results, headers=['Device', 'Normal', 'Attack', 'Accuracy', 'F1 score'], tablefmt="pretty"))
