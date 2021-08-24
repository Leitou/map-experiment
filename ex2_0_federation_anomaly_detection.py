import numpy as np
import torch

from custom_types import Behavior, RaspberryPi, ModelArchitecture
from devices import AutoEncoderParticipant, Server
from data_handler import DataHandler
from utils import calculate_metrics
from tabulate import tabulate

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    print("Use case federated Anomaly/Zero Day Detection\n"
          "Is the federated model able to detect attacks as anomalies,\nie. recognize the difference from attacks"
          " to normal samples? Which attacks are hardest to detect?\n")

    normals = [Behavior.NORMAL, Behavior.NORMAL_V2]
    train_devices = []
    test_devices = []
    for device in RaspberryPi:
        for normal in normals:
            train_devices.append((device, {normal: 2000}, {normal: 200}))
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

    train_sets, test_sets = DataHandler.scale(train_sets, test_sets)

    participants = [AutoEncoderParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                    x_train, y_train, x_valid, y_valid in train_sets]
    server = Server(participants, ModelArchitecture.AUTO_ENCODER)
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
