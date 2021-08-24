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

    print("Use case standalone devices: Anomaly/Zero Day Detection\n"
          "Is the standalone model able to detect attacks as anomalies,\nie. recognize the difference from attacks"
          " to normal samples? Which attacks are hardest to detect?\n")

    normals = [Behavior.NORMAL, Behavior.NORMAL_V2]
    for device in RaspberryPi:
        for normal in normals:
            results = []

            test_devices = [(device, {normal: 500, Behavior.DELAY: 150}),
                            (device, {normal: 500, Behavior.DISORDER: 150}),
                            (device, {normal: 500, Behavior.FREEZE: 150}),
                            (device, {normal: 500, Behavior.HOP: 150}),
                            (device, {normal: 500, Behavior.MIMIC: 150}),
                            (device, {normal: 500, Behavior.NOISE: 150}),
                            (device, {normal: 500, Behavior.REPEAT: 150}),
                            (device, {normal: 500, Behavior.SPOOF: 150})]
            train_sets, test_sets = DataHandler.get_all_clients_data(
                [(device, {normal: 1500}, {normal: 150})],
                test_devices)

            train_sets, test_sets = DataHandler.scale(train_sets, test_sets, True)

            participants = [AutoEncoderParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                            x_train, y_train, x_valid, y_valid in train_sets]
            server = Server(participants, ModelArchitecture.AUTO_ENCODER)
            server.train_global_model(aggregation_rounds=5)

            for i, (x_test, y_test) in enumerate(test_sets):
                y_predicted = server.predict_using_global_model(x_test)
                attack = list(set(test_devices[i][1].keys()) - {Behavior.NORMAL, Behavior.NORMAL_V2})[0].value
                acc, f1, _ = calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
                results.append([device.value, normal.value, attack, f'{acc * 100:.2f}%', f'{f1 * 100:.2f}%'])

            print(tabulate(results, headers=['Device', 'Normal', 'Attack', 'Accuracy', 'F1 score'], tablefmt="pretty"))
