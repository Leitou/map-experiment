import numpy as np
import torch
from tabulate import tabulate

from custom_types import Behavior, RaspberryPi, ModelArchitecture, Scaler
from data_handler import DataHandler
from devices import AutoEncoderParticipant, Server
from utils import calculate_metrics

# TODO: reformat output: Device as column
if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    print("Use case federated Anomaly/Zero Day Detection\n"
          "Is the federation able to transfer its knowledge to a new device?\n")

    results = []
    normals = [Behavior.NORMAL, Behavior.NORMAL_V2]
    for device in RaspberryPi:
        train_devices = []
        for device2 in RaspberryPi:
            if device2 != device:
                train_devices.append((device, {Behavior.NORMAL: 2000}, {Behavior.NORMAL: 200}))
        test_devices = []
        for behavior in Behavior:
            test_devices.append((device, {behavior: 150}))

        train_sets, test_sets = DataHandler.get_all_clients_data(
            train_devices,
            test_devices)

        train_sets, test_sets = DataHandler.scale(train_sets, test_sets, scaling=Scaler.MINMAX_SCALER)

        participants = [AutoEncoderParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                        x_train, y_train, x_valid, y_valid in train_sets]
        server = Server(participants, ModelArchitecture.AUTO_ENCODER)
        server.train_global_model(aggregation_rounds=5)

        for i, (x_test, y_test) in enumerate(test_sets):
            y_predicted = server.predict_using_global_model(x_test)
            behavior = list(test_devices[i][1].keys())[0]
            acc, f1, _ = calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
            results.append([device, behavior, f'{acc * 100:.2f}%'])

    print(tabulate(results, headers=['Device', 'Behavior', 'Accuracy'], tablefmt="pretty"))
