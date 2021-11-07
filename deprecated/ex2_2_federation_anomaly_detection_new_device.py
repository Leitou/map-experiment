from typing import Dict

import numpy as np
import torch
from tabulate import tabulate

from custom_types import Behavior, RaspberryPi, ModelArchitecture, Scaler
from data_handler import DataHandler
from aggregation import Server
from participants import AutoEncoderParticipant
from utils import FederationUtils

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    print("Use case federated Anomaly/Zero Day Detection\n"
          "Is the federation able to transfer its knowledge to a new device?\n")

    res_dict: Dict[RaspberryPi, Dict[Behavior, str]] = {}
    results = []

    for device in RaspberryPi:
        device_dict: Dict[Behavior, str] = {}
        train_devices = []
        for device2 in RaspberryPi:
            if device2 != device:
                train_devices.append((device2, {Behavior.NORMAL: 2000}, {Behavior.NORMAL: 200}))
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
            acc, f1, _ = FederationUtils.calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
            device_dict[behavior] = f'{acc * 100:.2f}%'
        res_dict[device] = device_dict
    for behavior in Behavior:
        results.append([behavior.value] + [res_dict[device][behavior] for device in RaspberryPi])

    print(tabulate(results, headers=["Behavior"] + [pi.value for pi in RaspberryPi], tablefmt="pretty"))
