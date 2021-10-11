from typing import Dict

import numpy as np
import torch
from tabulate import tabulate
from copy import deepcopy

from custom_types import Behavior, RaspberryPi, ModelArchitecture, Scaler
from data_handler import DataHandler
from aggregation import Server
from participants import AutoEncoderParticipant
from utils import calculate_metrics

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    print("Use case federated Anomaly/Zero Day Detection\n"
          "Is the federation able to transfer its knowledge to a new device?\n")

    results, results_central = [], []
    res_dict: Dict[RaspberryPi, Dict[Behavior, str]] = {}
    res_dict_central: Dict[RaspberryPi, Dict[Behavior, str]] = {}

    for device in RaspberryPi:
        train_devices = []
        for device2 in RaspberryPi:
            if device2 != device:
                train_devices.append((device2, {Behavior.NORMAL: 1500}, {Behavior.NORMAL: 500}))
                train_devices.append((device2, {Behavior.NORMAL: 1500}, {Behavior.NORMAL: 500}))
                train_devices.append((device2, {Behavior.NORMAL: 1500}, {Behavior.NORMAL: 500}))

        test_devices = []
        for behavior in Behavior:
            test_devices.append((device, {behavior: 150}))

        train_sets, test_sets = DataHandler.get_all_clients_data(
            train_devices,
            test_devices)

        train_sets_fed, test_sets_fed = deepcopy(train_sets), deepcopy(test_sets)
        train_sets_fed, test_sets_fed = DataHandler.scale(train_sets_fed, test_sets_fed, scaling=Scaler.MINMAX_SCALER)

        participants = [AutoEncoderParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                        x_train, y_train, x_valid, y_valid in train_sets_fed]
        server = Server(participants, ModelArchitecture.AUTO_ENCODER)
        server.train_global_model(aggregation_rounds=5)

        x_train_all = np.concatenate(tuple(x_train for x_train, y_train, x_valid, y_valid in train_sets))
        y_train_all = np.concatenate(tuple(y_train for x_train, y_train, x_valid, y_valid in train_sets))
        x_valid_all = np.concatenate(tuple(x_valid for x_train, y_train, x_valid, y_valid in train_sets))
        y_valid_all = np.concatenate(tuple(y_valid for x_train, y_train, x_valid, y_valid in train_sets))
        train_set_cen = [(x_train_all, y_train_all, x_valid_all, y_valid_all)]
        train_set_cen, test_sets_cen = DataHandler.scale(train_set_cen, test_sets, central=True,
                                                         scaling=Scaler.MINMAX_SCALER)
        central_participant = [
            AutoEncoderParticipant(train_set_cen[0][0], train_set_cen[0][1], train_set_cen[0][2], train_set_cen[0][3],
                                   batch_size_valid=1)]
        central_server = Server(central_participant, ModelArchitecture.AUTO_ENCODER)
        central_server.train_global_model(aggregation_rounds=5)

        device_dict: Dict[Behavior, str] = {}
        device_dict_central: Dict[Behavior, str] = {}

        for i, (tfed, tcen) in enumerate(zip(test_sets_fed, test_sets_cen)):
            y_predicted = server.predict_using_global_model(tfed[0])
            y_predicted_central = central_server.predict_using_global_model(tcen[0])
            behavior = list(test_devices[i][1].keys())[0]

            acc, f1, _ = calculate_metrics(tfed[1].flatten(), y_predicted.flatten().numpy())
            acc_cen, f1_cen, _ = calculate_metrics(tcen[1].flatten(), y_predicted_central.flatten().numpy())
            device_dict[behavior] = f'{acc * 100:.2f}%'
            device_dict_central[behavior] = f'{acc_cen * 100:.2f}%'

        res_dict[device] = device_dict
        res_dict_central[device] = device_dict_central

    for behavior in Behavior:
        results.append([behavior.value] + [res_dict[device][behavior] for device in RaspberryPi])
        results_central.append([behavior.value] + [res_dict_central[device][behavior] for device in RaspberryPi])

    print("Federated Results")
    print(tabulate(results, headers=["Behavior"] + [pi.value for pi in RaspberryPi], tablefmt="pretty"))
    print("Centralized Results")
    print(tabulate(results_central, headers=["Behavior"] + [pi.value for pi in RaspberryPi], tablefmt="pretty"))
