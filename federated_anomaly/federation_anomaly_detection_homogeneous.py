import os
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
from tabulate import tabulate

from aggregation import Server
from custom_types import Behavior, RaspberryPi, ModelArchitecture, Scaler
from data_handler import DataHandler
from participants import AutoEncoderParticipant
from utils import FederationUtils

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)
    os.chdir("..")

    print("Use case federated Anomaly/Zero Day Detection\n"
          "Is the federated model able to detect attacks as anomalies,\nie. recognize the difference from attacks"
          " to normal samples? Which attacks are hardest to detect?\n")

    normal = Behavior.NORMAL
    train_devices = []
    test_devices = []

    results, results_central = [], []
    res_dict: Dict[RaspberryPi, Dict[Behavior, str]] = {}

    participants_per_device_type: Dict[RaspberryPi, int] = {
        RaspberryPi.PI3_1GB: 4,
        RaspberryPi.PI4_2GB_BC: 4,
        RaspberryPi.PI4_4GB: 4
    }

    for device in participants_per_device_type:
        train_devices += [(device, {normal: 1350}, {normal: 150})] * participants_per_device_type[device]
    for device in RaspberryPi:
        for behavior in Behavior:
            test_devices.append((device, {behavior: 150}))

    train_sets, test_sets = DataHandler.get_all_clients_data(
        train_devices,
        test_devices)

    # central
    train_sets_cen, test_sets_cen = deepcopy(train_sets), deepcopy(test_sets)
    train_sets_cen, test_sets_cen = DataHandler.scale(train_sets_cen, test_sets_cen, scaling=Scaler.MINMAX_SCALER)

    # copy data for federation and then scale
    train_sets_fed, test_sets_fed = deepcopy(train_sets), deepcopy(test_sets)
    train_sets_fed, test_sets_fed = DataHandler.scale(train_sets_fed, test_sets_fed, scaling=Scaler.MINMAX_SCALER)

    # train federation
    participants = [AutoEncoderParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                    x_train, y_train, x_valid, y_valid in train_sets_fed]
    server = Server(participants, ModelArchitecture.AUTO_ENCODER)
    server.train_global_model(aggregation_rounds=15)

    # train central
    x_train_all = np.concatenate(tuple(x_train for x_train, y_train, x_valid, y_valid in train_sets_cen))
    y_train_all = np.concatenate(tuple(y_train for x_train, y_train, x_valid, y_valid in train_sets_cen))
    x_valid_all = np.concatenate(tuple(x_valid for x_train, y_train, x_valid, y_valid in train_sets_cen))
    y_valid_all = np.concatenate(tuple(y_valid for x_train, y_train, x_valid, y_valid in train_sets_cen))
    central_participant = [
        AutoEncoderParticipant(x_train_all, y_train_all, x_valid_all, y_valid_all,
                               batch_size_valid=1)]
    central_server = Server(central_participant, ModelArchitecture.AUTO_ENCODER)
    central_server.train_global_model(aggregation_rounds=1, local_epochs=15)

    test_tuples_thres_plot = []
    for i, (tfed, tcen) in enumerate(zip(test_sets_fed, test_sets_cen)):
        y_predicted = server.predict_using_global_model(tfed[0])
        y_predicted_central = central_server.predict_using_global_model(tcen[0])
        behavior = list(test_devices[i][1].keys())[0]
        device = test_devices[i][0]

        acc, f1, _ = FederationUtils.calculate_metrics(tfed[1].flatten(), y_predicted.flatten().numpy())
        acc_cen, f1_cen, _ = FederationUtils.calculate_metrics(tcen[1].flatten(), y_predicted_central.flatten().numpy())
        device_dict = res_dict[device] if device in res_dict else {}
        device_dict[behavior] = f'{acc * 100:.2f}% ({(acc - acc_cen) * 100:.2f}%)'

        res_dict[device] = device_dict
        test_tuples_thres_plot += ([(device, behavior)] * len(tfed[0]))

    for behavior in Behavior:
        results.append([behavior.value] + [res_dict[device][behavior] for device in RaspberryPi])

    print(tabulate(results, headers=["Behavior"] + [pi.value for pi in RaspberryPi], tablefmt="pretty"))

    print(f'Thresholds of federation: {server.participants_thresholds}')
    print(
        f'First 50 Prediction thresholds: {server.evaluation_thresholds[:50]}, full length: {len(server.evaluation_thresholds)}')
    print(len(test_tuples_thres_plot), len(server.evaluation_thresholds))
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.DataFrame.from_dict(
        {"threshold": server.evaluation_thresholds, "device": [a.value for a, b in test_tuples_thres_plot],
         "device-behavior": [f'{a.value} {b.value}' for a, b in test_tuples_thres_plot]})
    df_pi3 = df[df['device'] == RaspberryPi.PI3_1GB.value]
    sns.kdeplot(data=df_pi3, x="threshold", hue="device-behavior", log_scale=(True, True), common_norm=True,
                common_grid=True)
    plt.show()
    print(df.groupby("device-behavior").mean())
