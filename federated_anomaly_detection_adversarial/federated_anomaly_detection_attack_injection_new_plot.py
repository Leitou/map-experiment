import os

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt

from aggregation import Server
from custom_types import Behavior, RaspberryPi, Scaler, ModelArchitecture, AggregationMechanism
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
    print("Starting demo experiment: Adversarial Impact on Federated Anomaly Detection")

    pi_to_inject = RaspberryPi.PI4_4GB

    attack_devices = [(pi_to_inject, {Behavior.DISORDER: 300}, {Behavior.DISORDER: 30}),
                      (pi_to_inject, {Behavior.SPOOF: 300}, {Behavior.SPOOF: 30}),
                      (pi_to_inject, {Behavior.MIMIC: 300}, {Behavior.MIMIC: 30}),
                      (pi_to_inject, {Behavior.NOISE: 300}, {Behavior.NOISE: 30})]

    max_adversaries = 4
    participants_per_device_type = max_adversaries

    test_devices = []
    test_set_result_dict = {"device": [], "num_adversaries": [], "f1": []}

    for device in RaspberryPi:
        test_devices.append((device, {beh: 100 for beh in Behavior}))

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 4.8))
    fig.suptitle(f"Injecting device {pi_to_inject.value}")
    axs = axs.ravel().tolist()
    # Loop that handles aggregation
    agg_idx = 0
    for agg in AggregationMechanism:
        # Loop that handles 0 to n Adversaries, creating Test sets etc
        for i in range(0, max_adversaries + 1):
            train_devices = []
            for device in RaspberryPi:
                if device == RaspberryPi.PI4_2GB_WC:
                    continue
                elif device == pi_to_inject:
                    train_devices += attack_devices[0:i]
                    for j in range(participants_per_device_type - i):
                        train_devices.append((device, {Behavior.NORMAL: 1500}, {Behavior.NORMAL: 150}))
                else:
                    for j in range(participants_per_device_type):
                        train_devices.append((device, {Behavior.NORMAL: 1500}, {Behavior.NORMAL: 150}))
            train_sets, test_sets = DataHandler.get_all_clients_data(
                train_devices,
                test_devices)

            train_sets, test_sets = DataHandler.scale(train_sets, test_sets,
                                                      scaling=Scaler.MINMAX_SCALER)

            # participants contains adversaries already
            participants = [AutoEncoderParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                            x_train, y_train, x_valid, y_valid in train_sets]
            server = Server(participants, ModelArchitecture.AUTO_ENCODER,
                            aggregation_mechanism=agg)
            server.train_global_model(aggregation_rounds=15)

            for j, (tset) in enumerate(test_sets):
                y_predicted = server.predict_using_global_model(tset[0])
                device = test_devices[j][0]
                acc, f1, _ = FederationUtils.calculate_metrics(tset[1].flatten(), y_predicted.flatten().numpy())
                test_set_result_dict['device'].append(device.value)
                test_set_result_dict['num_adversaries'].append(i)
                test_set_result_dict['f1'].append(f1 * 100)

            all_train, all_test = FederationUtils.aggregate_test_sets(test_sets)
            y_predicted = server.predict_using_global_model(all_train)
            acc, f1, _ = FederationUtils.calculate_metrics(all_test.flatten(), y_predicted.flatten().numpy())
            test_set_result_dict['device'].append('All')
            test_set_result_dict['num_adversaries'].append(i)
            test_set_result_dict['f1'].append(f1 * 100)

        df = pd.DataFrame.from_dict(test_set_result_dict)
        sns.barplot(
            data=df, ci=None,
            x="device", y="f1", hue="num_adversaries",
            alpha=.6, ax=axs[agg_idx]
        )
        axs[agg_idx].set_ylim(0, 100)
        axs[agg_idx].set_title(f'{agg.value}')
        axs[agg_idx].get_legend().remove()

        axs[agg_idx].set_ylabel('Device')
        if agg_idx == 0:
            axs[agg_idx].set_ylabel('F1 Score (%)')
        else:
            axs[agg_idx].set_ylabel(None)
        agg_idx += 1
    # add legend
    handles, labels = axs[len(list(AggregationMechanism)) - 1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1, 0.9))
    fig.tight_layout()
    plt.show()
