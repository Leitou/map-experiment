import os

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt

from aggregation import Server
from custom_types import Behavior, RaspberryPi, Scaler, ModelArchitecture, AggregationMechanism
from data_handler import DataHandler
from participants import MLPParticipant, AllLabelFlipAdversary
from utils import FederationUtils

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)
    cwd = os.getcwd()
    os.chdir("..")

    print("Use case federated Anomaly/Zero Day Detection\n"
          "Is the federated model able to detect attacks as anomalies,\nie. recognize the difference from attacks"
          " to normal samples? Which attacks are hardest to detect?\n")
    print("Starting demo experiment: Adversarial Impact on Federated Anomaly Detection")

    pis_to_inject = [pi for pi in RaspberryPi if pi != RaspberryPi.PI4_2GB_WC]

    max_adversaries = 4
    participants_per_device_type = max_adversaries

    test_devices = []

    for device in RaspberryPi:
        test_devices.append((device, {beh: 100 for beh in Behavior}))

    test_set_result_dict = {"device": [], "num_adversaries": [], "f1": [], "aggregation": [], "injected": []}

    csv_result_path = cwd + os.sep + "binary_classification_label_flipping.csv"
    if os.path.isfile(csv_result_path):
        df = pd.read_csv(csv_result_path)
    else:
        for pi_to_inject in pis_to_inject:
            # Ordered ASC by threshold
            normal_devices = [(pi_to_inject, {Behavior.NORMAL: 250},
                               {Behavior.NORMAL: 25}),
                              (pi_to_inject, {Behavior.NORMAL: 250, Behavior.REPEAT: 250},
                               {Behavior.NORMAL: 25, Behavior.REPEAT: 25}),
                              (pi_to_inject, {Behavior.NORMAL: 250, Behavior.NOISE: 250},
                               {Behavior.NORMAL: 25, Behavior.NOISE: 25}),
                              (pi_to_inject, {Behavior.NORMAL: 250, Behavior.DELAY: 250},
                               {Behavior.NORMAL: 25, Behavior.DELAY: 25})]
            # Ordered DESC by threshold
            attack_devices = [
                (pi_to_inject, {Behavior.NORMAL: 250, Behavior.SPOOF: 250}, {Behavior.NORMAL: 25, Behavior.SPOOF: 25}),
                (pi_to_inject, {Behavior.NORMAL: 250, Behavior.MIMIC: 250}, {Behavior.NORMAL: 25, Behavior.MIMIC: 25}),
                (pi_to_inject, {Behavior.NORMAL: 250, Behavior.DELAY: 250}, {Behavior.NORMAL: 25, Behavior.DELAY: 25}),
                (pi_to_inject, {Behavior.NORMAL: 250, Behavior.DISORDER: 250},
                 {Behavior.NORMAL: 25, Behavior.DISORDER: 25})
            ]
            # Aggregation loop
            for agg in AggregationMechanism:
                # Adversary Loop -> here is the training
                for i in range(0, max_adversaries + 1):
                    cp_filename = f'{cwd}{os.sep}label_flipping_{pi_to_inject.value}_{agg.value}_{str(i)}.pt'
                    train_devices = []
                    for device in RaspberryPi:
                        if device == RaspberryPi.PI4_2GB_WC:
                            continue
                        elif device == pi_to_inject:
                            train_devices += attack_devices[:i]
                            train_devices += normal_devices[i:]
                        else:
                            train_devices += [(device, {Behavior.NORMAL: 250},
                                               {Behavior.NORMAL: 25}),
                                              (device, {Behavior.NORMAL: 250, Behavior.REPEAT: 250},
                                               {Behavior.NORMAL: 25, Behavior.REPEAT: 25}),
                                              (device, {Behavior.NORMAL: 250, Behavior.NOISE: 250},
                                               {Behavior.NORMAL: 25, Behavior.NOISE: 25}),
                                              (device, {Behavior.NORMAL: 250, Behavior.DELAY: 250},
                                               {Behavior.NORMAL: 25, Behavior.DELAY: 25})]
                    train_sets, test_sets = DataHandler.get_all_clients_data(
                        train_devices,
                        test_devices)

                    train_sets, test_sets = DataHandler.scale(train_sets, test_sets,
                                                              scaling=Scaler.MINMAX_SCALER)

                    inj_index = [dev[0] for dev in train_devices].index(pi_to_inject)
                    inj_indices = list(range(inj_index, inj_index + i))
                    participants = [(AllLabelFlipAdversary(x_train, y_train, x_valid, y_valid,
                                                   batch_size_valid=1) if loop_i in inj_indices else MLPParticipant(
                        x_train, y_train, x_valid, y_valid, batch_size_valid=1))
                                    for loop_i, (x_train, y_train, x_valid, y_valid) in enumerate(train_sets)]
                    server = Server(participants, ModelArchitecture.MLP_MONO_CLASS,
                                    aggregation_mechanism=agg)
                    if not os.path.isfile(cp_filename):
                        server.train_global_model(aggregation_rounds=15)
                        torch.save(server.global_model.state_dict(), cp_filename)
                    else:
                        server.global_model.load_state_dict(torch.load(cp_filename))
                        server.load_global_model_into_participants()
                        print(f'Loaded model for device {pi_to_inject.value} and {str(i)} attackers and {agg.value}')

                    for j, (tset) in enumerate(test_sets):
                        y_predicted = server.predict_using_global_model(tset[0])
                        device = test_devices[j][0]
                        acc, f1, _ = FederationUtils.calculate_metrics(tset[1].flatten(), y_predicted.flatten().numpy())
                        test_set_result_dict['device'].append(device.value)
                        test_set_result_dict['num_adversaries'].append(i)
                        test_set_result_dict['f1'].append(f1 * 100)
                        test_set_result_dict['aggregation'].append(agg.value)
                        test_set_result_dict['injected'].append(pi_to_inject.value)

                    all_train, all_test = FederationUtils.aggregate_test_sets(test_sets)
                    y_predicted = server.predict_using_global_model(all_train)
                    acc, f1, _ = FederationUtils.calculate_metrics(all_test.flatten(), y_predicted.flatten().numpy())
                    test_set_result_dict['device'].append('All')
                    test_set_result_dict['num_adversaries'].append(i)
                    test_set_result_dict['f1'].append(f1 * 100)
                    test_set_result_dict['aggregation'].append(agg.value)
                    test_set_result_dict['injected'].append(pi_to_inject.value)
        df = pd.DataFrame.from_dict(test_set_result_dict)
        df.to_csv(csv_result_path, index=False)

    # TODO move to Utils
    fig, axs = plt.subplots(nrows=len(pis_to_inject), ncols=len(list(AggregationMechanism)), figsize=(19.2, 19.2))
    axs = axs.ravel().tolist()
    agg_idx = 0

    for pi_to_inject in pis_to_inject:
        for agg in AggregationMechanism:
            df_loop = df[(df.injected == pi_to_inject.value) & (df.aggregation == agg.value)].drop(
                ['injected', 'aggregation'], axis=1)
            sns.barplot(
                data=df_loop, ci=None,
                x="device", y="f1", hue="num_adversaries",
                alpha=.6, ax=axs[agg_idx]
            )
            axs[agg_idx].set_ylim(0, 100)
            axs[agg_idx].set_title(f'{agg.value}')
            axs[agg_idx].get_legend().remove()

            axs[agg_idx].set_ylabel('Device')
            if agg_idx % len(list(AggregationMechanism)) == 0:
                axs[agg_idx].set_ylabel('F1 Score (%)')
            else:
                axs[agg_idx].set_ylabel(None)
            agg_idx += 1

    # add legend
    handles, labels = axs[len(list(AggregationMechanism)) - 1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1, 0.99), title="# of Adversaries")
    plt.tight_layout()
    plt.show()
    fig.savefig(f'result_plot_binary_classification_label_flipping_all.png', dpi=100)
