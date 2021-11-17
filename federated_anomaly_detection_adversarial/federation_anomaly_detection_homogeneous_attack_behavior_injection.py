import os
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from pathlib import Path

from aggregation import Server
from custom_types import Behavior, RaspberryPi, ModelArchitecture, Scaler, AggregationMechanism
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

    excluded_pi = RaspberryPi.PI4_2GB_WC
    normal = Behavior.NORMAL

    test_devices = []
    for device in RaspberryPi:
        if device == excluded_pi:
            continue
        bdict = {}
        for behavior in Behavior:
            if behavior == Behavior.NORMAL or behavior == Behavior.NORMAL_V2:
                bdict[behavior] = 950  # 95/5 split normal/attack behavior in test set
            else:
                bdict[behavior] = 50  # TODO: adapt this nr to change ratio
        test_devices.append((device, bdict))

    # train, predict and plot results
    bar_width = 1.5
    sep = 1
    colors = ['limegreen', 'gold', 'tab:orange', 'orangered', "violet"]
    colors = ['#87d64b', '#fae243', '#f8961e', '#ff4d36', '#8f00ff']
    text_colors = ['#456e25', '#998a28', '#b06a13', '#b33424', '#8f00ff']

    num_participants_per_device = 4
    inj_att_behavior = Behavior.DISORDER
    adv_device = RaspberryPi.PI4_4GB  # TODO: adapt this device for selecting another device type for the adversaries
    pis = list(RaspberryPi)
    pis_excl = pis[0:pis.index(excluded_pi)] + pis[pis.index(excluded_pi) + 1:]

    train_devices_base = []
    for i, device in enumerate(RaspberryPi):
        if device != adv_device and device != excluded_pi:
            train_devices_base += [(device, {normal: 1350}, {normal: 150})] * num_participants_per_device

    fig, axs = plt.subplots(4)
    max_num_adv = 4
    fig.suptitle(f'0-4 Adversarial {adv_device.name}s', fontsize=16)
    if not Path(
            f"{Path(__file__).parent}/results_inj_{inj_att_behavior.name}_adv_{adv_device.name}_{Path(__file__).stem}/").is_dir():
        Path(
            f"{Path(__file__).parent}/results_inj_{inj_att_behavior.name}_adv_{adv_device.name}_{Path(__file__).stem}").mkdir()

    for i, device in enumerate(pis_excl + ["ALL_DEVICES_ALL_BEHAVIORS"]):

        pos_x = 0
        title = device.name if i < 3 else device

        axs[i].set_title(title)
        axs[i].set_ylabel('F1-Score (%)')
        axs[i].set_ylim(0, 100.)
        # plt.gca().spines['top'].set_visible(False)
        # plt.gca().spines['right'].set_visible(False)
        #     plt.gca().spines['bottom'].set_visible(False)

        for agg in AggregationMechanism:
            for num_adv in range(max_num_adv + 1):
                print(f"Using {num_adv} adversaries")

                train_devices = deepcopy(train_devices_base)
                # inject adversarial participants via data
                train_devices += [(adv_device, {normal: 1350}, {normal: 150})] * (num_participants_per_device - num_adv)
                train_devices += [(adv_device, {inj_att_behavior: 130 // num_adv if num_adv != 0 else 130},
                                   {inj_att_behavior: 13 // num_adv if num_adv != 0 else 13})] * num_adv

                train_sets_fed, test_sets = DataHandler.get_all_clients_data(
                    train_devices,
                    test_devices)

                # copy data for federation and then scale
                test_sets_fed = deepcopy(test_sets)
                train_sets_fed, test_sets_fed = DataHandler.scale(train_sets_fed, test_sets_fed,
                                                                  scaling=Scaler.MINMAX_SCALER)

                # participants contains adversaries already
                participants = [AutoEncoderParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                                x_train, y_train, x_valid, y_valid in train_sets_fed]
                server = Server(participants, ModelArchitecture.AUTO_ENCODER, aggregation_mechanism=agg)

                filepath = f"{Path(__file__).parent}/results_inj_{inj_att_behavior.name}_adv_{adv_device.name}_{Path(__file__).stem}" \
                           f"/model_{agg.value}_{num_adv}_adv.pt"
                if not Path(filepath).is_file():
                    # train federation
                    server.train_global_model(aggregation_rounds=15)
                    torch.save(server.global_model.state_dict(), filepath)
                else:
                    server.global_model.load_state_dict(torch.load(filepath))
                    server.load_global_model_into_participants()

                if i < 3:
                    y_predicted = server.predict_using_global_model(test_sets_fed[i][0])
                    acc, f1, _ = FederationUtils.calculate_metrics(test_sets_fed[i][1].flatten(),
                                                                   y_predicted.flatten().numpy())
                else:
                    test_sets_fed_x = np.concatenate(tuple(x_test for x_test, y_test in
                                                           test_sets_fed))
                    test_sets_fed_y = np.concatenate(tuple(y_test for x_test, y_test in
                                                           test_sets_fed))
                    global_test_set = [(test_sets_fed_x, test_sets_fed_y)]
                    y_predicted = server.predict_using_global_model(global_test_set[0][0])
                    acc, f1, _ = FederationUtils.calculate_metrics(global_test_set[0][1].flatten(),
                                                                   y_predicted.flatten().numpy())

                print(f"f1 score: {f1:.2f}")

                # plotting grouped bar chart
                color = colors[num_adv]
                text_color = text_colors[num_adv]
                f1_height = f1 * 100
                if agg == AggregationMechanism.FED_AVG:
                    axs[i].bar(pos_x, height=f1_height, color=color, width=bar_width, lw=0.7, edgecolor='black',
                               label='f=' + repr(num_adv))
                else:
                    axs[i].bar(pos_x, height=f1_height, color=color, width=bar_width, lw=0.7,
                               edgecolor='black')  # yerr=[[yerr_down], [yerr_up]], capsize=11

                s = ("{:." + repr(0) + "f}").format(f1_height)
                if len(s) == 1:
                    text_x = pos_x - 0.2
                else:
                    text_x = pos_x - 0.5

                axs[i].text(x=text_x, y=f1_height + 1.2, s=s, fontsize='15.5', color=text_color)

                pos_x += bar_width
            pos_x += sep

        ticks = [(1.5 * bar_width) + (4 * bar_width + sep) * i for i in range(0, len(AggregationMechanism))]
        axs[i].set_xticks(ticks)
        aggs = [agg.value for agg in AggregationMechanism]
        axs[i].set_xticklabels(aggs)
        axs[i].legend(bbox_to_anchor=(1.12, 0.5), loc='right')
    plt.show()
    fig.savefig(f'{Path(__file__).parent}/results_adv_{adv_device.name}_{Path(__file__).stem}/'
                f'f1_scores_inject_{inj_att_behavior.name}_adv_{adv_device.name}.pdf', bbox_inches='tight')
