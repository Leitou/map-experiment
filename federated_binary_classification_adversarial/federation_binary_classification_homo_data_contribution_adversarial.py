import os
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import matplotlib.pyplot as plt

from aggregation import Server
from custom_types import Behavior, ModelArchitecture, Scaler, RaspberryPi, AggregationMechanism
from data_handler import DataHandler
from participants import MLPParticipant, AllLabelFlipAdversary
from utils import FederationUtils

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)
    os.chdir("..")

    print("Starting demo experiment: Adversarial Impact on Federations\n")

    train_devices = []
    test_devices = []

    results, results_central = [], []
    res_dict: Dict[RaspberryPi, Dict[Behavior, str]] = {}

    for device in RaspberryPi:
        bdict = {}
        for behavior in Behavior:
            bdict[behavior] = 80
        test_devices.append((device, bdict))


        if device == RaspberryPi.PI4_2GB_WC:
            continue
        train_devices += [(device, {Behavior.NORMAL: 300},
                           {Behavior.NORMAL: 30}),
                          (device, {Behavior.NORMAL: 300, Behavior.DELAY: 300},
                           {Behavior.NORMAL: 30, Behavior.DELAY: 30}),
                          (device, {Behavior.NORMAL: 300, Behavior.FREEZE: 300},
                           {Behavior.NORMAL: 30, Behavior.FREEZE: 30}),
                          (device, {Behavior.NORMAL: 300, Behavior.NOISE: 300},
                           {Behavior.NORMAL: 30, Behavior.NOISE: 30})]


    train_sets, test_sets = DataHandler.get_all_clients_data(
        train_devices,
        test_devices)

    # copy data for federation and then scale
    # independent of label flips
    train_sets_fed, test_sets_fed = deepcopy(train_sets), deepcopy(test_sets)
    train_sets_fed, test_sets_fed = DataHandler.scale(train_sets_fed, test_sets_fed, scaling=Scaler.MINMAX_SCALER)

    # TODO: implement first predictions on f1 score of concatenation of behaviors
    #  then implement predictions for each single behavior, valuable?

    # create global test set -
    # concat all test sets for all devices and behaviors
    test_sets_fed_x = np.concatenate(tuple(x_test for x_test, y_test in
                         test_sets_fed))
    test_sets_fed_y = np.concatenate(tuple(y_test for x_test, y_test in
                         test_sets_fed))
    global_test_set = [(test_sets_fed_x, test_sets_fed_y)]


    # train, predict and plot results
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_ylabel('F1-Score (%)')
    ax.set_ylim(0, 100.)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #     plt.gca().spines['bottom'].set_visible(False)

    pos_x = 0
    bar_width = 1.5
    sep = 1
    colors = ['limegreen', 'gold', 'tab:orange', 'orangered', "red"]
    colors = ['#87d64b', '#fae243', '#f8961e', '#ff4d36', '#ff0000']
    text_colors = ['#456e25', '#998a28', '#b06a13', '#b33424', '#ff0000']

    max_num_adv = 5
    for agg in AggregationMechanism:
        for num_adv in range(max_num_adv):
            print(f"Using {num_adv} adversaries")

            # train federation
            # TODO: implement first only adversaries of one device type -> rasp3, then maybe multiple device types
            adversaries = [AllLabelFlipAdversary(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                            x_train, y_train, x_valid, y_valid in train_sets_fed[:num_adv]]

            participants = [MLPParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                            x_train, y_train, x_valid, y_valid in train_sets_fed[num_adv:]]


            server = Server(adversaries + participants, ModelArchitecture.MLP_MONO_CLASS, aggregation_mechanism=agg)
            server.train_global_model(aggregation_rounds=15)

            # predict on all test data
            y_predicted = server.predict_using_global_model(global_test_set[0][0])
            acc, f1, _ = FederationUtils.calculate_metrics(global_test_set[0][1].flatten(), y_predicted.flatten().numpy())
            print(f1)


            # plotting grouped bar chart
            color = colors[num_adv]
            text_color = text_colors[num_adv]
            f1_height = f1 * 100
            if agg == AggregationMechanism.FED_AVG:
                ax.bar(pos_x, height=f1_height, color=color, width=bar_width, lw=0.7, edgecolor='black',
                       label='f=' + repr(num_adv))
            else:
                ax.bar(pos_x, height=f1_height, color=color, width=bar_width, lw=0.7,
                       edgecolor='black')  # yerr=[[yerr_down], [yerr_up]], capsize=11

        #         ax.errorbar(x=pos_x, y=mean_f1_bar, yerr=[[yerr_down], [yerr_up]],
        #                     capsize=6, color='black', elinewidth=0, lw=1.0, solid_capstyle='round')
        #
            s = ("{:."+repr(0)+"f}").format(f1_height)
            if len(s) == 1:
                text_x = pos_x - 0.2
            else:
                text_x = pos_x - 0.5

            ax.text(x=text_x, y=f1_height + 1.2, s=s, fontsize='15.5', color=text_color)

            pos_x += bar_width
        pos_x += sep

    ticks = [(1.5 * bar_width) + (4 * bar_width + sep) * i for i in range(0, len(AggregationMechanism))]

    ax.set_xticks(ticks)
    aggs = [agg.value for agg in AggregationMechanism]
    ax.set_xticklabels(aggs)
    ax.legend(bbox_to_anchor=(1.12, 0.5), loc='right')
    plt.show()
    fig.savefig('f1_scores_all_label_flip' + '.pdf', bbox_inches='tight')









    # not needed?
    # # train central
    # x_train_all = np.concatenate(tuple(x_train for x_train, y_train, x_valid, y_valid in train_sets_cen))
    # y_train_all = np.concatenate(tuple(y_train for x_train, y_train, x_valid, y_valid in train_sets_cen))
    # x_valid_all = np.concatenate(tuple(x_valid for x_train, y_train, x_valid, y_valid in train_sets_cen))
    # y_valid_all = np.concatenate(tuple(y_valid for x_train, y_train, x_valid, y_valid in train_sets_cen))
    # central_participant = [
    #     MLPParticipant(x_train_all, y_train_all, x_valid_all, y_valid_all,
    #                    batch_size_valid=1)]
    # central_server = Server(central_participant, ModelArchitecture.MLP_MONO_CLASS)
    # central_server.train_global_model(aggregation_rounds=1, local_epochs=15)
    #
    # for i, (tfed, tcen) in enumerate(zip(test_sets_fed, test_sets_cen)):
    #     y_predicted = server.predict_using_global_model(tfed[0])
    #     y_predicted_central = central_server.predict_using_global_model(tcen[0])
    #     behavior = list(test_devices[i][1].keys())[0]
    #     device = test_devices[i][0]
    #
    #     acc, f1, _ = FederationUtils.calculate_metrics(tfed[1].flatten(), y_predicted.flatten().numpy())
    #     acc_cen, f1_cen, _ = FederationUtils.calculate_metrics(tcen[1].flatten(), y_predicted_central.flatten().numpy())
    #     device_dict = res_dict[device] if device in res_dict else {}
    #     device_dict[behavior] = f'{acc * 100:.2f}% ({(acc - acc_cen) * 100:.2f}%)'
    #
    #     res_dict[device] = device_dict
    #
    # for behavior in Behavior:
    #     results.append([behavior.value] + [res_dict[device][behavior] for device in RaspberryPi])
    #
    # print(tabulate(results, headers=["Behavior"] + [pi.value for pi in RaspberryPi], tablefmt="pretty"))
