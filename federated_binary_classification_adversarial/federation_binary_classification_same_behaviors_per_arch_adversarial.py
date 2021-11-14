import os
from pathlib import Path
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

    if not Path(f"{Path(__file__).parent}/results_{Path(__file__).stem}/").is_dir():
        Path(f"{Path(__file__).parent}/results_{Path(__file__).stem}").mkdir()

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
    colors = ['limegreen', 'gold', 'tab:orange', 'orangered', "violet"]
    colors = ['#87d64b', '#fae243', '#f8961e', '#ff4d36', '#8f00ff']
    text_colors = ['#456e25', '#998a28', '#b06a13', '#b33424', '#8f00ff']

    max_num_adv = 5
    for agg in AggregationMechanism:
        for num_adv in range(max_num_adv):
            print(f"Using {num_adv} adversaries")

            # define adversarial/honest federation composition
            adversaries = [AllLabelFlipAdversary(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                           x_train, y_train, x_valid, y_valid in train_sets_fed[4:4 + num_adv]]

            participants = [MLPParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                            x_train, y_train, x_valid, y_valid in train_sets_fed[:4]] + [
                               MLPParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                               x_train, y_train, x_valid, y_valid in train_sets_fed[4 + num_adv:]]

            server = Server(adversaries + participants, ModelArchitecture.MLP_MONO_CLASS, aggregation_mechanism=agg)

            filepath = f"{Path(__file__).parent}/results_{Path(__file__).stem}/model_{agg.value}_{num_adv}_adv.pt"
            if not Path(filepath).is_file():
                # train federation
                # TODO: implement first only adversaries of one device type -> rasp3, then maybe multiple device types
                server.train_global_model(aggregation_rounds=15)
                torch.save(server.global_model.state_dict(), filepath)

            else:
                server.global_model.load_state_dict(torch.load(filepath))


            # predict on all test data
            y_predicted = server.predict_using_global_model(global_test_set[0][0])
            acc, f1, _ = FederationUtils.calculate_metrics(global_test_set[0][1].flatten(), y_predicted.flatten().numpy())
            print(f"{f1:.2f}")


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
