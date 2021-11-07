import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from aggregation import Server
from custom_types import Behavior, RaspberryPi, ModelArchitecture, Scaler
from data_handler import DataHandler
from participants import MLPParticipant
from utils import calculate_metrics

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)
    os.chdir("..")

    print("Similarity of attacks:\n"
          "Can the knowledge of one attack be used to detect another attack?")

    normals = [Behavior.NORMAL, Behavior.NORMAL_V2]
    attacks = [val for val in Behavior if val not in normals]
    all_accs = []
    device = RaspberryPi.PI4_2GB_WC
    test_devices = [(device, {other_attack: 100}) for other_attack in attacks]
    eval_labels = [other_attack.value for other_attack in attacks]
    train_labels = []
    for attack in attacks:
        train_sets, test_sets = DataHandler.get_all_clients_data(
            [(device, {Behavior.NORMAL: 250, attack: 250},
              {Behavior.NORMAL: 250, attack: 25})],
            test_devices)

        train_sets, test_sets = DataHandler.scale(train_sets, test_sets, scaling=Scaler.MINMAX_SCALER)
        participants = [MLPParticipant(x_train, y_train, x_valid, y_valid) for
                        x_train, y_train, x_valid, y_valid in train_sets]
        server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
        server.train_global_model(aggregation_rounds=1, local_epochs=15)
        att_accs = []
        for i, (x_test, y_test) in enumerate(test_sets):
            y_predicted = server.predict_using_global_model(x_test)
            acc, f1, conf_mat = calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
            att_accs.append(acc)
        all_accs.append(att_accs)
        train_labels.append(attack.value)

    train_sets, test_sets = DataHandler.get_all_clients_data(
        [(device, {Behavior.NORMAL: 500, Behavior.FREEZE: 250, Behavior.SPOOF: 250},
          {Behavior.NORMAL: 50, Behavior.FREEZE: 25, Behavior.SPOOF: 25})],
        test_devices)

    train_sets, test_sets = DataHandler.scale(train_sets, test_sets, scaling=Scaler.MINMAX_SCALER)
    participants = [MLPParticipant(x_train, y_train, x_valid, y_valid) for
                    x_train, y_train, x_valid, y_valid in train_sets]
    server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
    server.train_global_model(aggregation_rounds=1, local_epochs=15)
    att_accs = []
    for i, (x_test, y_test) in enumerate(test_sets):
        y_predicted = server.predict_using_global_model(x_test)
        acc, f1, conf_mat = calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
        att_accs.append(acc)
    all_accs.append(att_accs)
    train_labels.append("freeze, spoof")

    train_sets, test_sets = DataHandler.get_all_clients_data(
        [(device, {Behavior.NORMAL: 500, Behavior.REPEAT: 250, Behavior.HOP: 250},
          {Behavior.NORMAL: 50, Behavior.REPEAT: 25, Behavior.HOP: 25})],
        test_devices)

    train_sets, test_sets = DataHandler.scale(train_sets, test_sets, scaling=Scaler.MINMAX_SCALER)
    participants = [MLPParticipant(x_train, y_train, x_valid, y_valid) for
                    x_train, y_train, x_valid, y_valid in train_sets]
    server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
    server.train_global_model(aggregation_rounds=1, local_epochs=15)
    att_accs = []
    for i, (x_test, y_test) in enumerate(test_sets):
        y_predicted = server.predict_using_global_model(x_test)
        acc, f1, conf_mat = calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
        att_accs.append(acc)
    all_accs.append(att_accs)
    train_labels.append("repeat, hop")

    train_sets, test_sets = DataHandler.get_all_clients_data(
        [(device, {Behavior.NORMAL: 500, Behavior.DELAY: 166, Behavior.FREEZE: 166, Behavior.NOISE: 166},
          {Behavior.NORMAL: 50, Behavior.DELAY: 17, Behavior.FREEZE: 17, Behavior.NOISE: 17})],
        test_devices)

    train_sets, test_sets = DataHandler.scale(train_sets, test_sets, scaling=Scaler.MINMAX_SCALER)
    participants = [MLPParticipant(x_train, y_train, x_valid, y_valid) for
                    x_train, y_train, x_valid, y_valid in train_sets]
    server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
    server.train_global_model(aggregation_rounds=1, local_epochs=15)
    att_accs = []
    for i, (x_test, y_test) in enumerate(test_sets):
        y_predicted = server.predict_using_global_model(x_test)
        acc, f1, conf_mat = calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
        att_accs.append(acc)
    all_accs.append(att_accs)
    train_labels.append("delay, freeze, noise")

    hm = sns.heatmap(np.array(all_accs), xticklabels=eval_labels, yticklabels=train_labels)
    plt.title('Heatmap of Device ' + device.value, fontsize=15)
    plt.xlabel('Predicting', fontsize=12)
    plt.ylabel('Trained on', fontsize=12)
    plt.show()
    hm.get_figure().savefig(f"data_plot_class_similarity_{device.value}.png")
