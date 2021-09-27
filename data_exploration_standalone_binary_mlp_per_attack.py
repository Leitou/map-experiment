from typing import Dict

import numpy as np
import torch
from tabulate import tabulate

from custom_types import Behavior, RaspberryPi, ModelArchitecture
from devices import Server, Participant
from data_handler import DataHandler
from utils import calculate_metrics

# implicitly given by heatmaps
if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    print("Difficulty of detecting attacks with an mlp:\n"
          "What attacks are hardest to detect in what configuration?")

    normals = [Behavior.NORMAL, Behavior.NORMAL_V2]
    attacks = [val for val in Behavior if val not in normals]
    #attacks = [Behavior.FREEZE, Behavior.REPEAT]
    for normal in normals:
        other_normal = Behavior.NORMAL_V2 if normal == Behavior.NORMAL else Behavior.NORMAL
        labels = ["Behavior"] + [pi.value for pi in RaspberryPi]
        res_dict: Dict[RaspberryPi, Dict[Behavior, str]] = {}
        results = []
        for device in RaspberryPi:
            device_dict: Dict[Behavior, str] = {}
            for attack in attacks:
                train_sets, test_sets = DataHandler.get_all_clients_data(
                    [(device, {normal: 3000, attack: 500}, {normal: 300, attack: 50})],
                    [(device, {normal: 150}),
                     (device, {other_normal: 150}),
                     (device, {attack: 150})])

                train_sets, test_sets = DataHandler.scale(train_sets, test_sets, True)

                participants = [Participant(x_train, y_train, x_valid, y_valid) for
                                x_train, y_train, x_valid, y_valid in train_sets]
                server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
                server.train_global_model(aggregation_rounds=5)

                att_rows = []
                for i, (x_test, y_test) in enumerate(test_sets):
                    y_predicted = server.predict_using_global_model(x_test)
                    acc, f1, conf_mat = calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
                    att_rows.append([['nml', 'nml_v2', attack.value[:3]][i], f'{acc*100:.2f}%'])
                device_dict[attack] = tabulate(att_rows, tablefmt="latex")
            res_dict[device] = device_dict
        for attack in attacks:
            results.append([attack.value] + [res_dict[device][attack] for device in RaspberryPi])

        print("Normal Behavior:", normal)
        print(tabulate(results, headers=labels, tablefmt="latex_raw"))
