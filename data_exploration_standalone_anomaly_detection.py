from typing import Dict

import numpy as np
import torch

from custom_types import Behavior, RaspberryPi, ModelArchitecture, Scaler
from aggregation import Server
from participants import AutoEncoderParticipant
from data_handler import DataHandler
from utils import calculate_metrics
from tabulate import tabulate

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    print("Use case standalone devices: Anomaly/Zero Day Detection\n"
          "Is the standalone model able to detect attacks as anomalies,\nie. recognize the difference from attacks"
          " to normal samples? Which attacks are hardest to detect?\n")

    normals = [Behavior.NORMAL, Behavior.NORMAL_V2]
    for normal in normals:
        labels = ["Behavior"] + [pi.value for pi in RaspberryPi]
        results = []
        res_dict: Dict[RaspberryPi, Dict[Behavior, str]] = {}
        for device in RaspberryPi:
            device_dict: Dict[Behavior, str] = {}
            test_devices = [(device, {behavior: 150}) for behavior in Behavior]
            train_sets, test_sets = DataHandler.get_all_clients_data(
                [(device, {normal: 3000}, {normal: 1000})],
                test_devices)

            train_sets, test_sets = DataHandler.scale(train_sets, test_sets, True)

            participants = [AutoEncoderParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                            x_train, y_train, x_valid, y_valid in train_sets]
            server = Server(participants, ModelArchitecture.AUTO_ENCODER)
            server.train_global_model(aggregation_rounds=5)
            for i, (x_test, y_test) in enumerate(test_sets):
                y_predicted = server.predict_using_global_model(x_test)
                behavior = list(Behavior)[i]
                acc, f1, conf_mat = calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
                if acc == 1.0:
                    device_dict[behavior] = '100.00%'
                else:
                    (tn, fp, fn, tp) = conf_mat.ravel()
                    if behavior in normals:
                        device_dict[behavior] = f'{(100 * tn / (tn + fp)):.2f}%'
                    else:
                        device_dict[behavior] = f'{(100 * tp / (tp + fn)):.2f}%'
            res_dict[device] = device_dict
        for attack in [behavior for behavior in Behavior]:
            results.append([attack.value] + [res_dict[device][attack] for device in RaspberryPi])
        print("Normal Behavior:", normal)
        print(tabulate(results, headers=labels, tablefmt="latex"))
