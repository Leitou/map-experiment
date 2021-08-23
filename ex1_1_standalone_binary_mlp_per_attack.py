import numpy as np
import torch
from tabulate import tabulate

from custom_types import Attack, RaspberryPi, ModelArchitecture
from devices import Server, Participant
from data_handler import DataHandler
from utils import calculate_metrics

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    print("Difficulty of detecting attacks with an mlp:\n"
          "What attacks are hardest to detect in what configuration?")

    normals = [Attack.NORMAL, Attack.NORMAL_V2]
    attacks = [val for val in Attack if val not in normals]
    for device in RaspberryPi:
        for normal in normals:
            results = []
            for attack in attacks:
                train_sets, test_sets = DataHandler.get_all_clients_data(
                    [(device, {normal: 1500, attack: 500}, {normal: 150, attack: 50})],
                    [(device, {normal: 500, attack: 150})])

                train_sets, test_sets = DataHandler.scale(train_sets, test_sets, True)

                participants = [Participant(x_train, y_train, x_valid, y_valid) for
                                x_train, y_train, x_valid, y_valid in train_sets]
                server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
                server.train_global_model(aggregation_rounds=5)

                for i, (x_test, y_test) in enumerate(test_sets):
                    y_predicted = server.predict_using_global_model(x_test)
                    acc, f1, _ = calculate_metrics(y_test.flatten(), y_predicted.flatten().numpy())
                    results.append([device.value, normal.value, attack, f'{acc * 100:.2f}%', f'{f1 * 100:.2f}%'])

            print(tabulate(results, headers=['Device', 'Normal', 'Attack', 'Accuracy', 'F1 score'], tablefmt="pretty"))
