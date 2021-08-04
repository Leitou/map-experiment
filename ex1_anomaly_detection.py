import numpy as np
import torch

from custom_types import Attack, RaspberryPi, ModelArchitecture
from devices import AutoEncoderParticipant, Server
from sampling import DataSampler
from utils import print_experiment_scores

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    print(f'GPU available: {torch.cuda.is_available()}')
    print("Federated Anomaly Detection")

    print("Use case centralized: Anomaly/Zero Day Detection"
          "Is the Central model able to detect attacks as anomalies,\nie. recognize the difference from attacks"
          " to normal samples? Which attacks are hardest to detect?")

    device = RaspberryPi.PI4_2GB
    test_devices = [(device, {Attack.NORMAL: 150, Attack.DELAY: 150}),
                    (device, {Attack.NORMAL: 150, Attack.DISORDER: 150}),
                    (device, {Attack.NORMAL: 150, Attack.FREEZE: 150}),
                    (device, {Attack.NORMAL: 150, Attack.HOP: 150}),
                    (device, {Attack.NORMAL: 150, Attack.MIMIC: 150}),
                    (device, {Attack.NORMAL: 150, Attack.NOISE: 150}),
                    (device, {Attack.NORMAL: 150, Attack.REPEAT: 150}),
                    (device, {Attack.NORMAL: 150, Attack.SPOOF: 150})]
    train_sets, test_sets = DataSampler.get_all_clients_train_data_and_scaler(
        [(device, {Attack.NORMAL: 1500}, {Attack.NORMAL: 150}),
         #(device, {Attack.NORMAL: 1500}, {Attack.NORMAL: 150}),
         #(device, {Attack.NORMAL: 1500}, {Attack.NORMAL: 150}),
         #(device, {Attack.NORMAL: 1500}, {Attack.NORMAL: 150})
         ],
        test_devices)

    participants = [AutoEncoderParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                    x_train, y_train, x_valid, y_valid in train_sets]
    server = Server(participants, ModelArchitecture.AUTO_ENCODER)
    server.train_global_model(aggregation_rounds=5)

    # TODO: save and plot -> confusion matrix: seaborn, table: table..
    for i, (x_test, y_test) in enumerate(test_sets):
        print("-------------------------------------------")
        print("device:", test_devices[i][0], "attack:",
              set(test_devices[i][1].keys()) - {Attack.NORMAL, Attack.NORMAL_V2})
        y_predicted = server.predict_using_global_model(x_test)
        correct = (torch.from_numpy(y_test).flatten() == y_predicted).count_nonzero()
        print_experiment_scores(y_test.flatten(), y_predicted.flatten().numpy(), correct, federated=True)
