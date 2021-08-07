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
    print("Starting demo experiment: Federated vs Centralized Anomaly Detection")

    print("Use case FL: Anomaly/Zero Day Detection"
          "Is the Federated model able to detect attacks as anomalies,\nie. recognize the difference from attacks"
          " to normal samples? Which attacks are hardest to detect?")

    #  1. update_weights on exclusively normal samples + untrained model
    #  2. inference on new testdata with malicious samples + the trained model from 1.

    train_sets, test_sets = DataSampler.get_all_clients_data_and_scale(
        [(RaspberryPi.PI4_4GB, {Attack.NORMAL: 2000}, {Attack.NORMAL: 100}),
         (RaspberryPi.PI3_2GB, {Attack.NORMAL: 2000}, {Attack.NORMAL: 100}),
         (RaspberryPi.PI4_2GB, {Attack.NORMAL: 2000}, {Attack.NORMAL: 100})],
        [  # (RaspberryPi.PI3_2GB, {Attack.NORMAL: 500, Attack.SPOOF: 250, Attack.NOISE: 250}),
            # (RaspberryPi.PI4_2GB, {Attack.NORMAL: 500, Attack.SPOOF: 250, Attack.NOISE: 250}),
            (RaspberryPi.PI4_4GB, {Attack.NORMAL: 500, Attack.SPOOF: 250, Attack.NOISE: 250})])

    participants = [AutoEncoderParticipant(x_train, y_train, x_valid, y_valid, batch_size_valid=1) for
                    x_train, y_train, x_valid, y_valid in train_sets]
    server = Server(participants, ModelArchitecture.AUTO_ENCODER)
    server.train_global_model(aggregation_rounds=5)
    # no local inference
    x_test, y_test = test_sets[0]
    for x, y in test_sets[1:]:
        x_test = np.concatenate((x_test, x))
        y_test = np.concatenate((y_test, y))

    y_predicted = server.predict_using_global_model(x_test)
    print_experiment_scores(y_test.flatten(), y_predicted.flatten().numpy(), federated=True)

    print("------------------------------ CENTRALIZED BASELINE -----------------------")
    x_train_all = np.concatenate(tuple(x_train for x_train, y_train, x_valid, y_valid in train_sets))
    y_train_all = np.concatenate(tuple(y_train for x_train, y_train, x_valid, y_valid in train_sets))
    x_valid_all = np.concatenate(tuple(x_valid for x_train, y_train, x_valid, y_valid in train_sets))
    y_valid_all = np.concatenate(tuple(y_valid for x_train, y_train, x_valid, y_valid in train_sets))
    central_participants = [AutoEncoderParticipant(x_train_all, y_train_all,
                                                   x_valid_all, y_valid_all, batch_size_valid=1)]
    central_server = Server(central_participants, ModelArchitecture.AUTO_ENCODER)
    central_server.train_global_model(aggregation_rounds=5)

    y_predicted_central = central_server.predict_using_global_model(x_test)
    print_experiment_scores(y_test.flatten(), y_predicted_central.flatten().numpy(), federated=False)
