import numpy as np
import torch

from custom_types import Attack, RaspberryPi, ModelArchitecture
from devices import Participant, Server
from data_handler import DataHandler
from utils import print_experiment_scores

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    print(f'GPU available: {torch.cuda.is_available()}')
    print("Starting demo experiment: Federated vs Centralized Binary Classification")

    # print("Use case FL: Detect attacks on different architectures"
    #       "Is the Federated Model able to recognize attacks on a different device architecture than "
    #       "the one observed in training data?")
    # train_sets, test_sets = DataSampler.get_all_clients_train_data_and_scaler(
    #     [(RaspberryPi.PI4_4GB, {Attack.NORMAL: 2000, Attack.SPOOF: 2000}, {Attack.NORMAL: 50, Attack.SPOOF: 50}),
    #      (RaspberryPi.PI3_2GB, {Attack.NORMAL: 2000}, {Attack.NORMAL: 100}),
    #      (RaspberryPi.PI4_2GB, {Attack.NORMAL: 2000}, {Attack.NORMAL: 100})],
    #     [(RaspberryPi.PI3_2GB, {Attack.NORMAL: 750, Attack.SPOOF: 250})])

    print("Use case FL: Zero Day Detection, hoping for similarities\n"
          "Is the Federated model able to detect attacks that have not been observed previously,\n "
          "i.e. attacks that are not at all in the training data?")
    # e.g disorder and mimic, mimic and spoof, or spoof and noise could from the type of attack/behavior
    # or affected frequency be possibly expected to have a similar effect on different devices
    # This could be used to train e.g. frequency band sensitive predictors or
    # such that are sensitive to power spectrum swaps, value copying etc

    # 1.
    # train_sets, test_sets = DataSampler.get_all_clients_train_data_and_scaler(
    #     [(RaspberryPi.PI4_4GB, {Attack.NORMAL: 2000, Attack.DISORDER: 2000}, {Attack.NORMAL: 50, Attack.DISORDER: 50}),
    #      (RaspberryPi.PI3_2GB, {Attack.NORMAL: 2000}, {Attack.NORMAL: 100}),
    #      (RaspberryPi.PI4_2GB, {Attack.NORMAL: 2000, Attack.DISORDER: 2000}, {Attack.NORMAL: 50, Attack.DISORDER: 50})],
    #     [(RaspberryPi.PI3_2GB, {Attack.NORMAL: 750, Attack.SPOOF: 250})])
    # 2.
    # train_sets, test_sets = DataSampler.get_all_clients_train_data_and_scaler(
    #     [(RaspberryPi.PI4_4GB, {Attack.NORMAL: 2000, Attack.MIMIC: 2000}, {Attack.NORMAL: 50, Attack.MIMIC: 50}),
    #      (RaspberryPi.PI3_2GB, {Attack.NORMAL: 2000}, {Attack.NORMAL: 100}),
    #      (RaspberryPi.PI4_2GB, {Attack.NORMAL: 2000, Attack.DISORDER: 2000}, {Attack.NORMAL: 50, Attack.DISORDER: 50})],
    #     [(RaspberryPi.PI3_2GB, {Attack.NORMAL: 750, Attack.SPOOF: 250})])
    # 3. dependent on start seed few till many recognized
    # -> federated seems to have some benefits due to the averaging across devices compared to baseline
    # -> set different random seeds for testing multiple ways, resp. comment it out
    train_sets, test_sets = DataHandler.get_all_clients_data(
        [(RaspberryPi.PI4_4GB, {Attack.NORMAL: 2000, Attack.DISORDER: 2000}, {Attack.NORMAL: 50, Attack.DISORDER: 50}),
         (RaspberryPi.PI3_2GB, {Attack.NORMAL: 2000}, {Attack.NORMAL: 100}),
         (RaspberryPi.PI4_2GB_BC, {Attack.NORMAL: 2000, Attack.SPOOF: 2000}, {Attack.NORMAL: 50, Attack.SPOOF: 50})],
        [(RaspberryPi.PI3_2GB, {Attack.NORMAL: 750, Attack.NOISE: 250})])

    participants = [Participant(x_train, y_train, x_valid, y_valid) for
                    x_train, y_train, x_valid, y_valid in train_sets]
    server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
    server.train_global_model(aggregation_rounds=5)
    x_test, y_test = test_sets[0]

    y_predicted = server.predict_using_global_model(x_test)
    print_experiment_scores(y_test.flatten(), y_predicted.flatten().numpy(), federated=True)

    print("------------------------------ CENTRALIZED BASELINE -----------------------")
    x_train_all = np.concatenate(tuple(x_train for x_train, y_train, x_valid, y_valid in train_sets))
    y_train_all = np.concatenate(tuple(y_train for x_train, y_train, x_valid, y_valid in train_sets))
    x_valid_all = np.concatenate(tuple(x_valid for x_train, y_train, x_valid, y_valid in train_sets))
    y_valid_all = np.concatenate(tuple(y_valid for x_train, y_train, x_valid, y_valid in train_sets))
    central_participants = [Participant(x_train_all, y_train_all, x_valid_all, y_valid_all)]
    central_server = Server(central_participants, ModelArchitecture.MLP_MONO_CLASS)
    central_server.train_global_model(aggregation_rounds=5)

    y_predicted_central = central_server.predict_using_global_model(x_test)
    print_experiment_scores(y_test.flatten(), y_predicted_central.flatten().numpy(), federated=False)
