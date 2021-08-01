import numpy as np
import torch
from sklearn.metrics import f1_score

from custom_types import Attack, RaspberryPi, ModelArchitecture
from devices import Participant, Server
from sampling import DataSampler

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    print(f'GPU available: {torch.cuda.is_available()}')
    print("Starting demo multiclass experiment: Checking if new device can recognize known attacks")

    train_sets, test_sets = DataSampler.get_all_clients_train_data_and_scaler(
        [(RaspberryPi.PI4_4GB, {Attack.NORMAL: 2000, Attack.SPOOF: 1000, Attack.NOISE: 1000},
          {Attack.NORMAL: 100, Attack.SPOOF: 50, Attack.NOISE: 50}),
         (RaspberryPi.PI4_2GB, {Attack.NORMAL: 2000}, {Attack.NORMAL: 200}),
         (RaspberryPi.PI3_2GB, {Attack.NORMAL: 2000}, {Attack.NORMAL: 200})],
        [(RaspberryPi.PI3_2GB, {Attack.NORMAL: 700, Attack.SPOOF: 150, Attack.NOISE: 150})], multi_class=True)

    participants = [Participant(x_train, y_train.flatten(), x_valid, y_valid.flatten(), y_type=torch.long) for
                    x_train, y_train, x_valid, y_valid in train_sets]
    server = Server(participants, ModelArchitecture.MLP_MULTI_CLASS)
    server.train_global_model(aggregation_rounds=5)
    x_test, y_test = test_sets[0]
    y_predicted = server.predict_using_global_model(x_test)
    correct = (torch.from_numpy(y_test).flatten() == y_predicted).count_nonzero()

    f1 = f1_score(y_test.flatten(), y_predicted.flatten().numpy(), average='micro')
    print(f"Test Accuracy: {correct * 100 / len(y_predicted):.2f}%, F1 score: {f1 * 100:.2f}%")

    print("------------------------------ CENTRALIZED BASELINE -----------------------")
    x_train_all = np.concatenate(tuple(x_train for x_train, y_train, x_valid, y_valid in train_sets))
    y_train_all = np.concatenate(tuple(y_train for x_train, y_train, x_valid, y_valid in train_sets))
    x_valid_all = np.concatenate(tuple(x_valid for x_train, y_train, x_valid, y_valid in train_sets))
    y_valid_all = np.concatenate(tuple(y_valid for x_train, y_train, x_valid, y_valid in train_sets))
    central_participants = [
        Participant(x_train_all, y_train_all.flatten(), x_valid_all, y_valid_all.flatten(), y_type=torch.long)]
    central_server = Server(central_participants, ModelArchitecture.MLP_MULTI_CLASS)
    central_server.train_global_model(aggregation_rounds=5)

    y_predicted_central = central_server.predict_using_global_model(x_test)
    correct_central = (torch.from_numpy(y_test).flatten() == y_predicted_central).count_nonzero()
    f1_central = f1_score(y_test.flatten(), y_predicted_central.flatten().numpy(), average='micro')
    print(f"Test Accuracy: {correct_central * 100 / len(y_predicted_central):.2f}%, F1 score: {f1_central * 100:.2f}%")
