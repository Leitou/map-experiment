import torch

from custom_types import Attack, RaspberryPi, ModelArchitecture
from localops import Participant, Server
from sampling import DataSampler
import numpy as np

from sklearn.metrics import f1_score

if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    print(f'GPU available: {torch.cuda.is_available()}')
    print("Starting demo experiment: Checking if new device can recognize known attack")

    train_sets, test_sets = DataSampler.get_all_clients_train_data_and_scaler(
        [(RaspberryPi.PI4_4GB, {Attack.NORMAL: 2000, Attack.SPOOF: 2000}),
         (RaspberryPi.PI3_2GB, {Attack.NORMAL: 2000}),
         (RaspberryPi.PI3_2GB, {Attack.NORMAL_V2: 2000})],
        [(RaspberryPi.PI3_2GB, {Attack.NORMAL: 750, Attack.SPOOF: 250})])

    participants = [Participant(x, y) for x, y in train_sets]
    server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
    server.train_global_model(aggregation_rounds=5)
    x_test, y_test = test_sets[0]

    y_predicted = server.predict_using_global_model(x_test)
    correct = (torch.from_numpy(y_test).flatten() == y_predicted).count_nonzero()
    f1 = f1_score(y_test.flatten(), y_predicted.flatten().numpy())
    print(f"Test Accuracy: {correct * 100 / len(y_predicted):.2f}%, F1 score: {f1 * 100:.2f}%")

    print("------------------------------ CENTRALIZED BASELINE -----------------------")
    x_all = np.concatenate(tuple(x for x, y in train_sets))
    y_all = np.concatenate(tuple(y for x, y in train_sets))
    central_participants = [Participant(x_all, y_all)]
    central_server = Server(central_participants, ModelArchitecture.MLP_MONO_CLASS)
    central_server.train_global_model(aggregation_rounds=5)

    y_predicted_central = central_server.predict_using_global_model(x_test)
    correct_central = (torch.from_numpy(y_test).flatten() == y_predicted_central).count_nonzero()
    f1_central = f1_score(y_test.flatten(), y_predicted_central.flatten().numpy())
    print(f"Test Accuracy: {correct_central * 100 / len(y_predicted_central):.2f}%, F1 score: {f1_central * 100:.2f}%")
