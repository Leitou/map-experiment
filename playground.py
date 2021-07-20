import torch

from custom_types import Attack, RaspberryPi, ModelArchitecture
from localops import Participant, Server
from sampling import DataSampler
import numpy as np

from sklearn.metrics import f1_score

if __name__ == "__main__":
    print(f'GPU available: {torch.cuda.is_available()}')
    print("Checking if new device can recognize known attack")
    torch.random.manual_seed(0)
    np.random.seed(0)
    train_sets, test_sets = DataSampler.get_all_clients_train_data_and_scaler(
        [(RaspberryPi.PI4_4GB, {Attack.NORMAL: 2500, Attack.SPOOF: 2500}),
         (RaspberryPi.PI3_2GB, {Attack.NORMAL: 2500}),
         (RaspberryPi.PI3_2GB, {Attack.NORMAL_V2: 2500})],
        [(RaspberryPi.PI3_2GB, {Attack.NORMAL: 750, Attack.SPOOF: 250})])

    participants = [Participant(x, y) for x, y in train_sets]
    server = Server(participants, ModelArchitecture.MLP_MONO_CLASS)
    server.train_global_model(aggregation_rounds=5)
    x_test, y_test = test_sets[0]
    y_predicted = server.predict_using_global_model(x_test)
    correct = (torch.from_numpy(y_test).flatten() == y_predicted).count_nonzero()

    f1 = f1_score(y_test.flatten(), y_predicted.flatten().numpy())
    print(f"Test Accuracy: {correct * 100 / len(y_predicted):.2f}%, F1 score: {f1 * 100:.2f}%")
    print("Done!")
