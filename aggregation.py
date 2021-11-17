from tqdm import tqdm
from typing import List
from copy import deepcopy
from math import nan, isnan
import torch
from torch.utils.data import DataLoader

from custom_types import ModelArchitecture, AggregationMechanism
from models import mlp_model, auto_encoder_model
from participants import Participant, AutoEncoderParticipant


# TODO: https://github.com/fushuhao6/Attack-Resistant-Federated-Learning
class Server:
    def __init__(self, participants: List[Participant],
                 model_architecture: ModelArchitecture = ModelArchitecture.MLP_MONO_CLASS,
                 aggregation_mechanism: AggregationMechanism = AggregationMechanism.FED_AVG):
        assert len(participants) > 0, "At least one participant is required!"
        assert model_architecture is not None, "Model architecture has to be supplied!"
        self.aggregation_mechanism = aggregation_mechanism
        self.model_architecture = model_architecture
        self.participants = participants
        if model_architecture == ModelArchitecture.MLP_MONO_CLASS:
            self.global_model = mlp_model(in_features=68, out_classes=1)  # .cuda()
        elif model_architecture == ModelArchitecture.MLP_MULTI_CLASS:
            self.global_model = mlp_model(in_features=68, out_classes=9)  # .cuda()
        elif model_architecture == ModelArchitecture.AUTO_ENCODER:
            self.global_model = auto_encoder_model(in_features=68)  # .cuda()
        else:
            raise ValueError("Not yet implemented!")

        # initialize models on participants
        for p in self.participants:
            if self.model_architecture == ModelArchitecture.MLP_MONO_CLASS:
                p.set_model(mlp_model(in_features=68, out_classes=1))  # .cuda()
            elif self.model_architecture == ModelArchitecture.MLP_MULTI_CLASS:
                p.set_model(mlp_model(in_features=68, out_classes=9))  # .cuda()
            elif self.model_architecture == ModelArchitecture.AUTO_ENCODER:
                p.set_model(auto_encoder_model(in_features=68))
            else:
                raise ValueError("Not yet implemented!")

        self.global_threshold = nan
        self.participants_thresholds = []
        self.evaluation_thresholds = []

    def train_global_model(self, aggregation_rounds: int = 15, local_epochs: int = 5):
        for _ in tqdm(range(aggregation_rounds), unit="fedavg round", leave=False):
            for p in self.participants:
                p.train(optimizer=torch.optim.SGD(p.get_model().parameters(), lr=0.001, momentum=0.9),
                        loss_function=torch.nn.BCEWithLogitsLoss(reduction='sum') if
                        self.model_architecture == ModelArchitecture.MLP_MONO_CLASS
                        else (torch.nn.CrossEntropyLoss(reduction='sum')
                              if self.model_architecture == ModelArchitecture.MLP_MULTI_CLASS else
                              torch.nn.MSELoss(reduction='sum')),
                        num_local_epochs=local_epochs)

            if self.aggregation_mechanism == AggregationMechanism.TRIMMED_MEAN:
                new_weights = self.trimmed_mean_1()
            else:
                new_weights = self.fedavg()
            # TODO add & implement multiple ways of aggregation

            self.global_model.load_state_dict(deepcopy(new_weights))
            self.load_global_model_into_participants()


    def load_global_model_into_participants(self):
        for p in self.participants:
            p.get_model().load_state_dict(deepcopy(self.global_model.state_dict()))

    def predict_using_global_model(self, x):
        if self.model_architecture == ModelArchitecture.AUTO_ENCODER and isnan(self.global_threshold):
            for p in self.participants:
                # Quick and dirty casting
                p: AutoEncoderParticipant = p
                self.participants_thresholds.append(p.determine_threshold())
            self.global_threshold = max(self.participants_thresholds)

        test_data = torch.utils.data.TensorDataset(
            torch.from_numpy(x).type(torch.float)
        )
        data_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=16 if
                                                  self.model_architecture != ModelArchitecture.AUTO_ENCODER else 1,
                                                  shuffle=False)

        all_predictions = torch.tensor([])  # .cuda()

        self.global_model.eval()
        for idx, (batch_x,) in enumerate(data_loader):
            batch_x = batch_x  # .cuda()
            with torch.no_grad():
                model_predictions = self.global_model(batch_x)
                if self.model_architecture == ModelArchitecture.AUTO_ENCODER:
                    ae_loss = torch.nn.MSELoss(reduction="sum")
                    model_predictions = ae_loss(model_predictions, batch_x).unsqueeze(
                        0)  # unsqueeze as batch_size set to 1
                all_predictions = torch.cat((all_predictions, model_predictions))

        if self.model_architecture == ModelArchitecture.MLP_MONO_CLASS:
            sigmoid = torch.nn.Sigmoid()
            all_predictions = sigmoid(all_predictions).round().type(torch.long)
        elif self.model_architecture == ModelArchitecture.MLP_MULTI_CLASS:
            all_predictions = torch.argmax(all_predictions, dim=1).type(torch.long)
        elif self.model_architecture == ModelArchitecture.AUTO_ENCODER:
            self.evaluation_thresholds += all_predictions.tolist()
            all_predictions = (all_predictions > self.global_threshold).type(torch.long)
        else:
            raise ValueError("Not yet implemented!")

        return all_predictions.flatten()

    def fedavg(self):
        w_avg = deepcopy(self.participants[0].get_model().state_dict())
        for key in w_avg.keys():
            for p in self.participants[1:]:
                w_avg[key] += p.get_model().state_dict()[key]
            w_avg[key] = torch.div(w_avg[key], len(self.participants))
        return w_avg

    def trimmed_mean_1(self):
        return self.__trimmed_mean(1)

    # not so good with only few participants
    def trimmed_mean_2(self):
        return self.__trimmed_mean(2)

    def __trimmed_mean(self, n_trim: int) -> None:
        n = len(self.participants)
        n_remaining = n - 2 * n_trim

        with torch.no_grad():
            state_dict = self.global_model.state_dict()
            for key in state_dict:
                sorted_tensor, _ = torch.sort(
                    torch.stack([model.state_dict()[key] for model in [p.get_model() for p in self.participants]],
                                dim=-1),
                    dim=-1)
                trimmed_tensor = torch.narrow(sorted_tensor, -1, n_trim, n_remaining).type(torch.FloatTensor)
                state_dict[key] = trimmed_tensor.mean(dim=-1)
        return state_dict
