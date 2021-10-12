from tqdm import tqdm
from typing import List
from copy import deepcopy
from math import nan
import torch
from torch.utils.data import DataLoader

from custom_types import ModelArchitecture, AggregationMechanism
from models import mlp_model, auto_encoder_model
from participants import Participant, AutoEncoderParticipant


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
        self.global_threshold = nan

    def train_global_model(self, aggregation_rounds: int = 15, local_epochs: int = 5):
        # initialize model
        for p in self.participants:
            if self.model_architecture == ModelArchitecture.MLP_MONO_CLASS:
                p.set_model(mlp_model(in_features=68, out_classes=1))  # .cuda()
            elif self.model_architecture == ModelArchitecture.MLP_MULTI_CLASS:
                p.set_model(mlp_model(in_features=68, out_classes=9))  # .cuda()
            elif self.model_architecture == ModelArchitecture.AUTO_ENCODER:
                p.set_model(auto_encoder_model(in_features=68))
            else:
                raise ValueError("Not yet implemented!")
        for _ in tqdm(range(aggregation_rounds), unit="fedavg round", leave=False):
            for p in self.participants:
                p.train(optimizer=torch.optim.SGD(p.get_model().parameters(), lr=0.001, momentum=0.9),
                        loss_function=torch.nn.BCEWithLogitsLoss(reduction='sum') if
                        self.model_architecture == ModelArchitecture.MLP_MONO_CLASS
                        else (torch.nn.CrossEntropyLoss(reduction='sum')
                              if self.model_architecture == ModelArchitecture.MLP_MULTI_CLASS else
                              torch.nn.MSELoss(reduction='sum')),
                        num_local_epochs=local_epochs)

            if self.aggregation_mechanism == AggregationMechanism.FED_AVG:
                new_weights = self.fedavg()
            # TODO add & implement multiple ways of aggregation

            self.global_model.load_state_dict(new_weights)
            for p in self.participants:
                p.get_model().load_state_dict(deepcopy(new_weights))

    def predict_using_global_model(self, x):
        if self.model_architecture == ModelArchitecture.AUTO_ENCODER:
            thresholds = []
            for p in self.participants:
                # Quick and dirty casting
                p: AutoEncoderParticipant = p
                thresholds.append(p.determine_threshold())
            self.global_threshold = sum(thresholds) / len(thresholds)

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
