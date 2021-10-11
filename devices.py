from copy import deepcopy
from math import nan
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom_types import ModelArchitecture
from models import mlp_model, auto_encoder_model


class Participant:
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray,
                 valid_x: np.ndarray, valid_y: np.ndarray,
                 batch_size: int = 64, batch_size_valid=64, y_type: torch.dtype = torch.float):
        data_train = torch.utils.data.TensorDataset(
            torch.from_numpy(train_x).type(torch.float),
            torch.from_numpy(train_y).type(y_type)
        )
        self.data_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)

        # shuffling is not really necessary
        # and batch size should also be irrelevant, can be set to whatever fits in memory
        data_valid = torch.utils.data.TensorDataset(
            torch.from_numpy(valid_x).type(torch.float),
            torch.from_numpy(valid_y).type(y_type)
        )
        self.valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=batch_size_valid, shuffle=True)
        self.validation_losses = []

        self.model = None
        self.threshold = nan

    def train(self, optimizer, loss_function, num_local_epochs: int = 5):
        if self.model is None:
            raise ValueError("No model set on participant!")

        epoch_losses = []
        validation_losses = []
        for le in range(num_local_epochs):
            self.model.train()
            current_losses = []
            for batch_idx, (x, y) in enumerate(self.data_loader):
                x, y = x, y  # x.cuda(), y.cuda()
                optimizer.zero_grad()
                model_predictions = self.model(x)
                loss = loss_function(model_predictions, y)
                loss.backward()
                optimizer.step()
                current_losses.append(loss.item())
            epoch_losses.append(sum(current_losses) / len(current_losses))
            # print(f'Training Loss in epoch {le + 1}: {epoch_losses[le]}')

            with torch.no_grad():
                self.model.eval()
                current_losses = []
                for batch_idx, (x, y) in enumerate(self.valid_loader):
                    x, y = x, y  # x.cuda(), y.cuda()
                    model_predictions = self.model(x)
                    loss = loss_function(model_predictions, y)
                    current_losses.append(loss.item())
                validation_losses.append(sum(current_losses) / len(current_losses))
                # print(f'Validation Loss in epoch {le + 1}: {sum(current_losses) / len(current_losses)}')

            self.validation_losses = self.validation_losses + validation_losses

            if validation_losses[le] < 1e-4 or (le > 0 and (validation_losses[le] - validation_losses[le - 1]) > 1e-4):
                # print(f"Early stopping criterion reached in epoch {le + 1}")
                return

    def get_model(self):
        return self.model

    def set_model(self, model: torch.nn.Module):
        self.model = model


# TODO: check
#  robust AE methods for adversarial part: https://ieeexplore.ieee.org/document/9099561
class AutoEncoderParticipant(Participant):

    def train(self, optimizer, loss_function, num_local_epochs: int = 5):
        if self.model is None:
            raise ValueError("No model set on participant!")
        epoch_losses = []
        for le in range(num_local_epochs):
            self.model.train()
            current_losses = []
            for batch_idx, (x, _) in enumerate(self.data_loader):
                x = x  # x.cuda()
                optimizer.zero_grad()
                model_out = self.model(x)
                loss = loss_function(model_out, x)
                loss.backward()
                optimizer.step()
                current_losses.append(loss.item())
            epoch_losses.append(sum(current_losses) / len(current_losses))
            # print(f'Training Loss in epoch {le + 1}: {epoch_losses[le]}')

    def determine_threshold(self) -> float:
        mses = []
        self.model.eval()
        with torch.no_grad():
            loss_function = torch.nn.MSELoss(reduction='sum')
            for batch_idx, (x, _) in enumerate(self.valid_loader):
                x = x  # x.cuda()
                model_out = self.model(x)
                loss = loss_function(model_out, x)
                mses.append(loss.item())
        mses = np.array(mses)
        return mses.mean() + mses.std()


class Server:
    def __init__(self, participants: List[Participant],
                 model_architecture: ModelArchitecture = ModelArchitecture.MLP_MONO_CLASS):
        assert len(participants) > 0, "At least one participant is required!"
        assert model_architecture is not None, "Model architecture has to be supplied!"
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

            w_avg = deepcopy(self.participants[0].get_model().state_dict())
            for key in w_avg.keys():
                for p in self.participants[1:]:
                    w_avg[key] += p.get_model().state_dict()[key]
                w_avg[key] = torch.div(w_avg[key], len(self.participants))
            self.global_model.load_state_dict(w_avg)

            for p in self.participants:
                p.get_model().load_state_dict(deepcopy(w_avg))

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



# if threshold is selected like in the normal AutoEnc. Participant sending random weights is not an efficient attack
# a model with 100% malicious participants still recognizes some behaviors with 100% accuracy without attacking the threshold
class RandomWeightAdversary(AutoEncoderParticipant):
    def train(self, optimizer, loss_function, num_local_epochs: int = 5):
        state_dict = self.model.state_dict()
        new_dict = deepcopy(state_dict)
        for key in state_dict.keys():
            new_dict[key] = torch.rand(state_dict[key].size())
        self.model.load_state_dict(new_dict)

# inefficient attack
class ExaggerateThresholdAdversary(AutoEncoderParticipant):

    def determine_threshold(self) -> float:
        mses = []
        self.model.eval()
        with torch.no_grad():
            loss_function = torch.nn.MSELoss(reduction='sum')
            for batch_idx, (x, _) in enumerate(self.valid_loader):
                x = x  # x.cuda()
                model_out = self.model(x)
                loss = loss_function(model_out, x)
                mses.append(loss.item())
        mses = np.array(mses)
        # just way more than normal participants threshold
        return mses.mean() + 100*mses.std()

# inefficient attack
class UnderstateThresholdAdversary(AutoEncoderParticipant):

    def determine_threshold(self) -> float:
        mses = []
        self.model.eval()
        with torch.no_grad():
            loss_function = torch.nn.MSELoss(reduction='sum')
            for batch_idx, (x, _) in enumerate(self.valid_loader):
                x = x  # x.cuda()
                model_out = self.model(x)
                loss = loss_function(model_out, x)
                mses.append(loss.item())
        mses = np.array(mses)
        # just way more than normal participants threshold
        return mses.mean() - 100*mses.std()

