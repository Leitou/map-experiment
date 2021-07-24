from copy import deepcopy
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from custom_types import ModelArchitecture
from models import MLP
from sampling import DataSampler


class Participant:
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray,
                 valid_x: np.ndarray, valid_y: np.ndarray,
                 batch_size: int = 64, y_type: torch.dtype = torch.float):
        data_train = torch.utils.data.TensorDataset(
            torch.from_numpy(train_x).type(torch.float),
            torch.from_numpy(train_y).type(y_type)
        )
        self.data_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)

        # shuffling is not really necessary
        # and batch size should also be irrelevant, can be set to whatever fits in memory
        data_valid = torch.utils.data.TensorDataset(
            torch.from_numpy(valid_x).type(torch.float),
            torch.from_numpy(valid_y).type(y_type)
        )
        self.valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=batch_size, shuffle=True)
        self.validation_losses = []

        self.model = None

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
            print(f'Training Loss in epoch {le + 1}: {epoch_losses[le]}')

            with torch.no_grad():
                self.model.eval()
                current_losses = []
                for batch_idx, (x, y) in enumerate(self.valid_loader):
                    x, y = x, y  # x.cuda(), y.cuda()
                    model_predictions = self.model(x)
                    loss = loss_function(model_predictions, y)
                    current_losses.append(loss.item())
                validation_losses.append(sum(current_losses) / len(current_losses))
                print(f'Validation Loss in epoch {le + 1}: {sum(current_losses) / len(current_losses)}')

            self.validation_losses = self.validation_losses + validation_losses

            if validation_losses[le] < 1e-4 or (le > 0 and (validation_losses[le] - validation_losses[le - 1]) > 1e-4):
                print(f"Early stopping criterion reached in epoch {le + 1}")
                return

    def get_model(self):
        return self.model

    def set_model(self, model: torch.nn.Module):
        self.model = model


# TODO: either we redefine the constructor and use three sets if we also want early stopping here (splitting test)
#  or we do it as indicated below, using train for training and validation for treshold determination
class AutoEncoderParticipant(Participant):
    # TODO: Override train! Will only use train set?
    def train(self, optimizer, loss_function, num_local_epochs: int = 25):
        pass

    # TODO: determine threshold based on normal validation data
    def determine_threshold(self):
        pass


class Server:
    def __init__(self, participants: List[Participant],
                 model_architecture: ModelArchitecture = ModelArchitecture.MLP_MONO_CLASS):
        assert len(participants) > 0, "At least one participant is required!"
        assert model_architecture is not None, "Model architecture has to be supplied!"
        self.model_architecture = model_architecture
        self.participants = participants
        if model_architecture == ModelArchitecture.MLP_MONO_CLASS:
            self.global_model = MLP(in_features=75, out_classes=1)  # .cuda()
        elif model_architecture == ModelArchitecture.MLP_MULTI_CLASS:
            self.global_model = MLP(in_features=75, out_classes=9)  # .cuda()
        else:
            raise ValueError("Not yet implemented!")

    def train_global_model(self, aggregation_rounds: int = 15, local_epochs: int = 5):
        # initialize model
        for p in self.participants:
            if self.model_architecture == ModelArchitecture.MLP_MONO_CLASS:
                p.set_model(MLP(in_features=75, out_classes=1))  # .cuda()
            elif self.model_architecture == ModelArchitecture.MLP_MULTI_CLASS:
                p.set_model(MLP(in_features=75, out_classes=9))  # .cuda()
            else:
                raise ValueError("Not yet implemented!")
        for round_idx in range(aggregation_rounds):
            for p in self.participants:
                p.train(optimizer=torch.optim.SGD(p.get_model().parameters(), lr=0.001, momentum=0.9),
                        loss_function=torch.nn.BCEWithLogitsLoss(reduction='sum') if
                        self.model_architecture == ModelArchitecture.MLP_MONO_CLASS
                        else torch.nn.CrossEntropyLoss(reduction='sum'),
                        num_local_epochs=local_epochs)
            w_avg = deepcopy(self.global_model.state_dict())
            for key in w_avg.keys():
                for p in self.participants:
                    w_avg[key] += p.get_model().state_dict()[key]
                w_avg[key] = torch.div(w_avg[key], len(self.participants))
            self.global_model.load_state_dict(w_avg)

            for p in self.participants:
                p.get_model().load_state_dict(deepcopy(w_avg))

    def predict_using_global_model(self, x):
        test_data = torch.utils.data.TensorDataset(
            torch.from_numpy(x).type(torch.float)
        )
        data_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)

        sigmoid = torch.nn.Sigmoid()
        all_predictions = torch.tensor([])  # .cuda()

        self.global_model.eval()
        for idx, (batch_x,) in enumerate(data_loader):
            batch_x = batch_x  # .cuda()
            with torch.no_grad():
                model_predictions = self.global_model(batch_x)
                all_predictions = torch.cat((all_predictions, model_predictions))

        if self.model_architecture == ModelArchitecture.MLP_MONO_CLASS:
            all_predictions = sigmoid(all_predictions).round().type(torch.long)
        elif self.model_architecture == ModelArchitecture.MLP_MULTI_CLASS:
            all_predictions = torch.argmax(all_predictions, dim=1).type(torch.long)
        else:
            raise ValueError("Not yet implemented!")

        return all_predictions.flatten()


# TODO:
#  adaptions needed for multiclass (if desired)
#  adaptions for autoencoder -> threshold selection, different standardization?

# TODO: rename file, I did not get the LocalOps Meaning :D maybe call it Participant and Coordinator/Server?
class LocalOps(object):
    def __init__(self, pdata, batch_size, epochs, lr):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.trainloader, self.testloader = self.train_test_split(pdata)
        self.device = "cuda"
        # Default criterion set to BCEWithlogits loss function (combines BCEloss and sigmoid layer, numerically stable)

    def train_test_split(self, pdata):
        """
        Returns train and test dataloaders for a given data and targets
        """
        x_train, y_train, x_test, y_test = pdata
        x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).unsqueeze(1).float()
        x_test, y_test = torch.from_numpy(x_test).float(), torch.from_numpy(y_test).unsqueeze(1).float()

        # TODO: if needed split off validation set here
        # create loaders
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train.type(torch.FloatTensor))
        trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test.type(torch.FloatTensor))
        testloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return trainloader, testloader


class BinaryOps(LocalOps):
    def __init__(self, pdata, batch_size, epochs, lr):
        super(BinaryOps, self).__init__(pdata, batch_size, epochs, lr)
        self.criterion = nn.BCEWithLogitsLoss()

    def update_weights(self, model):
        # Set mode to train model
        model.train()
        epoch_loss = []
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

        for le in range(self.epochs):
            batch_loss = []
            for batch_idx, (x, y) in enumerate(self.trainloader):
                x, y = x.to(self.device), y.to(self.device)
                model.zero_grad()
                pred = model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    print('\r| Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        le + 1, batch_idx * len(x),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()), end="", flush=True)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        print()
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.testloader):
                x, y = x.to(self.device), y.to(self.device)

                # Inference
                pred = model(x)
                batch_loss = self.criterion(pred, y)
                loss += batch_loss.item()

                # Prediction Binary
                s = nn.Sigmoid()
                pred = s(pred)
                pred[pred < 0.5] = 0
                pred[pred > 0.5] = 1
                correct += (pred == y).type(torch.float).sum().item()
                total += len(y)

        return correct, total, loss


# TODO: use validation set to find threshold
#  data preprocessing specialties? values only in certain range?
#  different train and testloader needed for update_weights/inference
#  -> update_weights and inference should be done on two different instances of AutoencoderOps
#  1. update_weights on exclusively normal samples + untrained model
#  2. inference on new testdata with malicious samples + the trained model from 1.

class AutoencoderOps(LocalOps):
    def __init__(self, p: DataSampler, batch_size, epochs, lr):
        super(AutoencoderOps, self).__init__(p, batch_size, epochs, lr)
        self.criterion = nn.MSELoss()
        # TODO: adapt for validation/threshold selection split

    def update_weights(self, model):
        # Set mode to train model
        model.train()
        epoch_loss = []
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

        for le in range(self.epochs):
            batch_loss = []
            for batch_idx, (x, _) in enumerate(self.trainloader):
                # we don't care about targets here
                x = x.to(self.device)
                model.zero_grad()
                pred = model(x)
                loss = self.criterion(pred, x)
                loss.backward()
                optimizer.step()

                # if batch_idx % 10 == 0:
                #     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         global_round+1, le+1, batch_idx * len(x),
                #         len(self.trainloader.dataset),
                #         100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (x, y) in enumerate(self.loc_testloader):
            x, y = x.to(self.device), y.to(self.device)

            # Inference
            pred = model(x)
            batch_loss = self.criterion(pred, x)
            loss += batch_loss.item()

            # Prediction Autoencoder: if further than a certain threshold -> use validation to fin
            # correct += ((distance(pred,x) >= treshold) == y).type(torch.float).sum().item()
            total += len(y)

        return correct, total, loss
