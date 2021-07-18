import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from datasampling import DataSampler
import numpy as np


# TODO:
#  adaptions needed for multiclass (if desired)
#  adaptions for autoencoder -> threshold selection, different standardization?

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
