import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from datasampling import DataSampler
import numpy as np



# TODO:
#  adaptions needed for multiclass (if desired)
#  adaptions for autoencoder? not

class LocalOps(object):
    def __init__(self, p : DataSampler, batch_size, epochs, lr):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        data, targets = p.sample()
        self.trainloader, self.loc_testloader, self.glob_testloader = self.train_test_split(data, targets)
        self.device = "cuda"
        # Default criterion set to BCEWithlogits loss function (combines BCEloss and softmax layer, numerically stable)

    def train_test_split(self, x, y):
        """
        Returns train and test dataloaders for a given dataset, x & y
        """
        # split data
        idxs = np.arange(len(y))
        np.random.shuffle(idxs)
        train_split = 0.8
        loc_test_split = 0.1
        idxs_train = idxs[:int(train_split*len(idxs))]
        idxs_loc_test = idxs[int(train_split*len(idxs)):int((train_split+loc_test_split)*len(idxs))]
        idxs_test = idxs[int((train_split+loc_test_split)*len(idxs)):]
        x_train, y_train = x[idxs_train], y[idxs_train]
        x_loc_test, y_loc_test = x[idxs_loc_test], y[idxs_loc_test]
        x_test, y_test = x[idxs_test], y[idxs_test]

        # TODO: Check if standardization ok
        # scale with training split mean/std
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = torch.from_numpy(scaler.transform(x_train)).float()
        x_loc_test = torch.from_numpy(scaler.transform(x_loc_test)).float()
        x_test = torch.from_numpy(scaler.transform(x_test)).float()
        y_train = torch.from_numpy(y_train).float()
        y_loc_test = torch.from_numpy(y_loc_test).float()
        y_test = torch.from_numpy(y_test).float()

        # create loaders
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train.type(torch.FloatTensor))
        trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        loc_test_dataset = torch.utils.data.TensorDataset(x_loc_test, y_loc_test.type(torch.FloatTensor))
        loctestloader = DataLoader(loc_test_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test.type(torch.FloatTensor))
        testloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return trainloader, loctestloader, testloader

class BinaryUpdate(LocalOps):
    def __init__(self, p : DataSampler, batch_size, epochs, lr):
        super(BinaryUpdate, self).__init__(p, batch_size, epochs, lr)
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

                # if batch_idx % 10 == 0:
                #     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         global_round+1, le+1, batch_idx * len(x),
                #         len(self.trainloader.dataset),
                #         100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

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
            batch_loss = self.criterion(pred, y)
            loss += batch_loss.item()

            # Prediction Binary
            pred[pred < 0.5] = 0
            pred[pred > 0.5] = 1
            correct += (pred == y).type(torch.float).sum().item()
            total += len(y)

        accuracy = correct/total
        return accuracy, loss


# TODO: use validation set to find threshold
#  data preprocessing for values in range [0,1]
class AutoencoderUpdate(LocalOps):
    def __init__(self, p : DataSampler, batch_size, epochs, lr):
        super(AutoencoderUpdate, self).__init__(p, batch_size, epochs, lr)
        self.criterion = nn.MSELoss()

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
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

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
            #correct += ((distance(pred,x) <= treshold) == y).type(torch.float).sum().item()
            total += len(y)

        accuracy = correct/total
        return accuracy, loss