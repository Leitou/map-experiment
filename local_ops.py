import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from participant_data import ParticipantSampler
import numpy as np



# TODO: evaluate whether this is easiest as an abstract class + implementations for binary/multiclass & unsupervised

class LocalOps(object):
    def __init__(self, p : ParticipantSampler, batch_size, loc_epochs, lr):
        self.lr = lr
        self.loc_epochs = loc_epochs
        self.batch_size = batch_size
        data, targets = p.sample()
        self.trainloader, self.loc_testloader, self.glob_testloader = self.train_test_split(data, targets)
        self.device = "cuda"
        # Default criterion set to BCEWithlogits loss function (combines BCEloss and softmax layer, numerically stable)
        self.criterion = nn.BCEWithLogitsLoss() # nn.NLLLoss().to(self.device)

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

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

        for le in range(self.loc_epochs):
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

            # # Prediction Multiclass
            # _, pred_labels = torch.max(pred, 1)
            # pred_labels = pred_labels.view(-1)
            # correct += torch.sum(torch.eq(pred_labels, y)).item()

            total += len(y)

        accuracy = correct/total
        return accuracy, loss


def test_inference(participants: LocalOps, model):
    """ Returns the test accuracy and loss for the global test split of all participants
    """
    model.eval()
    losses, totals, corrects = [], [], []
    device = 'cuda'

    criterion = nn.BCEWithLogitsLoss().to(device)
    for p in participants:
        plosses, ptotal, pcorrect = 0.0, 0.0, 0.0
        for batch_idx, (x, y) in enumerate(p.glob_testloader):
            x, y = x.to(device), y.to(device)

            # Inference
            pred = model(x)
            batch_loss = criterion(pred, y)
            plosses += batch_loss.item()

            # Prediction Binary
            pred[pred < 0.5] = 0
            pred[pred > 0.5] = 1
            pcorrect += (pred == y).type(torch.float).sum().item()
            ptotal += len(y)
        losses.append(plosses / len(p.glob_testloader)) # to ensure to return average losses per participant
        totals.append(ptotal)
        corrects.append(pcorrect)

    return losses, corrects, totals