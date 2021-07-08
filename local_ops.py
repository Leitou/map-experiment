import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from participant_data import ParticipantSampler
import numpy as np



# TODO: evaluate whether this is easiest as an abstract class + implementations for binary/multiclass & unsupervised

class LocalUpdate(object):
    def __init__(self, p : ParticipantSampler, sample_size, batch_size, loc_epochs, lr):
        self.lr = lr
        self.loc_epochs = loc_epochs
        self.batch_size = batch_size
        data, targets = p.sample(sample_size)
        self.trainloader, self.testloader = self.train_test_split(data, targets)
        self.device = "cuda"
        # Default criterion set to BCEWithlogits loss function (combines BCEloss and softmax layer numerically stable)
        self.criterion = nn.BCEWithLogitsLoss() # nn.NLLLoss().to(self.device)

    def train_test_split(self, x, y):
        """
        Returns train and test dataloaders for a given dataset, x & y
        """
        idxs = np.arange(len(y))
        np.random.shuffle(idxs)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_test = idxs[int(0.8*len(idxs)):]

        x_train, y_train = torch.from_numpy(x[idxs_train]).float(), torch.from_numpy(y[idxs_train]).float()
        x_test, y_test = torch.from_numpy(x[idxs_test]).float(), torch.from_numpy(y[idxs_test]).float()

        train_dataset = torch.utils.data.TensorDataset(x_train, y_train.type(torch.FloatTensor))
        trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test.type(torch.FloatTensor))
        testloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return trainloader, testloader

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

        for batch_idx, (x, y) in enumerate(self.testloader):
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



def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss