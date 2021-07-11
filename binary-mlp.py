import csv
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from utils import read_data
from datasampling import DataSampler
from localops import LocalOps
from sys import exit

# TODO:
#   Refactor to global model experiment / Baseline for comparisons
#   make use of participant_data/samplers
#
# normal_path = "data/ras-3-data/samples_normal_2021-06-18-15-59_50s"
# normal_input, normal_targets = read_data(normal_path)
# print(normal_input.shape)
# disorder_path = "data/ras-3-data/samples_disorder_2021-06-30-23-54_50s"
# disorder_input, disorder_targets = read_data(disorder_path, True)
# print(disorder_input.shape)
# repeat_path = "data/ras-3-data/samples_repeat_2021-07-01-20-00_50s"
# repeat_input, repeat_targets = read_data(repeat_path, True)
# print(repeat_input.shape, "\n")
#
#
#
# # concat
# # all files have 79 features
# upsample_param = 10 # to adapt
# tot_input_data = np.vstack((normal_input, np.tile(disorder_input, (upsample_param, 1)), np.tile(repeat_input, (upsample_param, 1))))
# print(tot_input_data.shape) # 8212 = 7609 + 304 + 299
# tot_targets = np.concatenate((normal_targets, np.tile(disorder_targets, upsample_param), np.tile(repeat_targets, upsample_param)))
# print(tot_targets.shape) # 8212
# # shuffle data
# indices = np.arange(len(tot_targets))
# np.random.shuffle(indices)
# tot_input_data = tot_input_data[indices]
# tot_targets = tot_targets[indices]
#
# # binary classification
# # Get cpu or gpu device for training.
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("###### Using {} device ######".format(device))
#
# # get train/test split and dataloaders
# split = 0.8
# input_train, targets_train = tot_input_data[:int(len(tot_input_data)*split)], tot_targets[:int(len(tot_input_data)*split)]
# input_test, targets_test = tot_input_data[int(len(tot_input_data)*split):], tot_targets[int(len(tot_input_data)*split):]
# # standardize
# scaler = StandardScaler()
# scaler.fit(input_train)
# print("Standardscaler means: ", scaler.mean_.shape)
# print("Standardscaler standard deviations: ", scaler.scale_.shape)
# input_train = scaler.transform(input_train)
# input_test = scaler.transform(input_test)
#
#
# X_train, y_train = torch.from_numpy(input_train).float(), torch.from_numpy(targets_train).float()
# X_test, y_test = torch.from_numpy(input_test).float(), torch.from_numpy(targets_test).float()
# X_train, y_train = X_train.to(device), y_train.to(device)
# X_test, y_test = X_test.to(device), y_test.to(device)
#
# batch_size = 128
# train_dataset = torch.utils.data.TensorDataset(X_train, y_train.type(torch.FloatTensor))
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataset = torch.utils.data.TensorDataset(X_test, y_test.type(torch.FloatTensor))
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# define mlp
class NeuralNetwork(nn.Module):
    def __init__(self,K):
        super(NeuralNetwork, self).__init__()
        self.linstack = nn.Sequential(
            nn.Linear(K, 512), # bias=True is default
            nn.Sigmoid(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.6),
            nn.Linear(512, 1),
            # nn.Sigmoid(),
            # nn.BatchNorm1d(128),
            # nn.Dropout(0.6),
            # nn.Linear(128,1)
        )

    def forward(self, x):
        output = self.linstack(x)
        output = output.view(-1)
        return output

# select some hyperparameters and initialize network
learning_rate = 1e-5
epochs = 100
m = 0.9
device = "cuda"

net = NeuralNetwork(K=75).to(device)
#print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=m)
loss_fn = nn.BCEWithLogitsLoss()

baseline_sampler = DataSampler(6000, [["ras4-8gb", ["normal"]], ["ras3", ["normal", "delay", "disorder"]]])
aggregator = LocalOps(baseline_sampler, 64, 20, lr=learning_rate)


# 4 Train the network using categorical cross-entropy and SoftMax
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch %  30 == 0:
            loss, current = loss.item(), batch * batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    model.eval() # switch to testing mode, if theres a difference in training and testing, e.g. dropout,
    with torch.no_grad(): # tells that no need to build a derivative graph, performance
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred[pred < 0.5] = 0
            pred[pred > 0.5] = 1

            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

train = True

if train:
    acc = 0
    mdict, loss = aggregator.update_weights(net)
    acc, test_loss = aggregator.inference(net)
    # for e in range(epochs):
    #     print(f"Epoch {e + 1}: run train and test loop")
    #     train_loop(train_loader, net, loss_fn, optimizer)
    #     tacc = test_loop(test_loader, net, loss_fn)
    #     if tacc > acc:
    #         acc = tacc
    #         torch.save(net.state_dict(), "upsampling-mlp.model")
    print(f"Done! highest accuracy ever achieved is: {acc}")
else:
    net.load_state_dict(torch.load("global-mlp.model"))
    #tacc = test_loop(test_loader, net, loss_fn)
    #print(f"Done! highest accuracy ever achieved is: {tacc}")



