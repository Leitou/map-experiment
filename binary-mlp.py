import csv
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sys import exit

# Q:
# do we need to balance or shuffle data?
# How is it done in valerian


# what we have so far:
# for each pi: normal, normal_v2, disorder, repeat -> 2 normal, 2 attacks

# goal:
# set up an mlp with binary classification: either good or bad (infected)
# do this in both a local and a federated setting:


# read in and label data
def read_data(path, malicious=False):
    input = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        print(f"header: {next(reader)}")
        for row in reader:
            r = []
            for el in row:
                r.append(float(el))
            input.append(r)
    targets = np.ones(len(input), dtype=np.float32) if malicious else np.zeros(len(input), dtype=np.float32)
    return np.array(input), targets

scaler = StandardScaler()
normal_path = "ras-3-192.168.0.205/samples_normal_2021-06-18-15-59_50s"
normal_input, normal_targets = read_data(normal_path)
normal_input = scaler.fit_transform(normal_input)
print(normal_input.shape)
disorder_path = "ras-3-192.168.0.205/samples_disorder_2021-06-28-17-47_50s"
disorder_input, disorder_targets = read_data(disorder_path, True)
disorder_input = scaler.fit_transform(disorder_input)
print(disorder_input.shape)
repeat_path = "ras-3-192.168.0.205/samples_repeat_2021-06-29-09-12_50s"
repeat_input, repeat_targets = read_data(repeat_path, True)
repeat_input = scaler.fit_transform(repeat_input)
print(repeat_input.shape, "\n")
#
# print(normal_targets[:5])
# print(normal_targets.shape)
# print(disorder_targets[:5])
# print(disorder_targets.shape)
# print(repeat_targets[:5])
# print(repeat_targets.shape)


# TODO:
# done-concat data
# run a binary classif. mlp on top
# check results
# decide on balancing

# concat
# all files have 79 features
upsample_param = 5 # to adapt
tot_input_data = np.vstack((normal_input, np.tile(disorder_input, (upsample_param, 1)), np.tile(repeat_input, (upsample_param, 1))))
print(tot_input_data.shape) # 8212 = 7609 + 304 + 299
tot_targets = np.concatenate((normal_targets, np.tile(disorder_targets, upsample_param), np.tile(repeat_targets, upsample_param)))
print(tot_targets.shape) # 8212
# shuffle data
np.random.shuffle(tot_input_data)
np.random.shuffle(tot_targets)
# standardize
tot_input_data = scaler.fit_transform(tot_input_data)


# binary classification
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("###### Using {} device ######".format(device))

# get train/test split and dataloaders
split = 0.8
input_train, targets_train = tot_input_data[:int(len(tot_input_data)*split)], tot_targets[:int(len(tot_input_data)*split)]
input_test, targets_test = tot_input_data[int(len(tot_input_data)*split):], tot_targets[int(len(tot_input_data)*split):]

X_train, y_train = torch.from_numpy(input_train).float(), torch.from_numpy(targets_train).float()
X_test, y_test = torch.from_numpy(input_test).float(), torch.from_numpy(targets_test).float()
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

batch_size = 64
train_dataset = torch.utils.data.TensorDataset(X_train, y_train.type(torch.FloatTensor))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test.type(torch.FloatTensor))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# define mlp
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linstack = nn.Sequential(
            nn.Linear(79, 512), # bias=True is default
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
learning_rate = 1e-4
epochs = 100
m = 0.9

net = NeuralNetwork().to(device)
#print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=m)
loss_fn = nn.BCEWithLogitsLoss()


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
    for e in range(epochs):
        print(f"Epoch {e + 1}: run train and test loop")
        train_loop(train_loader, net, loss_fn, optimizer)
        tacc = test_loop(test_loader, net, loss_fn)
        if tacc > acc:
            acc = tacc
            torch.save(net.state_dict(), "upsampling-mlp.model")
    print(f"Done! highest accuracy ever achieved is: {acc}")
else:
    net.load_state_dict(torch.load("upsampling-mlp.model"))
    tacc = test_loop(test_loader, net, loss_fn)
    print(f"Done! highest accuracy ever achieved is: {tacc}")



