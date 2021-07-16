import torch
from torch import nn
from models import MLP
from datasampling import DataSampler
from localops import BinaryOps
from utils import get_sampler_data, get_baseline_data, scale_baseline, scale_federation
from federated import train_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import copy

## Test Experiment for centralized binary classification

# select some hyperparameters and initialize network
learning_rate = 1e-5
batch_size = 64
epochs = 30
num_tot_samples = 6000
device = "cuda"
split = 0.8

net = MLP(K=75).to(device)
print(net)

baseline_sampler = DataSampler(num_tot_samples, [("ras4-4gb", ["normal"]),
                                                 ("ras3", ["normal", "delay", "disorder"])])
# get the balanced data and targets per participant, split into train and testing parts
baseline_data = get_baseline_data(get_sampler_data([baseline_sampler], split=split))

# standardize data
scaler = MinMaxScaler() # TRY OUT with StandardScaler()
baseline_data, scaler = scale_baseline(baseline_data, scaler)
aggregator = BinaryOps(baseline_data, batch_size, epochs, learning_rate)

train = True
if train:
    acc = 0
    # training
    mdict, loss = aggregator.update_weights(net)
    # inference
    corrects, totals, test_loss = aggregator.inference(net)
    torch.save(net.state_dict(), "global-mlp.model")
    print(f"\n\nFinal accuracy after {epochs} epochs: {100*corrects / totals}%")
else:
    net.load_state_dict(torch.load("global-mlp.model"))

