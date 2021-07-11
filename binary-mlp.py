import torch
from torch import nn
from models import MLP
from datasampling import DataSampler
from localops import BinaryUpdate

## Test Experiment for centralized binary classification

# select some hyperparameters and initialize network
learning_rate = 1e-5
batch_size = 64
epochs = 20
num_tot_samples = 6000
m = 0.9
device = "cuda"

net = MLP(K=75).to(device)
#print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=m)
loss_fn = nn.BCEWithLogitsLoss()

baseline_sampler = DataSampler(num_tot_samples, [["ras4-8gb", ["normal"]], ["ras3", ["normal", "delay", "disorder"]]])
aggregator = BinaryUpdate(baseline_sampler, batch_size, epochs, learning_rate, loss_fn)

train = True
if train:
    acc = 0
    # training
    mdict, loss = aggregator.update_weights(net)
    # inference
    acc, test_loss = aggregator.inference(net)
    torch.save(net.state_dict(), "global-mlp.model")
    print(f"Final accuracy after {epochs} epochs: {acc}")
else:
    net.load_state_dict(torch.load("global-mlp.model"))
    #tacc = test_loop(test_loader, net, loss_fn)
    #print(f"Done! highest accuracy ever achieved is: {tacc}")



