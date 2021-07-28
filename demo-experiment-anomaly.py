import torch
from torch import nn
from models import AutoEncoder
from sampling import DataSampler
from devices import AutoencoderOps

## Test Experiment for centralized anomaly detection

# select some hyperparameters and initialize network
learning_rate = 1e-5
batch_size = 64
epochs = 20
num_tot_samples = 6000
m = 0.9
device = "cuda"
criterion = nn.MSELoss()


baseline_sampler = DataSampler(num_tot_samples, [["ras4-8gb", ["normal"]], ["ras3", ["normal", "delay", "disorder"]]])
aggregator = AutoencoderOps(baseline_sampler, batch_size, epochs, learning_rate)
net = AutoEncoder(K=75).to(device)
print(net)

# TODO: AE training, complete threshold selection etc

train = True
if train:
    acc = 0
    # training
    mdict, loss = aggregator.update_weights(net)
    # inference
    corrects, totals, test_loss = aggregator.inference(net)
    torch.save(net.state_dict(), "global-mlp.model")
    print(f"Final accuracy after {epochs} epochs: {corrects / totals}")
else:
    net.load_state_dict(torch.load("global-mlp.model"))