import copy
import torch
from torch import nn
import numpy as np
from utils import average_weights, get_total_sample_accuracy, get_averaged_accuracy_for_participants


def train_model(global_model, participants, epochs):
    # Training
    train_loss, train_accuracy = [], []
    # val_acc_list, net_list = [], []
    # cv_loss, cv_acc = [], []
    print_every = 2
    # val_loss_pre, counter = 0, 0

    for epoch in range(epochs):
        local_weights, local_losses = [], []
        print(f'\r| Global Training Epoch : {epoch + 1} |',end='', flush=True)

        global_model.train()
        # m = max(int(participation_rate * num_clients), 1)

        # calculate new weights locally for each participant and its data
        for p in participants:
            w, loss = p.update_weights(model=copy.deepcopy(global_model))
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # aggregate local weights and update global weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for p in participants:
            acc, loss = p.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f'\nAvg Training Statistics after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

    return train_accuracy, train_loss

def print_test_results(participants, global_model, epochs):
    test_losses, test_corrects, test_totals = test_inference_xdata(participants, global_model)
    avg_test_loss = sum(test_losses) / len(test_losses)
    test_accuracy_avg = get_averaged_accuracy_for_participants(test_corrects, test_totals)
    test_accuracy_tot = get_total_sample_accuracy(test_corrects, test_totals)
    print(f'\nResults after {epochs} global rounds of training:')
    print("|---- Avg Test Accuracy: {:.2f}%".format(100 * test_accuracy_avg))
    print("|---- Tot Test Accuracy: {:.2f}%\n".format(100 * test_accuracy_tot))


def test_inference_xdata(participants, model):
    """ Returns the test accuracy and loss for the global test split of all participants (list of LocalOps objects)
    It is possible to create an arbitrary dataset (e.g. not the one used for training) for testing
    In order to do this you must create samplers and localOps
    like this for a global aggregator test:
    baseline_sampler = DataSampler(6000, [["ras4-8gb", ["normal"]], ["ras3", ["normal", "delay", "disorder"]]])
    aggregator = [BinaryUpdate(baseline_sampler, 64, 20, lr=learning_rate)]
    call: test_inference_xdata(aggregator, model)

    like this for a federated test:
    # define participants and what data they contribute to the model
    samplers = [DataSampler(loc_sample_size, [["ras4-8gb", ["normal"]]]),
                DataSampler(loc_sample_size, [["ras3", ["normal", "delay", "disorder"]]])]
    # initialize list of participants and their own unique test splits
    participants = [BinaryUpdate(s, batch_size, loc_epochs=loc_epochs, lr=lr) for s in samplers]
    call: test_inference_xdata(participants, model)
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
        losses.append(plosses / len(p.glob_testloader))  # to ensure to return average losses per participant
        totals.append(ptotal)
        corrects.append(pcorrect)

    return losses, corrects, totals
