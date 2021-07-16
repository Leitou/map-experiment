import copy
import torch
from torch import nn
import numpy as np
from utils import average_weights


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

        # Calculate avg test accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for p in participants:
            corrects, totals, loss = p.inference(model=global_model)
            list_acc.append(corrects / totals)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        print(f"loss {list_loss}")

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f'\nAvg Training Statistics after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

    return train_accuracy, train_loss

def print_test_results(participants, global_model, epochs):
    if len(participants) == 1:
        print(f'\nBaseline model results after {epochs} global rounds of training:')
    else:
        print(f'\nFederated model results after {epochs} global rounds of training:')
    test_acc, test_loss = test_inference_xdata(participants, global_model)
    print("|---- Avg Test Loss: {:.4f}".format(test_loss))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))


def test_inference_xdata(participants, model, glob_avg=True):
    """
    participants is a list of LocalOps objects, upon which inference() can be called,
    model is a just a trained model

    Returns the test accuracy and loss for the global test split of all participants (list of LocalOps objects)
    It is possible to create an arbitrary dataset (e.g. not the one used for training) for testing.
    This is just a helper function.
    """

    # TODO: adapt this function & inference() functions to get TPR/FPR etc
    tot_acc, tot_loss = 0.0, 0.0
    list_corrects, list_totals, list_loss = [], [], []
    model.eval()
    for p in participants:
        corrects, totals, loss = p.inference(model=model)
        list_corrects.append(corrects)
        list_totals.append(totals)
        list_loss.append(loss)

    if glob_avg == True:
        tot_acc = get_total_sample_accuracy(list_corrects, list_totals)
    else:
        tot_acc = get_averaged_accuracy_for_participants(list_corrects, list_totals)
    # loss already batch averaged by torch
    tot_loss = sum(list_loss)

    return tot_acc, tot_loss



# choice 1 for federated accuracy calculation: distinction matters in case of unequal number of participant samples
def get_total_sample_accuracy(corrects, totals):
    return sum(corrects) / sum(totals)

# choice 2 for federated accuracy calculation
def get_averaged_accuracy_for_participants(corrects, totals):
    acc = 0.0
    for c, t in zip(corrects, totals):
        acc += c/t
    return acc / len(corrects)