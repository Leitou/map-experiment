import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from time import sleep
from participant_data import ParticipantSampler
from local_ops import LocalUpdate

import torch
#from tensorboardX import SummaryWriter

#from options import args_parser
#from update import LocalUpdate, test_inference
from sys import exit
from models import MLP
from utils import average_weights, read_data, exp_details


# TODO: preprocessing
# check all features for their unique values (np.unique()), print some statistics for each

# TODO: experiment options
# test up- vs downscaling - all classes equally represented
# use more attack data

# TODO: model building
# build federated binary mlp

# build autoencoder joining normal and normal_v2
# build federated autoencoder


# TODO: training loop federated
# determine means of assigning data for each participant
# is given by data collected? should it be artificially assigned?
# class for participant? manually defining data for each user and possibly after that evaluate how to do customizable
# adapt local update to each participant
# split datasets into training and test set, what belongs in each?
#

if __name__ == '__main__':
    #start_time = time.time()

    # define paths
    # path_project = os.path.abspath('..')
    # logger = SummaryWriter('../logs')

    #args = args_parser()
    #exp_details(args)

    device = 'cuda'
    method = "binary"

    if method == "binary":
        global_model = MLP(K=75)

    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)


    epochs = 10
    loc_epochs = 10
    participation_rate = 1
    num_clients = 2
    loc_sample_size = 3000
    batch_size = 64
    lr = 1e-4

    # copy weights
    global_weights = global_model.state_dict()

    # define participants and what data they contribute to the model
    samplers = [ParticipantSampler(3, ["normal", "delay", "disorder"]), ParticipantSampler(4, ["normal", "delay", "disorder"])]
    # initialize list of participants and their own unique test splits
    # can be done either at this position or within the loop over participants
    # choice to shuffle once, or at every global epoch
    participants = [LocalUpdate(s, loc_sample_size, batch_size, loc_epochs=loc_epochs, lr=lr) for s in samplers]

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0


    for epoch in tqdm(range(epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Epoch : {epoch+1} |\n')

        global_model.train()
        #m = max(int(participation_rate * num_clients), 1)

        # calculate new weights locally for each participant and its data
        for p in participants:
            # local_model = LocalUpdate(p, batch_size=batch_size, loc_epochs=local_epochs, lr=lr) # uncomment for epoch wise resampling/shuffling
            w, loss = p.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # aggregate local weights and update global weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    #print(train_loss)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for p in participants:
            acc, loss = p.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Statistics after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    print(train_accuracy)
    # # Test inference after completion of training
    # test_acc, test_loss = test_inference(args, global_model, test_dataset)
    #
    # print(f' \n Results after {args.epochs} global rounds of training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    # print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # # Saving the objects train_loss and train_accuracy:
    # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)
    #
    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)
    #
    # print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))