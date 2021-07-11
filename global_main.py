import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from datasampling import DataSampler
from localops import LocalOps, BinaryUpdate, AutoencoderUpdate
from federated import train_model, test_inference_xdata
from torch import nn

#from tensorboardX import SummaryWriter
#from options import args_parser
from sys import exit
from models import MLP, AutoEncoder
from utils import average_weights, get_averaged_accuracy_for_participants, get_total_sample_accuracy


# TODO: preprocessing
# if needed check all features for their unique values (np.unique()), print some statistics for each

# TODO: model building
# build and train, both aggregating and federated models, separate and compare


if __name__ == '__main__':

    # define model
    device = 'cuda'
    method = "binary"
    if method == "binary":
        global_model = MLP(K=75)
    elif method == "autoencoder":
        global_model = AutoEncoder(K=75)
    else:
        exit('Error: unrecognized model')
    global_model.to(device)
    global_model.train()
    print(global_model)

    # set hyperparameters
    epochs = 10
    loc_epochs = 10
    participation_rate = 1
    num_clients = 2
    loc_sample_size = 3000 # makes each participant contribute the same amount of data samples -> customize if unequal data
    batch_size = 64
    lr = 1e-4
    glob_sample_size = 6000

    # copy weights
    global_weights = global_model.state_dict()

    # TODO: define data, train the aggregated baseline, then the federated model, (evtl make util function to not type monitoring programs multiple times)
    # Baseline model training (uses )
    baseline_sampler = DataSampler(glob_sample_size, [["ras4-8gb", ["normal"]],
                                                      ["ras3", ["normal", "delay", "disorder"]]])
    aggregator = BinaryUpdate(baseline_sampler, batch_size, loc_epochs, lr=lr)
    train_base_accuracy, train_base_loss = train_model(copy.deepcopy(global_model), [aggregator], epochs)

    # Federated Training
    # define participants and what data they contribute to the model
    samplers = [DataSampler(loc_sample_size, [["ras4-8gb", ["normal"]]]),
                DataSampler(loc_sample_size, [["ras3", ["normal", "delay", "disorder"]]])]
    # initialize list of participants and their own unique test splits
    participants = [BinaryUpdate(s, batch_size, loc_epochs, lr) for s in samplers]
    train_fed_accuracy, train_fed_loss = train_model(global_model, participants, epochs)

    # comparing acc & loss progression
    print(train_base_accuracy)
    print(train_fed_accuracy, "\n")
    print(train_base_loss)
    print(train_fed_loss)

    exit()
    # Test inference after completion of training on unseen data

    test_losses, test_corrects, test_totals = test_inference_xdata(participants, global_model)
    avg_test_loss = sum(test_losses) / len(test_losses)
    test_accuracy_avg = get_averaged_accuracy_for_participants(test_corrects, test_totals)
    test_accuracy_tot = get_total_sample_accuracy(test_corrects, test_totals)
    print(f'\nResults after {epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_fed_accuracy[-1]))
    print("|---- Avg Test Accuracy: {:.2f}%".format(100*test_accuracy_avg))
    print("|---- Tot Test Accuracy: {:.2f}%".format(100*test_accuracy_tot))

    # TODO: use data completely independent of sampling for testing
    #tdata, ttargets =
    #tl, tc, tt = test_inference_randata(global_model, data)


    # TODO: make plots/save params/ensure reproducibility
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