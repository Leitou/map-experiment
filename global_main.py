import copy
import pickle
from sys import exit

from datasampling import DataSampler
from models import MLP, AutoEncoder
from localops import BinaryUpdate, AutoencoderUpdate
from federated import train_model, print_test_results
from utils import plot_results, get_sampler_data, get_baseline_data, minmax_scale_baseline, minmax_scale_federation
from time import time

# TODO: early stopping - https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
if __name__ == '__main__':
    start = time()
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
    split = 0.8

    # copy weights
    global_weights = global_model.state_dict()

    # TODO: evtl make util function to not type monitoring programs multiple times)
    # # Baseline model training
    # print("Baseline Training Start")
    # baseline_sampler = DataSampler(glob_sample_size, [("ras4-4gb", ["normal"]),
    #                                                   ("ras3", ["normal", "delay", "disorder"])])
    # aggregator = [BinaryUpdate(baseline_sampler, batch_size, loc_epochs, lr=lr)]
    # base_model = copy.deepcopy(global_model)
    # train_base_accuracy, train_base_loss = train_model(base_model, aggregator, epochs)
    # print_test_results(aggregator, base_model, epochs)


    # Data definition, balancing and standardization
    print("Federated Training Start")
    samplers = [DataSampler(loc_sample_size, [("ras4-4gb", ["normal"])]),
                DataSampler(loc_sample_size, [("ras3", ["normal", "delay", "disorder"])])]

    # get the data and targets per participant, split into train and testing parts
    data_per_participant = get_sampler_data(samplers, split=split)
    baseline_data = get_baseline_data(data_per_participant)
    exit()
    # TODO: fit only training data! transform both training and testing data for all
    # use baseline_data[0] for fit, transform baseline_data[0] and baseline_data[2]
    baseline_data, scaler = minmax_scale_baseline(baseline_data)
    data_per_participant = minmax_scale_federation(data_per_participant, scaler)
    # and call standardize() on the data
    #scaled_data_per_participant = minmax_scale(data_per_participant)

    exit()
    # TODO: minmax scaling vs standardizedscaler
    # initialize list of participants and their own unique test splits
    participants = [BinaryUpdate(s, batch_size, loc_epochs, lr) for s in samplers]
    # TODO: add TPR/FPR in training loop: from sklearn.metrics import f1_score, classification_report
    train_fed_accuracy, train_fed_loss = train_model(global_model, participants, epochs)
    print_test_results(participants, global_model, epochs)

    end = time()

    # TODO:
    #  make plots/results params/ensure reproducibility of experiments
    #  -> store models for baseline/federated, hyperparams, clients and data used
    #plot_results(train_base_loss, train_base_accuracy, train_fed_loss, train_fed_accuracy, ["exp1-loss", "exp1-acc"])

    print(f"total execution time: {end - start}")
    # pickle example:
    # # Saving the objects train_loss and train_accuracy:
    # file_name = '../results/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)
    #
    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)
    #
    # print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

