import copy
import pickle
from sys import exit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from datasampling import DataSampler
from models import MLP, AutoEncoder
from localops import BinaryOps, AutoencoderOps
from federated import train_model, print_test_results
from utils import plot_results, get_sampler_data, get_baseline_data, scale_baseline, scale_federation
from time import time

# TODO: early stopping - https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
if __name__ == '__main__':

    # Define model
    device = 'cuda'
    method = "binary"
    if method == "binary":
        global_model = MLP(in_features=75)
    elif method == "autoencoder":
        global_model = AutoEncoder(K=75)
    else:
        exit('Error: unrecognized model')
    global_model.to(device)
    global_model.train()
    print(global_model)

    ## Set hyperparameters
    epochs = 50
    loc_epochs = 6
    participation_rate = 1
    loc_sample_size = 6000 # makes each participant contribute the same amount of data samples -> customize if unequal data
    batch_size = 64
    lr = 1e-5
    split = 0.8

    ## Data definition, balancing and standardization
    print("Federated Training Start")
    samplers = [DataSampler(loc_sample_size, [("ras4-4gb", ["normal", "delay"])]),
                DataSampler(loc_sample_size, [("ras3", ["normal", "delay", "disorder"])])]

    # get the balanced data and targets per participant, split into train and testing parts
    data_per_participant = get_sampler_data(samplers, split=split)
    baseline_data = get_baseline_data(data_per_participant)

    # standardize data
    scaler = StandardScaler()#MinMaxScaler() # TRY OUT with StandardScaler()
    baseline_data, scaler = scale_baseline(baseline_data, scaler)
    data_per_participant = scale_federation(data_per_participant, scaler)

    ## Training
    # TODO: add TPR/FPR: from sklearn.metrics import f1_score, classification_report, confusion_matrix
    start = time()
    # Baseline model training: initialize one single operator performing weight updates and inference
    print("\nBaseline Training Start")
    base_model = copy.deepcopy(global_model)
    aggregator = BinaryOps(baseline_data, batch_size, 1, lr)

    # use train_model() with just one operator [aggregator]
    # aggregator.update_weights() and aggregate.inference() also work (see binary-mlp.py)
    # but they dont return lists of acc-/loss along the training for plotting
    train_base_accuracy, train_base_loss = train_model(base_model, [aggregator], epochs)

    # Federated model training: initialize list of operators building the federated model
    print("\nFederated Training Start")
    participants = [BinaryOps(pdata, batch_size, loc_epochs, lr) for pdata in data_per_participant]
    train_fed_accuracy, train_fed_loss = train_model(global_model, participants, epochs)

    end = time()
    # Directly compare results: (possibly use arbitrary data here for inference)
    print_test_results([aggregator], base_model, epochs)
    print_test_results(participants, global_model, epochs)
    print(f"\nElapsed time for baseline and federation training: {(end - start):.2f} s")

    # TODO:
    #  make plots/results params/ensure reproducibility of experiments
    #  -> store models for baseline/federated, hyperparams, clients and data used
    plot_results(train_base_loss, train_base_accuracy, train_fed_loss, train_fed_accuracy, ["exp1-loss", "exp1-acc"])


    # pickle example:
    # # Saving the objects train_loss and train_accuracy:
    # file_name = '../results/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)
    #
    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)


