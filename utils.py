import csv
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt



# TODO: further preprocessing needed? here or in localops classes
#  (removing columns with equal values, how to handle in federated setting? needed?)
# read in and attach binary labels to data
def read_data(path, malicious=False):
    input = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        #print(f"header: {header}")
        num_filtered = 0
        for row in reader:
            r = []
            if float(row[3]) == 1.: # connectivity == 1
                for el in row[4:]:
                    r.append(float(el))
                input.append(r)
            else:
                num_filtered += 1
    print(f"Number of rows filtered: {num_filtered}")
    targets = np.ones(len(input), dtype=np.float32) if malicious else np.zeros(len(input), dtype=np.float32)
    return np.array(input), targets


# receives a number of samplers, splits their data in training and testing parts
# and returns the data and targets per participant
def get_sampler_data(samplers, split):
    alldata = [s.sample() for s in samplers]
    alldata_splits = []
    for d, t in alldata:
        dlen = len(d)
        # shuffle
        idxs = np.arange(dlen)
        np.random.shuffle(idxs)
        d = d[idxs]
        t = t[idxs]
        # split in training and testing parts
        x_train, y_train = d[:int(split * dlen)], t[:int(split * dlen)]
        x_test, y_test = d[int(split * dlen):], t[int(split * dlen):]
        alldata_splits.append((x_train,y_train, x_test,y_test))

    return alldata_splits


# returns the baseline splits for data and targets resulting from
# the corresponding aggregated participant data and target splits
def get_baseline_data(data_per_participant):

    bx_train, by_train = data_per_participant[0][0], data_per_participant[0][1]
    bx_test, by_test = data_per_participant[0][2], data_per_participant[0][3]

    for x_train, y_train, x_test, y_test in data_per_participant[1:]:
        print(f"len bx_train {len(bx_train)}, len by_train {len(by_train)}")
        print(f"len bx_test {len(bx_test)}, len by_test {len(by_test)}")
        # just append all splits for the global model data
        bx_train = np.vstack((bx_train, x_train))
        by_train = np.concatenate((by_train, y_train))
        bx_test = np.vstack((bx_test, x_test))
        by_test = np.concatenate((by_test, y_test))

    return copy.deepcopy((bx_train,by_train, bx_test,by_test))


# scales the baseline_data and returns the scaler
def scale_baseline(alldata, scaler):
    print(f"scaling baseline")
    bx_train, by_train, bx_test, by_test = alldata
    scaler.fit(bx_train)
    bx_train = scaler.transform(bx_train)
    bx_test = scaler.transform(bx_test)

    baseline_data = (bx_train,copy.deepcopy(by_train), bx_test,copy.deepcopy(by_test))
    return baseline_data, scaler


# requires that scaler is already fitted with all data from the baseline
def scale_federation(data_per_participant, scaler):
    print(f"scaling federation")
    federation_data = []
    for x_train,y_train, x_test,y_test in data_per_participant:
        federation_data.append((scaler.transform(x_train),copy.deepcopy(y_train), scaler.transform(x_test),copy.deepcopy(y_test)))
    return federation_data


def average_weights(w):
    """
    Returns the average of the weights, w is a list of dicts of local model weights.
    Keys are the same for all w_i as the model architecture is the same for every participant
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg



def plot_results(bl_losses, bl_accs, fed_losses, fed_accs, filenames):
    # Plotting - loss curve
    plt.figure()
    plt.title('Training Loss vs Global Epochs')
    plt.plot(range(len(bl_losses)), bl_losses, label="Baseline", color='r')
    plt.plot(range(len(fed_losses)), fed_losses, label="Federated", color='b')
    plt.ylabel('Training loss')
    plt.xlabel('Global Epochs')
    plt.legend()
    plt.savefig(f'results/plots/{filenames[0]}.png')

    # Plotting - average accuracy
    plt.figure()
    plt.title('Average Accuracy vs Global Epochs')
    plt.plot(range(len(bl_accs)), bl_accs, label="Baseline", color='r')
    plt.plot(range(len(fed_accs)), fed_accs, label="Federated", color='b')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Global Epochs')
    plt.legend()
    plt.savefig(f'results/plots/{filenames[1]}.png')

# TODO: store trained model and dict of hyperparams at a given path
def save_model(model, hyperparams, path):
    return


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return