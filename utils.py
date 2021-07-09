import csv
import copy
import torch
import numpy as np
from torchvision import datasets, transforms
#from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
#from sampling import cifar_iid, cifar_noniid

# TODO: decide on how to include further preprocessing
#  (removing columns with equal values, how to handle in federated setting? needed?)
# read in and label data
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
    #print(f"Number of rows filtered: {num_filtered}")
    targets = np.ones(len(input)-num_filtered, dtype=np.float32) if malicious else np.zeros(len(input)-num_filtered, dtype=np.float32)
    return np.array(input), targets


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def get_total_sample_accuracy(corrects, totals):
    return sum(corrects) / sum(totals)

def get_averaged_accuracy_for_participants(corrects, totals):
    acc = 0.0
    for c, t in zip(corrects, totals):
        acc += c/t
    return acc / len(corrects)


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