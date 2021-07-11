import torch
from torch import nn
from localops import LocalOps

# TODO: Check needed? Move to another file as not exclusively localops, Subject to change for Other models,
#  Add a test_inference function for random samples that are also fed into a global model like in binary-mlp

def test_inference_pdata(participants: LocalOps, model):
    """ Returns the test accuracy and loss for the global test split of all participants
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




def test_inference_randata(model, tdata):
    # TODO: use global model to calculate accuracy and losses on tdata
    accuracy, loss = 0.0, 0.0
    return accuracy, loss