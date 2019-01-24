import torch
import torch.nn.functional as F

import numpy as np


def train_nll(model, optimizer, loader, device=torch.device('cuda'), weights=None):
    model.train()
    for batch_idx, (data, _, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output, -1)
        loss = F.nll_loss(output, target.long(), weight=weights)
        loss.backward()
        optimizer.step()


def test_nll(model, loader, device=torch.device('cuda'), weights=None):
    '''
    returns loss, accuracy
    '''
    model.eval()
    loss = 0
    accuracy = 0

    true_p = 0
    false_p = 0
    pos = 0

    pred_total = np.empty((0,1))
    true_total = np.empty((0,1))

    with torch.no_grad():
        for data, _, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, -1)
            loss += F.nll_loss(output, target.long(), size_average=False, weight=weights).item()
            pred = output.max(1, keepdim=True)[1]
            accuracy += pred.eq(target.long().view_as(pred)).sum().item()

            # calculate TP
            pred = pred.view(-1,).detach().cpu().numpy()
            true = target.view(-1,).detach().cpu().numpy()

            pred_total = np.concatenate((pred_total, pred))
            true_total = np.concatenate((true_total, true))



            pos += true.sum()
            true_p += np.logical_and(pred == 1, true == 1).sum()
            false_p += np.logical_and(pred == 1, true == 0).sum()

    num_objects = len(loader.dataset)
    loss /= num_objects
    accuracy /= num_objects

    MAP = sklearn.metrics.average_precision_score(true_total, pred_total)

    if pos != 0:
        true_p /= pos
        false_p /= pos

    return loss, accuracy, true_p, false_p, MAP
