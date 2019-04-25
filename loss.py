import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
from transformer import Constants


def compute_performance(pred, gold, smoothing, log=False):
    loss = compute_loss(pred, gold, smoothing)

    pred_max = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    #if log:
    #  print("pred", pred)
    #  print("pred", pred_max)
    #  print("gold", gold)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred_max.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct

def compute_loss(pred, gold, smoothing):
    gold = gold.contiguous().view(-1)
    if smoothing:
      eps = 0.1
      n_class = pred.size(1)

      one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
      one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
      log_prb = F.log_softmax(pred, dim=1)

      non_pad_mask = gold.ne(Constants.PAD)
      loss = -(one_hot * log_prb).sum(dim=1)
      loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
      loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')    
    return loss
  