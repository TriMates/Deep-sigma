from sklearn import metrics
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.autograd import grad as torch_grad
from torch.autograd import Variable

def NMI(y_true, y_pred):
    return metrics.normalized_mutual_info_score(y_true, y_pred)


def ARI(y_true, y_pred):
    return metrics.adjusted_rand_score(y_true, y_pred)

def ACC(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc