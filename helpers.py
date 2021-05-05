import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import importlib
import sklearn
import random
import torch
import math

from helpers import *

import time


def scale(x, out_range=(0, 29)):
    '''
    Takes as input the coordinate weights and scales them between 0 and len(weights)
    '''
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    to_ret = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
    return to_ret.astype(int)


def construct_indices(weights, dim, length):
    '''
    The coordinate weights are between 0-len(weights) but the size of X is len(weights) * batch size so the torch.taken
    function will only take items at indices between  0 & len(weights) meaning only the first item in the batch. This function
    adds len(weights) to each index so taken grabs from every batch
    ^^ fix that explanation yo lol
    '''
    print(dim)
    print(length)
    indices = []
    weights = scale(weights.clone().detach().numpy())
    print(weights.size)
    for i in range(0, dim):
        to_add = i * length
        cur_indices = [i + to_add for i in weights]
        indices.append(cur_indices)
    return torch.tensor(indices, dtype = torch.int64)

def scale_noOverlap(x, out_range=(0, 29)):
    '''
    Takes as input the coordinate weights and scales them between 0 and len(weights)
    Dan removed int rounding from this one.
    '''
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    to_ret = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
    return to_ret


def construct_noOverlap_indices(weights, dim, length):
    '''
    The coordinate weights are between 0-len(weights) but the size of X is len(weights) * batch size so the torch.taken
    function will only take items at indices between  0 & len(weights) meaning only the first item in the batch. This function
    adds len(weights) to each index so taken grabs from every batch
    Dan then modified whatever the above was to ensure the rounding only occurs to available indices, precluding
    drop out. 
    ^^ fix that explanation yo lol
    '''
    indices = []
    weights = scale_noOverlap(weights.clone().cpu().detach().numpy())
    indices = dim*[[x for _,x in sorted(zip(weights,range(0,length)))]]
    for i in range(0,len(indices)):
        indices[i] = [x+(i*length) for x in indices[i]]
    return torch.tensor(indices, dtype = torch.int64).cuda()#.to("cuda:0")
    