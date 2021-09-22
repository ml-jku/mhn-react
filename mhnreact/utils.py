# -*- coding: utf-8 -*-
"""
Author: Philipp Seidl
        ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning
        Johannes Kepler University Linz
Contact: seidl@ml.jku.at

General utility functions
"""

import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import math
import torch

# used and fastest version
def top_k_accuracy(y_true, y_pred, k=5, ret_arocc=False, ret_mrocc=False, verbose=False, count_equal_as_correct=False, eps_noise=0):
    """ partly from http://stephantul.github.io/python/pytorch/2020/09/18/fast_topk/
        count_equal counts equal values as beein a correct choice e.g. all preds = 0 --> T1acc = 1
        ret_mrocc ... also return median rank of correct choice
        eps_noise ... if >0 ads noise*eps to y_pred .. recommended e.g. 1e-10
    """
    if eps_noise>0:
        if torch.is_tensor(y_pred):
            y_pred = y_pred + torch.rand(y_pred.shape)*eps_noise
        else:
            y_pred = y_pred + np.random.rand(*y_pred.shape)*eps_noise

    if count_equal_as_correct:
        greater = (y_pred > y_pred[range(len(y_pred)), y_true][:,None]).sum(1) # how many are bigger
    else:
        greater = (y_pred >= y_pred[range(len(y_pred)), y_true][:,None]).sum(1) # how many are bigger or equal
    if torch.is_tensor(y_pred):
        greater = greater.long()
    if isinstance(k, int): k = [k]  # pack it into a list
    tkaccs = []
    for ki in k:
        if count_equal_as_correct:
            tkacc = (greater<=(ki-1))
        else:
            tkacc = (greater<=(ki))
        if torch.is_tensor(y_pred):
            tkacc = tkacc.float().mean().detach().cpu().numpy()
        else:
            tkacc = tkacc.mean()
        tkaccs.append(tkacc)
        if verbose: print('Top', ki, 'acc:\t', str(tkacc)[:6])

    if ret_arocc:
        arocc = greater.float().mean()+1
        if torch.is_tensor(arocc):
            arocc = arocc.detach().cpu().numpy()
        return (tkaccs[0], arocc) if len(tkaccs) == 1 else (tkaccs, arocc)
    if ret_mrocc:
        mrocc = greater.median()+1
        if torch.is_tensor(mrocc):
            mrocc = mrocc.float().detach().cpu().numpy()
        return (tkaccs[0], mrocc) if len(tkaccs) == 1 else (tkaccs, mrocc)
    

    return tkaccs[0] if len(tkaccs) == 1 else tkaccs


def seed_everything(seed=70135):
    """ does what it says ;) - from https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335"""
    import numpy as np
    import random
    import os
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_best_gpu():
    '''Get the gpu with most RAM on the machine. From P. Neves'''
    import torch
    if torch.cuda.is_available():
        gpus_ram = []
        for ind in range(torch.cuda.device_count()):
            gpus_ram.append(torch.cuda.get_device_properties(ind).total_memory/1e9)
        return f"cuda:{gpus_ram.index(max(gpus_ram))}"
    else:
        raise ValueError("No gpus were detected in this machine.")


def sort_by_template_and_flatten(template_scores, prod_idx_reactants, agglo_fun=sum):
    flat_results = []
    for ii in range(len(template_scores)):
        idx_prod_reactants = defaultdict(list)
        for k,v in prod_idx_reactants[ii].items():
            for iv in v:
                idx_prod_reactants[iv].append(template_scores[ii,k])
        d2 = {k: agglo_fun(v) for k, v in idx_prod_reactants.items()}
        if len(d2)==0:
            flat_results.append([])
        else:
            flat_results.append(pd.DataFrame.from_dict(d2, orient='index').sort_values(0, ascending=False).index.values.tolist())
    return flat_results


def str2bool(v):
    """adapted from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', '',' '):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

@np.vectorize
def lgamma(x):
    return math.lgamma(x)

def multinom_gk(array, axis=0):
    """Multinomial lgamma pooling over a given axis"""
    res = lgamma(np.sum(array,axis=axis)+2) - np.sum(lgamma(array+1),axis=axis)
    return res