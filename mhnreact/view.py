# -*- coding: utf-8 -*-
"""
Author: Philipp Seidl
        ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning
        Johannes Kepler University Linz
Contact: seidl@ml.jku.at

Loading log-files from training
"""

from pathlib import Path
import os
import datetime
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_experiments(EXP_DIR = Path('data/experiments/')):
    dfs = []
    for fn in os.listdir(EXP_DIR):
        print(fn, end='\r')
        if fn.split('.')[-1]=='tsv':
            df = pd.read_csv(EXP_DIR/fn, sep='\t', index_col=0)
            try:
                with open(df['fn_hist'][0]) as f:
                    hist = eval(f.readlines()[0] )
                df['hist'] = [hist]
                df['fn'] = fn
            except:
                print('err')
                #print(df['fn_hist'])
            dfs.append( df )
    df = pd.concat(dfs,ignore_index=True)
    return df

def get_x(k, kw, operation='max', index=None):
    operation = getattr(np,operation)
    try:
        if index is not None:
            return k[kw][index]

        return operation(k[kw])
    except:
        return 0

def get_min_val_loss_idx(k):
    return get_x(k, 'loss_valid', 'argmin') #changed from argmax to argmin!!

def get_tauc(hist):
    idx = get_min_val_loss_idx(hist)
    # takes max TODO take idx
    return np.mean([get_x(hist, f't100_acc_nte_{nt}') for nt in [*range(11),'>10']])

def get_stats_from_hist(df):
    df['0shot_acc'] = df['hist'].apply(lambda k: get_x(k, 't100_acc_nte_0'))
    df['1shot_acc'] = df['hist'].apply(lambda k: get_x(k, 't100_acc_nte_1'))
    df['>49shot_acc'] = df['hist'].apply(lambda k: get_x(k, 't100_acc_nte_>49'))
    df['min_loss_valid'] = df['hist'].apply(lambda k: get_x(k, 'loss_valid', 'min'))
    return df