# -*- coding: utf-8 -*-
"""
Author: Philipp Seidl, Philipp Renz
        ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning
        Johannes Kepler University Linz
Contact: seidl@ml.jku.at

Evaluation functions for single-step-retrosynthesis
"""
import sys

import rdchiral
from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants
import hashlib
from rdkit import Chem

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from copy import deepcopy
from glob import glob
import os
import pickle

from multiprocessing import Pool
import hashlib
import pickle
import logging

#import timeout_decorator


def _cont_hash(fn):
    with open(fn, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_templates_only(path, cache_dir='/tmp'):
    arg_hash_base = 'load_templates_only' + path
    arg_hash = hashlib.md5(arg_hash_base.encode()).hexdigest()
    matches = glob(os.path.join(cache_dir, arg_hash+'*'))
    
    if len(matches) > 1:
        raise RuntimeError('Too many matches')
    elif len(matches) == 1:
        fn = matches[0]
        content_hash = _cont_hash(path)
        content_hash_file = os.path.basename(fn).split('_')[1].split('.')[0]
        if content_hash_file == content_hash:
            with open(fn, 'rb') as f:
                return pickle.load(f)
    
    df = pd.read_json(path)
    template_dict = {}
    for row in range(len(df)):
        template_dict[df.iloc[row]['index']] = df.iloc[row].reaction_smarts
    
    # cache the file
    content_hash = _cont_hash(path)
    fn = os.path.join(cache_dir, f"{arg_hash}_{content_hash}.p")
    with open(fn, 'wb') as f:
        pickle.dump(template_dict, f)
        
def load_templates_v2(path, get_complete_df=False):
    if get_complete_df:
        df = pd.read_json(path)
        return df
    
    return load_templates_only(path)

def canonicalize_reactants(smiles, can_steps=2):
    if can_steps==0:
        return smiles
    
    mol = Chem.MolFromSmiles(smiles)
    for a in mol.GetAtoms():
        a.ClearProp('molAtomMapNumber')
    
    smiles = Chem.MolToSmiles(mol, True)
    if can_steps==1:
        return smiles

    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)
    if can_steps==2:
        return smiles
    
    raise ValueError("Invalid can_steps")
        


def load_test_set(fn):
    df = pd.read_csv(fn, index_col=0)
    test = df[df.dataset=='test']

    test_product_smarts = list(test.prod_smiles) # we make predictions for these
    for s in test_product_smarts:
        assert len(s.split('.')) == 1
        assert '>' not in s

    test_reactants = [] # we want to predict these
    for rs in list(test.rxn_smiles):
        rs = rs.split('>>')
        assert len(rs) == 2
        reactants_ori, products = rs
        reactants = reactants_ori.split('.')
        products = products.split('.')
        assert len(reactants) >= 1
        assert len(products) == 1

        test_reactants.append(reactants_ori)

    return test_product_smarts, test_reactants


#@timeout_decorator.timeout(1, use_signals=False)
def time_out_rdchiralRun(temp, prod_rct, combine_enantiomers=False):
    rxn = rdchiralReaction(temp)
    return rdchiralRun(rxn, prod_rct, combine_enantiomers=combine_enantiomers) 

def _run_templates_rdchiral(prod_appl):
    prod, applicable_templates = prod_appl
    prod_rct = rdchiralReactants(prod) # preprocess reactants with rdchiral
    
    results = {}
    for idx, temp in applicable_templates:
        temp = str(temp)
        try:
            results[(idx, temp)] = time_out_rdchiralRun(temp, prod_rct, combine_enantiomers=False)
        except:
            pass
            
    return results

def _run_templates_rdchiral_original(prod_appl):
    prod, applicable_templates = prod_appl
    prod_rct = rdchiralReactants(prod) # preprocess reactants with rdchiral
    
    results = {}    
    rxn_cache = {}
    for idx, temp in applicable_templates:
        temp = str(temp)
        if temp in rxn_cache:
            rxn = rxn_cache[(temp)]
        else:
            try:
              rxn = rdchiralReaction(temp)
              rxn_cache[temp] = rxn
            except:
              rxn_cache[temp] = None
              msg = temp+' error converting to rdchiralReaction'
              logging.debug(msg)
        try:
            res = rdchiralRun(rxn, prod_rct, combine_enantiomers=False)    
            results[(idx, temp)] =  res
        except:
            pass
            
    return results

def run_templates(test_product_smarts, templates, appl, njobs=32, cache_dir='/tmp'):
    appl_dict = defaultdict(list)
    for i,j in zip(*appl):
        appl_dict[i].append(j)
    
    prod_appl_list = []       
    for prod_idx, prod in enumerate(test_product_smarts):
        applicable_templates = [(idx, templates[idx]) for idx in appl_dict[prod_idx]]
        prod_appl_list.append((prod, applicable_templates))
    
    arg_hash = hashlib.md5(pickle.dumps(prod_appl_list)).hexdigest()
    cache_file = os.path.join(cache_dir, arg_hash+'.p')
    
    if os.path.isfile(cache_file):
        with open(cache_file, 'rb') as f:
            print('loading results from file',f)
            all_results = pickle.load(f)
            
    #find /tmp -type f \( ! -user root \) -atime +3 -delete
    # to delete the tmp files that havent been accessed 3 days

    else:    
        #with Pool(njobs) as pool:
        #    all_results = pool.map(_run_templates_rdchiral, prod_appl_list)
        
        from tqdm.contrib.concurrent import process_map
        all_results = process_map(_run_templates_rdchiral, prod_appl_list, max_workers=njobs, chunksize=1, mininterval=2)
        
        #with open(cache_file, 'wb') as f:
        #    print('saving applicable_templates to cache', cache_file)
        #    pickle.dump(all_results, f)
            

        
    prod_idx_reactants = []
    prod_temp_reactants = []

    for prod, idx_temp_reactants in zip(test_product_smarts, all_results):
        prod_idx_reactants.append({idx_temp[0]: r for idx_temp, r  in idx_temp_reactants.items()})
        prod_temp_reactants.append({idx_temp[1]: r for idx_temp, r  in idx_temp_reactants.items()})
    
    return prod_idx_reactants, prod_temp_reactants

def sort_by_template(template_scores, prod_idx_reactants):
    sorted_results = []
    for i, predictions in enumerate(prod_idx_reactants):
        score_row = template_scores[i]
        appl_idxs = np.array(list(predictions.keys()))
        if len(appl_idxs) == 0:
            sorted_results.append([])
            continue
        scores = score_row[appl_idxs]
        sorted_idxs = appl_idxs[np.argsort(scores)][::-1]
        sorted_reactants = [predictions[idx] for idx in sorted_idxs]
        sorted_results.append(sorted_reactants)
    return sorted_results

def no_dup_same_order(l):
    return list({r: 0 for r in l}.keys()) 

def flatten_per_product(sorted_results, remove_duplicates=True):
    flat_results = [sum((r for r in row), []) for row in sorted_results]
    if remove_duplicates:
        flat_results =  [no_dup_same_order(row) for row in flat_results]
    return flat_results


def topkaccuracy(test_reactants, predicted_reactants, ks=[1], ret_ranks=False):
    ks = [k if k is not None else 1e10 for k in ks]
    ranks = []
    for true, pred in zip(test_reactants, predicted_reactants):
        try:
            rank = pred.index(true) + 1
        except ValueError:
            rank = 1e15
        ranks.append(rank)
    ranks = np.array(ranks)
    if ret_ranks:
        return ranks
    
    return [np.mean([ranks <= k]) for k in ks]