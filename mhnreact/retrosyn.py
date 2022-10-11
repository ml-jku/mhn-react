from collections import defaultdict
from mhnreact.molutils import canonicalize_template, canonicalize_smi, remove_attom_mapping
from mhnreact.model import MHN

import os
import pickle
from collections import Counter
from urllib.request import urlretrieve
from time import time

from scipy.special import softmax
from rdkit.Chem import rdChemReactions
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

import logging
import warnings
from joblib import Memory
                
from treelib import Node, Tree

import torch

cachedir = 'data/cache/'
memory = Memory(cachedir, verbose=0, bytes_limit=80e9)

from mhnreact.molutils import getFingerprint
from mhnreact.molutils import convert_smiles_to_fp

reaction_superclass_names = {
    1 : 'Heteroatom alkylation and arylation',
    2 : 'Acylation and related processes',
    3 : 'C-C bond formation',
    4 : 'Heterocycle formation', #TODO check
    5 : 'Protections',
    6 : 'Deprotections',
    7 : 'Reductions',
    8 : 'Oxidations',
    9 : 'Functional group interconversoin (FGI)',
    10 :'Functional group addition (FGA)'
}

@memory.cache(ignore=[])
def getTemplateApplicabilityMatrix(t, fp_size=8096, fp_type='pattern'):
    only_left_side_of_templates = list(map(lambda k: k.split('>>')[0], t.values()))
    return convert_smiles_to_fp(only_left_side_of_templates,is_smarts=True, which=fp_type, fp_size=fp_size)

def FPF(smi, templates, fp_size=8096, fp_type='pattern'):
    """Fingerprint-Filter for applicability"""
    tfp = getTemplateApplicabilityMatrix(templates, fp_size=fp_size, fp_type=fp_type)
    if not isinstance(smi, list):
        smi = [smi]
    mfp = convert_smiles_to_fp(smi,which=fp_type, fp_size=fp_size)
    applicable = ((tfp&mfp).sum(1)==(tfp.sum(1)))
    return applicable

@memory.cache
def load_emol_bbs(DATA_DIR='data/cache/'):
    """Downloads emolecules building blocks"""
    os.makedirs(DATA_DIR, exist_ok=True)
    fn_bb_txt = os.path.join(DATA_DIR, 'emol_building_blocks.txt')
    fn_emol_csv = os.path.join(DATA_DIR, 'version.smi.gz')
    fn_emol_pickle = os.path.join(DATA_DIR, 'emol.pkl.gz')

    url_emol_csv = 'http://downloads.emolecules.com/free/2021-10-01/version.smi.gz'

    if os.path.isfile(fn_bb_txt):
        with open(fn_bb_txt, 'r') as fh:
            return fh.read().split()

    if not os.path.isfile(fn_emol_csv):
        print('Downloading emolecule data')
        urlretrieve(url_emol_csv, fn_emol_csv)

    print('Loading eMolecules data frame (may take a minute ;)')
    emol_df = pd.read_csv(fn_emol_csv, sep=' ', compression='gzip')

    bb_list = set(list(emol_df.isosmiles.values))

    return bb_list

def in_emol_or_trivial(smi):
    smi = canonicalize_smi(smi, remove_atom_mapping=True)
    if len(smi)<5:
        return True
    return smi in emol

def not_in_emol_nor_trivial(smi):
    return not in_emol_or_trivial(smi)

def link_mols_to_html(list_of_mols):
    final_str = '['
    for mol in list_of_mols:
        url = f'{mol}.html'
        final_str+=f'<a href="{url}">{mol}</a>, '
    return final_str+']'

def svg_to_html(svg):
    return svg.replace('<?xml version=\'1.0\' encoding=\'iso-8859-1\'?>\n','').replace('\n','')

def compute_html(res, target_smiles):
    res_df = pd.DataFrame(res)#.canon_reaction.unique()
    res_df['non_buyable_reactants'] = res_df.reaction_canonical.apply(lambda k: list(filter(not_in_emol_nor_trivial, k.split('>>')[0].split('.'))) )
    res_df['SVG'] = res_df.reaction.apply(lambda k: svg_to_html(smarts2svg(k)))
    res_df['non_buyable_reactants'] = res_df.non_buyable_reactants.apply(link_mols_to_html)
    res_df.to_html(f'data/cache/retrohtml/{target_smiles}.html', escape=False)
    return res_df

#@memory.cache(ignore=['clf','viz'])
def ssretro(target_smiles:str, clf, num_paths=1, try_max_temp=10, viz=False, use_FPF=False):
    """single-step-retrosynthesis"""
    if hasattr(clf, 'templates'):
        if clf.X is None:
            clf.X = clf.template_encoder(clf.templates)
    preds = clf.forward_smiles([target_smiles])
    if use_FPF:
        appl = FPF(target_smiles, t)
        preds = preds * torch.tensor(appl)
    preds = clf.softmax(preds)
    
    idxs = preds.argsort().detach().numpy().flatten()[::-1]
    preds = preds.detach().numpy().flatten()
    
    try:
        prod_rct = rdchiralReactants(target_smiles)
    except:
        print('target_smiles', target_smiles, 'not computebale')
        return []
    reactions = []
    
    i=0
    while len(reactions)<num_paths and (i<try_max_temp):
        resu = []
        while (not len(resu)) and (i<try_max_temp): #continue 
            #print(i, end='  \r')
            try:
                rxn = rdchiralReaction(t[idxs[i]])
                resu = rdchiralRun(rxn, prod_rct, keep_mapnums=True, combine_enantiomers=True, return_mapped=True)
            except:
                resu=['err']
            i+=1
        
        if len(resu)==2: # if there is a result
            res, mapped_res = resu

            rs = [AllChem.MolToSmiles(prod_rct.reactants)+'>>'+k[0] for k in list(mapped_res.values())]
            for r in rs:
                di = {
                    'template_used':t[idxs[i]],
                    'template_idx': idxs[i],
                    'template_rank': i+1, #get the acutal rank, not the one without non-executable
                    'reaction': r,
                    'reaction_canonical': canonicalize_template(r),
                    'prob': preds[idxs[i]]*100,
                    'template_class': reaction_superclass_names[df[df.reaction_smarts==t[idxs[i]]]["class"].unique()[0]] 
                    }
                di['template_num_train_samples'] = (y['train']==di['template_idx']).sum()
                reactions.append(di)
            if viz:
                for r in rs:
                    print('with template #',idxs[i], t[idxs[i]])
                    smarts2svg(r, useSmiles=True, highlightByReactant=True);
    return reactions

def main(target_smiles, clf_name, max_steps=3, model_type='mhn'):
    clf = load_clf(clf_name, model_type=model_type)
    clf.eval()
    res = ssretro(target_smiles, clf=clf, num_paths=10, viz=False, use_FPF=True)
    res_df = pd.DataFrame(res)
    emol = load_emol_bbs()
    print('check non-buyable')
    res_df['non_buyable_reactants'] = res_df.reaction_canonical.apply(lambda k: list(filter(not_in_emol_nor_trivial, k.split('>>')[0].split('.'))) )
    print('compute html-files')
    res_df = compute_html(res_df, 'mhn_main_'+target_smiles)
    
    import functools
    import operator
    non_buyable_reactant_set = ['_dummy']
    
    while (max_steps!=0) and (len(non_buyable_reactant_set)>0):
        max_steps-=1
        non_buyable_reactant_set = set(functools.reduce(operator.iconcat, res_df['non_buyable_reactants'].values.tolist(), []))
        for r in non_buyable_reactant_set:
            res_2 = ssretro(r, clf=clf)
            if len(res_2>0):
                compute_html(res_2, r)

                res_df = pd.DataFrame(res_2)
                res_df['non_buyable_reactants'] = res_df.reaction_canonical.apply(lambda k: list(filter(not_in_emol_nor_trivial, k.split('>>')[0].split('.'))) )


def recu(target_smi, clf, tree, n_iter=1, use_FPF=False):
        """recurrent part of multi-step-retrosynthesis"""
        if n_iter:
            res = ssretro(target_smi, clf, num_paths=5, try_max_temp=100, use_FPF=use_FPF)
            if len(res)==0:
                return tree
            reactants = res[0]['reaction_canonical'].split('>>')[1].split('.')
            if len(reactants)>=2:
                pass

            for r in reactants:
                try:
                    if check_availability:
                        avail = not_in_emol_nor_trivial(r)
                        tree.create_node(r+'_available' if avail else r,  r, parent=target_smi)
                    else:
                        tree.create_node(r,  r, parent=target_smi)
                except:
                    reactants = res[1]['reaction_canonical'].split('>>')[1].split('.')
                    for r in reactants:
                        try:
                            tree.create_node(r+'_available' if (check_availability and not_in_emol_nor_trivial(r)) else r,  r, parent=target_smi)
                        except: 
                            tree.create_node(r+'_faild',r+'_faild', parent=target_smi)
                tree = recu(r, clf, tree, n_iter=(n_iter-1), use_FPF=use_FPF)
            return tree
        else:
            return tree

def msretro(target_smi, clf, n_iter=3, use_FPF=False, check_availability=False):
    """Multi-step-retro-synthesis function"""
    tree = Tree()
    tree.create_node(target_smi, target_smi)

    tree = recu(target_smi, clf, tree, n_iter=n_iter, use_FPF=False)
    return tree
  
if __name__ == '__main__':
    msretro()