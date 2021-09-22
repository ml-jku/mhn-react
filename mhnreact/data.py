# -*- coding: utf-8 -*-
"""
Author: Philipp Seidl
        ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning
        Johannes Kepler University Linz
Contact: seidl@ml.jku.at

File contains functions that help prepare and download USPTO-related datasets
"""

import os
import gzip
import pickle
import requests
import subprocess
import pandas as pd
import numpy as np
from scipy import sparse
import json

def download_temprel_repo(save_path='data/temprel-fortunato', chunk_size=128):
    "downloads the template-relevance master branch"
    url = "https://gitlab.com/mefortunato/template-relevance/-/archive/master/template-relevance-master.zip"
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def unzip(path):
    "unzips a file given a path"
    import zipfile
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(path.replace('.zip',''))


def download_file(url, output_path=None):
    """
        # code from fortunato
        # could also import  from temprel.data.download import get_uspto_50k but slightly altered ;)

    """
    if not output_path:
        output_path = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def get_uspto_480k():
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('data/raw'):
        os.mkdir('data/raw')
    os.chdir('data/raw')
    download_file(
        'https://github.com/connorcoley/rexgen_direct/raw/master/rexgen_direct/data/train.txt.tar.gz',
        'train.txt.tar.gz'
    )
    subprocess.run(['tar', 'zxf', 'train.txt.tar.gz'])
    download_file(
        'https://github.com/connorcoley/rexgen_direct/raw/master/rexgen_direct/data/valid.txt.tar.gz',
        'valid.txt.tar.gz'
    )
    subprocess.run(['tar', 'zxf', 'valid.txt.tar.gz'])
    download_file(
        'https://github.com/connorcoley/rexgen_direct/raw/master/rexgen_direct/data/test.txt.tar.gz',
        'test.txt.tar.gz'
    )
    subprocess.run(['tar', 'zxf', 'test.txt.tar.gz'])

    with open('train.txt') as f:
        train = [
            {
                'reaction_smiles': line.strip(),
                'split': 'train'
            }
            for line in f.readlines()
        ]
    with open('valid.txt') as f:
        valid = [
            {
                'reaction_smiles': line.strip(),
                'split': 'valid'
            }
            for line in f.readlines()
        ]
    with open('test.txt') as f:
        test = [
            {
                'reaction_smiles': line.strip(),
                'split': 'test'
            }
            for line in f.readlines()
        ]

    df = pd.concat([
        pd.DataFrame(train),
        pd.DataFrame(valid),
        pd.DataFrame(test)
    ]).reset_index()
    df.to_json('uspto_lg_reactions.json.gz', compression='gzip')
    os.chdir('..')
    os.chdir('..')
    return df

def get_uspto_50k():
    '''
    get SI from:
    Nadine Schneider; Daniel M. Lowe; Roger A. Sayle; Gregory A. Landrum. J. Chem. Inf. Model.201555139-53
    '''
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('data/raw'):
        os.mkdir('data/raw')
    os.chdir('data/raw')
    subprocess.run(['wget', 'https://pubs.acs.org/doi/suppl/10.1021/ci5006614/suppl_file/ci5006614_si_002.zip'])
    subprocess.run(['unzip', '-o', 'ci5006614_si_002.zip'])
    data = []
    with gzip.open('ChemReactionClassification/data/training_test_set_patent_data.pkl.gz') as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
    reaction_smiles = [d[0] for d in data]
    reaction_reference = [d[1] for d in data]
    reaction_class = [d[2] for d in data]
    df = pd.DataFrame()
    df['reaction_smiles'] = reaction_smiles
    df['reaction_reference'] = reaction_reference
    df['reaction_class'] = reaction_class
    df.to_json('uspto_sm_reactions.json.gz', compression='gzip')
    os.chdir('..')
    os.chdir('..')
    return df

def get_uspto_golden():
    """ get uspto golden and convert it to smiles dataframe from
    Lin, Arkadii; Dyubankova, Natalia; Madzhidov, Timur; Nugmanov, Ramil;
    Rakhimbekova, Assima; Ibragimova, Zarina; Akhmetshin, Tagir; Gimadiev,
    Timur; Suleymanov, Rail; Verhoeven, Jonas; Wegner, Jörg Kurt;
    Ceulemans, Hugo; Varnek, Alexandre (2020):
    Atom-to-Atom Mapping: A Benchmarking Study of Popular Mapping Algorithms and Consensus Strategies.
    ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.13012679.v1
    """
    if os.path.exists('data/raw/uspto_golden.json.gz'):
        print('loading precomputed')
        return pd.read_json('data/raw/uspto_golden.json.gz', compression='gzip')
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('data/raw'):
        os.mkdir('data/raw')
    os.chdir('data/raw')
    subprocess.run(['wget', 'https://github.com/Laboratoire-de-Chemoinformatique/Reaction_Data_Cleaning/raw/master/data/golden_dataset.zip'])
    subprocess.run(['unzip', '-o', 'golden_dataset.zip']) #return golden_dataset.rdf

    from CGRtools.files import RDFRead
    import CGRtools
    from rdkit.Chem import AllChem
    def cgr2rxnsmiles(cgr_rx):
        smiles_rx = '.'.join([AllChem.MolToSmiles(CGRtools.to_rdkit_molecule(m)) for m in cgr_rx.reactants])
        smiles_rx += '>>'+'.'.join([AllChem.MolToSmiles(CGRtools.to_rdkit_molecule(m)) for m in cgr_rx.products])
        return smiles_rx

    data = {}
    input_file = 'golden_dataset.rdf'
    do_basic_standardization=True
    print('reading and converting the rdf-file')
    with RDFRead(input_file) as f:
            while True:
                try:
                    r = next(f)
                    key = r.meta['Reaction_ID']
                    if do_basic_standardization:
                        r.thiele()
                        r.standardize()
                    data[key] = cgr2rxnsmiles(r)
                except StopIteration:
                    break

    print('saving as a dataframe to data/uspto_golden.json.gz')
    df = pd.DataFrame([data],index=['reaction_smiles']).T
    df['reaction_reference'] = df.index
    df.index = range(len(df)) #reindex
    df.to_json('uspto_golden.json.gz', compression='gzip')

    os.chdir('..')
    os.chdir('..')
    return df

def load_USPTO_fortu(path='data/processed', which='uspto_sm_', is_appl_matrix=False):
    """
    loads the fortunato preprocessed data as
    dict X containing X['train'], X['valid'], and X['test']
    as well as the labels containing the corresponding splits
    returns X, y
    """

    X = {}
    y = {}

    for split in ['train','valid', 'test']:
        tmp = np.load(f'{path}/{which}{split}.input.smiles.npy', allow_pickle=True)
        X[split] = []
        for ii in range(len(tmp)):
            X[split].append( tmp[ii].split('.'))

        if is_appl_matrix:
            y[split] = sparse.load_npz(f'{path}/{which}{split}.appl_matrix.npz')
        else:
            y[split] = np.load(f'{path}/{which}{split}.labels.classes.npy', allow_pickle=True)
        print(split, y[split].shape[0], 'samples (', y[split].max() if not is_appl_matrix else y[split].shape[1],'max label)')
    return X, y

#TODO one should load in this file pd.read_json('uspto_R_retro.templates.uspto_R_.json.gz')
# this only holds the templates.. the other holds everything
def load_templates_sm(path = 'data/processed/uspto_sm_templates.df.json.gz', get_complete_df=False):
    "returns a dict mapping from class index to mapped reaction_smarts from the templates_df"
    df = pd.read_json(path)
    if get_complete_df: return df
    template_dict = {}
    for row in range(len(df)):
        template_dict[df.iloc[row]['index']] = df.iloc[row].reaction_smarts
    return template_dict

def load_templates_lg(path = 'data/processed/uspto_lg_templates.df.json.gz', get_complete_df=False):
    return load_templates_sm(path=path, get_complete_df=get_complete_df)

def load_USPTO_sm():
    "loads the default dataset"
    return load_USPTO_fortu(which='uspto_sm_')

def load_USPTO_lg():
    "loads the default dataset"
    return load_USPTO_fortu(which='uspto_lg_')

def load_USPTO_sm_pretraining():
    "loads the default application matrix label and dataset"
    return load_USPTO_fortu(which='uspto_sm_', is_appl_matrix=True)
def load_USPTO_lg_pretraining():
    "loads the default application matrix label and dataset"
    return load_USPTO_fortu(which='uspto_lg_', is_appl_matrix=True)

def load_USPTO_df_sm():
    "loads the USPTO small Sm dataset dataframe"
    return pd.read_json('data/raw/uspto_sm_reactions.json.gz')

def load_USPTO_df_lg():
    "loads the USPTO large Lg dataset dataframe"
    return pd.read_json('data/raw/uspto_sm_reactions.json.gz')

def load_USPTO_golden():
    "loads the golden USPTO dataset"
    return load_USPTO_fortu(which=f'uspto_golden_', is_appl_matrix=False)

def load_USPTO(which = 'sm', is_appl_matrix=False):
    return load_USPTO_fortu(which=f'uspto_{which}_', is_appl_matrix=is_appl_matrix)

def load_templates(which = 'sm',fdir='data/processed', get_complete_df=False):
    return load_templates_sm(path=f'{fdir}/uspto_{which}_templates.df.json.gz', get_complete_df=get_complete_df)

def load_data(dataset, path):
    splits = ['train', 'valid', 'test']
    split2smiles = {}
    split2label = {}
    split2reactants = {}
    split2appl = {}
    split2prod_idx_reactants = {}

    for split in splits:
        label_fn = os.path.join(path, f'{dataset}_{split}.labels.classes.npy')
        split2label[split] = np.load(label_fn, allow_pickle=True)

        smiles_fn = os.path.join(path, f'{dataset}_{split}.input.smiles.npy')
        split2smiles[split] = np.load(smiles_fn, allow_pickle=True)

        reactants_fn = os.path.join(path, f'uspto_R_{split}.reactants.canonical.npy')
        split2reactants[split] = np.load(reactants_fn, allow_pickle=True)


        split2appl[split] = np.load(os.path.join(path, f'{dataset}_{split}.applicability.npy'))

        pir_fn = os.path.join(path, f'{dataset}_{split}.prod.idx.reactants.p')
        if os.path.isfile(pir_fn):
            with open(pir_fn, 'rb') as f:
                split2prod_idx_reactants[split] = pickle.load(f)


    if len(split2prod_idx_reactants) == 0:
        split2prod_idx_reactants = None

    with open(os.path.join(path, f'{dataset}_templates.json'), 'r') as f:
        label2template = json.load(f)
        label2template = {int(k): v for k,v in label2template.items()}

    return split2smiles, split2label, split2reactants, split2appl, split2prod_idx_reactants, label2template


def load_dataset_from_csv(csv_path='', split_col='split', input_col='prod_smiles', ssretroeval=False, reactants_col='reactants_can', ret_df=False, **kwargs):
    """loads the dataset from a CSV file containing a split-column, and input-column which can be defined,
    as well as a 'reaction_smarts' column containing the extracted template, a 'label' column (the index of the template)
    :returns

    """
    print('loading X, y from csv')
    df = pd.read_csv(csv_path)
    X = {}
    y = {}

    for spli in set(df[split_col]):
        #X[spli] = list(df[df[split_col]==spli]['prod_smiles'].apply(lambda k: [k]))
        X[spli] = list(df[df[split_col]==spli][input_col].apply(lambda k: [k]))
        y[spli] = (df[df[split_col]==spli]['label']).values
        print(spli, len(X[spli]), 'samples')

    # template to dict
    tmp = df[['reaction_smarts','label']].drop_duplicates(subset=['reaction_smarts','label']).sort_values('label')
    tmp.index= tmp.label
    template_list = tmp['reaction_smarts'].to_dict()
    print(len(template_list),'templates')

    if ssretroeval:
        # setup for ttest
        test_reactants_can = list(df[df[split_col]=='test'][reactants_col])

        only_in_test = set(y['test']) - set(y['train']).union(set(y['valid']))
        print('obfuscating', len(only_in_test), 'templates because they are only in test')
        for ii in only_in_test:
                template_list[ii] = 'CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC.CCCCCCCCCCCCCCCCCCCCCCCCCCC.CCCCCCCCCCCCCCCCCCCCCC>>CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC.CCCCCCCCCCCCCCCCCCCCC' #obfuscate them
        if ret_df:
            return X, y, template_list, test_reactants_can, df
        return X, y, template_list, test_reactants_can

    if ret_df:
        return X, y, template_list, None, df
    return X, y, template_list, None