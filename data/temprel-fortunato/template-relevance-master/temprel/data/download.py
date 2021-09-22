import os
import gzip
import pickle
import requests
import subprocess
import pandas as pd

def download_file(url, output_path=None):
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

    pd.concat([
        pd.DataFrame(train),
        pd.DataFrame(valid),
        pd.DataFrame(test)
    ]).reset_index().to_json('reactions.json.gz', compression='gzip')

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
    df.to_json('reactions.json.gz', compression='gzip')
