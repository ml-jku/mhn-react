import time
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from joblib import Parallel, delayed

from rdkit import rdBase
rdBase.DisableLog('rdApp.*')

from rdkit import Chem
from rdkit.Chem import AllChem

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate pairwise template applicability between template and molecule sets')
    parser.add_argument('--templates', dest='templates', default='data/processed/retro.templates.uspto_50k.json.gz')
    parser.add_argument('--train-smiles', dest='train_smiles', default='data/processed/train.input.smiles.npy')
    parser.add_argument('--valid-smiles', dest='valid_smiles', default='data/processed/valid.input.smiles.npy')
    parser.add_argument('--test-smiles', dest='test_smiles', default='data/processed/test.input.smiles.npy')
    parser.add_argument('--output-prefix', dest='output_prefix', default='data/processed/')
    return parser.parse_args()

def smiles_to_mol(smiles):
    return [Chem.MolFromSmiles(smi) for smi in smiles]

def calc_appl(mols, templates):
    lil = sparse.lil_matrix((len(mols), len(templates)), dtype=int)
    for j, smarts in enumerate(templates):
        rxn = AllChem.ReactionFromSmarts(smarts)
        for i, mol in enumerate(mols):
            try:
                res = rxn.RunReactants([mol])
            except KeyboardInterrupt:
                print('Interrupted')
                raise KeyboardInterrupt
            except Exception as e:
                print(e)
                res = None
            if res:
                lil[i, j] = 1
    return lil.tocsr()


def start_timer(comm):
    rank = comm.Get_rank()
    if rank == 0:
        return time.time()
    else:
        return None

def print_timing(comm, t0, name=None):
    rank = comm.Get_rank()
    if rank == 0:
        print_str = 'elapsed'
        if name:
            print_str += ' [{}]'.format(name)
        print_str += ': {}'.format(time.time()-t0)
        print(print_str)
        return time.time()
    else:
        return None


def mpi_appl(comm, smiles, smarts, save_path):
    rank = comm.Get_rank()
    size = comm.Get_size()
    t0 = start_timer(comm)
    if rank == 0:  smiles = np.array_split(smiles, size)
    smiles = comm.scatter(smiles, root=0)
    t0 = print_timing(comm, t0, name='scatter')
    mols = smiles_to_mol(smiles)
    comm.Barrier()
    t0 = print_timing(comm, t0, name='convert')
    appl = calc_appl(mols, smarts)
    comm.Barrier()
    t0 = print_timing(comm, t0, name='appl')
    appl = comm.gather(appl, root=0)
    t0 = print_timing(comm, t0, name='gather')
    if rank == 0:
        appl = sparse.vstack(appl)
        sparse.save_npz(save_path, appl)
    t0 = print_timing(comm, t0, name='save')
    comm.Barrier()

if __name__ == '__main__':
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_arguments()

    t0 = start_timer(comm)

    templates = pd.read_json(args.templates)
    templates = templates.sort_values('index')
    template_smarts = templates['reaction_smarts'].values

    if rank == 0:
        train_smi = np.load(args.train_smiles)
        val_smi = np.load(args.valid_smiles)
        test_smi = np.load(args.test_smiles)
    else:
        train_smi = None
        val_smi = None
        test_smi = None

    t0 = print_timing(comm, t0, name='read')

    mpi_appl(comm, test_smi, template_smarts, args.output_prefix+'test.appl_matrix.npz')
    mpi_appl(comm, val_smi, template_smarts, args.output_prefix+'valid.appl_matrix.npz')
    mpi_appl(comm, train_smi, template_smarts, args.output_prefix+'train.appl_matrix.npz')