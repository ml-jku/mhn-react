import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import Parallel, delayed

def unmap(smiles, canonicalize=True):
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    return Chem.MolToSmiles(mol, isomericSmiles=canonicalize)

def smiles_to_fingerprint(smi, length=2048, radius=2, useFeatures=False, useChirality=True):
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        raise ValueError('Cannot parse {}'.format(smi))
    fp_bit = AllChem.GetMorganFingerprintAsBitVect(
        mol=mol, radius=radius, nBits = length, 
        useFeatures=useFeatures, useChirality=useChirality
    )
    return np.array(fp_bit)

def smiles_list_to_fingerprints(smiles_list, fp_length=2048, fp_radius=2, nproc=1):
    if nproc > 1:
        fps = Parallel(n_jobs=nproc, verbose=1)(
            delayed(smiles_to_fingerprint)(smi, length=fp_length, radius=fp_radius) for smi in smiles_list
        )
        fps = np.array(fps)
    else:
        fps = np.array([smiles_to_fingerprint(smi, length=fp_length, radius=fp_radius) for smi in smiles_list])
    return fps

def remove_spectating_reactants(reaction_smiles):
    reactants, spectators, products = reaction_smiles.split('>')
    rmols = [Chem.MolFromSmiles(r) for r in reactants.split('.')]
    pmol = Chem.MolFromSmiles(products)
    product_map_numbers = set([atom.GetAtomMapNum() for atom in pmol.GetAtoms()])
    react_strings = []
    if spectators:
        spectators = spectators.split('.')
    else:
        spectators = []
    for rmol in rmols:
        map_numbers = set([atom.GetAtomMapNum() for atom in rmol.GetAtoms()])
        intersection = map_numbers.intersection(product_map_numbers)
        if intersection:
            react_strings.append(Chem.MolToSmiles(rmol))
        else:
            spectators.append(Chem.MolToSmiles(rmol))
    return '.'.join(react_strings) + '>' + '.'.join(spectators) + '>' + products

def precursors_from_template(mol, template):
    precursors = set()
    results = template.RunReactants([mol])
    for res in results:
        res_smiles = '.'.join(sorted([Chem.MolToSmiles(m, isomericSmiles=True) for m in res]))
        if Chem.MolFromSmiles(res_smiles):
            precursors.add(res_smiles)
    return list(precursors)

def precursors_from_templates(target_smiles, templates, nproc=1):
    mol = Chem.MolFromSmiles(target_smiles)
    precursor_set_list = Parallel(n_jobs=nproc, verbose=1)(
        delayed(precursors_from_template)(mol, template) for template in templates
    )
    return list(set().union(*precursor_set_list))

def templates_from_smarts_list(smarts_list, nproc=1):
    templates = Parallel(n_jobs=nproc, verbose=1)(
        delayed(AllChem.ReactionFromSmarts)(smarts) for smarts in smarts_list
    )