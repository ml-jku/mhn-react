import numpy as np
from ..rdkit import smiles_list_to_fingerprints, precursors_from_templates

def tanimoto(fp1, fp2):
    a = fp1.sum()
    b = fp2.sum()
    c = float((fp1&fp2).sum())
    return c/(a+b-c)

def pairwise_tanimoto(arr1, arr2, metric=tanimoto):
    if arr1.size == 0:
        return np.array([[]])
    return cdist(arr1, arr2, metric=metric)

def diversity_from_smiles_list(smiles_list, fp_length=2048, fp_radius=2, nproc=1):
    fps = smiles_list_to_fingerprints(smiles_list, fp_length=fp_length, fp_radius=fp_radius, nproc=nproc)
    similarity = pairwise_tanimoto(fps, fps)
    diversity = 1 - similarity
    np.fill_diagonal(diversity, np.nan)
    return diversity

def diversity(model, test_smiles, templates, topk=100, fp_length=2048, fp_radius=2, nproc=1):
    div = []
    for smi in test_smiles:
        fp = smiles_to_fingerprint(smi, length=fp_length, radius=fp_radius)
        pred = model.predict(fp.reshape(1, -1)).reshape(-1)
        ind = np.argsort(-pred)[:topk]
        precursors = precursors_from_templates(smi, templates[ind], nproc=nproc)
        div.append(diversity_from_smiles_list(precursors, nproc=nproc))
    return div