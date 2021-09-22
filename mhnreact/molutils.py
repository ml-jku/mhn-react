# -*- coding: utf-8 -*-
"""
Author: Philipp Seidl, Philipp Renz
        ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning
        Johannes Kepler University Linz
Contact: seidl@ml.jku.at

Molutils contains functions that aid in handling molecules or templates
"""

import logging
import re
import warnings
from itertools import product, permutations

from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
from tqdm.notebook import tqdm
import swifter

import rdkit.RDLogger as rkl
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.Chem.rdmolops import FastFindRings
from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder

from scipy import sparse
from sklearn.feature_extraction import DictVectorizer

import warnings
import rdkit.RDLogger as rkl
import numpy as np

log = logging.getLogger(__name__)
logger = rkl.logger()

def remove_attom_mapping(smiles):
    """ removes a number after a ':' """
    return re.sub(r':\d+', '', str(smiles))


def canonicalize_smi(smi, is_smarts=False, remove_atom_mapping=True):
    r"""
    Canonicalize SMARTS from https://github.com/rxn4chemistry/rxnfp/blob/master/rxnfp/tokenization.py#L249
    """
    mol = Chem.MolFromSmarts(smi)
    if not mol:
        raise ValueError("Molecule not canonicalizable")
    if remove_atom_mapping:
        for atom in mol.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol)


def canonicalize_template(smarts):
    smarts = str(smarts)
    # remove attom-mapping
    #smarts = remove_attom_mapping(smarts)

    # order the list of smiles + canonicalize it
    results = []
    for part in smarts.split('>>'):
        a = part.split('.')
        a = [canonicalize_smi(x, is_smarts=True, remove_atom_mapping=True) for x in a]
        #a = [remove_attom_mapping(x) for x in a]
        a.sort()
        results.append( '.'.join(a) )
    return '>>'.join(results)

def ebv2np(ebv):
    """Explicit bit vector returned by rdkit to numpy array. """
    return np.frombuffer(bytes(ebv.ToBitString(), 'utf-8'), 'u1') - ord('0')

def smiles2morgan(smiles, radius=2):
    """ computes ecfp from smiles """
    return GetMorganFingerprint(smiles, radius)


def getFingerprint(smiles, fp_size=4096, radius=2, is_smarts=False, which='morgan', sanitize=True):
    """maccs+morganc+topologicaltorsion+erg+atompair+pattern+rdkc"""
    if isinstance(smiles, list):
        return np.array([getFingerprint(smi, fp_size, radius, is_smarts, which) for smi in smiles]).max(0) # max pooling if it's list of lists

    if is_smarts:
        mol = Chem.MolFromSmarts(str(smiles), mergeHs=False)
        #mol.UpdatePropertyCache() #Correcting valence info
        #FastFindRings(mol) #Providing ring info
    else:
        mol = Chem.MolFromSmiles(str(smiles), sanitize=False)

    if mol is None:
        msg = f"{smiles} couldn't be converted to a fingerprint using 0's instead"
        logger.warning(msg)
        #warnings.warn(msg)
        return np.zeros(fp_size).astype(np.bool)

    if sanitize:
        faild_op = Chem.SanitizeMol(mol, catchErrors=True)
        FastFindRings(mol) #Providing ring info

    mol.UpdatePropertyCache(strict=False) #Correcting valence info # important operation

    def mol2np(mol, which, fp_size):
        is_dict = False
        if which=='morgan':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size, useFeatures=False, useChirality=True)
        elif which=='rdk':
            fp = Chem.RDKFingerprint(mol, fpSize=fp_size, maxPath=6)
        elif which=='rdkc':
            # https://greglandrum.github.io/rdkit-blog/similarity/reference/2021/05/26/similarity-threshold-observations1.html
            # -- maxPath 6 found to be better for retrieval in databases
            fp = AllChem.UnfoldedRDKFingerprintCountBased(mol, maxPath=6).GetNonzeroElements()
            is_dict = True
        elif which=='morganc':
            fp = AllChem.GetMorganFingerprint(mol, radius, useChirality=True, useBondTypes=True, useFeatures=True,  useCounts=True).GetNonzeroElements()
            is_dict = True
        elif which=='topologicaltorsion':
            fp = AllChem.GetTopologicalTorsionFingerprint(mol).GetNonzeroElements()
            is_dict = True
        elif which=='maccs':
            fp = AllChem.GetMACCSKeysFingerprint(mol)
        elif which=='erg':
            v = AllChem.GetErGFingerprint(mol)
            fp = {idx:v[idx] for idx in np.nonzero(v)[0]}
            is_dict = True
        elif which=='atompair':
            fp = AllChem.GetAtomPairFingerprint(mol).GetNonzeroElements()
            is_dict = True
        elif which=='pattern':
            fp = Chem.PatternFingerprint(mol, fpSize=fp_size)
        elif which=='ecfp4':
            # roughly equivalent to ECFP4
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_size, useFeatures=False, useChirality=True)
        elif which=='layered':
            fp = AllChem.LayeredFingerprint(mol, fpSize=fp_size, maxPath=7)
        elif which=='mhfp':
            #TODO check if one can avoid instantiating the MHFP encoder
            fp = MHFPEncoder().EncodeMol(mol, radius=radius, rings=True, isomeric=False, kekulize=False, min_radius=1)
            fp = {f:1 for f in fp}
            is_dict = True
        elif not (type(which)==str):
            fp = which(mol)

        if is_dict:
            nd = np.zeros(fp_size)
            for k in fp:
                nk = k%fp_size #remainder
                #print(nk, k, fp_size)
                #3160 36322170 3730
                #print(nd[nk], fp[k])
                if nd[nk]!=0:
                    #print('c',end='')
                    nd[nk] = nd[nk]+fp[k] #pooling colisions
                nd[nk] = fp[k]

            return nd #np.log(1+nd) # discussion with segler

        return ebv2np(fp)

    """ + for folding * for concat """
    cc_symb = '*'
    if ('+' in which) or (cc_symb in which):
        concat = False
        split_sym = '+'
        if cc_symb in which:
            concat=True
            split_sym = '*'

        np_fp = np.zeros(fp_size)

        remaining_fps = (which.count(split_sym)+1)
        fp_length_remain = fp_size

        for fp_type in which.split(split_sym):
            if concat:
                fpp = mol2np(mol, fp_type, fp_length_remain//remaining_fps)
                np_fp[(fp_size-fp_length_remain):(fp_size-fp_length_remain+len(fpp))] += fpp
                fp_length_remain -= len(fpp)
                remaining_fps -=1
            else:
                try:
                  fpp = mol2np(mol, fp_type, fp_size)
                  np_fp[:len(fpp)] += fpp
                except:
                  pass
                  #print(fp_type,end='')

        return np.log(1 + np_fp)
    else:
        return mol2np(mol, which, fp_size)


def _getFingerprint(inp):
  return getFingerprint(inp[0], inp[1], inp[2], inp[3], inp[4])


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')


def convert_smiles_to_fp(list_of_smiles, fp_size=2048, is_smarts=False, which='morgan', radius=2, njobs=1, verbose=False):
    """
    list of smiles can be list of lists, than the resulting array will pe badded to the max list len
    which: morgan, rdk, ecfp4, or object
    NOTE: morgan or ecfp4 throws error for is_smarts
    """

    inp = [(smi, fp_size, radius, is_smarts, which) for smi in list_of_smiles]
    #print(inp)
    if verbose: print(f'starting pool with {njobs} workers')
    if njobs>1:
    #with Pool(njobs) as pool:
    #    fps = pool.map(_getFingerprint, inp)
        fps = process_map(_getFingerprint, inp, max_workers=njobs, chunksize=1, mininterval=0)
    else:
        fps = [getFingerprint(smi, fp_size=fp_size, radius=radius, is_smarts=is_smarts, which=which) for smi in list_of_smiles]
    return np.array(fps)


def convert_smartes_to_fp(list_of_smarts, fp_size=2048):
    if isinstance(list_of_smarts, np.ndarray):
        list_of_smarts = list_of_smarts.tolist()
    if isinstance(list_of_smarts, list):
        if isinstance(list_of_smarts[0], list):
            pad = len(max(list_of_smarts, key=len))
            fps = [[getTemplateFingerprint(smarts, fp_size=fp_size) for smarts in sample]
                   + [np.zeros(fp_size, dtype=np.bool)] * (pad - len(sample))  # zero padding
                   for sample in list_of_smarts]
        else:
            fps = [[getTemplateFingerprint(smarts, fp_size=fp_size) for smarts in list_of_smarts]]
    return np.asarray(fps)


def get_reactants_from_smarts(smarts):
    """
        from a (forward-)reaction given as a smart, only returns the reactants (not e.g. solvents or reagents)
        returns list of smiles or empty list
    """
    from rdkit.Chem import RDConfig
    import sys
    sys.path.append(RDConfig.RDContribDir)
    from RxnRoleAssignment import identifyReactants
    try:
        rdk_reaction = AllChem.ReactionFromSmarts(smarts)
        rx_idx = identifyReactants.identifyReactants(rdk_reaction)[0][0]
    except ValueError:
        return []
    # TODO what if a product is recognized as a reactanat.. is that possible??
    return [Chem.MolToSmiles(rdk_reaction.GetReactants()[i]) for i in rx_idx]


def smarts2rdkfp(smart, fp_size=2048):
    mol = Chem.MolFromSmarts(str(smart))
    if mol is None: return np.zeros(fp_size).astype(np.bool)
    return AllChem.RDKFingerprint(mol)
    # fp = np.asarray(fp).astype(np.bool) # takes ages =/


def smiles2rdkfp(smiles, fp_size=2048):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: return np.zeros(fp_size).astype(np.bool)
    return AllChem.RDKFingerprint(mol)


def mol2morganfp(mol, radius=2, fp_size=2048):
    try:
        Chem.SanitizeMol(mol)  # due to error --> see https://sourceforge.net/p/rdkit/mailman/message/34828604/
    except:
        pass
        # print(mol)
        # return np.zeros(fp_size).astype(np.bool)
        # TODO
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size)


def smarts2morganfp(smart, fp_size=2048, radius=2):
    mol = Chem.MolFromSmarts(str(smart))
    if mol is None: return np.zeros(fp_size).astype(np.bool)
    return mol2morganfp(mol)


def smiles2morganfp(smiles, fp_size=2048, radius=2):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: return np.zeros(fp_size).astype(np.bool)
    return mol2morganfp(mol)


def smarts2fp(smart, which='morgan', fp_size=2048, radius=2):
    if which == 'rdk':
        return smarts2rdkfp(smart, fp_size=fp_size)
    else:
        return smarts2morganfp(smart, fp_size=fp_size, radius=radius)


def smiles2fp(smiles, which='morgan', fp_size=2048, radius=2):
    if which == 'rdk':
        return smiles2rdkfp(smiles, fp_size=fp_size)
    else:
        return smiles2morganfp(smiles, fp_size=fp_size, radius=radius)


class FP_featurizer():
    "FP_featurizer: Fingerprint featurizer"
    def __init__(self,
                 fp_types = ['MACCS','Morgan2CBF', 'Morgan6CBF', 'ErG','AtomPair','TopologicalTorsion','RDK','ECFP6'],
                 max_features = 4096, counts=True, log_scale=True, folding=None, collision_pooling='max'):

        self.v = DictVectorizer(sparse=True, dtype=np.uint16)
        self.max_features = max_features
        self.idx_col = None
        self.counts = counts
        self.fp_types = [fp_types] if isinstance(fp_types, str) else fp_types

        self.log_scale = log_scale # from discussion with segler

        self.folding = None
        self.colision_pooling = collision_pooling

    def compute_fp_list(self, smiles_list, is_smarts=False):
        fp_list = []
        for smiles in smiles_list:
            try:
                if isinstance(smiles, list):
                    smiles = smiles[0]
                if is_smarts:
                    mol = Chem.MolFromSmarts(smiles)
                else:
                    mol = Chem.MolFromSmiles(smiles) #TODO small hack only applicable here!!!
                fp_dict = {}
                for fp_type in self.fp_types:
                    fp_dict.update( fingerprintTypes[fp_type](mol) ) #returns a dict
                fp_list.append(fp_dict)
            except:
                fp_list.append({})
        return fp_list

    def fit(self, x_train, is_smarts=False):
        fp_list = self.compute_fp_list(x_train, is_smarts=is_smarts)
        Xraw = self.v.fit_transform(fp_list)
        # compute variance of a csr_matrix E[x**2] - E[x]**2
        axis = 0
        Xraw_sqrd = Xraw.copy()
        Xraw_sqrd.data **= 2
        var_col = Xraw_sqrd.mean(axis) - np.square(Xraw.mean(axis))
        #idx_col = (-np.array((Xraw>0).var(axis=0)).argpartition(self.max_features))
        #idx_col = np.array((Xraw>0).sum(axis=0)>=self.min_fragm_occur).flatten()
        self.idx_col = (-np.array(var_col)).flatten().argpartition(min(self.max_features, Xraw.shape[1]-1))[:min(self.max_features, Xraw.shape[1])]
        print(f'from {var_col.shape[1]} to {len(self.idx_col)}')
        return self.scale(Xraw[:,self.idx_col].toarray())

    def transform(self, x_test, is_smarts=False):
        fp_list = self.compute_fp_list(x_test, is_smarts=is_smarts)
        X_raw = self.v.transform(fp_list)
        return self.scale(X_raw[:,self.idx_col].toarray())

    def scale(self, X):
        if self.log_scale:
            return np.log(1 + X)
        return X

    def save(self, path='data/fpfeat.pkl'):
        import pickle
        with open(path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load(self, path='data/fpfeat.pkl'):
        import pickle
        with open(path, 'rb') as input:
            self = pickle.load(input)


def getTemplateFingerprintOnBits(smarts, fp_size=2048):
    rxn = AllChem.ReactionFromSmarts(str(smarts))
    #construct a structural fingerprint for a ChemicalReaction by concatenating the reactant fingerprint and the product fingerprint
    return (AllChem.CreateStructuralFingerprintForReaction(rxn)).GetOnBits()


def calc_template_fingerprint_group_mapping(template_list, fp_size, save_path=''):
    """
    calculate the mapping from old idx to new idx for the templates
    returns a set with a numpy array with the mapping and the indices to take
    """

    templ_df = pd.DataFrame()
    templ_df['smarts'] = template_list
    templ_df['templ_emb'] = templ_df['smarts'].swifter.apply(lambda smarts: str(list(getTemplateFingerprintOnBits(smarts, fp_size))))
    templ_df['idx_orig'] = [ii for ii in range(len(templ_df))]

    grouped_templ = templ_df.groupby('templ_emb').apply(lambda x: x.index.tolist())

    grouped_templ = templ_df.groupby('templ_emb')
    grouped_templ = grouped_templ.min().sort_values('idx_orig')
    grouped_templ['new_idx'] = range(len(grouped_templ))

    new_templ_df = templ_df.join(grouped_templ, on='templ_emb',how='right', lsuffix='_l', rsuffix='_r').sort_values('idx_orig_l')

    map_orig2new = new_templ_df['new_idx'].values
    take_those_indices_from_orig = grouped_templ.idx_orig.values
    if save_path!='':
        suffix_maporig2new = '_maporig2new.npy'
        suffix_takethose = '_tfp_take_idxs.npy'
        np.save(f'{save_path}{suffix_maporig2new}', map_orig2new,allow_pickle=False)
        np.save(f'{save_path}{suffix_takethose}', take_those_indices_from_orig,allow_pickle=False)
    return (map_orig2new, take_those_indices_from_orig)


class ECFC_featurizer():
    def __init__(self, radius=6, min_fragm_occur=50, useChirality=True, useFeatures=False):
        self.v = DictVectorizer(sparse=True, dtype=np.uint16)
        self.min_fragm_occur=min_fragm_occur
        self.idx_col = None
        self.radius=radius
        self.useChirality = useChirality
        self.useFeatures = useFeatures

    def compute_fp_list(self, smiles_list):
        fp_list = []
        for smiles in smiles_list:
            try:
                if isinstance(smiles, list):
                    smiles = smiles[0]
                mol = Chem.MolFromSmiles(smiles) #TODO small hack only applicable here!!!
                fp_list.append( AllChem.GetMorganFingerprint(mol, self.radius, useChirality=self.useChirality,
                                                             useFeatures=self.useFeatures).GetNonzeroElements() ) #returns a dict
            except:
                fp_list.append({})
        return fp_list

    def fit(self, x_train):
        fp_list = self.compute_fp_list(x_train)
        Xraw = self.v.fit_transform(fp_list)
        idx_col = np.array((Xraw>0).sum(axis=0)>=self.min_fragm_occur).flatten()
        self.idx_col = idx_col
        return Xraw[:,self.idx_col].toarray()

    def transform(self, x_test):
        fp_list = self.compute_fp_list(x_test)
        X_raw = self.v.transform(fp_list)
        return X_raw[:,self.idx_col].toarray()


def ecfp2dict(mol, radius=3):
    #SECFP (SMILES Extended Connectifity Fingerprint)
    # from mhfp.encoder import MHFPEncoder
    from mhfp.encoder import MHFPEncoder
    v = MHFPEncoder.secfp_from_mol(mol, length=4068, radius=radius, rings=True, kekulize=True, min_radius=1)
    return {f'ECFP{radius*2}_'+str(idx):1 for idx in np.nonzero(v)[0]}


def erg2dict(mol):
    v = AllChem.GetErGFingerprint(mol)
    return {'erg'+str(idx):v[idx] for idx in np.nonzero(v)[0]}


def morgan2dict(mol, radius=2, useChirality=True, useBondTypes=True, useFeatures=True, useConts=True):
    mdic = AllChem.GetMorganFingerprint(mol, radius=radius, useChirality=useChirality, useBondTypes=True,
                                    useFeatures=True, useCounts=True).GetNonzeroElements()
    return {f'm{radius}{useChirality}{useBondTypes}{useFeatures}'+str(kk):mdic[kk]for kk in mdic}


def atompair2dict(mol):
    mdic = AllChem.GetAtomPairFingerprint(mol).GetNonzeroElements()
    return {f'ap'+str(kk):mdic[kk]for kk in mdic}


def tt2dict(mol):
    mdic = AllChem.GetTopologicalTorsionFingerprint(mol).GetNonzeroElements()
    return {f'tt'+str(kk):mdic[kk]for kk in mdic}


def rdk2dict(mol):
    mdic = AllChem.UnfoldedRDKFingerprintCountBased(mol).GetNonzeroElements()
    return {f'rdk'+str(kk):mdic[kk]for kk in mdic}


def pattern2dict(mol):
    mdic = AllChem.PatternFingerprint(mol, fpSize=16384).GetOnBits()
    return {'pt'+str(kk):1 for kk in mdic}


fingerprintTypes = {
    'MACCS' : lambda k: {'MCCS'+str(ob):1 for ob in AllChem.GetMACCSKeysFingerprint(k).GetOnBits()},
    'Morgan2CBF' : lambda mol: morgan2dict(mol, 2, True, True, True, True),
    'Morgan4CBF' : lambda mol: morgan2dict(mol, 4, True, True, True, True),
    'Morgan6CBF' : lambda mol: morgan2dict(mol, 6, True, True, True, True),
    'ErG' :  erg2dict,
    'AtomPair' : atompair2dict,
    'TopologicalTorsion' : tt2dict,
    #'RDK' : lambda k: {'MCCS'+str(ob):1 for ob in AllChem.RDKFingerprint(k).GetOnBits()},
    'RDK' : rdk2dict,
    'ECFP6' : lambda mol: ecfp2dict(mol, radius=3),
    'Pattern': pattern2dict,
}


def smarts2appl(product_smarts, template_product_smarts, fpsize=2048, v=False, use_tqdm=False, njobs=1, nsplits=1):
    """This takes in a list of product smiles (misnamed in code) and a list of product sides
    of templates and calculates which templates are applicable to which product.
    This is basically a substructure search. Maybe there are faster versions but I wrote this one.

    Args:
        product_smarts: List of smiles of molecules to check.
        template_product_smarts: List of substructures to check
        fpsize: fingerprint size to use in screening
        v: if v then information will be printed
        use_tdqm: if True then a progressbar will be displayed but slows down the computation.
        njobs: how many parallel jobs to run in parallel.
        nsplits: how many splits should be made along the product_smarts list. Useful to avoid memory
            explosion.
    Returns: list of tuples (i,j) that indicates the product i has substructure j.
    """
    if v: print("Calculating template molecules")
    template_mols = [Chem.MolFromSmarts(s) for s in template_product_smarts]
    if v: print("Calculating template fingerprints")
    template_ebvs = [Chem.PatternFingerprint(m, fpSize=fpsize) for m in template_mols]
    if v: print(f'Building template ints: [{len(template_mols)}, {fpsize}]')
    template_ints = [int(e.ToBitString(), base=2) for e in template_ebvs]
    del template_ebvs

    if njobs == 1 and nsplits == 1:
        return _smarts2appl(product_smarts, template_product_smarts, template_ints, fpsize, v, use_tqdm)
    elif nsplits == 1:
        nsplits = njobs


    # split products into batches
    product_splits = np.array_split(np.array(product_smarts), nsplits)
    ioffsets = [0] + list(np.cumsum([p.shape[0] for p in product_splits[:-1]]))
    inps = [(ps, template_product_smarts, template_ints, fpsize, v, use_tqdm, ioff, 0) for ps, ioff in zip(product_splits, ioffsets)]

    if v: print("Creating workers")
    #results = process_map(__smarts2appl, inps, max_workers=njobs, chunksize=1)
    with Pool(njobs) as pool:
        results = pool.starmap(_smarts2appl, inps)
    imatch = np.concatenate([r[0] for r in results])
    jmatch = np.concatenate([r[1] for r in results])
    return imatch, jmatch


def __smarts2appl(inp):
    return _smarts2appl(*inp)


def _smarts2appl(product_smarts, template_product_smarts, template_ints, fpsize=2048, v=False, use_tqdm=True, ioffset=0, joffset=0):
    """See smarts2appl for a description"""

    if v: print("Calculating product molecules")
    product_mols = [Chem.MolFromSmiles(s) for s in product_smarts]
    if v: print("Calculating product fingerprints")
    product_ebvs = [Chem.PatternFingerprint(m, fpSize=fpsize) for m in product_mols]
    if v: print(f'Building product ints: [{len(product_mols)}, {fpsize}]')
    # This loads each fingerprint into a python integer on which we can use bitwise operations.
    product_ints = [int(e.ToBitString(), base=2) for e in product_ebvs]
    del product_ebvs

    # product_mols = {i: m for i,m in enumerate(product_mols)}


    if v: print('Checking symbolically')
    # buffer for template molecules. This are handed over as smarts as they are slow to pickle
    template_mols = {}

    # create iterator and add progressbar if use_tqdm is True
    iterator = product(enumerate(product_ints), enumerate(template_ints))
    if use_tqdm:
        nelem = len(product_ints) * len(template_ints)
        iterator = tqdm(iterator, total=nelem, miniters=1_000_000)

    imatch = []
    jmatch = []
    for (i, p_int), (j, t_int) in iterator:
        if (p_int & t_int) == t_int:        # fingerprint based screen
            p = product_mols[i]
            t = template_mols.get(j, False)
            if not t:
                t = Chem.MolFromSmarts(template_product_smarts[j])
                template_mols[j] = t
            if p.HasSubstructMatch(t):
                imatch.append(i)
                jmatch.append(j)
    if v: print("Finished loop")
    return np.array(imatch)+ioffset, np.array(jmatch)+joffset


def extract_from_reaction(reaction, radius=1, verbose=False):
    """adapted from rdchiral package"""
    from rdchiral.template_extractor import mols_from_smiles_list, replace_deuterated, get_fragments_for_changed_atoms, expand_changed_atom_tags, canonicalize_transform, get_changed_atoms
    reactants = mols_from_smiles_list(replace_deuterated(reaction['reactants']).split('.'))
    products = mols_from_smiles_list(replace_deuterated(reaction['products']).split('.'))

    # if rdkit cant understand molecule, return
    if None in reactants: return {'reaction_id': reaction['_id']}
    if None in products: return {'reaction_id': reaction['_id']}

    # try to sanitize molecules
    try:
        #for i in range(len(reactants)):
        #    reactants[i] = AllChem.RemoveHs(reactants[i]) # *might* not be safe
        #for i in range(len(products)):
        #    products[i] = AllChem.RemoveHs(products[i]) # *might* not be safe

        #[Chem.SanitizeMol(mol) for mol in reactants + products] # redundant w/ RemoveHs
        for mol in reactants + products:
            Chem.SanitizeMol(mol, catchErrors=True)
            FastFindRings(mol) #Providing ring info
            mol.UpdatePropertyCache(strict=False) #Correcting valence info # important operation

        #changed
        #[Chem.SanitizeMol(mol, catchErrors=True) for mol in reactants + products] # redundant w/ RemoveHs

        #[mol.UpdatePropertyCache() for mol in reactants + products]
    except Exception as e:
        # can't sanitize -> skip
        print(e)
        print('Could not load SMILES or sanitize')
        print('ID: {}'.format(reaction['_id']))
        return {'reaction_id': reaction['_id']}

    are_unmapped_product_atoms = False
    extra_reactant_fragment = ''
    for product in products:
        prod_atoms = product.GetAtoms()
        if sum([a.HasProp('molAtomMapNumber') for a in prod_atoms]) < len(prod_atoms):
            if verbose: print('Not all product atoms have atom mapping')
            if verbose: print('ID: {}'.format(reaction['_id']))
            are_unmapped_product_atoms = True

    if are_unmapped_product_atoms: # add fragment to template
        for product in products:
            prod_atoms = product.GetAtoms()
            # Get unmapped atoms
            unmapped_ids = [
                a.GetIdx() for a in prod_atoms if not a.HasProp('molAtomMapNumber')
            ]
            if len(unmapped_ids) > MAXIMUM_NUMBER_UNMAPPED_PRODUCT_ATOMS:
                # Skip this example - too many unmapped product atoms!
                return
            # Define new atom symbols for fragment with atom maps, generalizing fully
            atom_symbols = ['[{}]'.format(a.GetSymbol()) for a in prod_atoms]
            # And bond symbols...
            bond_symbols = ['~' for b in product.GetBonds()]
            if unmapped_ids:
                extra_reactant_fragment += AllChem.MolFragmentToSmiles(
                    product, unmapped_ids,
                    allHsExplicit = False, isomericSmiles = USE_STEREOCHEMISTRY,
                    atomSymbols = atom_symbols, bondSymbols = bond_symbols
                ) + '.'
        if extra_reactant_fragment:
            extra_reactant_fragment = extra_reactant_fragment[:-1]
            if verbose: print('    extra reactant fragment: {}'.format(extra_reactant_fragment))

        # Consolidate repeated fragments (stoichometry)
        extra_reactant_fragment = '.'.join(sorted(list(set(extra_reactant_fragment.split('.')))))


    if None in reactants + products:
        print('Could not parse all molecules in reaction, skipping')
        print('ID: {}'.format(reaction['_id']))
        return {'reaction_id': reaction['_id']}

    # Calculate changed atoms
    changed_atoms, changed_atom_tags, err = get_changed_atoms(reactants, products)
    if err:
        if verbose:
            print('Could not get changed atoms')
            print('ID: {}'.format(reaction['_id']))
        return
    if not changed_atom_tags:
        if verbose:
            print('No atoms changed?')
            print('ID: {}'.format(reaction['_id']))
        # print('Reaction SMILES: {}'.format(example_doc['RXN_SMILES']))
        return {'reaction_id': reaction['_id']}

    try:
        # Get fragments for reactants
        reactant_fragments, intra_only, dimer_only = get_fragments_for_changed_atoms(reactants, changed_atom_tags,
            radius = radius, expansion = [], category = 'reactants')
        # Get fragments for products
        # (WITHOUT matching groups but WITH the addition of reactant fragments)
        product_fragments, _, _  = get_fragments_for_changed_atoms(products, changed_atom_tags,
            radius = radius-1, expansion = expand_changed_atom_tags(changed_atom_tags, reactant_fragments),
            category = 'products')
    except ValueError as e:
        if verbose:
            print(e)
            print(reaction['_id'])
        return {'reaction_id': reaction['_id']}

    # Put together and canonicalize (as best as possible)
    rxn_string = '{}>>{}'.format(reactant_fragments, product_fragments)
    rxn_canonical = canonicalize_transform(rxn_string)
    # Change from inter-molecular to intra-molecular
    rxn_canonical_split = rxn_canonical.split('>>')
    rxn_canonical = rxn_canonical_split[0][1:-1].replace(').(', '.') + \
        '>>' + rxn_canonical_split[1][1:-1].replace(').(', '.')

    reactants_string = rxn_canonical.split('>>')[0]
    products_string  = rxn_canonical.split('>>')[1]

    retro_canonical = products_string + '>>' + reactants_string

    # Load into RDKit
    rxn = AllChem.ReactionFromSmarts(retro_canonical)
    # edited
    #if rxn.Validate()[1] != 0:
    #    print('Could not validate reaction successfully')
    #    print('ID: {}'.format(reaction['_id']))
    #    print('retro_canonical: {}'.format(retro_canonical))
    #    if VERBOSE: raw_input('Pausing...')
    #    return {'reaction_id': reaction['_id']}
    n_warning, n_errors = rxn.Validate()
    if n_errors:
      # resolves some errors
      rxn = AllChem.ReactionFromSmarts(AllChem.ReactionToSmiles(rxn))
      n_warning, n_errors = rxn.Validate()

    template = {
        'products': products_string,
        'reactants': reactants_string,
        'reaction_smarts': retro_canonical,
        'intra_only': intra_only,
        'dimer_only': dimer_only,
        'reaction_id': reaction['_id'],
        'necessary_reagent': extra_reactant_fragment,
        'num_errors': n_errors,
        'num_warnings': n_warning,
    }

    return template


def extract_template(rxn_smi, radius=1):
    if isinstance(rxn_smi, str):
        reaction = {
            'reactants': rxn_smi.split('>')[0],
            'products': rxn_smi.split('>')[-1],
            'id': rxn_smi,
            '_id': rxn_smi
        }
    else:
        reaction = rxn_smi
    try:
        res = extract_from_reaction(reaction, radius=radius)
        return res['reaction_smarts'] # returns a retro-template
    except:
        msg = f'failed to extract template from "{rxn_smi}"'
        log.warning(msg)
        return None


def getTemplateFingerprint(smarts, fp_size=4096):
    """ CreateStructuralFingerprintForReaction """
    if isinstance(smarts, (list,)):
        return np.vstack([getTemplateFingerprint(sm) for sm in smarts])

    rxn = AllChem.ReactionFromSmarts(str(smarts))
    if rxn is None:
        msg = f"{smarts} couldn't be converted to a fingerprint using 0's instead"
        log.warning(msg)
        #warnings.warn(msg)
        return np.zeros(fp_size).astype(np.bool)

    return np.array(list(AllChem.CreateStructuralFingerprintForReaction(rxn, )), dtype=np.bool)