from rdkit import Chem
from rdkit.Chem import AllChem

def validate_template(template):
    rxn = AllChem.ReactionFromSmarts(template['reaction_smarts'])
    if not rxn:
        return False
    prod_mol = Chem.MolFromSmiles(template['products'])
    if not prod_mol:
        return False
    react_mol = Chem.MolFromSmiles(template['reactants'])
    if not react_mol:
        return False
    for atom in react_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        rxn_res = rxn.RunReactants([prod_mol])
    except ValueError:
        return False
    if not rxn_res:
        return False
    for res in rxn_res:
        react_smi = '.'.join([Chem.MolToSmiles(react) for react in res])
        res_mol = Chem.MolFromSmiles(react_smi)
        if not res_mol:
            continue
        for atom in res_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        react_smi = Chem.MolToSmiles(res_mol)
        if react_smi in Chem.MolToSmiles(react_mol):
            return True
    return False

def mapping_validity(reaction_smiles):
    reactants, spectators, products = reaction_smiles.split('>')
    reactant_mol = Chem.MolFromSmiles(reactants)
    product_mol = Chem.MolFromSmiles(products)
    
    reactant_mapping = {}
    for atom in reactant_mol.GetAtoms():
        map_number = atom.GetAtomMapNum()
        if not map_number: continue
        if map_number in reactant_mapping:
            return 'DUPLICATE_REACTANT_MAPPING'
        reactant_mapping[map_number] = atom.GetIdx()
    
    product_mapping = {}
    for atom in product_mol.GetAtoms():
        map_number = atom.GetAtomMapNum()
        if not map_number: continue
        if map_number in product_mapping:
            return 'DUPLICATE_PRODUCT_MAPPING'
        product_mapping[map_number] = atom.GetIdx()
    
    if len(reactant_mapping) < len(product_mapping):
        return 'UNMAPPED_REACTANT_ATOM(S)'

    for map_number in product_mapping.keys():
        if map_number not in reactant_mapping:
            return 'UNMAPPED_PRODUCT_ATOM'
        reactant_atom = reactant_mol.GetAtomWithIdx(reactant_mapping[map_number])
        product_atom = product_mol.GetAtomWithIdx(product_mapping[map_number])

        if reactant_atom.GetSymbol() != product_atom.GetSymbol():
            return 'ALCHEMY'

    return 'VALID'