from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
from py2opsin import py2opsin
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from functools import partial
from pathlib import Path

def name_to_smiles(name):
    smiles = py2opsin(str(name))
    if smiles:
        return smiles
    else: 
        return None

def calculate_descriptor(smiles, descriptor_func):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return descriptor_func(mol)
    
def main():

    product_file = "data/raw/amika_perk_up_dry_shampoo.csv"
    df = pd.read_csv(product_file)
    df.columns = ["IUPAC_NAME", "GIVEN_NAME"]

    df['SMILES'] = df['IUPAC_NAME'].apply(lambda x: name_to_smiles(x))

    mol_weight = partial(calculate_descriptor, descriptor_func=Descriptors.MolWt)
    h_acceptor = partial(calculate_descriptor, descriptor_func=Descriptors.NumHAcceptors)
    h_donor = partial(calculate_descriptor, descriptor_func=Lipinski.NumHDonors)
    tpsa = partial(calculate_descriptor, descriptor_func=Chem.rdMolDescriptors.CalcTPSA)

    descriptors = {
        'MW': mol_weight,
        'H_ACCEPTOR': h_acceptor,
        'H_DONOR': h_donor,
        'TPSA': tpsa
    }

    for col, func in descriptors.items():
        df[col] = df['SMILES'].apply(func)
    
    df.to_csv(f"data/{Path(product_file).name}")

if __name__ == "__main__":
    main()
