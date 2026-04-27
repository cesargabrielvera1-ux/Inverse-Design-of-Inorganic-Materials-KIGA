"""
Dataset Unique Elements Extractor
====================================
Extracts the list of unique chemical elements present in a dataset
from composition strings.

NOTE: File paths are set to '/content/' for Google Colab compatibility.
Please update these paths according to your local environment.
"""

import re
import pandas as pd

# Load training dataframe directly to ensure it is defined
df_train = pd.read_excel('/content/Compositions.xlsx')

comp_col = 'composition' if 'composition' in df_train.columns else 'final_composition'
compositions = df_train[comp_col].dropna().tolist()

# List of all element symbols (1 to 118)
all_elements = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]

# For each composition in the dataset, extract elements
unique_elements = set()
for comp in compositions:
    # comp is a string such as "Cd0.06 In0.94 Te0.06 As0.94"
    candidates = re.findall(r'[A-Z][a-z]?', comp)
    for cand in candidates:
        if cand in all_elements:
            unique_elements.add(cand)

print(unique_elements)
