"""
Hybrid Element Classification System
======================================
Hybrid classification combining chemical knowledge and dataset analysis
to categorize elements into metals, semiconductors, and insulators.

NOTE: File paths are set to '/content/' for Google Colab compatibility.
Please update these paths according to your local environment.
"""

import pandas as pd
import re
from collections import defaultdict


def get_element_categories_hybrid(file_path):
    """
    Hybrid classification: chemical knowledge + dataset analysis.
    """
    # Read dataset
    df = pd.read_csv(file_path)

    # Find columns
    comp_col = None
    gap_num_col = None
    gap_cat_col = None

    for col in df.columns:
        col_lower = col.lower()
        if 'composition' in col_lower:
            comp_col = col
        elif 'gap expt' in col_lower or 'gap_expt' in col_lower:
            gap_num_col = col
        elif 'gap_category' in col_lower:
            gap_cat_col = col

    pattern = re.compile(r'([A-Z][a-z]?)(?=\d|\.|$)')

    # For each element, collect statistics
    element_stats = defaultdict(lambda: {'gaps': [], 'insulator_count': 0, 'total': 0})

    for _, row in df.iterrows():
        comp = str(row[comp_col])
        elements = set(pattern.findall(comp))

        # Get gap
        gap_value = None
        if gap_num_col and pd.notna(row.get(gap_num_col)):
            gap_value = float(row[gap_num_col])

        # Count if insulator
        is_insulator = False
        if gap_value is not None and gap_value >= 3.0:
            is_insulator = True

        # Update statistics
        for element in elements:
            element_stats[element]['total'] += 1
            if gap_value is not None:
                element_stats[element]['gaps'].append(gap_value)
            if is_insulator:
                element_stats[element]['insulator_count'] += 1

    # Hybrid classification
    element_categories = {
        'metals': [],
        'semiconductors': [],
        'insulators': [],
        'transition_metals': []
    }

    # Base chemical knowledge
    KNOWN_INSULATOR_ELEMENTS = {
        'O': 'Forms insulating oxides (MgO, Al2O3, SiO2)',
        'F': 'Most electronegative halogen, forms ionic insulators',
        'Cl': 'Forms insulating chlorides (NaCl, KCl)',
        'Br': 'Forms insulating bromides',
        'I': 'Forms insulating iodides',
        'N': 'Forms insulating nitrides (AlN, GaN, BN)',
        'S': 'Forms insulating sulfides (ZnS, CdS)',
        'H': 'In many insulating compounds (H2O, NH3)',
        'B': 'Boron oxide (B2O3) is an insulator',
        'C': 'Diamond is a perfect insulator',
        'P': 'Some phosphates are insulators',
        'Se': 'Some selenides are insulators',
        'Te': 'Some tellurides are insulators'
    }

    KNOWN_METAL_ELEMENTS = {
        'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',
        'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',
        'Al', 'Ga', 'In', 'Tl',
        'Sn', 'Pb', 'Bi', 'Po'
    }

    KNOWN_SEMICONDUCTOR_ELEMENTS = {
        'Si', 'Ge',
        'As', 'Sb',
        'Se', 'Te',
        'Ga', 'In', 'Cd', 'Zn', 'Hg', 'Sn', 'Pb'
    }

    # Transition metals (by periodic table)
    TRANSITION_METALS = {
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'
    }

    # Process each element
    all_elements = set(element_stats.keys())

    for element in all_elements:
        stats = element_stats[element]

        # If the element has few appearances, use chemical knowledge
        if stats['total'] < 5:
            # Classify by chemical knowledge
            if element in KNOWN_INSULATOR_ELEMENTS:
                element_categories['insulators'].append(element)
            elif element in KNOWN_METAL_ELEMENTS:
                element_categories['metals'].append(element)
            elif element in KNOWN_SEMICONDUCTOR_ELEMENTS:
                element_categories['semiconductors'].append(element)
            else:
                # Default to metal
                element_categories['metals'].append(element)

        else:
            # Classify by dataset statistics
            insulator_pct = (stats['insulator_count'] / stats['total']) * 100

            if stats['gaps']:
                avg_gap = sum(stats['gaps']) / len(stats['gaps'])
            else:
                avg_gap = 0

            # CLASSIFICATION RULES:
            # 1. If it appears as insulator in > 20% of cases -> INSULATOR
            # 2. If average gap >= 2.5 -> INSULATOR
            # 3. If average gap <= 0.5 -> METAL
            # 4. Others -> SEMICONDUCTOR

            if insulator_pct >= 20 or avg_gap >= 2.5:
                element_categories['insulators'].append(element)
            elif avg_gap <= 0.5:
                element_categories['metals'].append(element)
            else:
                element_categories['semiconductors'].append(element)

        # Transition metals (always based on periodic table)
        if element in TRANSITION_METALS:
            element_categories['transition_metals'].append(element)

    # Ensure known insulator elements are in the list
    for insulator in KNOWN_INSULATOR_ELEMENTS:
        if insulator in all_elements and insulator not in element_categories['insulators']:
            # Verify it is not already in another category by mistake
            if insulator in element_categories['metals']:
                element_categories['metals'].remove(insulator)
            if insulator in element_categories['semiconductors']:
                element_categories['semiconductors'].remove(insulator)
            element_categories['insulators'].append(insulator)

    # Remove duplicates and sort
    for category in element_categories:
        element_categories[category] = sorted(list(set(element_categories[category])))

    return element_categories


# Usage
file_path = "/content/train_balanced_stratified.csv"
categories = get_element_categories_hybrid(file_path)

print("=" * 60)
print("HYBRID CLASSIFICATION (Chemical Knowledge + Dataset)")
print("=" * 60)

for cat, elems in categories.items():
    print(f"\n{cat.upper()} ({len(elems)} elements):")
    if elems:
        # Display in columns
        for i in range(0, len(elems), 10):
            chunk = elems[i:i+10]
            print("  " + ", ".join(chunk))
    else:
        print("  (none)")

# Generate code for copying
print("\n" + "=" * 60)
print("CODE FOR YOUR PROJECT:")
print("=" * 60)

print("\n        # Map elements by type (hybrid classification)")
print("        element_categories = {")
for i, (cat, elems) in enumerate(categories.items()):
    elems_str = "', '".join(elems)
    if i < len(categories) - 1:
        print(f"            '{cat}': ['{elems_str}'],")
    else:
        print(f"            '{cat}': ['{elems_str}']")
print("        }")
