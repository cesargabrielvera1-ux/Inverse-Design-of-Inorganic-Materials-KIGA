"""
Final Element Classification
=============================
Definitive element classification based on the dataset and chemical knowledge.

NOTE: Update paths and imports according to your local environment.
"""

# DEFINITIVE CLASSIFICATION BASED ON YOUR DATASET AND CHEMICAL KNOWLEDGE
element_categories = {
    'metals': [
        # Alkali metals
        'Li', 'Na', 'K', 'Rb', 'Cs',
        # Alkaline earth metals
        'Be', 'Mg', 'Ca', 'Sr', 'Ba',
        # Transition metals with low gap in the dataset
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        # Lanthanides (metallic)
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
        # Actinides
        'Th', 'U',
        # Other p-block metals
        'Al', 'Ga', 'In', 'Tl', 'Sn', 'Pb', 'Bi'
    ],

    'semiconductors': [
        # Elemental semiconductors
        'Si', 'Ge',
        # III-V and II-VI semiconductors
        'Ga', 'As', 'In', 'P', 'Sb',
        # Chalcogen semiconductors
        'Se', 'Te',
        # Others that appear as semiconductors in the dataset
        'B', 'C', 'S', 'Zn', 'Cd', 'Hg', 'Sn', 'Pb',
        # Elements that can behave as metals or semiconductors
        'Ag', 'Au', 'Cu', 'Fe', 'Mn', 'Mo', 'Ni', 'Ti', 'V', 'W', 'Zr'
    ],

    'insulators': [
        # HALOGENS (form ionic insulating compounds)
        'F', 'Cl', 'Br', 'I',
        # OXYGEN (forms insulating oxides: MgO, Al2O3, SiO2, etc.)
        'O',
        # NITROGEN (forms insulating nitrides: AlN, GaN, BN)
        'N',
        # HYDROGEN (in many insulating compounds)
        'H',
        # SULFUR (some sulfides are insulators: ZnS, CdS)
        'S',
        # BORON (B2O3 is an insulator)
        'B',
        # CARBON (diamond is a perfect insulator)
        'C',
        # PHOSPHORUS (some phosphates are insulators)
        'P',
        # Alkaline earth metals in ionic compounds
        'Be', 'Mg', 'Ca', 'Sr', 'Ba',
        # Aluminum in insulating oxides
        'Al'
    ],

    'transition_metals': [
        # Transition metals (d-block)
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        # Lanthanides (f-block)
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
        # Selected actinides
        'Th', 'U'
    ]
}

# Note: Some elements appear in multiple categories because:
# 1. They can behave differently in different compounds
# 2. Your dataset shows that some elements appear in several types of materials
# 3. It is more useful for your model to have this information than a strict classification

print("Code to copy into your project:")
print("\nelement_categories = {")
for i, (cat, elems) in enumerate(element_categories.items()):
    elems_str = "', '".join(elems)
    if i < len(element_categories) - 1:
        print(f"    '{cat}': ['{elems_str}'],")
    else:
        print(f"    '{cat}': ['{elems_str}']")
print("}")
