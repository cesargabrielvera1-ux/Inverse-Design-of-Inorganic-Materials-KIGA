"""
Ablation Study: KIGA vs. Baseline Genetic Algorithm
=====================================================
Compares the full Knowledge-Informed Genetic Algorithm (KIGA)
against a chemistry-agnostic baseline using identical GA operators and
initialization. Both are evaluated on the same stability metric post-hoc.

NOTE: File paths are set to '/content/' for Google Colab compatibility.
Please update these paths according to your local environment.
"""

import subprocess
import sys

# Install dependencies if needed
print("Installing dependencies...")
for pkg in ["tensorflow", "scikit-learn", "pymatgen", "matminer", "pandas", "numpy", "matplotlib", "seaborn", "joblib", "tqdm", "scipy"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
print("Done.")

import os
import random
import hashlib
import itertools
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from scipy import stats

from pymatgen.core import Composition, Element
from matminer.featurizers.composition import ElementProperty
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Global style for plots
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# =============================================================================
# Magpie Calculator
# =============================================================================

class MagpieCalculator:
    def __init__(self):
        self.featurizer = ElementProperty.from_preset("magpie")
        self.descriptor_names = self.featurizer.feature_labels()
        self.composition_cache = {}
        print(f"Magpie initialized: {len(self.descriptor_names)} descriptors")

    def calculate(self, composition: Composition, use_cache: bool = True) -> np.ndarray:
        formula = composition.reduced_formula
        formula_key = f"{formula}_{hashlib.md5(str(composition).encode()).hexdigest()[:8]}"
        if use_cache and formula_key in self.composition_cache:
            return self.composition_cache[formula_key]
        try:
            descriptors = self.featurizer.featurize(composition)
            if len(descriptors) != 132:
                descriptors = list(descriptors) + [0.0] * (132 - len(descriptors))
            descriptors = np.array(descriptors, dtype=np.float32)
            descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
            if use_cache:
                self.composition_cache[formula_key] = descriptors
            return descriptors
        except Exception:
            return np.zeros(132, dtype=np.float32)

    def composition_from_formula(self, formula: str) -> Optional[Composition]:
        try:
            return Composition(formula)
        except Exception:
            return None

# =============================================================================
# Chemical Knowledge Base & Stability Score
# =============================================================================

def get_valence_rules():
    return {
        'O': [-2], 'H': [+1], 'F': [-1], 'Cl': [-1], 'Br': [-1], 'I': [-1],
        'S': [-2, +4, +6], 'N': [-3, +3, +5], 'C': [-4, +4], 'P': [-3, +3, +5],
        'Se': [-2, +4, +6], 'Te': [-2, +4, +6], 'As': [-3, +3, +5], 'Sb': [-3, +3, +5],
        'Li': [+1], 'Na': [+1], 'K': [+1], 'Rb': [+1], 'Cs': [+1],
        'Be': [+2], 'Mg': [+2], 'Ca': [+2], 'Sr': [+2], 'Ba': [+2],
        'Al': [+3], 'Ga': [+3], 'In': [+3], 'Tl': [+1, +3], 'Sn': [+2, +4], 'Pb': [+2, +4], 'Bi': [+3],
        'Sc': [+3], 'Ti': [+2, +3, +4], 'V': [+2, +3, +4, +5], 'Cr': [+2, +3, +6],
        'Mn': [+2, +3, +4, +6, +7], 'Fe': [+2, +3], 'Co': [+2, +3], 'Ni': [+2, +3],
        'Cu': [+1, +2], 'Zn': [+2], 'Y': [+3], 'Zr': [+4], 'Nb': [+3, +5],
        'Mo': [+3, +4, +5, +6], 'Tc': [+4, +7], 'Ru': [+3, +4, +6, +8],
        'Rh': [+3], 'Pd': [+2, +4], 'Ag': [+1], 'Cd': [+2],
        'Hf': [+4], 'Ta': [+5], 'W': [+4, +5, +6], 'Re': [+4, +6, +7],
        'Os': [+3, +4, +6, +8], 'Ir': [+3, +4], 'Pt': [+2, +4], 'Au': [+1, +3], 'Hg': [+1, +2],
        'La': [+3], 'Ce': [+3, +4], 'Pr': [+3], 'Nd': [+3], 'Pm': [+3],
        'Sm': [+2, +3], 'Eu': [+2, +3], 'Gd': [+3], 'Tb': [+3, +4],
        'Dy': [+3], 'Ho': [+3], 'Er': [+3], 'Tm': [+3], 'Yb': [+2, +3], 'Lu': [+3],
        'Th': [+4], 'U': [+4, +6], 'B': [+3], 'Si': [+4], 'Ge': [+2, +4],
    }


def estimate_stoichiometry_penalty(composition_dict: Dict) -> float:
    """Simplified charge-balance check for binary systems."""
    elements = list(composition_dict.keys())
    if len(elements) == 2:
        elem1, elem2 = elements[0], elements[1]
        valence_rules = get_valence_rules()
        valences1 = valence_rules.get(elem1, [1])
        valences2 = valence_rules.get(elem2, [1])
        try:
            m1 = Element(elem1).atomic_mass
            m2 = Element(elem2).atomic_mass
        except Exception:
            return 1.0
        prop1 = composition_dict[elem1]
        prop2 = composition_dict[elem2]
        n1_rel = prop1 / m1
        n2_rel = prop2 / m2
        best_balance = float('inf')
        for v1 in valences1:
            for v2 in valences2:
                total_charge = v1 * n1_rel + v2 * n2_rel
                charge_imbalance = abs(total_charge)
                if charge_imbalance < best_balance:
                    best_balance = charge_imbalance
        if best_balance > 0.5:
            return 0.3
        elif best_balance > 0.3:
            return 0.5
        elif best_balance > 0.1:
            return 0.8
    return 1.0


def compute_stability_score(composition_dict: Dict) -> float:
    """
    Compute chemical stability score (identical logic to original KIGA).
    Returns a float in [0, 1].
    """
    elements = list(composition_dict.keys())
    score = 0.5

    # 1. Stoichiometry penalty
    stoich_penalty = estimate_stoichiometry_penalty(composition_dict)
    score *= stoich_penalty

    # 2. Electronegativity check
    electronegativities = []
    for elem in elements:
        try:
            electronegativities.append(Element(elem).X)
        except Exception:
            electronegativities.append(2.0)

    if len(electronegativities) > 1:
        electroneg_diff = max(electronegativities) - min(electronegativities)
        if 0.5 < electroneg_diff < 2.0:
            score += 0.2
        elif electroneg_diff > 3.0:
            score -= 0.1

    # 3. Known stable pairs (empirical rules from original)
    stable_pairs = {
        ('Li', 'O'), ('Cu', 'O'), ('Mn', 'O'), ('Cu', 'S'), ('In', 'S'),
        ('Ba', 'O'), ('Cu', 'Se'), ('Ga', 'Se'), ('In', 'Se'), ('B', 'O'),
        ('O', 'P'), ('O', 'V'), ('O', 'Se'), ('Bi', 'O'), ('Ni', 'O'),
        ('S', 'Sb'), ('Ag', 'S'), ('O', 'Ti'), ('Fe', 'O'), ('Ga', 'S'),
        ('Se', 'Sn'), ('Pb', 'Se'), ('S', 'Sn'), ('K', 'S'), ('Co', 'O'),
        ('Pb', 'Te'), ('Cs', 'Se'), ('In', 'Te'), ('Sb', 'Te'), ('Ge', 'Se'),
        ('Ba', 'Se'), ('O', 'Te'), ('P', 'S'), ('As', 'Ga'), ('Bi', 'Se'),
        ('Nb', 'O'), ('Ge', 'S'), ('Li', 'Mn'), ('K', 'Se'), ('Ga', 'Te'),
        ('Cd', 'Se'), ('Ga', 'P'), ('O', 'Sr'), ('Se', 'Zn'), ('Sb', 'Se'),
        ('Fe', 'Li'), ('Ba', 'S'), ('Na', 'O'), ('Li', 'Ni'), ('H', 'O'),
        ('La', 'O'), ('K', 'O'), ('P', 'Se'), ('As', 'In'), ('Cd', 'Te'),
        ('O', 'Zn'), ('O', 'Pb'), ('O', 'Sb'), ('Mo', 'O'), ('Pb', 'Sn'),
        ('Hg', 'Se'), ('Bi', 'S'), ('Bi', 'Te'), ('As', 'S'), ('Ga', 'Sb'),
        ('Se', 'Te'), ('Ba', 'Ga'), ('Cs', 'S'), ('O', 'Rb'), ('Ca', 'O'),
        ('Cl', 'O'), ('O', 'Ta'), ('F', 'O'), ('O', 'W'), ('Hg', 'S'),
        ('Rb', 'S'), ('S', 'Se'), ('Cr', 'O'), ('As', 'Se'), ('Ga', 'Zn'),
        ('In', 'P'), ('Pb', 'S'), ('Co', 'Li'), ('S', 'Zn'), ('S', 'Tl'),
        ('La', 'S'), ('Na', 'S'), ('Ce', 'O'), ('Ge', 'Pb'), ('Cu', 'In'),
        ('Ba', 'In'), ('I', 'O'), ('As', 'P'), ('O', 'S'), ('N', 'O'),
        ('Cd', 'O'), ('Sn', 'Te'), ('Ga', 'In'), ('Cu', 'Te'), ('O', 'Y'),
    }
    for elem1, elem2 in itertools.combinations(elements, 2):
        if (elem1, elem2) in stable_pairs or (elem2, elem1) in stable_pairs:
            score += 0.1

    return float(max(0.0, min(1.0, score)))


# Global set of known stable pairs with alphabetically sorted tuples
STABLE_PAIRS_SET = frozenset(
    tuple(sorted(pair)) for pair in {
        ('Li', 'O'), ('Cu', 'O'), ('Mn', 'O'), ('Cu', 'S'), ('In', 'S'),
        ('Ba', 'O'), ('Cu', 'Se'), ('Ga', 'Se'), ('In', 'Se'), ('B', 'O'),
        ('O', 'P'), ('O', 'V'), ('O', 'Se'), ('Bi', 'O'), ('Ni', 'O'),
        ('S', 'Sb'), ('Ag', 'S'), ('O', 'Ti'), ('Fe', 'O'), ('Ga', 'S'),
        ('Se', 'Sn'), ('Pb', 'Se'), ('S', 'Sn'), ('K', 'S'), ('Co', 'O'),
        ('Pb', 'Te'), ('Cs', 'Se'), ('In', 'Te'), ('Sb', 'Te'), ('Ge', 'Se'),
        ('Ba', 'Se'), ('O', 'Te'), ('P', 'S'), ('As', 'Ga'), ('Bi', 'Se'),
        ('Nb', 'O'), ('Ge', 'S'), ('Li', 'Mn'), ('K', 'Se'), ('Ga', 'Te'),
        ('Cd', 'Se'), ('Ga', 'P'), ('O', 'Sr'), ('Se', 'Zn'), ('Sb', 'Se'),
        ('Fe', 'Li'), ('Ba', 'S'), ('Na', 'O'), ('Li', 'Ni'), ('H', 'O'),
        ('La', 'O'), ('K', 'O'), ('P', 'Se'), ('As', 'In'), ('Cd', 'Te'),
        ('O', 'Zn'), ('O', 'Pb'), ('O', 'Sb'), ('Mo', 'O'), ('Pb', 'Sn'),
        ('Hg', 'Se'), ('Bi', 'S'), ('Bi', 'Te'), ('As', 'S'), ('Ga', 'Sb'),
        ('Se', 'Te'), ('Ba', 'Ga'), ('Cs', 'S'), ('O', 'Rb'), ('Ca', 'O'),
        ('Cl', 'O'), ('O', 'Ta'), ('F', 'O'), ('O', 'W'), ('Hg', 'S'),
        ('Rb', 'S'), ('S', 'Se'), ('Cr', 'O'), ('As', 'Se'), ('Ga', 'Zn'),
        ('In', 'P'), ('Pb', 'S'), ('Co', 'Li'), ('S', 'Zn'), ('S', 'Tl'),
        ('La', 'S'), ('Na', 'S'), ('Ce', 'O'), ('Ge', 'Pb'), ('Cu', 'In'),
        ('Ba', 'In'), ('I', 'O'), ('As', 'P'), ('O', 'S'), ('N', 'O'),
        ('Cd', 'O'), ('Sn', 'Te'), ('Ga', 'In'), ('Cu', 'Te'), ('O', 'Y'),
    }
)


def compute_composite_plausibility(composition_dict: Dict) -> float:
    """
    Composite plausibility score for fair ablation comparison.

    Corrects the bias in the raw stability score where compositions with many
    elements automatically accumulate more pair bonuses (C(n,2) grows
    quadratically). The pair bonus is averaged per pair, and the final score
    is multiplied by a complexity penalty so that simpler compositions are
    preferred, matching real-world synthesis difficulty.

    Returns a float in [0, 1].
    """
    elements = list(composition_dict.keys())
    n_elements = len(elements)

    # 1. Stoichiometry penalty
    score = 0.5
    stoich_penalty = estimate_stoichiometry_penalty(composition_dict)
    score *= stoich_penalty

    # 2. Electronegativity adjustment
    electronegativities = []
    for elem in elements:
        try:
            electronegativities.append(Element(elem).X)
        except Exception:
            electronegativities.append(2.0)

    if len(electronegativities) > 1:
        electroneg_diff = max(electronegativities) - min(electronegativities)
        if 0.5 < electroneg_diff < 2.0:
            score += 0.2
        elif electroneg_diff > 3.0:
            score -= 0.1

    # 3. Normalized pair bonus (average per pair, not sum)
    n_pairs = 0
    pair_bonus = 0.0
    for e1, e2 in itertools.combinations(elements, 2):
        n_pairs += 1
        if tuple(sorted([e1, e2])) in STABLE_PAIRS_SET:
            pair_bonus += 0.1
    avg_pair_bonus = pair_bonus / max(1, n_pairs)
    score += avg_pair_bonus

    # Clip base score
    score = max(0.0, min(1.0, score))

    # 4. Complexity penalty (fewer elements = higher penalty value = preferred)
    complexity_penalty = max(0.0, 1.0 - 0.1 * (n_elements - 2))

    return float(score * complexity_penalty)


def filter_elements_by_material_type(elements: List[str], target_gap: float) -> List[str]:
    """Identical filtering logic as original KIGA."""
    if target_gap == 0:
        material_type = 'metal'
    elif target_gap < 0.1:
        material_type = 'semimetal'
    elif target_gap < 1.5:
        material_type = 'narrow_gap'
    elif target_gap < 3.0:
        material_type = 'semiconductor'
    else:
        material_type = 'insulator'

    element_categories = {
        'metals': ['Li','Na','K','Rb','Cs','Be','Mg','Ca','Sr','Ba','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
                   'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg',
                   'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Th','U',
                   'Al','Ga','In','Tl','Sn','Pb','Bi'],
        'semiconductors': ['Si','Ge','Ga','As','In','P','Sb','Se','Te','B','C','S'],
        'insulators': ['F','Cl','Br','I','O','N','H','Be','Mg','Ca','Sr','Ba','Al'],
        'transition_metals': ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
                              'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho',
                              'Er','Tm','Yb','Lu','Th','U']
    }

    if material_type in ('metal', 'semimetal'):
        selected = element_categories['metals'] + element_categories['transition_metals']
    elif material_type == 'narrow_gap':
        selected = element_categories['semiconductors'] + element_categories['transition_metals']
    elif material_type == 'semiconductor':
        selected = element_categories['semiconductors'] + ['O','S','Se','Te']
    else:
        selected = element_categories['insulators'] + element_categories['metals'][:5]

    filtered = list({elem for elem in selected if elem in elements})
    if len(filtered) < 10:
        additional = [e for e in elements if e not in filtered]
        filtered.extend(additional[:20])
    return list({elem for elem in filtered})[:30]

# =============================================================================
# Model Wrapper
# =============================================================================

class BandGapPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        self.magpie = MagpieCalculator()
        print(f"Loading model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = keras.models.load_model(model_path, compile=False)
        print("Model loaded.")
        print(f"Loading scaler from {scaler_path}...")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        print("Scaler loaded.")
        self.cache = {}
        self.call_count = 0

    def _hash_composition(self, composition_dict: Dict) -> str:
        sorted_items = sorted(composition_dict.items(), key=lambda x: x[0])
        s = "_".join([f"{e}_{p:.6f}" for e, p in sorted_items])
        return hashlib.md5(s.encode()).hexdigest()[:16]

    def predict(self, composition_dict: Dict) -> float:
        h = self._hash_composition(composition_dict)
        if h in self.cache:
            return self.cache[h]
        # Build pymatgen Composition directly from fractions (pymatgen normalizes internally)
        comp_dict = {elem: float(prop) for elem, prop in composition_dict.items() if prop > 0.0}
        if not comp_dict:
            return 0.0
        try:
            comp = Composition(comp_dict)
        except Exception:
            return 0.0
        desc = self.magpie.calculate(comp)
        # Scale
        if hasattr(self.scaler, 'transform'):
            desc = self.scaler.transform(desc.reshape(1, -1)).flatten()
        pred = self.model.predict(desc.reshape(1, -1), verbose=0).flatten()[0]
        pred = float(max(0.0, pred))
        self.cache[h] = pred
        self.call_count += 1
        return pred

# =============================================================================
# Genetic Algorithm
# =============================================================================

def generate_composition_proportions(n_elements: int, min_frac: float = 0.05) -> np.ndarray:
    """Generate random normalized proportions with a minimum fraction for each element."""
    base = np.full(n_elements, min_frac)
    remainder = 1.0 - base.sum()
    if remainder < 0:
        return np.ones(n_elements) / n_elements
    extras = np.random.random(n_elements)
    extras = extras / extras.sum() * remainder
    props = base + extras
    return props / props.sum()


class CompositionGeneticAlgorithm:
    def __init__(self, population_size: int = 150, generations: int = 80,
                 mutation_rate: float = 0.25, crossover_rate: float = 0.7,
                 elite_size: int = 15, tournament_size: int = 4,
                 early_stop_patience: int = 15):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.early_stop_patience = early_stop_patience
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []

    def initialize_population(self, elements: List[str], max_elements: int = 4, min_elements: int = 2) -> List[Dict]:
        population = []
        for _ in range(self.population_size):
            n_elements = random.randint(min_elements, max_elements)
            selected_elements = random.sample(elements, n_elements)
            proportions = generate_composition_proportions(n_elements, min_frac=0.05)
            individual = {
                'elements': {elem: float(prop) for elem, prop in zip(selected_elements, proportions)},
                'fitness': 0.0,
                'predicted_gap': 0.0,
                'error': 0.0,
                'age': 0,
            }
            population.append(individual)
        return population

    def tournament_selection(self, population: List[Dict]) -> List[Dict]:
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(population, self.tournament_size)
            winner = max(tournament, key=lambda x: x['fitness'])
            selected.append(self._copy_individual(winner))
        return selected

    @staticmethod
    def _copy_individual(ind: Dict) -> Dict:
        return {
            'elements': ind['elements'].copy(),
            'fitness': ind['fitness'],
            'predicted_gap': ind['predicted_gap'],
            'error': ind['error'],
            'age': ind['age'],
        }

    @staticmethod
    def arithmetic_crossover(parent1: Dict, parent2: Dict) -> Dict:
        child_elements = {}
        all_elements = set(parent1['elements'].keys()) | set(parent2['elements'].keys())
        for elem in all_elements:
            p1 = parent1['elements'].get(elem, 0.0)
            p2 = parent2['elements'].get(elem, 0.0)
            alpha = random.random()
            cp = alpha * p1 + (1 - alpha) * p2
            if cp > 0.001:
                child_elements[elem] = cp
        total = sum(child_elements.values())
        if total > 0:
            child_elements = {k: v / total for k, v in child_elements.items()}
        return {
            'elements': child_elements,
            'fitness': 0.0,
            'predicted_gap': 0.0,
            'error': 0.0,
            'age': max(parent1['age'], parent2['age']) + 1,
        }

    @staticmethod
    def geometric_crossover(parent1: Dict, parent2: Dict) -> Dict:
        child_elements = {}
        all_elements = set(parent1['elements'].keys()) | set(parent2['elements'].keys())
        for elem in all_elements:
            p1 = parent1['elements'].get(elem, 0.001)
            p2 = parent2['elements'].get(elem, 0.001)
            cp = np.sqrt(p1 * p2)
            if cp > 0.001:
                child_elements[elem] = cp
        total = sum(child_elements.values())
        if total > 0:
            child_elements = {k: v / total for k, v in child_elements.items()}
        return {
            'elements': child_elements,
            'fitness': 0.0,
            'predicted_gap': 0.0,
            'error': 0.0,
            'age': max(parent1['age'], parent2['age']) + 1,
        }

    def mutate_composition(self, individual: Dict, elements: List[str], mutation_strength: float = 0.3) -> Dict:
        mutated = self._copy_individual(individual)
        elements_dict = mutated['elements'].copy()
        mutation_type = random.choices(
            ['adjust', 'add', 'remove', 'replace'],
            weights=[0.6, 0.2, 0.1, 0.1]
        )[0]

        if mutation_type == 'adjust':
            if elements_dict:
                elem_to_adjust = random.choice(list(elements_dict.keys()))
                adjustment = random.uniform(-mutation_strength, mutation_strength)
                new_prop = max(0.01, elements_dict[elem_to_adjust] * (1 + adjustment))
                delta = new_prop - elements_dict[elem_to_adjust]
                elements_dict[elem_to_adjust] = new_prop
                other_elements = [e for e in elements_dict.keys() if e != elem_to_adjust]
                if other_elements and delta != 0:
                    for elem in other_elements:
                        elements_dict[elem] = max(0.01, elements_dict[elem] * (1 - delta))

        elif mutation_type == 'add' and len(elements_dict) < 6:
            available = [e for e in elements if e not in elements_dict]
            if available:
                new_element = random.choice(available)
                reduction = random.uniform(0.05, 0.2) / len(elements_dict)
                for elem in elements_dict:
                    elements_dict[elem] *= (1 - reduction)
                elements_dict[new_element] = reduction * len(elements_dict)

        elif mutation_type == 'remove' and len(elements_dict) > 2:
            elem_to_remove = random.choice(list(elements_dict.keys()))
            removed_prop = elements_dict.pop(elem_to_remove)
            if elements_dict:
                for elem in elements_dict:
                    elements_dict[elem] /= (1 - removed_prop)

        elif mutation_type == 'replace' and len(elements_dict) > 1:
            elem_to_replace = random.choice(list(elements_dict.keys()))
            available = [e for e in elements if e not in elements_dict]
            if available:
                new_element = random.choice(available)
                prop = elements_dict.pop(elem_to_replace)
                elements_dict[new_element] = prop

        total = sum(elements_dict.values())
        if total > 0:
            mutated['elements'] = {k: max(0.0, v / total) for k, v in elements_dict.items()}
        mutated['age'] = individual['age'] + 1
        return mutated

    @staticmethod
    def calculate_population_diversity(population: List[Dict]) -> float:
        if len(population) < 2:
            return 0.0
        all_elements = set()
        for ind in population:
            all_elements.update(ind['elements'].keys())
        element_list = sorted(list(all_elements))
        n_elements = len(element_list)
        if n_elements == 0:
            return 0.0
        comp_matrix = np.zeros((len(population), n_elements))
        for i, ind in enumerate(population):
            for elem, prop in ind['elements'].items():
                if elem in element_list:
                    j = element_list.index(elem)
                    comp_matrix[i, j] = prop
        diversity = 0.0
        count = 0
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = np.linalg.norm(comp_matrix[i] - comp_matrix[j])
                diversity += distance
                count += 1
        return diversity / count if count > 0 else 0.0

    def evolve(self, population: List[Dict], elements: List[str], fitness_function) -> Tuple[List[Dict], Dict]:
        # Evaluate fitness
        for ind in population:
            ind['fitness'], ind['predicted_gap'], ind['error'] = fitness_function(ind)
        population.sort(key=lambda x: x['fitness'], reverse=True)
        elite = [self._copy_individual(ind) for ind in population[:self.elite_size]]
        parents = self.tournament_selection(population[self.elite_size:])
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents) and random.random() < self.crossover_rate:
                p1, p2 = parents[i], parents[i + 1]
                child = self.arithmetic_crossover(p1, p2) if random.random() < 0.5 else self.geometric_crossover(p1, p2)
                offspring.append(child)
        mutated_offspring = []
        for child in offspring:
            if random.random() < self.mutation_rate:
                mutated_offspring.append(self.mutate_composition(child, elements))
            else:
                mutated_offspring.append(child)
        new_population = elite.copy()
        new_population.extend(mutated_offspring)
        while len(new_population) < self.population_size:
            n_elements = random.randint(2, 4)
            selected_elements = random.sample(elements, n_elements)
            proportions = generate_composition_proportions(n_elements, min_frac=0.05)
            new_individual = {
                'elements': {elem: float(prop) for elem, prop in zip(selected_elements, proportions)},
                'fitness': 0.0,
                'predicted_gap': 0.0,
                'error': 0.0,
                'age': 0,
            }
            new_population.append(new_individual)
        # Evaluate only new individuals
        for ind in new_population:
            if ind['fitness'] == 0.0 and ind['predicted_gap'] == 0.0:
                ind['fitness'], ind['predicted_gap'], ind['error'] = fitness_function(ind)
        fitness_values = [ind['fitness'] for ind in new_population]
        avg_fitness = float(np.mean(fitness_values))
        best_fitness = float(max(fitness_values))
        diversity = self.calculate_population_diversity(new_population)
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        self.diversity_history.append(diversity)
        stats = {
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'diversity': diversity,
        }
        return new_population, stats

# =============================================================================
# Fitness Builders
# =============================================================================

def build_fitness_function(predictor: BandGapPredictor, target_gap: float, mode: str = 'baseline', tolerance: float = 0.1, use_complexity_penalty: bool = True):
    """
    mode='baseline': fitness = base_fitness(prediction, target)
    mode='kiga':     fitness = base_fitness * complexity_penalty * (1 + stability_bonus)

    use_complexity_penalty=False allows isolating the pure chemical knowledge effect
    from the parsimony bias (useful for ablation interpretation).
    """
    def evaluate(individual: Dict):
        predicted_gap = predictor.predict(individual['elements'])
        error = abs(predicted_gap - target_gap)
        if error <= tolerance:
            base_fitness = 10.0 * (1.0 - error / tolerance)
        else:
            base_fitness = 1.0 / (1.0 + error)

        if mode == 'baseline':
            fitness = base_fitness
        else:
            n_elements = len(individual['elements'])
            if use_complexity_penalty:
                complexity_penalty = max(0.0, 1.0 - 0.1 * (n_elements - 2))
            else:
                complexity_penalty = 1.0  # disable parsimony bias for pure chemistry ablation
            stability_bonus = compute_stability_score(individual['elements'])
            fitness = base_fitness * complexity_penalty * (1.0 + stability_bonus)
        return float(fitness), float(predicted_gap), float(error)
    return evaluate

# =============================================================================
# Experiment Runner
# =============================================================================

def run_ga_experiment(target_gap: float, seed: int, predictor: BandGapPredictor, mode: str = 'baseline', use_complexity_penalty: bool = True) -> Dict:
    """
    Runs one GA execution and returns final population metrics.
    """
    # Seed everything deterministically
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    # Clear caches to ensure isolated, reproducible predictions per run
    predictor.cache.clear()
    predictor.magpie.composition_cache.clear()

    # Define element pool exactly as original KIGA does
    default_elements = [
        'Li','Na','K','Rb','Cs','Be','Mg','Ca','Sr','Ba','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
        'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg',
        'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Th','U',
        'Al','Ga','In','Tl','Sn','Pb','Bi','B','C','Si','Ge','P','As','Sb',
        'H','N','O','F','S','Cl','Br','I','Se','Te','Xe'
    ]
    # Validate with pymatgen
    valid_elements = []
    for elem in default_elements:
        try:
            Element(elem)
            valid_elements.append(elem)
        except Exception:
            pass

    filtered_elements = filter_elements_by_material_type(valid_elements, target_gap)

    ga = CompositionGeneticAlgorithm(
        population_size=150,
        generations=80,
        mutation_rate=0.25,
        crossover_rate=0.7,
        elite_size=15,
        tournament_size=4,
        early_stop_patience=15,
    )

    population = ga.initialize_population(filtered_elements, max_elements=4, min_elements=2)
    fitness_func = build_fitness_function(predictor, target_gap, mode=mode, use_complexity_penalty=use_complexity_penalty)

    best_fitness_so_far = -np.inf
    patience_counter = 0

    for gen in range(ga.generations):
        population, stats = ga.evolve(population, filtered_elements, fitness_func)
        current_best = stats['best_fitness']
        if current_best > best_fitness_so_far + 1e-9:
            best_fitness_so_far = current_best
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= ga.early_stop_patience:
            break

    # Final evaluation across population for stability, plausibility, and MAE
    final_preds = []
    final_stabilities = []   # raw score used inside KIGA fitness (kept for reference)
    final_plausibilities = []  # composite score for fair ablation comparison
    final_n_elements = []
    for ind in population:
        pred = predictor.predict(ind['elements'])
        final_preds.append(pred)
        final_stabilities.append(compute_stability_score(ind['elements']))
        final_plausibilities.append(compute_composite_plausibility(ind['elements']))
        final_n_elements.append(len(ind['elements']))
    final_maes = [abs(p - target_gap) for p in final_preds]

    mean_stability = float(np.mean(final_stabilities))
    mean_plausibility = float(np.mean(final_plausibilities))
    mean_mae = float(np.mean(final_maes))
    best_mae = float(min(final_maes))
    best_fitness = max(ind['fitness'] for ind in population)
    mean_n_elements = float(np.mean(final_n_elements))

    return {
        'target': target_gap,
        'seed': seed,
        'mode': mode,
        'mean_stability': mean_stability,
        'mean_plausibility': mean_plausibility,
        'mean_mae': mean_mae,
        'best_mae': best_mae,
        'best_fitness': best_fitness,
        'mean_n_elements': mean_n_elements,
        'final_population_size': len(population),
    }

# =============================================================================
# Main Ablation Experiment
# =============================================================================

def main_ablation(model_path: str = '/content/bandgap_nn_model.keras',
                  scaler_path: str = '/content/scaler.pkl',
                  use_complexity_penalty: bool = True):
    predictor = BandGapPredictor(model_path, scaler_path)

    targets = [0.0, 0.05, 0.8, 1.8, 2.5, 4.0]
    seeds = [0, 1, 2, 3, 4]
    modes = ['baseline', 'kiga']

    records = []

    # CRITICAL: loop seed OUTSIDE mode so both algorithms start from the same RNG state for each seed.
    for target in targets:
        print(f"\n{'='*60}")
        print(f"Target Band Gap: {target} eV")
        print(f"{'='*60}")
        for seed in seeds:
            for mode in modes:
                label = f"{mode.upper()}_cpx" if (mode == 'kiga' and use_complexity_penalty) else mode.upper()
                print(f"  Running {label} | seed={seed} | target={target} eV ...", end=" ")
                result = run_ga_experiment(target, seed, predictor, mode=mode, use_complexity_penalty=use_complexity_penalty)
                records.append(result)
                print(f"stab={result['mean_stability']:.3f}, plaus={result['mean_plausibility']:.3f}, mae={result['mean_mae']:.3f}, n_el={result['mean_n_elements']:.2f}")

    df = pd.DataFrame(records)
    return df

# =============================================================================
# Statistical Analysis & Plots
# =============================================================================

def analyze_and_plot(df: pd.DataFrame, output_dir: str = "/content/ablation_results", suffix: str = ""):
    os.makedirs(output_dir, exist_ok=True)

    # Aggregate by target and mode
    summary = []
    for target in sorted(df['target'].unique()):
        for mode in df['mode'].unique():
            subset = df[(df['target'] == target) & (df['mode'] == mode)]
            if len(subset) == 0:
                continue
            summary.append({
                'target': target,
                'mode': mode,
                'mean_plausibility': subset['mean_plausibility'].mean(),
                'std_plausibility': subset['mean_plausibility'].std(ddof=1),
                'mean_stability': subset['mean_stability'].mean(),
                'std_stability': subset['mean_stability'].std(ddof=1),
                'mean_mae': subset['mean_mae'].mean(),
                'std_mae': subset['mean_mae'].std(ddof=1),
                'mean_n_elements': subset['mean_n_elements'].mean(),
                'std_n_elements': subset['mean_n_elements'].std(ddof=1),
                'n_runs': len(subset),
            })
    df_summary = pd.DataFrame(summary)

    # Print table
    print("\n" + "="*90)
    print("ABLATION STUDY SUMMARY (mean +/- std over 5 seeds)")
    print("="*90)
    for target in sorted(df['target'].unique()):
        print(f"\nTarget: {target} eV")
        for mode in df['mode'].unique():
            row = df_summary[(df_summary['target']==target) & (df_summary['mode']==mode)].iloc[0]
            print(f"  {mode.upper():12s}: Plausibility = {row['mean_plausibility']:.4f} +/- {row['std_plausibility']:.4f} | "
                  f"MAE = {row['mean_mae']:.4f} +/- {row['std_mae']:.4f} | "
                  f"N_elements = {row['mean_n_elements']:.2f} +/- {row['std_n_elements']:.2f}")

    # Independent t-tests per target
    print("\n" + "="*90)
    print("INDEPENDENT t-TEST RESULTS (Baseline vs KIGA)")
    print("="*90)
    for target in sorted(df['target'].unique()):
        base_plaus = df[(df['target']==target) & (df['mode']=='baseline')]['mean_plausibility'].values
        kiga_plaus = df[(df['target']==target) & (df['mode']=='kiga')]['mean_plausibility'].values
        base_mae = df[(df['target']==target) & (df['mode']=='baseline')]['mean_mae'].values
        kiga_mae = df[(df['target']==target) & (df['mode']=='kiga')]['mean_mae'].values
        base_nel = df[(df['target']==target) & (df['mode']=='baseline')]['mean_n_elements'].values
        kiga_nel = df[(df['target']==target) & (df['mode']=='kiga')]['mean_n_elements'].values

        if len(kiga_plaus) == 0:
            continue

        t_plaus, p_plaus = stats.ttest_ind(base_plaus, kiga_plaus, equal_var=False)
        t_mae, p_mae = stats.ttest_ind(base_mae, kiga_mae, equal_var=False)
        t_nel, p_nel = stats.ttest_ind(base_nel, kiga_nel, equal_var=False)

        plaus_improv = (kiga_plaus.mean() - base_plaus.mean()) / max(1e-9, base_plaus.mean()) * 100
        mae_improv = (base_mae.mean() - kiga_mae.mean()) / max(1e-9, base_mae.mean()) * 100

        print(f"\nTarget {target} eV:")
        print(f"  Plausibility: t={t_plaus:.3f}, p={p_plaus:.4f}, improvement={plaus_improv:+.1f}%")
        print(f"  MAE:          t={t_mae:.3f}, p={p_mae:.4f}, improvement={mae_improv:+.1f}%")
        print(f"  N_elements:   t={t_nel:.3f}, p={p_nel:.4f}, delta={kiga_nel.mean() - base_nel.mean():+.2f}")

    # Plotting
    targets_sorted = sorted(df['target'].unique())
    x = np.arange(len(targets_sorted))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Mean Composite Plausibility
    for mode, color in [('baseline', 'steelblue'), ('kiga', 'coral')]:
        means = [df_summary[(df_summary['target']==t) & (df_summary['mode']==mode)]['mean_plausibility'].values[0] for t in targets_sorted]
        stds = [df_summary[(df_summary['target']==t) & (df_summary['mode']==mode)]['std_plausibility'].values[0] for t in targets_sorted]
        offset = -width/2 if mode == 'baseline' else width/2
        axes[0].bar(x + offset, means, width, yerr=stds, label=mode.upper(), color=color, capsize=4)
    axes[0].set_ylabel('Mean Composite Plausibility')
    axes[0].set_xlabel('Target Band Gap (eV)')
    axes[0].set_title('Mean Population Plausibility: Baseline vs KIGA')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{t}' for t in targets_sorted])
    axes[0].legend()
    axes[0].set_ylim(bottom=0)

    # 2. Mean MAE
    for mode, color in [('baseline', 'steelblue'), ('kiga', 'coral')]:
        means = [df_summary[(df_summary['target']==t) & (df_summary['mode']==mode)]['mean_mae'].values[0] for t in targets_sorted]
        stds = [df_summary[(df_summary['target']==t) & (df_summary['mode']==mode)]['std_mae'].values[0] for t in targets_sorted]
        offset = -width/2 if mode == 'baseline' else width/2
        axes[1].bar(x + offset, means, width, yerr=stds, label=mode.upper(), color=color, capsize=4)
    axes[1].set_ylabel('Mean Absolute Error (eV)')
    axes[1].set_xlabel('Target Band Gap (eV)')
    axes[1].set_title('Mean Population MAE')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{t}' for t in targets_sorted])
    axes[1].legend()
    axes[1].set_ylim(bottom=0)

    # 3. Mean N_elements
    for mode, color in [('baseline', 'steelblue'), ('kiga', 'coral')]:
        means = [df_summary[(df_summary['target']==t) & (df_summary['mode']==mode)]['mean_n_elements'].values[0] for t in targets_sorted]
        stds = [df_summary[(df_summary['target']==t) & (df_summary['mode']==mode)]['std_n_elements'].values[0] for t in targets_sorted]
        offset = -width/2 if mode == 'baseline' else width/2
        axes[2].bar(x + offset, means, width, yerr=stds, label=mode.upper(), color=color, capsize=4)
    axes[2].set_ylabel('Mean Number of Elements')
    axes[2].set_xlabel('Target Band Gap (eV)')
    axes[2].set_title('Mean Population Complexity')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f'{t}' for t in targets_sorted])
    axes[2].legend()
    axes[2].set_ylim(bottom=0)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'ablation_comparison{suffix}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved to {plot_path}")

    # Box plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(data=df, x='target', y='mean_plausibility', hue='mode', ax=axes[0], palette=['steelblue', 'coral'])
    axes[0].set_title('Plausibility Distribution by Target')
    axes[0].set_xlabel('Target Band Gap (eV)')
    axes[0].set_ylabel('Mean Composite Plausibility')
    sns.boxplot(data=df, x='target', y='mean_mae', hue='mode', ax=axes[1], palette=['steelblue', 'coral'])
    axes[1].set_title('MAE Distribution by Target')
    axes[1].set_xlabel('Target Band Gap (eV)')
    axes[1].set_ylabel('Mean MAE (eV)')
    sns.boxplot(data=df, x='target', y='mean_n_elements', hue='mode', ax=axes[2], palette=['steelblue', 'coral'])
    axes[2].set_title('Complexity Distribution by Target')
    axes[2].set_xlabel('Target Band Gap (eV)')
    axes[2].set_ylabel('Mean N Elements')
    plt.suptitle('')
    plt.tight_layout()
    box_path = os.path.join(output_dir, f'ablation_distributions{suffix}.png')
    plt.savefig(box_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Box plot saved to {box_path}")

    # Save CSVs
    csv_path = os.path.join(output_dir, f'ablation_results{suffix}.csv')
    df.to_csv(csv_path, index=False)
    summary_path = os.path.join(output_dir, f'ablation_summary{suffix}.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"CSVs saved to {csv_path} and {summary_path}")

    return df_summary

# =============================================================================
# Execute
# =============================================================================
if __name__ == '__main__':
    # Adjust paths if your files are elsewhere
    MODEL_PATH = '/content/bandgap_nn_model.keras'
    SCALER_PATH = '/content/scaler.pkl'

    # ============================================================
    # EXPERIMENT A: KIGA with original complexity penalty (parsimony bias)
    # ============================================================
    print("\n" + "="*80)
    print("EXPERIMENT A: KIGA with complexity penalty (original design)")
    print("="*80)
    df_results_a = main_ablation(model_path=MODEL_PATH, scaler_path=SCALER_PATH, use_complexity_penalty=True)
    df_summary_a = analyze_and_plot(df_results_a, suffix="_with_cpx")

    # ============================================================
    # EXPERIMENT B: KIGA WITHOUT complexity penalty (pure chemistry ablation)
    # ============================================================
    # This isolates the effect of the stability bonus from the parsimony bias.
    print("\n" + "="*80)
    print("EXPERIMENT B: KIGA without complexity penalty (pure chemical knowledge)")
    print("="*80)
    df_results_b = main_ablation(model_path=MODEL_PATH, scaler_path=SCALER_PATH, use_complexity_penalty=False)
    df_summary_b = analyze_and_plot(df_results_b, suffix="_no_cpx")

    print("\nDone.")
