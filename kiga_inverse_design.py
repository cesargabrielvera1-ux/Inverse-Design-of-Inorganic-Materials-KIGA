"""
KIGA - Knowledge-Informed Genetic Algorithm for Inverse Band Gap Design
===========================================================================
Complete inverse design system for materials based on target band gap.
Combines a neural network model, genetic algorithm, and chemical validation.

Originally designed for Google Colab with multiple sequential cells.
Merged into a single executable script.

NOTE: File paths are set to '/content/' for Google Colab compatibility.
Please update these paths according to your local environment.
"""

# =============================================================================
# DEPENDENCIES
# =============================================================================
# In a local environment, install the following packages:
#   pip install tensorflow scikit-learn pymatgen matminer optuna pandas numpy matplotlib seaborn joblib tqdm scipy

import numpy as np
from math import gcd
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import json
import warnings
import os
import random
import hashlib
import itertools
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter

from pymatgen.core import Composition, Element
from matminer.featurizers.composition import ElementProperty
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

print("Libraries imported successfully")
print(f"TensorFlow: {tf.__version__}")
print(f"pymatgen: {Composition('H2O').formula}")  # Simple verification

# =============================================================================
# REPRODUCIBILITY CONFIGURATION
# =============================================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# TensorFlow performance configuration
tf.config.optimizer.set_jit(True)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

print("Reproducibility configuration completed")

# =============================================================================
# MAGPIE CALCULATOR
# =============================================================================

class MagpieCalculator:
    """
    Complete calculator for the 132 Magpie descriptors for chemical compositions.
    Based on matminer and pymatgen for precise calculations.
    """

    def __init__(self):
        """Initializes the MagpieData featurizer from matminer."""
        # FIX: MagpieData() is no longer directly importable in matminer 0.10.0.
        # It is accessed via ElementProperty.from_preset("magpie").
        self.featurizer = ElementProperty.from_preset("magpie")
        self.descriptor_names = self.featurizer.feature_labels()
        self.element_data_cache = {}  # Cache for individual element calculations
        self.composition_cache = {}   # Cache for already calculated compositions
        self._initialize_element_properties()

        print(f"Magpie calculator initialized with {len(self.descriptor_names)} descriptors")
        print(f"   First 5 descriptors: {self.descriptor_names[:5]}...")

    def _initialize_element_properties(self):
        """Preloads element properties for optimization."""
        print("Loading element properties...")
        self.element_properties = {}

        # For each element in the periodic table
        for atomic_number in range(1, 95):  # Up to Plutonium
            try:
                element = Element.from_Z(atomic_number)
                symbol = element.symbol

                # Calculate basic properties for cache
                self.element_properties[symbol] = {
                    'atomic_number': atomic_number,
                    'atomic_mass': element.atomic_mass,
                    'row': element.row,
                    'group': element.group,
                    'electronegativity': element.X,
                    'ionic_radii': element.ionic_radii,
                    'atomic_radius': element.atomic_radius,
                    'mendeleev_no': element.mendeleev_no,
                }
            except Exception:
                continue  # Skip problematic elements

        print(f"   Properties loaded for {len(self.element_properties)} elements")

    def calculate(self, composition: Composition, use_cache: bool = True) -> np.ndarray:
        """
        Calculates the 132 Magpie descriptors for a chemical composition.

        Args:
            composition: pymatgen Composition object
            use_cache: Use cache to speed up repeated calculations

        Returns:
            Numpy array with the 132 descriptors
        """
        # Generate unique key for the composition
        formula = composition.reduced_formula
        formula_key = f"{formula}_{hashlib.md5(str(composition).encode()).hexdigest()[:8]}"

        # Check cache
        if use_cache and formula_key in self.composition_cache:
            return self.composition_cache[formula_key]

        try:
            # Calculate descriptors using matminer
            descriptors = self.featurizer.featurize(composition)

            # Verify that we have 132 descriptors
            if len(descriptors) != 132:
                print(f"Warning: {formula} has {len(descriptors)} descriptors, padding...")
                descriptors = self._pad_descriptors(descriptors)

            # Convert to numpy array and handle NaN
            descriptors = np.array(descriptors, dtype=np.float32)
            descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)

            # Save to cache
            if use_cache:
                self.composition_cache[formula_key] = descriptors

            return descriptors

        except Exception as e:
            print(f"Error calculating descriptors for {formula}: {str(e)}")
            # Return zero vector of correct size
            return np.zeros(132, dtype=np.float32)

    def _pad_descriptors(self, descriptors: List) -> List:
        """Pads missing descriptors with zeros."""
        if len(descriptors) < 132:
            return list(descriptors) + [0.0] * (132 - len(descriptors))
        return descriptors[:132]  # Truncate if more than 132

    def calculate_batch(self, compositions: List[Composition],
                        batch_size: int = 100) -> np.ndarray:
        """
        Calculates descriptors for a batch of compositions.

        Args:
            compositions: List of Composition objects
            batch_size: Batch size for processing

        Returns:
            Numpy array of shape (n_compositions, 132)
        """
        print(f"Calculating batch of {len(compositions)} compositions...")

        all_descriptors = []
        for i in tqdm(range(0, len(compositions), batch_size),
                      desc="Calculating descriptors"):
            batch = compositions[i:i+batch_size]
            batch_descriptors = [self.calculate(comp) for comp in batch]
            all_descriptors.extend(batch_descriptors)

        return np.array(all_descriptors)

    def composition_from_formula(self, formula: str) -> Optional[Composition]:
        """Converts chemical formula to Composition object with error handling."""
        try:
            return Composition(formula)
        except Exception as e:
            print(f"Error parsing formula {formula}: {str(e)}")
            return None

    def get_descriptor_names(self) -> List[str]:
        """Returns the names of the 132 descriptors."""
        return self.descriptor_names

    def get_descriptor_statistics(self, dataset: pd.DataFrame = None) -> Dict:
        """
        Calculates statistics of the descriptors.

        Args:
            dataset: DataFrame with data to calculate statistics

        Returns:
            Dictionary with min, max, mean, std for each descriptor
        """
        stats = {}

        if dataset is not None:
            for i, name in enumerate(self.descriptor_names):
                if i < dataset.shape[1]:
                    col_data = dataset.iloc[:, i]
                    stats[name] = {
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std())
                    }

        return stats

    def clear_cache(self):
        """Clears the composition cache."""
        self.composition_cache.clear()
        print("Composition cache cleared")

# =============================================================================
# COMPOSITION GENETIC ALGORITHM
# =============================================================================

class CompositionGeneticAlgorithm:
    """
    Specialized genetic algorithm for chemical composition optimization.
    """

    def __init__(self,
                 population_size: int = 100,
                 generations: int = 50,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.7,
                 elite_size: int = 10,
                 tournament_size: int = 3):

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size

        # Evolution statistics
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []

        print(f"Genetic Algorithm initialized:")
        print(f"   Population: {population_size}")
        print(f"   Generations: {generations}")
        print(f"   Mutation rate: {mutation_rate}")
        print(f"   Crossover rate: {crossover_rate}")

    def initialize_population(self, elements: List[str],
                             max_elements: int = 4,
                             min_elements: int = 2) -> List[Dict]:
        """
        Initializes a random population of compositions.

        Args:
            elements: List of allowed elements
            max_elements: Maximum number of elements per composition
            min_elements: Minimum number of elements per composition

        Returns:
            List of individuals (compositions)
        """
        population = []

        for _ in range(self.population_size):
            # Random number of elements
            n_elements = random.randint(min_elements, max_elements)

            # Select unique elements
            selected_elements = random.sample(elements, n_elements)

            # Generate random proportions (normalized to 1)
            proportions = np.random.random(n_elements)
            proportions = proportions / proportions.sum()

            # Create individual
            individual = {
                'elements': {elem: prop for elem, prop in zip(selected_elements, proportions)},
                'fitness': 0.0,
                'predicted_gap': 0.0,
                'error': 0.0,
                'age': 0  # For tracking age
            }

            population.append(individual)

        return population

    def tournament_selection(self, population: List[Dict]) -> List[Dict]:
        """Tournament selection."""
        selected = []

        for _ in range(len(population)):
            # Select k random individuals
            tournament = random.sample(population, self.tournament_size)

            # Select the best (highest fitness)
            winner = max(tournament, key=lambda x: x['fitness'])
            selected.append(winner.copy())  # Copy to avoid references

        return selected

    def arithmetic_crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """
        Arithmetic crossover for compositions.
        Combines proportions from both parents.
        """
        child_elements = {}

        # Get all unique elements from both parents
        all_elements = set(parent1['elements'].keys()) | set(parent2['elements'].keys())

        for elem in all_elements:
            prop1 = parent1['elements'].get(elem, 0.0)
            prop2 = parent2['elements'].get(elem, 0.0)

            # Arithmetic average with random weight
            alpha = random.random()
            child_prop = alpha * prop1 + (1 - alpha) * prop2

            if child_prop > 0.001:  # Ignore very small proportions
                child_elements[elem] = child_prop

        # Renormalize to sum = 1
        if child_elements:
            total = sum(child_elements.values())
            child_elements = {k: v/total for k, v in child_elements.items()}

        return {
            'elements': child_elements,
            'fitness': 0.0,
            'predicted_gap': 0.0,
            'error': 0.0,
            'age': max(parent1['age'], parent2['age']) + 1
        }

    def geometric_crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """
        Geometric crossover for compositions.
        Better for logarithmic spaces.
        """
        child_elements = {}

        all_elements = set(parent1['elements'].keys()) | set(parent2['elements'].keys())

        for elem in all_elements:
            prop1 = parent1['elements'].get(elem, 0.001)  # Avoid zero
            prop2 = parent2['elements'].get(elem, 0.001)

            # Geometric mean
            child_prop = np.sqrt(prop1 * prop2)

            if child_prop > 0.001:
                child_elements[elem] = child_prop

        # Renormalize
        if child_elements:
            total = sum(child_elements.values())
            child_elements = {k: v/total for k, v in child_elements.items()}

        return {
            'elements': child_elements,
            'fitness': 0.0,
            'predicted_gap': 0.0,
            'error': 0.0,
            'age': max(parent1['age'], parent2['age']) + 1
        }

    def mutate_composition(self, individual: Dict,
                          elements: List[str],
                          mutation_strength: float = 0.3) -> Dict:
        """
        Applies mutation to a composition.

        Mutation types:
        1. Proportion change
        2. Element addition
        3. Element removal
        4. Element substitution
        """
        mutated = individual.copy()
        elements_dict = mutated['elements'].copy()

        # Determine mutation type
        mutation_type = random.choices(
            ['adjust', 'add', 'remove', 'replace'],
            weights=[0.6, 0.2, 0.1, 0.1]
        )[0]

        if mutation_type == 'adjust':
            # Adjust proportions randomly
            elem_to_adjust = random.choice(list(elements_dict.keys()))
            adjustment = random.uniform(-mutation_strength, mutation_strength)

            new_prop = max(0.01, elements_dict[elem_to_adjust] * (1 + adjustment))
            delta = new_prop - elements_dict[elem_to_adjust]
            elements_dict[elem_to_adjust] = new_prop

            # Redistribute difference proportionally
            other_elements = [e for e in elements_dict.keys() if e != elem_to_adjust]
            if other_elements and delta != 0:
                for elem in other_elements:
                    elements_dict[elem] = max(0.01, elements_dict[elem] * (1 - delta))

        elif mutation_type == 'add' and len(elements_dict) < 6:
            # Add new element
            available_elements = [e for e in elements if e not in elements_dict]
            if available_elements:
                new_element = random.choice(available_elements)
                # Take a little from each existing element
                reduction = random.uniform(0.05, 0.2) / len(elements_dict)
                for elem in elements_dict:
                    elements_dict[elem] *= (1 - reduction)
                elements_dict[new_element] = reduction * len(elements_dict)

        elif mutation_type == 'remove' and len(elements_dict) > 2:
            # Remove random element
            elem_to_remove = random.choice(list(elements_dict.keys()))
            removed_prop = elements_dict.pop(elem_to_remove)

            # Redistribute proportion
            if elements_dict:
                for elem in elements_dict:
                    elements_dict[elem] /= (1 - removed_prop)

        elif mutation_type == 'replace' and len(elements_dict) > 1:
            # Replace one element with another
            elem_to_replace = random.choice(list(elements_dict.keys()))
            available_elements = [e for e in elements if e not in elements_dict]

            if available_elements:
                new_element = random.choice(available_elements)
                prop = elements_dict.pop(elem_to_replace)
                elements_dict[new_element] = prop

        # Renormalize
        total = sum(elements_dict.values())
        mutated['elements'] = {k: v/total for k, v in elements_dict.items()}
        mutated['age'] = individual['age'] + 1

        return mutated

    def calculate_population_diversity(self, population: List[Dict]) -> float:
        """Calculates the diversity of the population."""
        if len(population) < 2:
            return 0.0

        # Extract composition vectors
        all_elements = set()
        for ind in population:
            all_elements.update(ind['elements'].keys())

        element_list = sorted(list(all_elements))
        n_elements = len(element_list)

        if n_elements == 0:
            return 0.0

        # Create composition matrix
        comp_matrix = np.zeros((len(population), n_elements))

        for i, ind in enumerate(population):
            for elem, prop in ind['elements'].items():
                if elem in element_list:
                    j = element_list.index(elem)
                    comp_matrix[i, j] = prop

        # Calculate diversity as average distance between individuals
        diversity = 0.0
        count = 0

        for i in range(len(population)):
            for j in range(i+1, len(population)):
                distance = np.linalg.norm(comp_matrix[i] - comp_matrix[j])
                diversity += distance
                count += 1

        if count > 0:
            diversity /= count

        return diversity

    def evolve(self, population: List[Dict],
               elements: List[str],
               fitness_function: callable) -> Tuple[List[Dict], Dict]:
        """
        Executes one complete generation of evolution.

        Returns:
            New population and generation statistics
        """
        # Evaluate fitness of current population
        for ind in population:
            ind['fitness'], ind['predicted_gap'], ind['error'] = fitness_function(ind)

        # Sort by descending fitness
        population.sort(key=lambda x: x['fitness'], reverse=True)

        # Save elite
        elite = population[:self.elite_size]

        # Select parents (excluding elite for diversity)
        parents = self.tournament_selection(population[self.elite_size:])

        # Crossover
        offspring = []
        for i in range(0, len(parents), 2):
            if i+1 < len(parents) and random.random() < self.crossover_rate:
                parent1, parent2 = parents[i], parents[i+1]

                # Choose crossover type randomly
                if random.random() < 0.5:
                    child = self.arithmetic_crossover(parent1, parent2)
                else:
                    child = self.geometric_crossover(parent1, parent2)

                offspring.append(child)

        # Mutation
        mutated_offspring = []
        for child in offspring:
            if random.random() < self.mutation_rate:
                mutated_child = self.mutate_composition(child, elements)
                mutated_offspring.append(mutated_child)
            else:
                mutated_offspring.append(child)

        # Create new population: elite + mutated offspring + new random individuals
        new_population = elite.copy()
        new_population.extend(mutated_offspring)

        # Complete with new random individuals if necessary
        while len(new_population) < self.population_size:
            n_elements = random.randint(2, 4)
            selected_elements = random.sample(elements, n_elements)
            proportions = np.random.random(n_elements)
            proportions = proportions / proportions.sum()

            new_individual = {
                'elements': {elem: prop for elem, prop in zip(selected_elements, proportions)},
                'fitness': 0.0,
                'predicted_gap': 0.0,
                'error': 0.0,
                'age': 0
            }
            new_population.append(new_individual)

        # Evaluate new population
        for ind in new_population:
            if ind['fitness'] == 0.0:  # Only evaluate new individuals
                ind['fitness'], ind['predicted_gap'], ind['error'] = fitness_function(ind)

        # Calculate statistics
        fitness_values = [ind['fitness'] for ind in new_population]
        avg_fitness = np.mean(fitness_values)
        best_fitness = max(fitness_values)
        diversity = self.calculate_population_diversity(new_population)

        # Update history
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        self.diversity_history.append(diversity)

        stats = {
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'diversity': diversity,
            'population_size': len(new_population),
            'unique_formulas': len(set(self._individual_to_formula(ind) for ind in new_population))
        }

        return new_population, stats

    def _individual_to_formula(self, individual: Dict) -> str:
        """Converts individual to approximate chemical formula."""
        elements = individual['elements']
        # Sort by descending proportion
        sorted_elements = sorted(elements.items(), key=lambda x: x[1], reverse=True)

        formula_parts = []
        for elem, prop in sorted_elements:
            if prop > 0.05:  # Ignore traces below 5%
                # Convert proportion to approximate integer coefficient
                if prop > 0.33:
                    coeff = 1
                elif prop > 0.25:
                    coeff = ""
                elif prop > 0.10:
                    coeff = 2
                else:
                    coeff = 3

                formula_parts.append(f"{elem}{coeff if coeff != '' else ''}")

        return "".join(formula_parts)

    def plot_evolution(self, save_path: str = None):
        """Plots the evolution of the genetic algorithm."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Fitness vs Generation
        axes[0, 0].plot(self.best_fitness_history, 'b-', linewidth=2, label='Best Fitness')
        axes[0, 0].plot(self.avg_fitness_history, 'r--', linewidth=2, label='Average Fitness')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].set_title('Fitness Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Diversity vs Generation
        axes[0, 1].plot(self.diversity_history, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Diversity')
        axes[0, 1].set_title('Population Diversity')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Fitness distribution in the last generation
        if len(self.best_fitness_history) > 0:
            axes[1, 0].hist([self.best_fitness_history[-1], self.avg_fitness_history[-1]],
                           bins=20, alpha=0.7, label=['Best', 'Average'])
            axes[1, 0].set_xlabel('Fitness')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Fitness Distribution (Last Gen)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Improvement trend
        if len(self.best_fitness_history) > 1:
            improvements = []
            for i in range(1, len(self.best_fitness_history)):
                improvement = self.best_fitness_history[i] - self.best_fitness_history[i-1]
                improvements.append(improvement)

            axes[1, 1].plot(improvements, 'purple', linewidth=2)
            axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Generation')
            axes[1, 1].set_ylabel('Improvement Delta')
            axes[1, 1].set_title('Generational Improvement')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

# =============================================================================
# INVERSE BAND GAP DESIGNER
# =============================================================================

class InverseBandgapDesigner:
    """
    Complete inverse design system for materials based on target band gap.
    Combines neural network model, genetic algorithm, and chemical validation.
    """

    def __init__(self,
                 model_path: str = None,
                 scaler_path: str = None,
                 elements: List[str] = None,
                 max_elements: int = 4,
                 device: str = 'auto',
                 validation_data_path: str = None):
        """
        Initializes the inverse designer.

        Args:
            model_path: Path to the NN model (.keras)
            scaler_path: Path to the scaler (.pkl)
            elements: List of allowed elements
            max_elements: Maximum elements per composition
            validation_data_path: Path to external validation dataset (CSV)
            device: 'auto', 'cpu', or 'gpu'
        """
        print("="*80)
        print("INVERSE BANDGAP DESIGNER - INITIALIZATION")
        print("="*80)

        # 1. Configure device
        self.device = self._setup_device(device)

        # 2. Load model and scaler
        self.model, self.scaler = self._load_model_and_scaler(model_path, scaler_path)

        # 3. Initialize Magpie calculator
        self.magpie_calc = MagpieCalculator()

        # 4. Configure chemical space
        self.elements = elements or self._get_default_elements()
        self.max_elements = max_elements
        self.chemical_rules = self._load_chemical_rules()

        # 5. Initialize genetic algorithm
        self.ga = CompositionGeneticAlgorithm(
            population_size=150,
            generations=80,
            mutation_rate=0.25,
            crossover_rate=0.7,
            elite_size=15,
            tournament_size=4
        )

        # 6. Cache and statistics
        self.prediction_cache = {}
        self.stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'unique_formulas': set()
        }

        # 7. Load validation dataset if provided
        self.validation_dict = {}
        if validation_data_path:
            self._load_validation_data(validation_data_path)

        print(f"\nInverse Designer initialized successfully:")
        print(f"   - Allowed elements: {len(self.elements)} elements")
        print(f"   - Maximum elements per composition: {max_elements}")
        print(f"   - Magpie descriptors: 132 features")
        print(f"   - NN model: {self.model.input_shape[1]} -> {self.model.output_shape[1]}")
        if self.validation_dict:
            print(f"   - Validation dataset: {len(self.validation_dict)} compounds")
        print("="*80)

    def _load_validation_data(self, path: str):
        """Loads the external validation dataset for comparisons."""
        try:
            df_val = pd.read_csv(path)
            # Identify composition and band gap columns
            comp_col = 'composition' if 'composition' in df_val.columns else 'final_composition'
            gap_col = 'gap expt' if 'gap expt' in df_val.columns else 'bandgap'

            self.validation_dict = {}

            for idx, row in df_val.iterrows():
                formula = row[comp_col]
                try:
                    comp = Composition(formula)
                    reduced = comp.reduced_formula
                    bg = row[gap_col]
                    # If duplicates exist, keep the first one
                    if reduced not in self.validation_dict:
                        self.validation_dict[reduced] = bg
                except Exception:
                    continue

            print(f"Validation dataset loaded: {len(self.validation_dict)} unique compounds")
        except Exception as e:
            print(f"Could not load validation dataset: {e}")
            self.validation_dict = {}

    def validate_candidate_against_experimental(self, candidate_dict: Dict) -> Dict:
        """
        Checks if the candidate (or a similar composition) exists in the validation dataset.

        Args:
            candidate_dict: Dictionary {element: proportion}

        Returns:
            Dictionary with validation information:
            - 'in_validation': bool
            - 'formula': reduced formula
            - 'experimental_bg': experimental band gap (if exists)
            - 'predicted_bg': model-predicted band gap
            - 'error': absolute error between experimental and predicted
        """
        if not hasattr(self, 'validation_dict') or not self.validation_dict:
            return {'in_validation': False, 'formula': ''}

        # Get reduced formula of the candidate
        formula = self._dict_to_formula(candidate_dict)
        try:
            comp = Composition(formula)
            reduced = comp.reduced_formula
        except Exception:
            return {'in_validation': False, 'formula': formula}

        # Search in the dictionary
        if reduced in self.validation_dict:
            exp_bg = self.validation_dict[reduced]
            # Predict with the model (may be cached)
            pred_bg = self.predict_bandgap(candidate_dict)
            return {
                'in_validation': True,
                'formula': reduced,
                'experimental_bg': exp_bg,
                'predicted_bg': pred_bg,
                'error': abs(pred_bg - exp_bg)
            }
        else:
            return {
                'in_validation': False,
                'formula': reduced
            }

    def predict_formula(self, formula: str) -> float:
        """Predicts the band gap of a given chemical formula."""
        try:
            comp = Composition(formula)
            # Get atomic fractions
            elem_dict = {str(el): comp.get_atomic_fraction(el) for el in comp.elements}
            return self.predict_bandgap(elem_dict)
        except Exception as e:
            print(f"Error processing formula {formula}: {e}")
            return None

    def _setup_device(self, device: str) -> str:
        """Configures the device (CPU/GPU)."""
        if device == 'auto':
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    device = 'gpu'
                    print(f"GPU detected: {len(gpus)} device(s)")
                except Exception:
                    device = 'cpu'
            else:
                device = 'cpu'

        if device == 'cpu':
            print("Using CPU (GPU not available)")
        elif device == 'gpu':
            print("GPU configured for optimal use")

        return device

    def _load_model_and_scaler(self, model_path: str, scaler_path: str):
        """Loads the neural network model and scaler."""
        # If no paths are provided, use defaults from your training
        if model_path is None:
            possible_paths = [
                '/content/bandgap_nn_model.keras',
                '/content/bandgap_nn_model.h5',
                './model.keras'
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break

        if scaler_path is None:
            possible_paths = [
                '/content/scaler.pkl',
                '/content/scaler_advanced.pkl',
                './scaler.pkl'
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    scaler_path = path
                    break

        # Load model
        print(f"Loading model from: {model_path}")
        try:
            model = keras.models.load_model(model_path)
            print(f"   Model loaded: {model.summary()}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("   Creating dummy model for testing...")
            model = self._create_dummy_model()

        # Load scaler
        print(f"Loading scaler from: {scaler_path}")
        try:
            scaler = joblib.load(scaler_path)
            print("   Scaler loaded")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            print("   Creating dummy scaler...")
            scaler = StandardScaler()

        return model, scaler

    def _create_dummy_model(self):
        """Creates a dummy model if no model is available."""
        model = keras.Sequential([
            layers.Input(shape=(132,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='softplus')
        ])
        model.compile(optimizer='adam', loss='mae')
        return model

    def _get_default_elements(self) -> List[str]:
        """Returns a list of common elements for search."""
        common_elements = [
            # Alkali metals
            'Li', 'Na', 'K', 'Rb', 'Cs',
            # Alkaline earth metals
            'Be', 'Mg', 'Ca', 'Sr', 'Ba',
            # Transition metals (first series)
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            # Transition metals (second series)
            'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
            # Transition metals (third series)
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            # Lanthanides
            'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            # Actinides
            'Th', 'U',
            # p-block metals
            'Al', 'Ga', 'In', 'Tl', 'Sn', 'Pb', 'Bi',
            # Semimetals and semiconductors
            'B', 'C', 'Si', 'Ge', 'P', 'As', 'Sb',
            # Nonmetals
            'H', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I',
            # Chalcogens
            'Se', 'Te',
            # Noble gases
            'Xe'
        ]

        # Filter elements that actually exist in pymatgen
        valid_elements = []
        for elem in common_elements:
            try:
                Element(elem)  # Verify that it exists
                valid_elements.append(elem)
            except Exception:
                continue

        return valid_elements

    def _load_chemical_rules(self) -> Dict:
        """Loads chemical rules for validation."""
        return {
            'max_elements_per_composition': self.max_elements,
            'min_elements_per_composition': 2,
            'allowed_valences': self._get_valence_rules(),
            'electronegativity_ranges': self._get_electronegativity_ranges(),
            'forbidden_combinations': [
                ('He', 'X'),  # Helium does not form compounds
                ('Ne', 'X'),
                ('Ar', 'X'),
                ('Kr', 'X'),
                ('Rn', 'X')
                # Xe is kept because it does form some compounds
            ],
            'common_oxidation_states': self._get_common_oxidation_states()
        }

    def _get_valence_rules(self) -> Dict:
        """Valence rules for common elements."""
        return {
            # Nonmetals and halogens
            'O': [-2], 'H': [+1], 'F': [-1], 'Cl': [-1], 'Br': [-1], 'I': [-1],
            'S': [-2, +4, +6], 'N': [-3, +3, +5], 'C': [-4, +4], 'P': [-3, +3, +5],
            'Se': [-2, +4, +6], 'Te': [-2, +4, +6], 'As': [-3, +3, +5], 'Sb': [-3, +3, +5],
            # Alkali metals
            'Li': [+1], 'Na': [+1], 'K': [+1], 'Rb': [+1], 'Cs': [+1],
            # Alkaline earth metals
            'Be': [+2], 'Mg': [+2], 'Ca': [+2], 'Sr': [+2], 'Ba': [+2],
            # p-block metals
            'Al': [+3], 'Ga': [+3], 'In': [+3], 'Tl': [+1, +3], 'Sn': [+2, +4], 'Pb': [+2, +4], 'Bi': [+3],
            # Transition metals (first series)
            'Sc': [+3], 'Ti': [+2, +3, +4], 'V': [+2, +3, +4, +5], 'Cr': [+2, +3, +6],
            'Mn': [+2, +3, +4, +6, +7], 'Fe': [+2, +3], 'Co': [+2, +3], 'Ni': [+2, +3],
            'Cu': [+1, +2], 'Zn': [+2],
            # Transition metals (second series)
            'Y': [+3], 'Zr': [+4], 'Nb': [+3, +5], 'Mo': [+3, +4, +5, +6], 'Tc': [+4, +7],
            'Ru': [+3, +4, +6, +8], 'Rh': [+3], 'Pd': [+2, +4], 'Ag': [+1], 'Cd': [+2],
            # Transition metals (third series)
            'Hf': [+4], 'Ta': [+5], 'W': [+4, +5, +6], 'Re': [+4, +6, +7],
            'Os': [+3, +4, +6, +8], 'Ir': [+3, +4], 'Pt': [+2, +4], 'Au': [+1, +3], 'Hg': [+1, +2],
            # Lanthanides
            'La': [+3], 'Ce': [+3, +4], 'Pr': [+3], 'Nd': [+3], 'Pm': [+3],
            'Sm': [+2, +3], 'Eu': [+2, +3], 'Gd': [+3], 'Tb': [+3, +4],
            'Dy': [+3], 'Ho': [+3], 'Er': [+3], 'Tm': [+3], 'Yb': [+2, +3], 'Lu': [+3],
            # Actinides
            'Th': [+4], 'U': [+4, +6],
            # Semimetals
            'B': [+3], 'Si': [+4], 'Ge': [+2, +4],
        }

    def _get_electronegativity_ranges(self) -> Dict:
        """Electronegativity ranges for stability."""
        return {
            'ionic': (1.0, 3.0),     # For ionic compounds
            'covalent': (0.0, 1.0),  # For covalent compounds
            'metallic': (0.0, 0.5)   # For metallic compounds
        }

    def _get_common_oxidation_states(self) -> Dict:
        """Common oxidation states."""
        return {
            # Nonmetals and halogens
            'O': -2, 'H': +1, 'F': -1, 'Cl': -1, 'Br': -1, 'I': -1,
            'S': +6, 'N': -3, 'C': +4, 'P': +5, 'Se': +4, 'Te': +4, 'As': +3, 'Sb': +3,
            # Alkali metals
            'Li': +1, 'Na': +1, 'K': +1, 'Rb': +1, 'Cs': +1,
            # Alkaline earth metals
            'Be': +2, 'Mg': +2, 'Ca': +2, 'Sr': +2, 'Ba': +2,
            # p-block metals
            'Al': +3, 'Ga': +3, 'In': +3, 'Tl': +3, 'Sn': +4, 'Pb': +2, 'Bi': +3,
            # Common transition metals
            'Sc': +3, 'Ti': +4, 'V': +5, 'Cr': +3, 'Mn': +2, 'Fe': +3, 'Co': +2, 'Ni': +2,
            'Cu': +2, 'Zn': +2, 'Zr': +4, 'Nb': +5, 'Mo': +6, 'Tc': +7, 'Ru': +3, 'Rh': +3,
            'Pd': +2, 'Ag': +1, 'Cd': +2, 'Hf': +4, 'Ta': +5, 'W': +6, 'Re': +7, 'Os': +4,
            'Ir': +3, 'Pt': +2, 'Au': +3, 'Hg': +2,
            # Lanthanides (most common +3)
            'La': +3, 'Ce': +3, 'Pr': +3, 'Nd': +3, 'Pm': +3, 'Sm': +3, 'Eu': +3, 'Gd': +3,
            'Tb': +3, 'Dy': +3, 'Ho': +3, 'Er': +3, 'Tm': +3, 'Yb': +3, 'Lu': +3,
            # Actinides
            'Th': +4, 'U': +6,
            # Semimetals
            'B': +3, 'Si': +4, 'Ge': +4,
        }

    def composition_to_descriptors(self, composition_dict: Dict) -> np.ndarray:
        """
        Converts composition dictionary to Magpie descriptors.

        Args:
            composition_dict: Dictionary {element: proportion}

        Returns:
            Array of 132 scaled descriptors
        """
        # 1. Convert to formula
        formula = self._dict_to_formula(composition_dict)

        # 2. Create Composition object
        composition = self.magpie_calc.composition_from_formula(formula)
        if composition is None:
            return np.zeros(132, dtype=np.float32)

        # 3. Calculate descriptors
        descriptors = self.magpie_calc.calculate(composition)

        # 4. Scale (as in training)
        if hasattr(self.scaler, 'transform'):
            descriptors = descriptors.reshape(1, -1)
            descriptors = self.scaler.transform(descriptors)
            descriptors = descriptors.flatten()

        return descriptors

    def predict_bandgap(self, composition_dict: Dict) -> float:
        """
        Predicts band gap for a composition.

        Args:
            composition_dict: Dictionary {element: proportion}

        Returns:
            Predicted band gap in eV
        """
        # Generate unique cache key
        cache_key = self._get_composition_hash(composition_dict)

        # Check cache
        if cache_key in self.prediction_cache:
            self.stats['cache_hits'] += 1
            return self.prediction_cache[cache_key]

        # Calculate descriptors
        descriptors = self.composition_to_descriptors(composition_dict)

        # Predict
        prediction = self.model.predict(
            descriptors.reshape(1, -1),
            verbose=0
        ).flatten()[0]

        # Ensure non-negative
        prediction = max(0.0, float(prediction))

        # Update statistics
        self.stats['total_predictions'] += 1
        self.stats['unique_formulas'].add(cache_key)

        # Save to cache
        self.prediction_cache[cache_key] = prediction

        return prediction

    def fitness_function(self, target_gap: float, tolerance: float = 0.1) -> callable:
        """
        Creates the fitness function for the genetic algorithm.

        Args:
            target_gap: Target band gap in eV
            tolerance: Acceptable tolerance in eV

        Returns:
            Fitness function that takes an individual and returns (fitness, predicted_gap, error)
        """
        def evaluate(individual: Dict):
            # Predict band gap
            predicted_gap = self.predict_bandgap(individual['elements'])

            # Calculate absolute error
            error = abs(predicted_gap - target_gap)

            # Calculate fitness (higher is better)
            # Fitness combines accuracy and penalty for large errors
            if error <= tolerance:
                # Within tolerance: high fitness
                base_fitness = 10.0 * (1.0 - error/tolerance)
            else:
                # Outside tolerance: fitness decays exponentially
                base_fitness = 1.0 / (1.0 + error)

            # Penalize compositions with many elements
            n_elements = len(individual['elements'])
            complexity_penalty = max(0, 1.0 - 0.1 * (n_elements - 2))

            # Bonus for chemical stability (simplified)
            stability_bonus = self._estimate_stability(individual['elements'])

            # Final fitness
            fitness = base_fitness * complexity_penalty * (1.0 + stability_bonus)

            return fitness, predicted_gap, error

        return evaluate

    def _estimate_stability(self, composition_dict: Dict) -> float:
        """
        Simplified chemical stability estimation.

        Args:
            composition_dict: Composition dictionary

        Returns:
            Stability score between 0.0 and 1.0
        """
        elements = list(composition_dict.keys())

        # 1. Check element pairs with simple rules
        score = 0.5  # Base score

        # 1.1. Penalty for invalid stoichiometry
        stoichiometry_penalty = self._estimate_stoichiometry_penalty(composition_dict)
        score *= stoichiometry_penalty

        # 2. Check electronegativities
        electronegativities = []
        for elem in elements:
            try:
                el = Element(elem)
                electronegativities.append(el.X)
            except Exception:
                electronegativities.append(2.0)  # Default value

        if len(electronegativities) > 1:
            electroneg_diff = max(electronegativities) - min(electronegativities)
            # Moderate difference is good for stability
            if 0.5 < electroneg_diff < 2.0:
                score += 0.2
            elif electroneg_diff > 3.0:
                score -= 0.1

        # 3. Check elements known to form stable compounds
        stable_pairs = [
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
            ('Cd', 'O'), ('Sn', 'Te'), ('Ga', 'In'), ('Cu', 'Te'), ('O', 'Y')
        ]

        for elem1, elem2 in itertools.combinations(elements, 2):
            if (elem1, elem2) in stable_pairs or (elem2, elem1) in stable_pairs:
                score += 0.1

        # 4. Normalize to [0, 1]
        return max(0.0, min(1.0, score))

    def _validate_stoichiometry(self, elements_dict: Dict, formula: str = None) -> Tuple[bool, str]:
        """
        Simplified version without recursion.
        Only performs basic checks.
        """
        elements = list(elements_dict.keys())

        # If the formula already looks reasonable, accept it
        if formula and not self._is_formula_unusual(formula):
            return True, formula

        # For binary compounds with very high coefficients
        if len(elements) == 2:
            import re
            # Search for high coefficients (>4)
            if re.search(r'[A-Z][a-z]?[5-9]', formula) or re.search(r'[A-Z][a-z]?\d{2,}', formula):
                # Generate simple formula
                simple_formula = self._simple_rational_formula(elements_dict)
                return False, simple_formula

        return True, formula if formula else self._simple_rational_formula(elements_dict)

    def _adjust_stoichiometry(self, elements_dict: Dict,
                             valence_rules: Dict,
                             atomic_masses: Dict) -> Optional[Dict]:
        """
        Adjusts proportions to balance charges.
        """
        elements = list(elements_dict.keys())

        # Only for binary and ternary systems for simplicity
        if len(elements) > 3:
            return elements_dict

        # Calculate current charges
        charges = {}
        for elem in elements:
            if elem in valence_rules and valence_rules[elem]:
                charges[elem] = valence_rules[elem][0]  # Most common valence
            else:
                charges[elem] = 1

        # For binary system: A_xB_y -> |charge_A| * x = |charge_B| * y
        if len(elements) == 2:
            elem1, elem2 = elements[0], elements[1]
            charge1 = abs(charges[elem1])
            charge2 = abs(charges[elem2])

            # Find simplest integer ratio
            # Least common multiple to obtain integer numbers
            ratio1 = charge2
            ratio2 = charge1

            # Simplify ratio if possible
            try:
                from math import gcd
                gcd_val = gcd(ratio1, ratio2)
                ratio1 //= gcd_val
                ratio2 //= gcd_val
            except Exception:
                pass

            # Calculate proportions by mass
            m1 = atomic_masses.get(elem1, 1.0)
            m2 = atomic_masses.get(elem2, 1.0)

            total_mass = (ratio1 * m1) + (ratio2 * m2)
            new_prop1 = (ratio1 * m1) / total_mass
            new_prop2 = (ratio2 * m2) / total_mass

            return {elem1: new_prop1, elem2: new_prop2}

        # For ternary systems, simplified approach
        return elements_dict

    def _simple_rational_formula(self, composition_dict: Dict) -> str:
        """
        Generates formula with simple rational proportions based on valences.
        Improved version that considers multiple possible valences.
        """
        elements = list(composition_dict.keys())

        if len(elements) == 0:
            return ""

        # Get valence rules
        valence_rules = self._get_valence_rules()
        common_oxidation = self._get_common_oxidation_states()

        # For binary systems
        if len(elements) == 2:
            elem1, elem2 = elements
            prop1, prop2 = composition_dict[elem1], composition_dict[elem2]

            # Get possible valences for each element
            valences1 = valence_rules.get(elem1, [1])
            valences2 = valence_rules.get(elem2, [1])

            # If both elements have multiple valences, try combinations
            possible_formulas = []

            for v1 in valences1[:2]:  # Take only the first 2 valences
                for v2 in valences2[:2]:
                    # For neutral compound: |v1| * n1 = |v2| * n2
                    v1_abs = abs(v1)
                    v2_abs = abs(v2)

                    # Find simplest integer ratio
                    n1 = v2_abs
                    n2 = v1_abs

                    # Simplify using gcd
                    try:
                        from math import gcd
                        gcd_val = gcd(n1, n2)
                        n1 //= gcd_val
                        n2 //= gcd_val
                    except Exception:
                        pass

                    # Limit to small coefficients (maximum 4)
                    while n1 > 4 or n2 > 4:
                        if n1 > 1:
                            n1 //= 2
                        if n2 > 1:
                            n2 //= 2

                    # Calculate expected mass ratio
                    try:
                        m1 = Element(elem1).atomic_mass
                        m2 = Element(elem2).atomic_mass
                        expected_ratio = (n1 * m1) / (n2 * m2)
                        current_ratio = prop1 / prop2 if prop2 > 0 else float('inf')

                        # Save formula with its error
                        formula = f"{elem1}{n1 if n1 > 1 else ''}{elem2}{n2 if n2 > 1 else ''}"
                        error = abs(current_ratio - expected_ratio)
                        possible_formulas.append((formula, error, n1 + n2))
                    except Exception:
                        pass

            # Choose best formula (lowest error, simplest)
            if possible_formulas:
                # Sort by error and then by simplicity (sum of coefficients)
                possible_formulas.sort(key=lambda x: (x[1], x[2]))
                return possible_formulas[0][0]

        # For more elements or fallback
        formula_parts = []
        for elem, prop in sorted(composition_dict.items(), key=lambda x: x[1], reverse=True):
            if prop > 0.1:  # Only main elements
                # Simple coefficient based on proportion
                if prop > 0.33:
                    coeff = 1
                elif prop > 0.25:
                    coeff = ""
                elif prop > 0.20:
                    coeff = 2
                elif prop > 0.15:
                    coeff = 3
                else:
                    coeff = 4

                formula_parts.append(f"{elem}{coeff if coeff != '' else ''}")

        # If we did not generate enough parts, add at least 2 elements
        if len(formula_parts) < 2 and len(elements) >= 2:
            top_elements = sorted(composition_dict.items(), key=lambda x: x[1], reverse=True)[:2]
            formula_parts = [f"{elem}{1 if i == 0 else 1}" for i, (elem, _) in enumerate(top_elements)]

        return "".join(formula_parts)

    def _estimate_stoichiometry_penalty(self, composition_dict: Dict) -> float:
        """
        Penalizes compositions with invalid stoichiometry.
        More flexible version that allows multiple combinations.
        """
        elements = list(composition_dict.keys())

        # For binary systems
        if len(elements) == 2:
            elem1, elem2 = elements[0], elements[1]

            # Get all possible valences
            valence_rules = self._get_valence_rules()
            valences1 = valence_rules.get(elem1, [1])
            valences2 = valence_rules.get(elem2, [1])

            # Calculate atomic proportions
            try:
                m1 = Element(elem1).atomic_mass
                m2 = Element(elem2).atomic_mass
            except Exception:
                return 1.0  # We cannot verify

            prop1 = composition_dict[elem1]
            prop2 = composition_dict[elem2]

            n1_rel = prop1 / m1
            n2_rel = prop2 / m2

            # Check if any valence combination balances charges
            best_balance = float('inf')

            for v1 in valences1:
                for v2 in valences2:
                    total_charge = v1 * n1_rel + v2 * n2_rel
                    charge_imbalance = abs(total_charge)
                    if charge_imbalance < best_balance:
                        best_balance = charge_imbalance

            # Penalize based on best balance found
            if best_balance > 0.5:
                return 0.3  # Strong penalty
            elif best_balance > 0.3:
                return 0.5  # Moderate penalty
            elif best_balance > 0.1:
                return 0.8  # Light penalty

        return 1.0  # No penalty

    def design_materials(self,
                        target_gap: float,
                        material_type: str = None,
                        n_candidates: int = 10,
                        show_progress: bool = True) -> Dict:
        """
        Designs materials for a target band gap.

        Args:
            target_gap: Target band gap in eV
            material_type: Optional: 'metal', 'semiconductor', 'insulator'
            n_candidates: Number of candidates to return
            show_progress: Show progress bar

        Returns:
            Dictionary with complete results
        """
        print("\n" + "="*80)
        print(f"MATERIAL DESIGN FOR BAND GAP = {target_gap:.3f} eV")
        print("="*80)

        # Filter elements by material type if specified
        filtered_elements = self._filter_elements_by_material_type(
            self.elements, target_gap, material_type
        )

        print(f"Search space:")
        print(f"   - Elements: {len(filtered_elements)} available")
        print(f"   - Possible combinations: ~10^{self._estimate_combinatorial_space(filtered_elements):.1f}")
        print(f"   - Target tolerance: +/- 0.1 eV")

        # Create fitness function
        fitness_func = self.fitness_function(target_gap, tolerance=0.1)

        # Initialize population
        population = self.ga.initialize_population(
            filtered_elements,
            max_elements=min(self.max_elements, 4),
            min_elements=2
        )

        # Evolution across generations
        best_candidates = []
        generation_stats = []

        print("\nGENETIC EVOLUTION IN PROGRESS...")
        for gen in tqdm(range(self.ga.generations), disable=not show_progress):
            # Execute one generation
            population, stats = self.ga.evolve(population, filtered_elements, fitness_func)

            # Save statistics
            generation_stats.append(stats)

            # Save best candidates of this generation
            population_sorted = sorted(population, key=lambda x: x['fitness'], reverse=True)
            best_of_gen = population_sorted[:5]

            for cand in best_of_gen:
                if cand['error'] <= 0.15:  # Acceptable error
                    formula = self._dict_to_formula(cand['elements'])
                    cand['formula'] = formula
                    best_candidates.append(cand.copy())

            # Show progress every 10 generations
            if show_progress and gen % 10 == 9:
                print(f"   Generation {gen+1}: Best fitness = {stats['best_fitness']:.3f}, "
                      f"Error = {population_sorted[0]['error']:.3f} eV")

        # Process results
        final_results = self._process_results(
            best_candidates, target_gap, n_candidates
        )

        # Add statistics
        final_results['evolution_stats'] = generation_stats
        final_results['search_space_info'] = {
            'elements_used': filtered_elements,
            'n_elements': len(filtered_elements),
            'total_predictions': self.stats['total_predictions'],
            'cache_hits': self.stats['cache_hits'],
            'unique_formulas_generated': len(self.stats['unique_formulas'])
        }

        # Show summary
        self._print_summary(final_results)

        return final_results

    def _filter_elements_by_material_type(self, elements: List[str],
                                         target_gap: float,
                                         material_type: str = None) -> List[str]:
        """Filters elements based on the target band gap."""
        if material_type is None:
            # Determine type based on gap
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

        # Map elements by type (empirical rules)
        element_categories = {
            'metals': ['Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Th', 'U', 'Al', 'Ga', 'In', 'Tl', 'Sn', 'Pb', 'Bi'],
            'semiconductors': ['Si', 'Ge', 'Ga', 'As', 'In', 'P', 'Sb', 'Se', 'Te', 'B', 'C', 'S'],
            'insulators': ['F', 'Cl', 'Br', 'I', 'O', 'N', 'H', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Al'],
            'transition_metals': ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Th', 'U']
        }

        # Combine categories according to material type
        if material_type in ['metal', 'semimetal']:
            selected = element_categories['metals'] + element_categories['transition_metals']
        elif material_type == 'narrow_gap':
            selected = element_categories['semiconductors'] + element_categories['transition_metals']
        elif material_type == 'semiconductor':
            selected = element_categories['semiconductors'] + ['O', 'S', 'Se', 'Te']
        else:  # insulator
            selected = element_categories['insulators'] + element_categories['metals'][:5]

        # Filter only available elements (using set to remove duplicates)
        filtered = list(set([elem for elem in selected if elem in elements]))

        # Ensure at least 10 elements
        if len(filtered) < 10:
            # Add additional common elements
            additional = [e for e in elements if e not in filtered]
            filtered.extend(additional[:20])

        return list(set(filtered))[:30]  # Limit to 30 elements for efficiency

    def _estimate_combinatorial_space(self, elements: List[str]) -> float:
        """Estimates the size of the combinatorial search space."""
        total_combinations = 0
        for k in range(2, self.max_elements + 1):
            # Combinations of k elements
            comb_k = len(list(itertools.combinations(elements, k)))
            # Each combination has infinite continuous proportions
            total_combinations += comb_k * 1000  # Approximation

        return np.log10(total_combinations)

    def _process_results(self, candidates: List[Dict],
                        target_gap: float,
                        n_candidates: int) -> Dict:
        """Processes and filters final candidates."""
        # Remove duplicates based on formula
        unique_candidates = {}
        for cand in candidates:
            # Use non-recursive version
            formula = cand.get('formula', self._simple_rational_formula(cand['elements']))
            if formula not in unique_candidates:
                unique_candidates[formula] = cand
            elif cand['fitness'] > unique_candidates[formula]['fitness']:
                unique_candidates[formula] = cand

        # Sort by fitness
        sorted_candidates = sorted(
            unique_candidates.values(),
            key=lambda x: x['fitness'],
            reverse=True
        )

        # Take top n candidates
        final_candidates = sorted_candidates[:n_candidates]

        # Calculate additional metrics
        for cand in final_candidates:
            # Ensure it has a formula
            if 'formula' not in cand:
                cand['formula'] = self._simple_rational_formula(cand['elements'])

            cand['stability_score'] = self._estimate_stability(cand['elements'])
            cand['novelty_score'] = self._calculate_novelty_score(cand['formula'])
            cand['synthesis_hints'] = self._generate_synthesis_hints(cand['elements'])

        # Create results structure
        results = {
            'target_bandgap': target_gap,
            'generation_timestamp': datetime.now().isoformat(),
            'n_candidates_generated': len(sorted_candidates),
            'n_candidates_returned': len(final_candidates),
            'candidates': final_candidates,
            'summary_metrics': self._calculate_summary_metrics(final_candidates, target_gap)
        }

        return results

    def _calculate_novelty_score(self, formula: str) -> float:
        """Calculates novelty score based on formula rarity."""
        # Simple rule: more elements = more novel
        composition = self.magpie_calc.composition_from_formula(formula)
        if composition:
            n_elements = len(composition.elements)
            avg_atomic_number = np.mean([e.Z for e in composition.elements])

            # Score between 0 and 1
            novelty = min(1.0, (n_elements / 5.0) * (avg_atomic_number / 50.0))
            return novelty

        return 0.5  # Default value

    def _generate_synthesis_hints(self, composition_dict: Dict) -> List[str]:
        """
        Generates synthesis hints based on the composition.

        Args:
            composition_dict: Composition dictionary

        Returns:
            List of strings with synthesis hints
        """
        elements = list(composition_dict.keys())
        hints = []

        # Basic synthesis rules
        if 'O' in elements and composition_dict['O'] > 0.3:
            hints.append("Consider synthesis by oxidation (sintering, sol-gel)")

        if len(elements) == 2:
            hints.append("Binary system: possible by direct element fusion")

        if any(elem in ['Ti', 'Zr', 'Hf', 'W', 'Mo'] for elem in elements):
            hints.append("Contains refractory elements: requires high temperature (>1000 C)")

        if 'H' in elements:
            hints.append("Contains hydrogen: consider synthesis in controlled atmosphere")

        if 'F' in elements or 'Cl' in elements:
            hints.append("Contains halogens: consider chemical transport methods")

        if all(elem in ['C', 'H', 'N', 'O'] for elem in elements):
            hints.append("Organic/Organometallic: consider solution synthesis")

        if not hints:
            hints.append("General synthesis: mix precursors + calcination")

        return hints[:3]  # Maximum 3 hints

    def _calculate_summary_metrics(self, candidates: List[Dict],
                                  target_gap: float) -> Dict:
        """
        Calculates summary metrics of the candidates.

        Args:
            candidates: List of candidates
            target_gap: Target band gap

        Returns:
            Dictionary with summary metrics
        """
        if not candidates:
            return {}

        errors = [c['error'] for c in candidates]
        fitnesses = [c['fitness'] for c in candidates]
        gaps = [c['predicted_gap'] for c in candidates]

        return {
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'std_error': float(np.std(errors)),
            'mean_fitness': float(np.mean(fitnesses)),
            'mean_predicted_gap': float(np.mean(gaps)),
            'success_rate': sum(1 for e in errors if e <= 0.1) / len(errors),
            'diversity_score': self._calculate_diversity_score(candidates)
        }

    def _calculate_diversity_score(self, candidates: List[Dict]) -> float:
        """
        Calculates diversity score among candidates.

        Args:
            candidates: List of candidates

        Returns:
            Diversity score between 0.0 and 1.0
        """
        if len(candidates) < 2:
            return 0.0

        # Extract element vectors
        all_elements = set()
        for cand in candidates:
            all_elements.update(cand['elements'].keys())

        element_list = sorted(list(all_elements))
        if not element_list:
            return 0.0

        # Create presence/absence matrix as INTEGERS
        presence_matrix = np.zeros((len(candidates), len(element_list)), dtype=int)

        for i, cand in enumerate(candidates):
            for elem in cand['elements']:
                if elem in element_list:
                    j = element_list.index(elem)
                    presence_matrix[i, j] = 1  # Integer

        # Calculate diversity as average Jaccard distance
        diversity = 0.0
        count = 0

        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                # Use bitwise operators with integers
                intersection = np.sum(presence_matrix[i] & presence_matrix[j])
                union = np.sum(presence_matrix[i] | presence_matrix[j])

                if union > 0:
                    similarity = intersection / union
                    diversity += (1 - similarity)
                    count += 1

        if count > 0:
            diversity /= count

        return diversity

    def _print_summary(self, results: Dict):
        """Prints results summary."""
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)

        metrics = results['summary_metrics']
        print(f"\nTARGET: Band Gap = {results['target_bandgap']:.3f} eV")
        print(f"METRICS:")
        print(f"   - Average error: {metrics.get('mean_error', 0):.3f} eV")
        print(f"   - Minimum error: {metrics.get('min_error', 0):.3f} eV")
        print(f"   - Success rate (error <= 0.1 eV): {metrics.get('success_rate', 0)*100:.1f}%")
        print(f"   - Candidate diversity: {metrics.get('diversity_score', 0):.2f}/1.0")

        print(f"\nTOP {min(3, len(results['candidates']))} CANDIDATES:")

        for i, cand in enumerate(results['candidates'][:3]):
            formula = cand.get('formula', self._dict_to_formula(cand['elements']))
            print(f"\n   #{i+1}: {formula}")
            print(f"      Predicted: {cand['predicted_gap']:.3f} eV (Error: {cand['error']:.3f} eV)")
            print(f"      Fitness: {cand['fitness']:.3f}")
            print(f"      Stability: {cand.get('stability_score', 0):.2f}/1.0")
            print(f"      Synthesis hints: {', '.join(cand.get('synthesis_hints', ['N/A']))}")

        print(f"\nSEARCH STATISTICS:")
        print(f"   - Total predictions: {self.stats['total_predictions']}")
        print(f"   - Cache hits: {self.stats['cache_hits']}")
        print(f"   - Unique formulas: {len(self.stats['unique_formulas'])}")
        print("="*80)

    def _get_composition_hash(self, composition_dict: Dict) -> str:
        """Generates a unique hash for a composition."""
        # Sort elements alphabetically for consistency
        sorted_items = sorted(composition_dict.items(), key=lambda x: x[0])
        composition_str = "_".join([f"{elem}_{prop:.6f}" for elem, prop in sorted_items])
        return hashlib.md5(composition_str.encode()).hexdigest()[:16]

    def _dict_to_formula(self, composition_dict: Dict) -> str:
        """
        Converts composition dictionary to a valid chemical formula.
        Simplified version that avoids recursion.
        """
        if not composition_dict:
            return ""

        # First try to generate simple rational formula
        simple_formula = self._simple_rational_formula(composition_dict)

        # Check if formula is unusual
        if self._is_formula_unusual(simple_formula):
            # If unusual, try a second strategy
            # Sort elements by proportion
            sorted_elements = sorted(composition_dict.items(), key=lambda x: x[1], reverse=True)

            # Take only the top 2-3 elements
            main_elements = sorted_elements[:min(3, len(sorted_elements))]

            formula_parts = []
            for i, (elem, prop) in enumerate(main_elements):
                if prop > 0.1:
                    # Assign coefficients 1, 2, 3 according to order
                    coeff = i + 1 if i < 3 else 1
                    formula_parts.append(f"{elem}{coeff if coeff > 1 else ''}")

            if formula_parts:
                return "".join(formula_parts)

        return simple_formula

    def _is_formula_unusual(self, formula: str) -> bool:
        """Detects chemically unusual formulas."""
        import re

        # Check for very high coefficients
        if re.search(r'[A-Z][a-z]?[5-9]', formula) or re.search(r'[A-Z][a-z]?\d{2,}', formula):
            return True

        # Check for solitary elements without coefficient (only one)
        if len(formula) <= 2 and formula[0].isupper():
            return True

        # Check for repetition of the same element with different coefficients
        elements = re.findall(r'([A-Z][a-z]?)\d*', formula)
        if len(elements) != len(set(elements)):
            # Duplicate elements in the formula
            return True

        return False

    def save_results(self, results: Dict, output_dir: str = "/content/results"):
        """Saves results to JSON and CSV files."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Save complete results in JSON
        json_path = os.path.join(output_dir, f"inverse_design_{timestamp}.json")
        with open(json_path, 'w') as f:
            # Convert numpy arrays to lists for JSON
            def json_serializer(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                raise TypeError(f"Type {type(obj)} not serializable")

            json.dump(results, f, indent=2, default=json_serializer)

        # 2. Save candidates in CSV
        csv_data = []
        for cand in results['candidates']:
            csv_data.append({
                'formula': cand.get('formula', self._dict_to_formula(cand['elements'])),
                'predicted_bandgap': cand['predicted_gap'],
                'target_bandgap': results['target_bandgap'],
                'error': cand['error'],
                'fitness': cand['fitness'],
                'stability_score': cand.get('stability_score', 0),
                'elements': ",".join(cand['elements'].keys()),
                'compositions': str(cand['elements'])
            })

        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, f"candidates_{timestamp}.csv")
        df.to_csv(csv_path, index=False)

        # 3. Save evolution statistics
        if 'evolution_stats' in results:
            evolution_df = pd.DataFrame(results['evolution_stats'])
            evolution_path = os.path.join(output_dir, f"evolution_{timestamp}.csv")
            evolution_df.to_csv(evolution_path, index=False)

        print(f"\nRESULTS SAVED:")
        print(f"   - Complete JSON: {json_path}")
        print(f"   - Candidates (CSV): {csv_path}")
        if 'evolution_stats' in results:
            print(f"   - Evolution (CSV): {evolution_path}")

        return {
            'json': json_path,
            'candidates_csv': csv_path,
            'evolution_csv': evolution_path if 'evolution_stats' in results else None
        }

    def plot_results(self, results: Dict, save_path: str = None):
        """Generates visualizations of the results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Error distribution
        errors = [c['error'] for c in results['candidates']]
        axes[0, 0].hist(errors, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(results['target_bandgap'], color='red', linestyle='--', label='Target')
        axes[0, 0].set_xlabel('Error (eV)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Fitness vs Error
        fitnesses = [c['fitness'] for c in results['candidates']]
        axes[0, 1].scatter(errors, fitnesses, alpha=0.6, s=50)
        axes[0, 1].set_xlabel('Error (eV)')
        axes[0, 1].set_ylabel('Fitness')
        axes[0, 1].set_title('Fitness vs Error')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Predictions vs Target
        predicted = [c['predicted_gap'] for c in results['candidates']]
        axes[0, 2].scatter([results['target_bandgap']]*len(predicted), predicted, alpha=0.6, s=50)
        axes[0, 2].plot([0, max(predicted)], [0, max(predicted)], 'r--', alpha=0.5)
        axes[0, 2].set_xlabel('Target Band Gap (eV)')
        axes[0, 2].set_ylabel('Predicted Band Gap (eV)')
        axes[0, 2].set_title('Predictions vs Target')
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Fitness evolution (if available)
        if 'evolution_stats' in results:
            evolution = results['evolution_stats']
            generations = list(range(len(evolution)))
            best_fitness = [s['best_fitness'] for s in evolution]
            avg_fitness = [s['avg_fitness'] for s in evolution]

            axes[1, 0].plot(generations, best_fitness, 'b-', linewidth=2, label='Best')
            axes[1, 0].plot(generations, avg_fitness, 'r--', linewidth=2, label='Average')
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Fitness')
            axes[1, 0].set_title('Fitness Evolution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Most frequent elements
        element_counts = Counter()
        for cand in results['candidates']:
            for elem in cand['elements']:
                element_counts[elem] += 1

        if element_counts:
            top_elements = element_counts.most_common(10)
            elements, counts = zip(*top_elements)
            axes[1, 1].bar(elements, counts, alpha=0.7)
            axes[1, 1].set_xlabel('Element')
            axes[1, 1].set_ylabel('Frequency in Candidates')
            axes[1, 1].set_title('Most Frequent Elements')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Stability vs Error
        stability_scores = [c.get('stability_score', 0.5) for c in results['candidates']]
        axes[1, 2].scatter(errors, stability_scores, alpha=0.6, s=50, c=fitnesses, cmap='viridis')
        axes[1, 2].set_xlabel('Error (eV)')
        axes[1, 2].set_ylabel('Stability Score')
        axes[1, 2].set_title('Stability vs Error (colored by fitness)')
        axes[1, 2].grid(True, alpha=0.3)
        plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2], label='Fitness')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

# =============================================================================
# MAIN EXECUTION SCRIPT
# =============================================================================

def main():
    """
    Main function to execute the inverse design.
    """
    print("="*80)
    print("INVERSE MATERIALS DESIGN - MAIN EXECUTION")
    print("="*80)

    # CONFIGURABLE PARAMETERS
    CONFIG = {
        # 1. PATHS (adjust according to your files)
        'model_path': '/content/bandgap_nn_model.keras',  # Your base NN model
        'scaler_path': '/content/scaler.pkl',             # Your scaler

        # 2. DESIGN PARAMETERS
        'target_gap': 2.5,           # Target band gap in eV
        'material_type': 'semiconductor',  # Optional: 'metal', 'semiconductor', 'insulator', 'narrow_gap_semiconductor'
        'n_candidates': 10,          # Number of candidates to generate

        # 3. ADVANCED PARAMETERS
        'max_elements': 4,           # Maximum elements per composition
        'device': 'auto',            # 'auto', 'cpu', or 'gpu'
        'save_results': True,        # Save results to files
        'plot_results': True,        # Generate plots
    }

    # ============================
    # INVERSE DESIGN EXECUTION
    # ============================

    # 1. INITIALIZE DESIGNER
    print("\n1. INITIALIZING INVERSE DESIGNER...")
    designer = InverseBandgapDesigner(
        model_path=CONFIG['model_path'],
        scaler_path=CONFIG['scaler_path'],
        validation_data_path='/content/validation_balanced_stratified.csv',
        max_elements=CONFIG['max_elements'],
        device=CONFIG['device']
    )

    # 2. EXECUTE DESIGN
    print(f"\n2. EXECUTING DESIGN FOR BAND GAP = {CONFIG['target_gap']} eV...")
    results = designer.design_materials(
        target_gap=CONFIG['target_gap'],
        material_type=CONFIG['material_type'],
        n_candidates=CONFIG['n_candidates'],
        show_progress=True
    )

    # 3. SHOW DETAILED RESULTS
    print(f"\n3. DETAILED RESULTS:")
    print("-"*60)

    for i, cand in enumerate(results['candidates']):
        formula = cand.get('formula', designer._dict_to_formula(cand['elements']))
        print(f"\nCANDIDATE #{i+1}:")
        print(f"   Formula: {formula}")
        print(f"   Predicted Band Gap: {cand['predicted_gap']:.3f} eV")
        print(f"   Error: {cand['error']:.3f} eV")
        print(f"   Fitness: {cand['fitness']:.3f}")
        print(f"   Stability: {cand.get('stability_score', 0):.2f}/1.0")
        print(f"   Elements: {', '.join(cand['elements'].keys())}")

        if 'synthesis_hints' in cand and cand['synthesis_hints']:
            print(f"   Synthesis hints:")
            for hint in cand['synthesis_hints']:
                print(f"      - {hint}")

    # 4. SAVE RESULTS
    if CONFIG['save_results']:
        print(f"\n4. SAVING RESULTS...")
        saved_files = designer.save_results(results, output_dir="/content/inverse_design_results")

        # Create configuration file
        config_path = "/content/inverse_design_results/config.json"
        with open(config_path, 'w') as f:
            json.dump(CONFIG, f, indent=2)
        print(f"   Configuration: {config_path}")

    # 5. VISUALIZATIONS
    if CONFIG['plot_results']:
        print(f"\n5. GENERATING VISUALIZATIONS...")
        designer.plot_results(results)

        # Plot genetic algorithm evolution
        if hasattr(designer, 'ga'):
            designer.ga.plot_evolution()

    # 6. FINAL RECOMMENDATIONS
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS:")
    print("="*80)

    if results['candidates']:
        best = results['candidates'][0]
        formula = best.get('formula', designer._dict_to_formula(best['elements']))

        print(f"\nPRIORITY CANDIDATE FOR SYNTHESIS:")
        print(f"   Formula: {formula}")
        print(f"   Error: {best['error']:.3f} eV")
        print(f"   Estimated stability: {best.get('stability_score', 0):.2f}/1.0")

        print(f"\nRECOMMENDED STEPS:")
        print(f"   1. Validate thermodynamic stability with DFT")
        print(f"   2. Optimize synthesis parameters")
        print(f"   3. Synthesize and characterize experimentally")
        print(f"   4. Measure experimental band gap for validation")

        print(f"\nSYNTHESIS SUGGESTIONS:")
        if 'synthesis_hints' in best:
            for j, hint in enumerate(best['synthesis_hints'], 1):
                print(f"   {j}. {hint}")
    else:
        print("No satisfactory candidates found.")
        print("   Consider adjusting search parameters.")

    print("\n" + "="*80)
    print("INVERSE DESIGN COMPLETED SUCCESSFULLY")
    print("="*80)

    return designer, results


# =============================================================================
# EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    try:
        designer, results = main()

        # Optional: Run additional test cases.
        print("\nADDITIONAL TEST CASES:")
        print("="*60)

        test_cases = [
            # (0.0, "metal"),
            # (0.05, "semimetal"),
            (0.8, "narrow_gap")
            # (1.8, "semiconductor"),
            # (4.0, "insulator")
        ]

        for target_gap, mat_type in test_cases:
            print(f"\nTesting: {target_gap} eV ({mat_type})...")
            try:
                test_results = designer.design_materials(
                    target_gap=target_gap,
                    material_type=mat_type,
                    n_candidates=3,
                    show_progress=False
                )

                if test_results['candidates']:
                    best = test_results['candidates'][0]
                    formula = best.get('formula', designer._dict_to_formula(best['elements']))
                    print(f"   Best candidate: {formula} (target error: {best['error']:.3f} eV)")

                    # Validation against experimental dataset
                    val_info = designer.validate_candidate_against_experimental(best['elements'])
                    if val_info['in_validation']:
                        print(f"      Found in validation dataset: {val_info['formula']} "
                              f"exp={val_info['experimental_bg']:.3f} eV, "
                              f"pred={val_info['predicted_bg']:.3f} eV, "
                              f"error vs exp={val_info['error']:.3f} eV")
                    else:
                        print(f"      Not found in validation dataset (possible new material)")
                else:
                    print("   No candidates found")

            except Exception as e:
                print(f"   Error: {str(e)}")
                continue

    except Exception as e:
        print(f"\nCRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

# =============================================================================
# ADVANCED ANALYSIS FUNCTIONS
# =============================================================================

def analyze_candidate_stability(designer: InverseBandgapDesigner,
                               candidate: Dict) -> Dict:
    """
    Advanced stability analysis for a candidate.
    """
    elements = candidate['elements']
    formula = candidate.get('formula', designer._dict_to_formula(elements))

    print(f"\nADVANCED ANALYSIS FOR: {formula}")
    print("-"*50)

    analysis = {
        'formula': formula,
        'predicted_gap': candidate['predicted_gap'],
        'error': candidate['error'],
        'stability_analysis': {},
        'synthesis_recommendations': []
    }

    # 1. Electronegativity analysis
    electronegativities = []
    atomic_radii = []

    for elem in elements:
        try:
            el = Element(elem)
            electronegativities.append(el.X)
            atomic_radii.append(el.atomic_radius)
        except Exception:
            continue

    if electronegativities:
        en_diff = max(electronegativities) - min(electronegativities)
        analysis['stability_analysis']['electronegativity'] = {
            'range': (min(electronegativities), max(electronegativities)),
            'difference': en_diff,
            'bond_type': 'ionic' if en_diff > 1.7 else 'covalent'
        }
        print(f"   Electronegativity: {en_diff:.2f} ({analysis['stability_analysis']['electronegativity']['bond_type']})")

    # 2. Atomic radius analysis
    if atomic_radii:
        radius_ratio = min(atomic_radii) / max(atomic_radii) if max(atomic_radii) > 0 else 0
        analysis['stability_analysis']['atomic_radii'] = {
            'min_radius': min(atomic_radii),
            'max_radius': max(atomic_radii),
            'ratio': radius_ratio
        }
        print(f"   Minimum atomic radius: {min(atomic_radii):.2f} pm")
        print(f"   Maximum atomic radius: {max(atomic_radii):.2f} pm")

    # 3. Check known databases
    analysis['stability_analysis']['known_analogs'] = find_similar_compounds(formula)

    # 4. Specific recommendations
    if 'O' in elements and elements['O'] > 0.5:
        analysis['synthesis_recommendations'].append(
            "Synthesis by oxidation: calcination at 800-1200 C for 4-12 hours"
        )

    if len(elements) == 2:
        analysis['synthesis_recommendations'].append(
            "Direct fusion: mix powders in stoichiometric ratio and melt"
        )

    if any(elem in ['Ti', 'Zr', 'Nb', 'Ta', 'W'] for elem in elements):
        analysis['synthesis_recommendations'].append(
            "Refractory material: requires high-temperature furnace (>1500 C)"
        )

    return analysis


def find_similar_compounds(formula: str) -> List[str]:
    """
    Searches for similar compounds in a local database (simulated).
    In production, connect to the Materials Project API.
    """
    # Simulated database of known compounds
    known_compounds = {
        'TiO2': 'Rutile/anatase - Semiconductor ~3.0-3.2 eV',
        'SiO2': 'Quartz - Insulator ~9.0 eV',
        'GaAs': 'III-V Semiconductor ~1.42 eV',
        'Si': 'Elemental semiconductor ~1.12 eV',
        'ZnO': 'Semiconductor ~3.37 eV',
        'Al2O3': 'Corundum - Insulator ~8.8 eV',
        'Fe2O3': 'Hematite - Semiconductor ~2.2 eV',
        'CuO': 'Semiconductor ~1.2-1.9 eV',
        'PbS': 'Galena - Narrow-gap semiconductor ~0.41 eV',
        'CdTe': 'Semiconductor ~1.44 eV'
    }

    # Search by common elements
    composition = Composition(formula)
    elements = set([str(e) for e in composition.elements])

    similar = []
    for known_formula, description in known_compounds.items():
        known_comp = Composition(known_formula)
        known_elements = set([str(e) for e in known_comp.elements])

        # If it shares at least 50% of the elements
        common_elements = elements.intersection(known_elements)
        if common_elements and len(common_elements) >= len(elements) * 0.5:
            similar.append(f"{known_formula}: {description}")

    return similar[:3]  # Maximum 3 similar compounds


def batch_design_study(designer: InverseBandgapDesigner,
                      target_gaps: List[float],
                      output_dir: str = "/content/batch_study"):
    """
    Executes inverse design for multiple target band gaps.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nBATCH STUDY - {len(target_gaps)} TARGETS")
    print("="*60)

    all_results = []

    for target_gap in target_gaps:
        print(f"\nDesigning for {target_gap} eV...")

        try:
            results = designer.design_materials(
                target_gap=target_gap,
                n_candidates=5,
                show_progress=False
            )

            all_results.append({
                'target_gap': target_gap,
                'results': results,
                'best_candidate': results['candidates'][0] if results['candidates'] else None
            })

            # Save individual results
            gap_str = f"{target_gap:.2f}".replace('.', 'p')
            designer.save_results(
                results,
                output_dir=os.path.join(output_dir, f"gap_{gap_str}")
            )

        except Exception as e:
            print(f"   Error: {str(e)}")
            continue

    # Comparative analysis
    print(f"\nCOMPARATIVE ANALYSIS:")
    print("-"*60)

    comparison_data = []
    for study in all_results:
        if study['best_candidate']:
            cand = study['best_candidate']
            comparison_data.append({
                'target': study['target_gap'],
                'predicted': cand['predicted_gap'],
                'error': cand['error'],
                'formula': cand.get('formula', designer._dict_to_formula(cand['elements'])),
                'fitness': cand['fitness']
            })

    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))

        # Save comparison
        csv_path = os.path.join(output_dir, "comparison.csv")
        df_comparison.to_csv(csv_path, index=False)
        print(f"\nComparison saved in: {csv_path}")

    return all_results

# =============================================================================
# USAGE EXAMPLE
# =============================================================================
"""
USAGE EXAMPLE STEP BY STEP FOR GOOGLE COLAB

1. Upload your model and scaler files to /content/
2. Execute all previous cells
3. Execute this section to test the system
"""

# Configuration for quick test
TEST_CONFIG = {
    'target_gap': 1.8,  # eV - typical for visible solar cells
    'material_type': 'semiconductor',
    'n_candidates': 2,
    'quick_test': True  # Reduce parameters for quick test
}

print("EXECUTING SYSTEM QUICK TEST")
print("="*60)

# 1. Initialize designer (will use default paths)
print("\n1. Initializing designer...")
designer = InverseBandgapDesigner(
    max_elements=3,  # Reduced for quick test
    device='auto',
    validation_data_path='/content/validation_balanced_stratified.csv'  # Adjust path if necessary
)

# 2. Test prediction for a known compound
print("\n2. Testing predictions for known compounds:")
known_compounds = [
    ('Si', 1.12),
    ('GaAs', 1.42),
    ('ZnO', 3.37),
    ('TiO2', 3.0),
    ('SiO2', 9.0),
    ('Cu2O', 2.0),   # Semiconductor
    ('InN', 0.7),     # Narrow-gap semiconductor
    ('PbS', 0.41),    # Narrow-gap semiconductor
    ('MgO', 7.8),     # Insulator
    ('Fe2O3', 2.2),   # Semiconductor
]
for formula, exp_gap in known_compounds:
    pred = designer.predict_formula(formula)
    if pred is not None:
        error = abs(pred - exp_gap)
        print(f"   {formula:6s}: pred={pred:.2f} eV, exp={exp_gap:.2f} eV, error={error:.2f} eV")
    else:
        print(f"   {formula:6s}: prediction error")


# 3. Execute inverse design
print(f"\n3. Designing materials for {TEST_CONFIG['target_gap']} eV...")
results = designer.design_materials(
    target_gap=TEST_CONFIG['target_gap'],
    material_type=TEST_CONFIG['material_type'],
    n_candidates=TEST_CONFIG['n_candidates'],
    show_progress=True
)

# 4. Show results and validate against experimental dataset
print("\n4. Results obtained:")
for i, cand in enumerate(results['candidates']):
    formula = cand.get('formula', designer._dict_to_formula(cand['elements']))
    print(f"   {i+1}. {formula}: {cand['predicted_gap']:.3f} eV (target error: {cand['error']:.3f} eV)")

    # Validation
    val_info = designer.validate_candidate_against_experimental(cand['elements'])
    if val_info['in_validation']:
        print(f"      Found in validation: {val_info['formula']} "
              f"exp={val_info['experimental_bg']:.3f} eV, "
              f"error vs exp={val_info['error']:.3f} eV")
    else:
        print(f"      Not found in validation dataset (possible new material)")

# 5. Save results
print("\n5. Saving results...")
saved_files = designer.save_results(results, output_dir="/content/test_results")
print(f"   Results saved in: {saved_files['json']}")

print("\n" + "="*60)
print("TEST COMPLETED SUCCESSFULLY")
print("="*60)

# Show final statistics
print(f"\nFINAL STATISTICS:")
print(f"   - Total predictions: {designer.stats['total_predictions']}")
print(f"   - Cache hits: {designer.stats['cache_hits']}")
print(f"   - Cache efficiency: {designer.stats['cache_hits']/max(1, designer.stats['total_predictions'])*100:.1f}%")
print(f"   - Unique formulas generated: {len(designer.stats['unique_formulas'])}")
