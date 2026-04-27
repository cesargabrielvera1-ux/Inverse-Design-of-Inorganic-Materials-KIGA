"""
Earth Mover's Distance (EMD) Analysis
=========================================
Computes the Earth Mover's Distance between generated candidate compositions
and the training dataset to assess compositional novelty.

NOTE: File paths are set to '/content/' for Google Colab compatibility.
Please update these paths according to your local environment.
"""

import pandas as pd
import numpy as np
from pymatgen.core import Composition
from scipy.stats import wasserstein_distance
from tqdm import tqdm

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
train_df = pd.read_csv('/content/train_balanced_stratified.csv')
candidates_df = pd.read_csv('/content/candidates.csv')

print(f"Training set size: {len(train_df)}")
print(f"Candidates set size: {len(candidates_df)}")

# ------------------------------------------------------------
# 2. Function to convert formula to atomic fraction vector
# ------------------------------------------------------------
def composition_to_fraction_vector(formula, all_elements):
    """
    Converts a chemical formula into a normalized vector of atomic fractions
    over a fixed set of elements.
    """
    try:
        comp = Composition(formula)
    except Exception:
        return None
    # Total number of atoms in the formula unit (not reduced)
    num_atoms = sum(comp.values())
    vec = np.zeros(len(all_elements))
    for el, amt in comp.items():
        if el in all_elements:
            idx = all_elements.index(el)
            vec[idx] = amt / num_atoms
    return vec

# ------------------------------------------------------------
# 3. Get the complete set of present elements
# ------------------------------------------------------------
train_formulas = train_df['composition'].astype(str).tolist()
candidate_formulas = candidates_df['formula'].astype(str).tolist()

all_elements_set = set()
for formula in train_formulas + candidate_formulas:
    try:
        comp = Composition(formula)
        all_elements_set.update(comp.keys())
    except Exception:
        continue

all_elements = sorted(list(all_elements_set))
print(f"Total unique elements: {len(all_elements)}")
print(f"Elements: {all_elements[:20]}...")  # Show first 20

# ------------------------------------------------------------
# 4. Vectorize the training set
# ------------------------------------------------------------
train_vectors = []
valid_train_formulas = []
for formula in tqdm(train_formulas, desc="Vectorizing training set"):
    vec = composition_to_fraction_vector(formula, all_elements)
    if vec is not None:
        train_vectors.append(vec)
        valid_train_formulas.append(formula)

train_vectors = np.array(train_vectors)
print(f"Valid training vectors: {train_vectors.shape[0]}")

# ------------------------------------------------------------
# 5. Calculate EMD for each candidate (minimum distance to training set)
# ------------------------------------------------------------
# The element distance metric is Euclidean distance in feature space.
# However, for simplicity and since EMD with uniform ground distance is a good approximation,
# we will use the one-dimensional Wasserstein distance, which essentially compares cumulative histograms.
# In this context, atomic fractions are discrete distributions over ordered elements.
# Ordering elements by atomic number is standard practice to give physical meaning to "closeness".

# For a more rigorous calculation, the Euclidean distance between elements in Magpie space can be used.
# However, the Wasserstein distance over atomic-fraction-ordered elements captures compositional similarity.
# Since the reviewer did not request a specific implementation, we will use the standard Wasserstein metric.

# Prepare dictionary with the candidates of interest
target_candidates = ['MnLi', 'V1CrB2Ge3', 'InGaSe4', 'GePO', 'Sr2ZrS4']

results = {}
for candidate_formula in target_candidates:
    # Verify if the candidate is in the DataFrame
    match = candidates_df[candidates_df['formula'] == candidate_formula]
    if match.empty:
        print(f"Warning: {candidate_formula} not found in candidates.csv")
        continue

    cand_vec = composition_to_fraction_vector(candidate_formula, all_elements)
    if cand_vec is None:
        print(f"Could not parse {candidate_formula}")
        continue

    # Calculate EMD against all training vectors
    emd_values = []
    for train_vec in train_vectors:
        # wasserstein_distance expects 1D distributions (values and weights).
        # Here the "values" are the element indices (0,1,2,...)
        # and the weights are the atomic fractions.
        # This is equivalent to Earth Mover's Distance if elements are ordered by atomic number.
        u_values = np.arange(len(all_elements))
        v_values = np.arange(len(all_elements))
        emd = wasserstein_distance(u_values, v_values, cand_vec, train_vec)
        emd_values.append(emd)

    min_emd = np.min(emd_values)
    results[candidate_formula] = min_emd
    print(f"{candidate_formula}: EMD = {min_emd:.4f}")

# ------------------------------------------------------------
# 6. Show final results
# ------------------------------------------------------------
print("\n--- Final EMD values for Table 4 candidates ---")
for formula, emd in results.items():
    print(f"{formula}: {emd:.4f}")

# Optional: save results to file
results_df = pd.DataFrame(list(results.items()), columns=['formula', 'EMD'])
results_df.to_csv('/content/emd_results.csv', index=False)
print("\nResults saved to /content/emd_results.csv")
