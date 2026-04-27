"""
External Validation Suite
==========================
Set of scripts for additional external validation:
1. Compare generated candidates with the experimental validation dataset.
2. Verify candidates against the Materials Project API.

NOTE: File paths are set to '/content/' for Google Colab compatibility.
Please update these paths according to your local environment.
"""

import pandas as pd
import numpy as np
from pymatgen.core import Composition

# =============================================================================
# PART 1: COMPARE CANDIDATES WITH EXPERIMENTAL DATASET
# =============================================================================
"""
This script compares the formulas generated in candidates.csv
with the experimental validation dataset (validation_balanced_stratified.csv)
and shows the matches found along with error statistics.
"""

# -------------------------------------------
# CONFIGURATION - UPDATE PATHS IF NECESSARY
# -------------------------------------------
CANDIDATES_FILE = '/content/candidates.csv'                # File with generated candidates
VALIDATION_FILE = '/content/validation_balanced_stratified.csv'  # Validation dataset

# -------------------------------------------
# LOAD FILES
# -------------------------------------------
print("Loading files...")
df_candidates = pd.read_csv(CANDIDATES_FILE)
df_val = pd.read_csv(VALIDATION_FILE)

# Identify formula column in validation
if 'composition' in df_val.columns:
    formula_col_val = 'composition'
elif 'final_composition' in df_val.columns:
    formula_col_val = 'final_composition'
else:
    raise ValueError("No formula column found in the validation dataset.")

# Identify band gap column in validation
if 'gap expt' in df_val.columns:
    gap_col_val = 'gap expt'
else:
    raise ValueError("No 'gap expt' column found in the validation dataset.")

print(f"Candidates: {len(df_candidates)} records")
print(f"Validation: {len(df_val)} records")

# -------------------------------------------
# BUILD EXPERIMENTAL VALUES DICTIONARY (reduced formula -> band gap)
# -------------------------------------------
exp_dict = {}
parse_errors = 0
for idx, row in df_val.iterrows():
    try:
        comp = Composition(row[formula_col_val])
        reduced = comp.reduced_formula
        exp_dict[reduced] = row[gap_col_val]
    except Exception:
        parse_errors += 1
        continue

print(f"Unparseable formulas in validation: {parse_errors}")

# -------------------------------------------
# SEARCH FOR MATCHES IN CANDIDATES
# -------------------------------------------
matches = []
for idx, row in df_candidates.iterrows():
    formula_raw = row['formula']
    try:
        comp = Composition(formula_raw)
        reduced = comp.reduced_formula
        if reduced in exp_dict:
            exp_bg = exp_dict[reduced]
            pred_bg = row['predicted_bandgap']
            error = abs(pred_bg - exp_bg)
            matches.append({
                'formula': reduced,
                'predicted (eV)': pred_bg,
                'experimental (eV)': exp_bg,
                'error (eV)': error,
                'fitness': row.get('fitness', np.nan),
                'stability': row.get('stability_score', np.nan)
            })
    except Exception:
        continue

# -------------------------------------------
# SHOW RESULTS
# -------------------------------------------
print("\n" + "="*80)
print("COMPARISON WITH EXPERIMENTAL VALIDATION DATASET")
print("="*80)

if matches:
    df_matches = pd.DataFrame(matches)
    # Sort by ascending error
    df_matches = df_matches.sort_values('error (eV)')

    # Show table
    print(df_matches.to_string(index=False))

    # Statistics
    print("\nERROR STATISTICS:")
    print(f"   - Number of matches: {len(df_matches)}")
    print(f"   - MAE: {df_matches['error (eV)'].mean():.4f} eV")
    print(f"   - RMSE: {np.sqrt((df_matches['error (eV)']**2).mean()):.4f} eV")
    print(f"   - Maximum error: {df_matches['error (eV)'].max():.4f} eV")
    print(f"   - Minimum error: {df_matches['error (eV)'].min():.4f} eV")

    # Optional: save comparison results
    output_file = '/content/comparacion_validacion.csv'
    df_matches.to_csv(output_file, index=False)
    print(f"\nResults saved in: {output_file}")
else:
    print("No matches found with the validation dataset.")
    print("   Possible causes:")
    print("   - The generated candidates are all new (not in the dataset).")
    print("   - The formulas in candidates.csv could not be parsed correctly.")
    print("   - The validation dataset does not contain those compounds.")


# =============================================================================
# PART 2: VERIFICATION WITH MATERIALS PROJECT API
# =============================================================================
"""
This section queries the Materials Project API to compare predicted band gaps
with DFT-calculated values from Materials Project.

Requires an MP API key. Set your API key below.
"""

# Install mp-api if not already installed
import subprocess
import sys

try:
    from mp_api.client import MPRester
except ImportError:
    print("Installing mp-api...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mp-api"])
    from mp_api.client import MPRester

import time

# -------------------------------------------
# CONFIGURATION
# -------------------------------------------
API_KEY = "YOUR_API_KEY_HERE"  # <-- REPLACE WITH YOUR MATERIALS PROJECT API KEY
CANDIDATES_FILE_MP = '/content/candidates.csv'   # File with generated candidates
OUTPUT_FILE_MP = '/content/comparacion_mp.csv'   # Output file

# -------------------------------------------
# LOAD CANDIDATES
# -------------------------------------------
print("\nLoading candidates for Materials Project verification...")
df_cand = pd.read_csv(CANDIDATES_FILE_MP)
print(f"Total candidates: {len(df_cand)}")


def get_reduced_formula(formula_str):
    try:
        comp = Composition(formula_str)
        return comp.reduced_formula
    except Exception:
        return None


df_cand['reduced_formula'] = df_cand['formula'].apply(get_reduced_formula)
df_cand = df_cand.dropna(subset=['reduced_formula'])
print(f"After parsing: {len(df_cand)} valid formulas")

# -------------------------------------------
# QUERY MATERIALS PROJECT API (using summary)
# -------------------------------------------
print("\nQuerying Materials Project API (summary)...")
print("   (This may take a few seconds per candidate)")

results_mp = []
with MPRester(API_KEY) as mpr:
    for idx, row in df_cand.iterrows():
        formula = row['reduced_formula']
        pred_bg = row['predicted_bandgap']
        print(f"   Searching {formula}...", end='')

        try:
            # Use summary.search which includes band_gap
            docs = mpr.summary.search(
                formula=formula,
                fields=['material_id', 'formula_pretty', 'band_gap']
            )

            if docs:
                doc = docs[0]
                mp_bg = doc.band_gap
                mp_id = doc.material_id
                mp_formula = doc.formula_pretty
                error = abs(pred_bg - mp_bg) if mp_bg is not None else np.nan

                results_mp.append({
                    'formula': formula,
                    'predicted (eV)': pred_bg,
                    'MP_bandgap (eV)': mp_bg,
                    'error (eV)': error,
                    'MP_id': mp_id,
                    'MP_formula': mp_formula
                })
                print(f" found (gap={mp_bg:.3f} eV)")
            else:
                results_mp.append({
                    'formula': formula,
                    'predicted (eV)': pred_bg,
                    'MP_bandgap (eV)': np.nan,
                    'error (eV)': np.nan,
                    'MP_id': None,
                    'MP_formula': None
                })
                print(f" not found")
        except Exception as e:
            print(f" error: {str(e)}")
            results_mp.append({
                'formula': formula,
                'predicted (eV)': pred_bg,
                'MP_bandgap (eV)': np.nan,
                'error (eV)': np.nan,
                'MP_id': None,
                'MP_formula': None
            })

        # Small pause to avoid saturating the API
        time.sleep(0.5)

# -------------------------------------------
# CREATE RESULTS DATAFRAME
# -------------------------------------------
df_results_mp = pd.DataFrame(results_mp)
df_found_mp = df_results_mp.dropna(subset=['MP_bandgap (eV)'])

print("\n" + "="*80)
print("MATERIALS PROJECT VERIFICATION RESULTS")
print("="*80)

if len(df_found_mp) > 0:
    print(f"\nMatches found: {len(df_found_mp)} out of {len(df_cand)}")
    print("\nResults table (first 10 rows):")
    print(df_found_mp.head(10).to_string(index=False))

    print("\nERROR STATISTICS (vs MP):")
    print(f"   MAE: {df_found_mp['error (eV)'].mean():.4f} eV")
    print(f"   RMSE: {np.sqrt((df_found_mp['error (eV)']**2).mean()):.4f} eV")
    print(f"   Maximum error: {df_found_mp['error (eV)'].max():.4f} eV")
    print(f"   Minimum error: {df_found_mp['error (eV)'].min():.4f} eV")
else:
    print("No matches found in Materials Project.")

# -------------------------------------------
# SAVE RESULTS
# -------------------------------------------
df_results_mp.to_csv(OUTPUT_FILE_MP, index=False)
print(f"\nComplete results saved in: {OUTPUT_FILE_MP}")

# Optional: show summary of not found candidates
df_not_found_mp = df_results_mp[df_results_mp['MP_id'].isna()]
if len(df_not_found_mp) > 0:
    print(f"\nCandidates not found in MP: {len(df_not_found_mp)}")
    print(df_not_found_mp[['formula', 'predicted (eV)']].head(10).to_string(index=False))
    if len(df_not_found_mp) > 10:
        print(f"   ... and {len(df_not_found_mp)-10} more")
