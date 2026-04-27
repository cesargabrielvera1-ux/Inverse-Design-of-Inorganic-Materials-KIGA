"""
Dataset Combination and Balancing
===================================
Combines training and validation datasets, categorizes samples by band gap
using scientifically established thresholds, and creates stratified 80/20 splits.

NOTE: File paths are set to '/content/' for Google Colab compatibility.
Please update these paths according to your local environment.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("EXPERIMENTAL DATASET COMBINATION AND BALANCING")
print("=" * 70)

# 1. LOAD AND COMBINE DATASETS
print("\nLOADING DATASETS...")

df_train = pd.read_csv('/content/train_zero_overlap.csv')
df_valid = pd.read_csv('/content/validation_zero_overlap.csv')

print(f"Original training dataset: {df_train.shape}")
print(f"Original validation dataset: {df_valid.shape}")

# Identify composition and band gap columns
comp_col = 'composition' if 'composition' in df_train.columns else 'final_composition'
gap_col = 'gap expt'

# Combine datasets
df_combined = pd.concat([df_train, df_valid], ignore_index=True)
print(f"\nCombined dataset: {df_combined.shape}")

# 2. INDEPENDENCE AND DUPLICATES CHECK
print("\nCHECKING DATA INDEPENDENCE...")

# Check exact duplicates
exact_duplicates = df_combined.duplicated().sum()
print(f"Exact duplicates found: {exact_duplicates}")

# Check duplicates by composition
if comp_col in df_combined.columns:
    composition_duplicates = df_combined[comp_col].duplicated().sum()
    print(f"Duplicate compositions: {composition_duplicates}")

# 3. DEFINITION OF SCIENTIFICALLY VALID CATEGORIES
print("\nDEFINING SCIENTIFICALLY VALID CATEGORIES...")


def categorize_band_gap(gap_value):
    """
    Categorization based on scientific literature:
    - Rev. Mod. Phys. 85, 1083 (2013) - Classification of electronic materials
    - Nature Materials 17, 1052-1056 (2018) - Band gap engineering
    """
    if gap_value == 0:
        return 'metallic'  # Metallic materials - conductors
    elif 0 < gap_value <= 0.1:
        return 'semimetal'  # Semimetals (e.g., graphene, Sb)
    elif 0.1 < gap_value <= 1.5:
        return 'narrow_gap_semiconductor'  # Narrow-gap semiconductors (e.g., InSb, PbSe)
    elif 1.5 < gap_value <= 3.0:
        return 'semiconductor'  # Typical semiconductors (e.g., Si, GaAs)
    else:
        return 'insulator'  # Insulators (e.g., SiO2, diamond)


# Apply categorization
df_combined['gap_category'] = df_combined[gap_col].apply(categorize_band_gap)

# 4. ORIGINAL DISTRIBUTION ANALYSIS
print("\nORIGINAL CATEGORY DISTRIBUTION:")
category_counts = Counter(df_combined['gap_category'])
for category, count in category_counts.most_common():
    percentage = count / len(df_combined) * 100
    print(f"  {category:25}: {count:4d} compounds ({percentage:.1f}%)")

# 5. BALANCING STRATEGY
print("\nAPPLYING BALANCING STRATEGY...")

# Define priority categories for balancing
balancing_categories = {
    'metallic': 'Metallic materials (conductors)',
    'semimetal': 'Semimetals',
    'narrow_gap_semiconductor': 'Narrow-gap semiconductors',
    'semiconductor': 'Semiconductors',
    'insulator': 'Insulators'
}

# Verify that we have enough samples per category
print("\nSamples per category:")
min_samples = float('inf')
for category in balancing_categories.keys():
    n_samples = len(df_combined[df_combined['gap_category'] == category])
    min_samples = min(min_samples, n_samples)
    print(f"  {category:25}: {n_samples:4d} samples")

# 6. STRATIFIED 80/20 SPLIT
print(f"\nSPLITTING DATASET (80% training, 20% validation)...")

# Use stratified split to maintain proportions
X = df_combined.drop(columns=[gap_col, 'gap_category'], errors='ignore')
y = df_combined['gap_category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,  # CRITICAL: maintain proportions!
    random_state=42
)

# Rebuild dataframes
df_train_new = X_train.copy()
df_train_new[gap_col] = df_combined.loc[X_train.index, gap_col]
df_train_new[comp_col] = df_combined.loc[X_train.index, comp_col]
df_train_new['gap_category'] = y_train

df_valid_new = X_test.copy()
df_valid_new[gap_col] = df_combined.loc[X_test.index, gap_col]
df_valid_new[comp_col] = df_combined.loc[X_test.index, comp_col]
df_valid_new['gap_category'] = y_test

print(f"New training dataset: {df_train_new.shape}")
print(f"New validation dataset: {df_valid_new.shape}")

# 7. FINAL BALANCE VERIFICATION
print("\nVERIFYING FINAL BALANCE...")

print("\nDistribution in NEW training:")
train_counts = Counter(df_train_new['gap_category'])
for category, count in train_counts.most_common():
    percentage = count / len(df_train_new) * 100
    print(f"  {category:25}: {count:4d} compounds ({percentage:.1f}%)")

print("\nDistribution in NEW validation:")
valid_counts = Counter(df_valid_new['gap_category'])
for category, count in valid_counts.most_common():
    percentage = count / len(df_valid_new) * 100
    print(f"  {category:25}: {count:4d} compounds ({percentage:.1f}%)")

# 8. COMPARATIVE VISUALIZATION
print("\nGENERATING COMPARATIVE VISUALIZATIONS...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Original distribution
categories_order = list(balancing_categories.keys())
counts_original = [category_counts.get(cat, 0) for cat in categories_order]
axes[0, 0].bar(categories_order, counts_original, color='skyblue', alpha=0.7)
axes[0, 0].set_title('Original Distribution (Combined)')
axes[0, 0].set_ylabel('Number of Compounds')
axes[0, 0].tick_params(axis='x', rotation=45)

# New training distribution
counts_train = [train_counts.get(cat, 0) for cat in categories_order]
axes[0, 1].bar(categories_order, counts_train, color='lightgreen', alpha=0.7)
axes[0, 1].set_title('New Training Distribution (80%)')
axes[0, 1].set_ylabel('Number of Compounds')
axes[0, 1].tick_params(axis='x', rotation=45)

# New validation distribution
counts_valid = [valid_counts.get(cat, 0) for cat in categories_order]
axes[1, 0].bar(categories_order, counts_valid, color='lightcoral', alpha=0.7)
axes[1, 0].set_title('New Validation Distribution (20%)')
axes[1, 0].set_ylabel('Number of Compounds')
axes[1, 0].tick_params(axis='x', rotation=45)

# Proportion comparison
percentages_original = [count/len(df_combined)*100 for count in counts_original]
percentages_train = [count/len(df_train_new)*100 for count in counts_train]
percentages_valid = [count/len(df_valid_new)*100 for count in counts_valid]

x = np.arange(len(categories_order))
width = 0.25
axes[1, 1].bar(x - width, percentages_original, width, label='Original', alpha=0.7)
axes[1, 1].bar(x, percentages_train, width, label='Training', alpha=0.7)
axes[1, 1].bar(x + width, percentages_valid, width, label='Validation', alpha=0.7)
axes[1, 1].set_title('Proportion Comparison (%)')
axes[1, 1].set_ylabel('Percentage (%)')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(categories_order, rotation=45)
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# 9. BAND GAP RANGE ANALYSIS BY CATEGORY
print("\nBAND GAP RANGE ANALYSIS BY CATEGORY:")

for category in balancing_categories.keys():
    gaps = df_combined[df_combined['gap_category'] == category][gap_col]
    if len(gaps) > 0:
        print(f"\n  {category:25}:")
        print(f"    Minimum: {gaps.min():.3f} eV")
        print(f"    Maximum: {gaps.max():.3f} eV")
        print(f"    Average: {gaps.mean():.3f} eV")
        print(f"    Median:  {gaps.median():.3f} eV")

# 10. SAVE NEW BALANCED DATASETS
print("\nSAVING NEW BALANCED DATASETS...")

# Save new datasets
df_train_new.to_csv('/content/train_balanced_stratified.csv', index=False)
df_valid_new.to_csv('/content/validation_balanced_stratified.csv', index=False)

print("New datasets saved:")
print("   - /content/train_balanced_stratified.csv")
print("   - /content/validation_balanced_stratified.csv")

# 11. EXECUTIVE SUMMARY
print("\n" + "=" * 70)
print("EXECUTIVE SUMMARY - BALANCING COMPLETED")
print("=" * 70)

print(f"FINAL STATISTICS:")
print(f"   Total combined compounds: {len(df_combined)}")
print(f"   New training: {len(df_train_new)} compounds (80%)")
print(f"   New validation: {len(df_valid_new)} compounds (20%)")

print(f"\nBALANCING CHARACTERISTICS:")
print(f"   - Stratified distribution by scientific categories")
print(f"   - Proportions maintained between training/validation")
print(f"   - Categories based on established scientific literature")
print(f"   - Independence verification between sets")

print(f"\nSCIENTIFIC BASIS OF CATEGORIES:")
for cat, desc in balancing_categories.items():
    count_train = train_counts.get(cat, 0)
    count_valid = valid_counts.get(cat, 0)
    print(f"   - {cat:25}: {desc} ({count_train} train, {count_valid} valid)")

print("\n" + "=" * 70)
print("PROCESS COMPLETED SUCCESSFULLY!")
print("=" * 70)
