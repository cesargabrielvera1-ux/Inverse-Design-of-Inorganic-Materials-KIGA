"""
Dataset Elements by Band Gap Category
========================================
Classifies elements into categories based on the experimental band gap
of the compounds they appear in.

NOTE: File paths are set to '/content/' for Google Colab compatibility.
Please update these paths according to your local environment.
"""

import pandas as pd
import re
from collections import defaultdict
import numpy as np


def classify_elements_by_bandgap(file_path):
    """
    Classifies elements into categories based on the experimental band gap.

    Args:
        file_path: Path to the dataset CSV file

    Returns:
        Dictionary with elements classified by category
    """
    try:
        # 1. Read the dataset
        df = pd.read_csv(file_path)
        print(f"Dataset loaded: {len(df)} records")

        # 2. Find columns
        composition_column = None
        gap_numeric_column = None
        gap_category_column = None

        for col in df.columns:
            col_lower = col.lower()
            if 'composition' in col_lower or 'comp' in col_lower:
                composition_column = col
            elif 'gap expt' in col_lower or 'gap_expt' in col_lower or ('gap' in col_lower and 'expt' in col_lower):
                gap_numeric_column = col
            elif 'gap_category' in col_lower or 'gap category' in col_lower:
                gap_category_column = col

        print(f"Composition column: '{composition_column}'")
        print(f"Numeric band gap column: '{gap_numeric_column}'")
        print(f"Gap category column: '{gap_category_column}'")

        # 3. Pattern to extract elements
        element_pattern = re.compile(r'([A-Z][a-z]?)(?=\d|\.|$)')

        # 4. Initialize dictionaries for classification
        element_data = defaultdict(lambda: {
            'metal_count': 0,
            'semimetal_count': 0,
            'narrow_gap_count': 0,
            'semiconductor_count': 0,
            'insulator_count': 0,
            'total_appearances': 0,
            'avg_gap': 0,
            'gap_sum': 0
        })

        # 5. Map categories to numeric values for classification
        category_to_gap = {
            'metallic': 0,
            'narrow_gap_semiconductor': 0.75,  # Midpoint between 0.1 and 1.5
            'semiconductor': 2.25,  # Midpoint between 1.5 and 3.0
            'insulator': 3.5  # Greater than 3.0
        }

        # 6. Process each compound
        for idx, row in df.iterrows():
            comp_str = str(row[composition_column])

            # Determine the gap value to use
            gap_value = None
            material_type = None

            # Priority 1: Use numeric value if available
            if gap_numeric_column and pd.notna(row.get(gap_numeric_column)):
                gap_value = float(row[gap_numeric_column])

                # Determine type based on gap (using your criteria)
                if gap_value == 0:
                    material_type = 'metal'
                elif gap_value < 0.1:
                    material_type = 'semimetal'
                elif gap_value < 1.5:
                    material_type = 'narrow_gap'
                elif gap_value < 3.0:
                    material_type = 'semiconductor'
                else:
                    material_type = 'insulator'

            # Priority 2: Use category if available
            elif gap_category_column and pd.notna(row.get(gap_category_column)):
                category = str(row[gap_category_column]).lower()

                # Map category to type
                if 'metallic' in category:
                    material_type = 'metal'
                    gap_value = 0
                elif 'narrow_gap' in category:
                    material_type = 'narrow_gap'
                    gap_value = 0.75
                elif 'semiconductor' in category and 'narrow' not in category:
                    material_type = 'semiconductor'
                    gap_value = 2.25
                elif 'insulator' in category:
                    material_type = 'insulator'
                    gap_value = 3.5
                else:
                    # Default to semiconductor
                    material_type = 'semiconductor'
                    gap_value = 2.25

            if material_type is None:
                continue

            # Extract elements
            elements = element_pattern.findall(comp_str)
            unique_elements = set(elements)

            # Update statistics for each element
            for element in unique_elements:
                element_data[element]['total_appearances'] += 1
                if gap_value is not None:
                    element_data[element]['gap_sum'] += gap_value

                if material_type == 'metal':
                    element_data[element]['metal_count'] += 1
                elif material_type == 'semimetal':
                    element_data[element]['semimetal_count'] += 1
                elif material_type == 'narrow_gap':
                    element_data[element]['narrow_gap_count'] += 1
                elif material_type == 'semiconductor':
                    element_data[element]['semiconductor_count'] += 1
                elif material_type == 'insulator':
                    element_data[element]['insulator_count'] += 1

        # 7. Calculate averages and percentages
        for element in element_data:
            if element_data[element]['total_appearances'] > 0:
                if element_data[element]['gap_sum'] > 0:
                    element_data[element]['avg_gap'] = (
                        element_data[element]['gap_sum'] / element_data[element]['total_appearances']
                    )
                else:
                    element_data[element]['avg_gap'] = 0

                # Calculate percentages
                total = element_data[element]['total_appearances']
                element_data[element]['metal_pct'] = (
                    element_data[element]['metal_count'] / total * 100
                ) if total > 0 else 0

                element_data[element]['semiconductor_pct'] = (
                    (element_data[element]['semimetal_count'] +
                     element_data[element]['narrow_gap_count'] +
                     element_data[element]['semiconductor_count']) / total * 100
                ) if total > 0 else 0

                element_data[element]['insulator_pct'] = (
                    element_data[element]['insulator_count'] / total * 100
                ) if total > 0 else 0

        # 8. Classify elements into categories
        element_categories = {
            'metals': [],
            'semiconductors': [],
            'insulators': [],
            'transition_metals': []
        }

        # Transition metals (by periodic table)
        transition_metals_set = {
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'
        }

        # Classification thresholds (adjustable)
        METAL_THRESHOLD = 40  # % of appearances as metal
        SEMICONDUCTOR_THRESHOLD = 40  # % of appearances as semiconductor
        INSULATOR_THRESHOLD = 40  # % of appearances as insulator

        for element, data in element_data.items():
            total_appearances = data['total_appearances']

            # Only classify elements with enough appearances
            if total_appearances < 3:
                continue

            # Classify as metal
            if data['metal_pct'] >= METAL_THRESHOLD:
                element_categories['metals'].append(element)

            # Classify as semiconductor
            elif data['semiconductor_pct'] >= SEMICONDUCTOR_THRESHOLD:
                element_categories['semiconductors'].append(element)

            # Classify as insulator
            elif data['insulator_pct'] >= INSULATOR_THRESHOLD:
                element_categories['insulators'].append(element)

            # If clear thresholds are not met, classify by average gap
            else:
                avg_gap = data['avg_gap']
                if avg_gap == 0:
                    element_categories['metals'].append(element)
                elif avg_gap < 1.5:
                    element_categories['semiconductors'].append(element)
                elif avg_gap >= 3.0:
                    element_categories['insulators'].append(element)
                else:
                    element_categories['semiconductors'].append(element)

            # Transition metals (based on periodic table)
            if element in transition_metals_set:
                element_categories['transition_metals'].append(element)

        # 9. Sort lists and remove duplicates
        for category in element_categories:
            element_categories[category] = sorted(list(set(element_categories[category])))

        # 10. Ensure all elements are in at least one category
        all_elements = set(element_data.keys())
        categorized_elements = set()
        for category in ['metals', 'semiconductors', 'insulators', 'transition_metals']:
            categorized_elements.update(element_categories[category])

        # Uncategorized elements (add to metals by default)
        uncategorized = all_elements - categorized_elements
        if uncategorized:
            print(f"\nUncategorized elements ({len(uncategorized)}): {', '.join(sorted(uncategorized))}")
            element_categories['metals'].extend(uncategorized)
            element_categories['metals'] = sorted(list(set(element_categories['metals'])))

        # 11. Print statistics
        print(f"\n{'='*60}")
        print("CLASSIFICATION STATISTICS")
        print(f"{'='*60}")

        print(f"\nTotal unique elements in dataset: {len(element_data)}")

        for category in ['metals', 'semiconductors', 'insulators', 'transition_metals']:
            elements = element_categories[category]
            print(f"\n{category.upper()} ({len(elements)} elements):")
            if elements:
                # Display in columns
                for i in range(0, len(elements), 10):
                    chunk = elements[i:i+10]
                    print("  " + ", ".join(chunk))
            else:
                print("  (none)")

        # 12. Show some detailed examples
        print(f"\n{'='*60}")
        print("DETAILED CLASSIFICATION EXAMPLES")
        print(f"{'='*60}")

        # Most common elements
        common_elements = sorted(
            [(elem, data['total_appearances']) for elem, data in element_data.items()],
            key=lambda x: x[1], reverse=True
        )[:10]

        for element, count in common_elements:
            data = element_data[element]
            print(f"\n{element} ({count} occurrences):")
            print(f"  Average gap: {data['avg_gap']:.3f}")
            print(f"  % as metal: {data['metal_pct']:.1f}%")
            print(f"  % as semiconductor: {data['semiconductor_pct']:.1f}%")
            print(f"  % as insulator: {data['insulator_pct']:.1f}%")

            # Determine main category
            max_pct = max(data['metal_pct'], data['semiconductor_pct'], data['insulator_pct'])
            if max_pct == data['metal_pct']:
                print(f"  Main category: METAL")
            elif max_pct == data['semiconductor_pct']:
                print(f"  Main category: SEMICONDUCTOR")
            else:
                print(f"  Main category: INSULATOR")

        return element_categories

    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


def generate_element_categories_code(element_categories):
    """
    Generates Python code with the element categories dictionary.

    Args:
        element_categories: Dictionary with classified elements

    Returns:
        Ready-to-copy Python code
    """
    if not element_categories:
        return "# Could not classify elements"

    code_lines = []
    code_lines.append("        # Map elements by type (based on dataset analysis)")
    code_lines.append("        element_categories = {")

    categories_order = ['metals', 'semiconductors', 'insulators', 'transition_metals']

    for i, category in enumerate(categories_order):
        elements = element_categories.get(category, [])
        elements_str = "', '".join(elements)
        if i < len(categories_order) - 1:
            code_lines.append(f"            '{category}': ['{elements_str}'],")
        else:
            code_lines.append(f"            '{category}': ['{elements_str}']")

    code_lines.append("        }")

    return "\n".join(code_lines)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Path to your dataset
    file_path = "/content/train_balanced_stratified.csv"

    print("=" * 60)
    print("ELEMENT CLASSIFICATION BY BAND GAP")
    print("=" * 60)

    # Classify elements
    element_categories = classify_elements_by_bandgap(file_path)

    if element_categories:
        # Generate code
        print("\n" + "=" * 60)
        print("CODE FOR element_categories (ready to copy)")
        print("=" * 60)

        element_categories_code = generate_element_categories_code(element_categories)
        print(element_categories_code)

        # Optional: Save results
        save_results = input("\nDo you want to save results to a file? (y/n): ")
        if save_results.lower() == 'y':
            with open('element_classification_results.txt', 'w') as f:
                f.write("ELEMENT CLASSIFICATION BY BAND GAP\n")
                f.write("=" * 50 + "\n\n")

                for category in ['metals', 'semiconductors', 'insulators', 'transition_metals']:
                    elements = element_categories.get(category, [])
                    f.write(f"{category.upper()} ({len(elements)} elements):\n")
                    f.write(", ".join(elements) + "\n\n")

                f.write("\nCode for element_categories:\n")
                f.write(element_categories_code)

            print("Results saved in 'element_classification_results.txt'")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print("=" * 60)
