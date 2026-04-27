"""
Dataset Element Pairs Analyzer
=================================
Extracts the most common element pairs from a chemical composition dataset.

NOTE: File paths are set to '/content/' for Google Colab compatibility.
Please update these paths according to your local environment.
"""

import pandas as pd
import re
from collections import Counter
import itertools


def extract_element_pairs_from_dataset(file_path, top_n=50):
    """
    Extracts the most common element pairs from a chemical composition dataset.

    Args:
        file_path: Path to the dataset CSV file
        top_n: Number of most common pairs to return

    Returns:
        List of tuples with the most common pairs and their frequencies
    """
    try:
        # 1. Read the dataset
        df = pd.read_csv(file_path)
        print(f"Dataset loaded: {len(df)} records")

        # 2. Identify the composition column
        composition_column = None
        for col in df.columns:
            if 'composition' in col.lower():
                composition_column = col
                break

        if composition_column is None:
            raise ValueError("No composition column found in the dataset")

        print(f"Using column: '{composition_column}'")

        # 3. Regular expression to extract chemical elements from the specific format
        # Example: "Cd0.06" or "In0.94" or "Te1" or "Hf2"
        # Pattern: 1-2 letters (uppercase + optional lowercase) followed by numbers/dot
        element_pattern = re.compile(r'([A-Z][a-z]?)(?=\d|\.|$)')

        # 4. Process all compositions
        all_pairs = []

        for idx, comp_str in enumerate(df[composition_column]):
            # Extract elements (ignoring numbers)
            elements = element_pattern.findall(str(comp_str))

            # Remove duplicates within the same composition
            unique_elements = list(set(elements))

            # Generate all possible pairs (alphabetical order for consistency)
            for elem1, elem2 in itertools.combinations(sorted(unique_elements), 2):
                # Create ordered pair to avoid duplicates (A,B) and (B,A)
                pair = (elem1, elem2)
                all_pairs.append(pair)

            # Show progress for large datasets
            if idx > 0 and idx % 1000 == 0:
                print(f"Processed {idx} records...")
                # Print example for debugging
                if idx == 1000:
                    print(f"  Example: '{comp_str}' -> elements: {elements}")

        # 5. Count pair frequencies
        pair_counter = Counter(all_pairs)

        # 6. Get the most common pairs
        most_common_pairs = pair_counter.most_common(top_n)

        # 7. Print statistics
        print(f"\nStatistics:")
        print(f"Total unique pairs found: {len(pair_counter)}")
        print(f"Total pair occurrences: {len(all_pairs)}")

        # 8. Return results
        return most_common_pairs

    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return []
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def generate_stable_pairs_code(most_common_pairs, min_frequency=5):
    """
    Generates Python code with the list of stable pairs.

    Args:
        most_common_pairs: List of pairs and their frequencies
        min_frequency: Minimum frequency to include a pair

    Returns:
        Ready-to-copy Python code
    """
    if not most_common_pairs:
        return "# No pairs found with the minimum frequency"

    # Filter pairs by minimum frequency
    filtered_pairs = [(pair, freq) for pair, freq in most_common_pairs if freq >= min_frequency]

    # Generate code
    code_lines = ["        # Stable pairs based on dataset analysis"]
    code_lines.append(f"        # Total unique pairs: {len(set([pair for pair, _ in most_common_pairs]))}")
    code_lines.append(f"        # Pairs with frequency >= {min_frequency}: {len(filtered_pairs)}")
    code_lines.append("        stable_pairs = [")

    for i, ((elem1, elem2), freq) in enumerate(filtered_pairs):
        comment = f"  # Frequency: {freq}"
        if i < len(filtered_pairs) - 1:
            code_lines.append(f"            ('{elem1}', '{elem2}'),{comment}")
        else:
            code_lines.append(f"            ('{elem1}', '{elem2}'){comment}")

    code_lines.append("        ]")

    return "\n".join(code_lines)


def analyze_element_distribution(file_path, top_elements=20):
    """
    Analyzes the distribution of individual elements in the dataset.
    """
    try:
        df = pd.read_csv(file_path)

        # Find composition column
        composition_column = None
        for col in df.columns:
            if 'composition' in col.lower():
                composition_column = col
                break

        if composition_column is None:
            return None

        # Pattern to extract elements
        element_pattern = re.compile(r'([A-Z][a-z]?)(?=\d|\.|$)')

        all_elements = []
        for comp_str in df[composition_column]:
            elements = element_pattern.findall(str(comp_str))
            all_elements.extend(elements)

        # Count frequencies
        element_counter = Counter(all_elements)

        print("\nIndividual element distribution (top 20):")
        print("-" * 40)
        for elem, freq in element_counter.most_common(top_elements):
            percentage = (freq / len(all_elements)) * 100
            print(f"{elem:3}: {freq:6} occurrences ({percentage:.2f}%)")

        print(f"\nTotal unique elements: {len(element_counter)}")
        print(f"Total element occurrences: {len(all_elements)}")

        return element_counter

    except Exception as e:
        print(f"Error in distribution analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Path to your dataset
    file_path = "/content/train_balanced_stratified.csv"

    print("=" * 60)
    print("ELEMENT PAIR ANALYSIS IN DATASET")
    print("=" * 60)

    # 1. Individual element distribution analysis
    element_counter = analyze_element_distribution(file_path)

    # 2. Extract most common pairs
    print("\n" + "=" * 60)
    print("EXTRACTION OF MOST COMMON ELEMENT PAIRS")
    print("=" * 60)

    most_common_pairs = extract_element_pairs_from_dataset(
        file_path,
        top_n=100  # Get top 100 for analysis
    )

    if most_common_pairs:
        print("\nTop 20 most common pairs:")
        print("-" * 40)
        for i, ((elem1, elem2), freq) in enumerate(most_common_pairs[:20]):
            print(f"{i+1:2}. ({elem1}, {elem2}): {freq} occurrences")

        # 3. Generate code for stable pairs
        print("\n" + "=" * 60)
        print("CODE FOR STABLE PAIRS (ready to copy)")
        print("=" * 60)

        # You can adjust min_frequency according to your needs
        stable_pairs_code = generate_stable_pairs_code(most_common_pairs, min_frequency=5)
        print(stable_pairs_code)

        # 4. Optional: Save results to file
        save_results = input("\nDo you want to save results to a file? (y/n): ")
        if save_results.lower() == 'y':
            with open('element_pairs_analysis.txt', 'w') as f:
                f.write("ELEMENT PAIR ANALYSIS\n")
                f.write("=" * 50 + "\n\n")

                f.write("Most common individual elements:\n")
                for elem, freq in element_counter.most_common(30):
                    f.write(f"{elem}: {freq}\n")

                f.write("\n\nMost common pairs (top 50):\n")
                for i, ((elem1, elem2), freq) in enumerate(most_common_pairs[:50]):
                    f.write(f"{i+1}. ({elem1}, {elem2}): {freq}\n")

                f.write("\n\nCode for stable_pairs:\n")
                f.write(stable_pairs_code)

            print("Results saved in 'element_pairs_analysis.txt'")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print("=" * 60)
