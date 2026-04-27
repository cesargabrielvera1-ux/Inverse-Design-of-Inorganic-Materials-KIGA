# Inverse Design of Inorganic Materials with Targeted Band Gaps

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![TensorFlow 2.18](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Neural Network-based band gap prediction combined with a Knowledge-Informed Genetic Algorithm (KIGA) for the discovery of inorganic materials with targeted electronic properties.**

This repository contains the complete implementation, datasets, pre-trained models, and candidate results associated with the manuscript:

**"Inverse Design of Inorganic Materials with Targeted Band Gaps Using Neural Networks and Knowledge-Informed Genetic Algorithm"**  
*Cesar Gabriel Vera de la Garza\*, Serguei Fomine*  
National Autonomous University of Mexico (UNAM), Materials Research Institute.

---

## Overview

The band gap (Eg) is a fundamental electronic property that determines the suitability of materials for electronics, photovoltaics, and optoelectronics. Traditional materials discovery relies on trial-and-error or high-throughput screening, which is expensive and slow. This project presents a hybrid inverse-design framework that combines:

1. **Robust Neural Network regression** trained on the largest experimental band-gap dataset compiled to date (6,489 unique compositions from Matbench and Zhuo et al.)
2. **A Knowledge-Informed Genetic Algorithm (KIGA)** that embeds chemical valence rules, stoichiometric constraints, and empirical stability criteria directly into the evolutionary search.
3. **Multi-source validation** against experimental data and DFT calculations from The Materials Project.

The framework achieves a mean absolute error (MAE) of **0.33 eV** on external validation (R² = 0.77) and generates chemically plausible novel candidates with prediction errors below **0.02 eV** for targeted band gaps ranging from 0.0 to 4.0 eV.

---

## Key Highlights

- **Largest experimental band-gap dataset to date:** 6,489 unique inorganic compositions with scientifically validated electronic categories (metals, semimetals, narrow-gap semiconductors, semiconductors, insulators).
- **Rigorous neural-network development:** Three architectures evaluated (Base, Two-Phase Optimized, Extended Search) with Bayesian hyperparameter optimization (Optuna), 5- to 10-fold cross-validation, and per-category performance analysis.
- **Composition-only descriptors:** 132 Magpie descriptors calculated via `matminer` + `pymatgen`; no structural information required, enabling sub-millisecond inference per composition.
- **Chemically constrained inverse design:** KIGA incorporates 80+ element valence rules, electronegativity checks, and a database of 100+ empirically stable element pairs to ensure synthesizable candidates.
- **External validation:** Comparison with The Materials Project database yields MAE = 0.35 eV against DFT band gaps, confirming physical consistency.

---

## Repository Structure

```
.
├── dataset_combination_balancing.py      # Dataset merging, category stratification, and 80/20 split
├── neural_network_base.py                # Model 1: Base architecture + Optuna + 5-fold CV
├── neural_network_two_phase.py           # Model 2: Two-phase Bayesian optimization + 10-fold CV
├── neural_network_advanced.py            # Model 3: Extended search space + advanced preprocessing
├── dataset_unique_elements.py            # Extract unique chemical elements from the dataset
├── dataset_element_pairs.py              # Analyze the most common element pairs in the dataset
├── dataset_elements_by_category.py       # Classify elements by band-gap category
├── element_classification_hybrid.py       # Hybrid classification (chemical knowledge + data-driven)
├── element_classification_final.py        # Final curated element categories used in KIGA
├── kiga_inverse_design.py                # Full KIGA pipeline: GA + NN + chemical validation
├── external_validation.py               # Validation against experimental dataset and Materials Project API
├── earth_movers_distance.py             # Compute EMD between candidates and training set
├── ablation_study.py                    # Ablation: KIGA vs. baseline GA (no chemical constraints)
│
├── models/                               # Pre-trained model weights and scalers
│   ├── bandgap_nn_model.keras
│   ├── scaler.pkl
│   └── ...
├── data/                                 # Processed datasets (not included in repo; see Data Availability)
│   ├── train_balanced_stratified.csv
│   └── validation_balanced_stratified.csv
├── results/                              # Generated candidates, logs, and figures
│   └── ...
└── README.md                             # This file
```

> **Note:** Absolute paths in the scripts default to `/content/` for compatibility with Google Colab. Please update them to your local working directory before execution.

---

## Installation

### Prerequisites

- Python >= 3.10 (tested on 3.12.0)
- A CUDA-capable GPU is recommended for neural-network training (optional for inference)

### Dependencies

Install the required packages via `pip`:

```bash
pip install tensorflow==2.18.0 scikit-learn==1.6.0 pymatgen==2023.12.18 \
        matminer==0.8.0 optuna==4.2.1 pandas==2.2.2 numpy==1.26.4 \
        matplotlib==3.10.0 seaborn==0.13.2 joblib==1.4.2 scipy tqdm
```

For a full environment specification, see the `requirements.txt` file (to be added).

---

## Usage

### 1. Data Preparation

Run the dataset preparation script to combine sources, remove duplicates, categorize by electronic class, and generate stratified splits:

```bash
python dataset_combination_balancing.py
```

This produces:
- `train_balanced_stratified.csv`
- `validation_balanced_stratified.csv`

### 2. Neural Network Training

Train the base model (Model 1) with Bayesian optimization and cross-validation:

```bash
python neural_network_base.py
```

For the extended architecture (Model 3):

```bash
python neural_network_advanced.py
```

For the two-phase optimization strategy (Model 2):

```bash
python neural_network_two_phase.py
```

Each script saves:
- The trained Keras model (`.keras`)
- The fitted `StandardScaler` (`.pkl`)
- JSON metrics and CSV predictions

### 3. Element & Pair Analysis (Optional)

These scripts support the chemical knowledge base used by KIGA:

```bash
python dataset_unique_elements.py          # List all elements present
python dataset_element_pairs.py            # Rank element pairs by frequency
python dataset_elements_by_category.py     # Category-driven classification
python element_classification_hybrid.py   # Hybrid data + knowledge rules
python element_classification_final.py     # Final curated categories
```

### 4. Inverse Design (KIGA)

Run the full inverse-design pipeline to generate novel candidates for a target band gap:

```bash
python kiga_inverse_design.py
```

Edit the `CONFIG` dictionary inside the script (or pass arguments if refactored) to set:
- `target_gap`: Desired band gap in eV (e.g., 2.5)
- `material_type`: Optional filter (`'metal'`, `'semiconductor'`, `'insulator'`)
- `n_candidates`: Number of top candidates to return

Outputs:
- `candidates_YYYYMMDD_HHMMSS.csv`
- `inverse_design_YYYYMMDD_HHMMSS.json`
- Evolution logs and fitness plots

### 5. Validation & Analysis

Compare generated candidates against experimental and DFT data:

```bash
python external_validation.py            # Requires a Materials Project API key
python earth_movers_distance.py          # Compute compositional novelty (EMD)
python ablation_study.py                 # Reproduce KIGA vs. baseline GA comparison
```

---

## Dataset

The training and validation datasets were compiled from two authoritative sources of **experimental** band-gap data:

- **Matbench benchmark dataset** (Ward et al., 2016)
- **Zhuo et al. dataset**

**Final statistics:**
- **6,489** unique inorganic compositions
- **80 / 20** stratified split (5,191 train / 1,298 validation)
- **Zero composition overlap** between splits to ensure true generalization

| Category | Threshold | Train % |
|---|---|---|
| Metallic | Eg = 0 eV | 42.7 % |
| Semiconductors | 1.5 < Eg ≤ 3.0 eV | 23.6 % |
| Narrow-gap semiconductors | 0.1 < Eg ≤ 1.5 eV | 22.1 % |
| Insulators | Eg > 3.0 eV | 11.1 % |
| Semimetals | 0 < Eg ≤ 0.1 eV | 0.5 % |

---

## Model Performance

**Selected Model (Model 1 - Base Architecture):**

| Metric | Value |
|---|---|
| Test MAE | **0.3329 eV** |
| Test R² | **0.7650** |
| CV MAE | 0.3517 ± 0.0282 eV |
| Per-category R² | 0.73 – 0.89 (all positive) |
| Inference time | < 1 ms / composition (CPU) |

*Model 3 achieved a lower raw MAE (0.2943 eV) but exhibited severe per-category overfitting (negative R² in multiple classes); Model 1 was therefore selected for deployment due to superior generalization and stability.*

---

## Inverse Design Results

Top candidates generated for representative target band gaps:

| Target (eV) | Formula | Predicted (eV) | Error (eV) | Stability | Known in MP? |
|---|---|---|---|---|---|
| 0.0 | MnLi | 0.000 | 0.000 | 0.75 | No |
| 0.8 | V₁CrB₂Ge₃ | 0.756 | 0.044 | 0.50 | No |
| 1.8 | InGaSe₄ | 1.799 | 0.001 | 0.85 | No |
| 2.5 | OGeP | 2.504 | 0.004 | 0.80 | No |
| 4.0 | Sr₂ZrS₄ | 3.987 | 0.013 | 0.72 | No |

**Ablation study:** KIGA consistently outperforms a chemistry-agnostic baseline GA in terms of **chemical plausibility** (Composite Plausibility Score improvements of 13.5 % – 32.2 %, p < 0.05) without sacrificing band-gap accuracy.

---

## Citation

If you use this code, the dataset, or the generated candidates in your research, please cite:

```bibtex
@article{veradelagarza2026inverse,
  title={Inverse Design of Inorganic Materials with Targeted Band Gaps Using Neural Networks and Knowledge-Informed Genetic Algorithm},
  author={Vera de la Garza, Cesar Gabriel and Fomine, Serguei},
  journal={...},
  year={2026},
  publisher={...},
  doi={...}
}
```

---

## Data & Code Availability

All Python scripts, pre-trained model weights, scaler objects, and complete candidate lists are provided in this repository under the **MIT License**.

- **Pre-trained Model:** `models/bandgap_nn_model.keras` + `models/scaler.pkl`
- **Candidate Data:** Available in `results/` as CSV and JSON files after running `kiga_inverse_design.py`.
- **Environment:** Google Colab Pro (GPU) + Python 3.12.0. See Table 1 of the manuscript for the full software stack.

> **Disclaimer:** The dataset reflects experimentally characterized compounds and therefore carries a "positive-data" bias. The stability scores should be interpreted as plausibility indicators, not as definitive thermodynamic predictions. All candidates recommended for experimental synthesis should be pre-screened with DFT.

---

## Authors & Contact

- **Cesar Gabriel Vera de la Garza** \* — [cesargabriel.vera@live.com](mailto:cesargabriel.vera@live.com)
- **Serguei Fomine** — [fomine@unam.mx](mailto:fomine@unam.mx)

National Autonomous University of Mexico (UNAM)  
Materials Research Institute  
Circuito de la Investigación Científica, Circuito Exterior, Investigación Científica S/N, C.U., 04510, Mexico City, Mexico.

---

## Acknowledgments

C. G. V. de la G. received support from CONAHCyT under postdoctoral fellowship I1200/311/2023. Support from DGAPA-UNAM (PAPIIT IN200125) is also acknowledged. The authors thank the developers of `pymatgen`, `matminer`, `TensorFlow`, `Optuna`, and `scikit-learn`, as well as the original dataset creators (Matbench team and Zhuo et al.) for making their data publicly available.
