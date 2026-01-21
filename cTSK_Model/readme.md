# cTSK: Causal-weighted TSK Fuzzy System for scRNA-seq Analysis

This repository contains the source code for the paper submitted to *Nature Communications*.

The **cTSK** framework combines causal discovery (WGCNA + FCI) with interpretable fuzzy neural networks to analyze gene regulatory networks.

## 📂 Repository Structure

* `data/`: Sample datasets (processed .h5ad files).
* `lib/` & `tsk_model/`: Core model libraries.
* `data_precess.py`: Script for causal module detection.
* `02_train_cTSK_explainability.py`: Main training script with explainability analysis.
* `03_ablation_study.py`: Validation of causal components.
* `04_benchmark_comparison.py`: Comparison with SVM, RF, XGBoost, etc.
* `05_plot_results.py`: Visualization generation.

## 🚀 Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
