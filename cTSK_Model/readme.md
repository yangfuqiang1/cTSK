# cTSK: Causal-weighted TSK Fuzzy System for scRNA-seq Analysis

This repository contains the source code for the paper submitted to *Nature Communications*.

The **cTSK** framework combines causal discovery (WGCNA + FCI) with interpretable fuzzy neural networks to analyze gene regulatory networks.

## 📂 Repository Structure

* `data/`: Sample datasets (processed .h5ad files).
* 'data_all':# Raw cancer datasets 
（链接: https://pan.baidu.com/s/1Caw8GQ-FMz2tWCCOTyypuQ 提取码: b65r)

## 🚀 Usage

You can choose between reproducing the figures directly or retraining the model from scratch.

### Option 1: Reproduce Figures (Quick Start)
To reproduce the figures presented in the paper using the pre-computed results, please run `05_plot_results.py` directly:

python 05_plot_results.py

### Option 2: Retrain Model (Full Workflow)
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

2.Data Preprocessing (Optional): If you have downloaded the raw data into the data_all/ folder, run this script to generate processed .h5ad files
python data_process.py

Please run the files in the following order:
* `lib/` & `tsk_model/`: Core model libraries.
* `data_precess.py`: Script for causal module detection.
* `02_train_cTSK_explainability.py`: Main training script with explainability analysis.
* `03_ablation_study.py`: Validation of causal components.
* `04_benchmark_comparison.py`: Comparison with SVM, RF, XGBoost, etc.
* `05_plot_results.py`: Visualization generation.
