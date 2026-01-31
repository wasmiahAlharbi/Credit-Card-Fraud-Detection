# Credit-Card-Fraud-Detection

A machine learning project to detect fraudulent credit card transactions using the well-known Kaggle dataset. This repository implements and compares several classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM) and common preprocessing and class-imbalance strategies.

- Dataset: "Credit Card Fraud Detection" (Kaggle) — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
- [Models Implemented](#models-implemented)
- [Evaluation](#evaluation)
- [Results & Comparison](#results--comparison)
- [Reproducibility](#reproducibility)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview
This project explores methods to detect fraudulent credit card transactions. The dataset is highly imbalanced (very few fraudulent transactions), so the project focuses on both model selection and techniques to handle class imbalance. The goal is to compare model performance using robust evaluation metrics (precision, recall, F1-score, ROC AUC) and produce a deployable pipeline for inference.

## Key Features
- Data loading and exploratory data analysis (EDA)
- Data cleaning and preprocessing (scaling, encoding where necessary)
- Multiple classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - LightGBM
- Class imbalance handling (undersampling, oversampling, SMOTE, class weights)
- Model training, evaluation, and comparison
- Scripts/notebooks for reproducibility and experimentation

## Dataset
This repository uses the Kaggle "Credit Card Fraud Detection" dataset:
- Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Format: CSV containing anonymized numerical features (V1..V28), `Time`, `Amount`, and `Class` (0 = legitimate, 1 = fraud).
- Notes: The features V1..V28 are the result of a PCA transformation performed by the dataset authors for confidentiality. Typical preprocessing includes scaling `Amount` and `Time` or creating derived features.

## Requirements
- Python 3.8+ recommended
- Typical libraries:
  - numpy, pandas, scikit-learn
  - matplotlib, seaborn (visualization)
  - imbalanced-learn (SMOTE)
  - xgboost, lightgbm
  - jupyter / jupyterlab (if using notebooks)

Install from requirements.txt if present:
```bash
pip install -r requirements.txt
```

Or create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate         # macOS/Linux
venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/wasmiahAlharbi/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```
2. Prepare the environment as above.
3. Download the Kaggle dataset and place `creditcard.csv` into the `data/` folder (create it if missing).

## Usage
There are usually two main ways to run experiments: Jupyter notebooks for exploration, and scripts for training/evaluation.

- To explore with notebooks:
```bash
jupyter notebook notebooks/
# or
jupyter lab
```

- To run training/evaluation scripts (example command — adapt to your repo's scripts):
```bash
python src/train.py --data data/creditcard.csv --model lightgbm --output results/
python src/evaluate.py --predictions results/predictions.csv --truth data/creditcard.csv
```

If your repo uses different filenames or structure, open the notebook(s) or the `src/` directory to see exact entry points.

## Preprocessing & Feature Engineering
Common preprocessing steps implemented or recommended:
- Handle missing values (the Kaggle dataset typically has none).
- Scale `Amount` and `Time` (StandardScaler or RobustScaler).
- Optionally apply feature selection or dimensionality reduction if desired.
- Handle class imbalance:
  - Random undersampling (reduce majority class)
  - Random oversampling (duplicate minority class)
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - Use class weights in classifiers (e.g., `class_weight='balanced'` in scikit-learn)

Document and persist preprocessing pipeline (scalers, encoders) to ensure consistent inference.

## Models Implemented
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

Each model supports:
- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV or custom)
- Cross-validation
- Use of class weights or resampled training data

## Evaluation
Recommended metrics for imbalanced classification:
- Confusion matrix
- Precision, Recall (Sensitivity), F1-score
- ROC AUC
- Precision-Recall (PR) curve and Average Precision (AP)
- Specificity and false positive rate (as needed for business constraints)

Always report both per-class metrics and aggregated summaries. When fraud is rare, prioritize recall (catching fraud) while keeping false positives manageable.

Example code to compute metrics (conceptual):
```python
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
print(classification_report(y_true, y_pred))
print("ROC AUC:", roc_auc_score(y_true, y_prob))
```

## Results & Comparison
- Train each model using the same preprocessing pipeline and cross-validation strategy.
- Compare models by ROC AUC, F1-score, and precision@k (business-relevant thresholds).
- Save model artifacts, metrics, and confusion matrices to a `results/` directory for later review.

(Replace this section with tables/figures produced by your experiments.)

## Reproducibility
- Set fixed random seeds for data splitting, resampling, and model training:
```python
RANDOM_STATE = 42
```
- Save environment dependencies to `requirements.txt` or `environment.yml`.
- Persist preprocessing objects and trained model files using `joblib` or `pickle`.

## Repository Structure (suggested / typical)
- data/                     # Place dataset here (not committed)
- notebooks/                # EDA and experiments
- src/                      # Training, evaluation scripts, utilities
- models/                   # Saved model artifacts
- results/                  # Evaluation results, plots, logs
- requirements.txt
- README.md

Adjust above listing to match the actual repository files.

## Contributing
Contributions, bug reports, and feature requests are welcome.
- Fork the repository
- Create a feature branch: `git checkout -b feature/your-feature`
- Commit changes and open a Pull Request
- Ensure new code includes tests where appropriate and updates the README if needed

## License
This project is released under the MIT License. See LICENSE for details.

## Contact
For questions or suggestions, open an issue in the repository or contact the maintainer:
- GitHub: [wasmiahAlharbi](https://github.com/wasmiahAlharbi)

## Acknowledgements & References
- Dataset: Kaggle — "Credit Card Fraud Detection" (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Typical techniques and references:
  - Chawla et al., SMOTE: Synthetic Minority Over-sampling Technique
  - XGBoost and LightGBM official docs
