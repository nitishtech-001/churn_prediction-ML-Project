# Churn Prediction

This repository contains a churn prediction project using the European Bank dataset. The project demonstrates data analysis, feature preparation, model training and evaluation, and a lightweight application entrypoint.

## Project Flow

1. Data ingestion: the raw CSV is stored in `Data/European_Bank.csv` and used for analysis and modeling.
2. Exploratory Data Analysis (EDA): notebooks in `Notebooks/` perform EDA to understand and visualize feature distributions and relationships.
3. Preprocessing: data cleaning, encoding categorical variables, feature scaling, and splitting into train/test sets are performed in the notebooks and scripts used for modelling.
4. Model training & evaluation: multiple model experiments are performed, results are compared and the best model(s) are saved under `Models/`.
5. Inference / App: `app.py` provides a simple entry point for running predictions or serving the model (see usage below).

## Repository Structure

- **app.py**: application entrypoint for running or demoing the model.
- **requirements.txt**: Python dependencies required to run the project.
- **Data/European_Bank.csv**: source dataset used for analysis and training.
- **Models/churn_prediction.pkl**: folder for saving trained model artifacts and related files.
- **Notebooks/**: Jupyter notebooks used for analysis and experimentation:
  - `analyzing.ipynb` — initial EDA and feature exploration.
  - `checking_models.ipynb` — training multiple models and quick comparisons.
  - `final_notebook.ipynb` — consolidated preprocessing pipeline and final model evaluation.

## Analysis Details

- EDA: check missing values, class imbalance, correlations, and distribution of numeric features.
- Feature engineering: create or transform features as identified in EDA (scaling, encodings, aggregations).
- Model experiments: try baseline models (Logistic Regression, Random Forest, XGBoost, etc.), use cross-validation, and compare metrics like accuracy, precision, recall, F1, and ROC AUC.
- Model selection: choose the best model based on validation performance and stability, then retrain on combined training data and save to `Models/`.

## How to Set Up

1. Create a Python environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run notebooks for analysis and training (recommended):

```bash
jupyter lab
```

4. Run the app (example):

```bash
python3 app.py
```
## To run the Notebook you have to uncommit the module or library name
example 1:
#matplotlib==3.9.0 
after uncommit
matplotlib==1.0.0
then again run the pip install -r requiremets.txt

## How to Reproduce Training

1. Open `Notebooks/checking_models.ipynb` to reproduce model training experiments.
2. Follow preprocessing steps shown in the notebooks or extract the preprocessing pipeline into a script before training at scale.
3. Save resulting model artifacts into `Models/` with a clear naming convention (e.g., `model_<name>_<date>.pkl`).
