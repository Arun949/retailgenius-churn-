# RetailGenius — Customer Churn Prediction

**EPITA International Programs | AI Project Methodology 2025-2026 | Group 7**

An end-to-end production-ready ML project to predict customer churn using the
[E-Commerce Churn Dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction).

---

## Project Structure

```
retailgenius-churn/
├── conf/
│   └── config.yaml            # All hyperparameters and paths
├── data/
│   ├── raw/                   # Place raw dataset here
│   └── processed/             # Auto-generated splits and feature files
├── docs/                      # Sphinx documentation source
├── mlruns/                    # MLflow tracking output (auto-generated)
├── notebooks/                 # Exploratory analysis notebooks
├── src/
│   ├── data/
│   │   └── data_preparation.py
│   ├── features/
│   │   └── feature_engineering.py
│   └── models/
│       ├── train.py
│       └── inference.py
├── tests/                     # Unit tests
├── environment.yml            # Conda environment
├── MLproject                  # MLflow Projects entry points
├── pyproject.toml             # Black formatter config
└── setup.cfg                  # flake8 config
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-group>/retailgenius-churn.git
cd retailgenius-churn
```

### 2. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate retailgenius-churn
```

### 3. Download the dataset

Download the **E-Commerce Customer Churn** dataset from Kaggle and place the
CSV file at:

```
data/raw/E Commerce.csv
```

---

## Running the Pipeline

Run each step in order:

```bash
# Step 1 — Data preparation (clean, encode, split)
python src/data/data_preparation.py

# Step 2 — Feature engineering (RFM, engagement, satisfaction features)
python src/features/feature_engineering.py

# Step 3 — Train XGBoost model (logged to MLflow)
python src/models/train.py --model_type xgboost

# Step 4 — Train Logistic Regression baseline (second MLflow run)
python src/models/train.py --model_type logistic_regression
```

### Using MLflow Projects

```bash
mlflow run . -e train -P model_type=xgboost
mlflow run . -e train -P model_type=logistic_regression
```

---

## View MLflow UI

```bash
mlflow ui --backend-store-uri mlruns
```

Open http://localhost:5000 to compare runs, metrics, and registered models.

---

## Serve the Model Locally

```bash
# Replace <run_id> with the run ID from the MLflow UI
mlflow models serve -m "models:/churn-predictor/latest" --port 5001 --no-conda
```

The model is now available at `http://localhost:5001/invocations`.

---

## Run Inference

```bash
python src/models/inference.py \
    --model_uri "models:/churn-predictor/latest" \
    --input_path data/processed/test_features.csv
```

Output is saved to `data/processed/test_features_scored.csv`.

---

## Generate Documentation

```bash
cd docs
sphinx-apidoc -o . ../src
make html
```

Open `docs/_build/html/index.html` in your browser.

---

## Code Quality

```bash
# Format code
black src/

# Lint
flake8 src/

# Run tests
pytest tests/
```

---

## References

- E-Commerce Churn Dataset — Kaggle
- MLflow Documentation — https://mlflow.org
- Cookiecutter Data Science — https://drivendata.github.io/cookiecutter-data-science/
- Sphinx Documentation — https://www.sphinx-doc.org
