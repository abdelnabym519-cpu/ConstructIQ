# Reproducibility Guide

## Requirements

- Docker Desktop
- Git
- sufficient local disk space
- approximately 8–12 GB RAM recommended

## Data

Source: San Francisco DataSF Building Permits

Dataset ID: `i98e-djp9`

Raw downloaded data is intentionally excluded from Git.

## Execution Order

1. `00_spark_smoke.py`
2. `01_bronze_ingest.py`
3. `02_silver_clean.py`
4. `03_gold_features.py`
5. `04_eda_analysis.py`
6. `05_ml_train.py`
7. `05a_ml_diagnostic.py`
8. `06_ml_train_v2.py`
9. `07_xai_explain.py`

## Generated Data Layers

- `data/source/`
- `data/bronze/`
- `data/silver/`
- `data/gold/`

## Model Artifacts

- `models/constructiq_spark/`
- `models/constructiq_spark_v2/`

Model artifacts are not committed to GitHub.

## Reproducible Reports

- `reports/eda/`
- `reports/ml/`
- `reports/ml_v2/`
- `reports/xai/`
