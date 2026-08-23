# ConstructIQ Architecture

## High-Level Flow

DataSF → Raw CSV → Apache Spark → Bronze → Silver → Gold

Gold feeds:

- EDA / Business Analytics
- ML / NLP
- Explainable AI
- Streamlit Dashboard

## Bronze

Purpose:

- preserve source records
- add ingestion metadata
- write compressed Parquet

## Silver

Purpose:

- normalize field names
- normalize missing values
- parse dates
- cast numeric fields
- deduplicate records
- create data-quality indicators

## Gold

Purpose:

- create the ML target
- prevent target leakage
- engineer project-scale features
- preserve project descriptions
- create the ML-ready dataset

## ML Architecture

Production target:

`log10(revised_cost)`

Temporal policy:

- Train: 2015–2023
- Validation: 2024
- Test: 2025+

Production model:

Structured features + Description TF-IDF → Linear Regression

## Explainability

The XAI layer combines:

- structured-vs-text ablation analysis
- vocabulary-based companion model
- global term associations
- documented limitations in a Model Card

## Serving

The Streamlit dashboard loads the serialized Spark `PipelineModel` and performs live inference through a local SparkSession.
