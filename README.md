# 🏗️ ConstructIQ

**Big Data & AI Construction Cost Intelligence**

ConstructIQ is an end-to-end academic Big Data and AI platform for processing real-world construction permit data, analyzing construction trends, and estimating revised construction costs for projects in San Francisco.

## Project Highlights

| Metric | Result |
|---|---:|
| Raw source | DataSF Building Permits |
| Raw source size | ~666 MB |
| Silver verified rows | 1,294,263 |
| ML-ready Gold rows | 1,231,991 |
| Gold columns | 29 |
| Production model | Structured + Description TF-IDF Linear Regression |
| Validation RMSE | $92,834 |
| Validation R² | 0.2179 |
| Validation Log-R² | 0.4785 |
| Final Test RMSE | $105,812 |
| Final Test R² | 0.2291 |
| Final Test Log-R² | 0.4641 |
| Improvement vs median baseline | 15.87% |

## Big Data Pipeline

The project uses the official San Francisco Building Permits dataset from DataSF (`i98e-djp9`).

```text
DataSF Building Permits
        |
        v
Bronze — Raw / Immutable
        |
        v
Silver — Cleaned / Validated / Typed
        |
        v
Gold — ML-ready Features
        |
        +----> EDA / Business Analytics
        +----> Machine Learning + NLP
        +----> Explainable AI
        |
        v
ConstructIQ Streamlit Dashboard
```

Verified Silver rows: **1,294,263**

Final Gold dataset: **1,231,991 rows × 29 columns**

## Machine Learning

Temporal evaluation:

- Train: 2015–2023
- Validation: 2024
- Test: 2025+

V2 policy:

- target source: `revised_cost` only
- supported cost domain: `$100–$1,000,000`
- target: `log10(revised construction cost)`

Selected model:

**Structured + Description TF-IDF Linear Regression**

### Final Performance

| Metric | Validation | Test |
|---|---:|---:|
| MAE | $34,906 | $41,718 |
| RMSE | $92,834 | $105,812 |
| R² | 0.2179 | 0.2291 |
| Log-RMSE | 0.4495 | 0.4773 |
| Log-R² | 0.4785 | 0.4641 |

Validation RMSE improved by **15.87%** versus the median baseline.

## Explainable AI

ConstructIQ adds:

1. structured-vs-text ablation analysis
2. an interpretable CountVectorizer companion model
3. a Model Card

Examples of terms associated with higher predicted cost include:

- remodel
- MEP
- stories
- kitchen
- crane
- shoring
- mechanical

These are predictive associations, not causal effects.

## Dashboard

The final dashboard contains:

1. Executive Overview
2. Construction Analytics
3. AI Cost Intelligence
4. Explainable AI
5. Model Performance
6. Architecture

A verified live dashboard prediction produced **$71,057**.

## Technology Stack

**Big Data:** Apache Spark 3.5.9, PySpark, Parquet, Snappy, Medallion Architecture

**AI / ML:** Spark MLlib, Linear Regression, Random Forest, Gradient Boosted Trees

**NLP:** RegexTokenizer, StopWordsRemover, HashingTF, IDF, CountVectorizer

**Analytics:** Pandas, Plotly, Streamlit

**Engineering:** Docker, Docker Compose, Git, GitHub

## Run Spark Foundation

```bash
docker compose -f docker-compose.spark.yml run --rm spark
```

Expected result:

`SPARK_FOUNDATION_OK`

## Run the Dashboard

Build:

```bash
docker build -t constructiq-dashboard:latest -f Dockerfile.dashboard .
```

Run:

```bash
docker run --name constructiq-dashboard   -p 8502:8501   -v "$(pwd)/models:/workspace/models:ro"   constructiq-dashboard:latest
```

Open:

`http://localhost:8502`

## Limitations

- Predictions are analytical estimates, not contractor quotations.
- The model is trained primarily on San Francisco permit data.
- Construction costs are highly right-skewed.
- Very unusual projects can have larger errors.
- Predictions are constrained to the supported model domain.
- Permit valuation may differ from final delivered construction cost.
- Explainability terms represent association, not causation.

## Open-Source Attribution

ConstructIQ was bootstrapped from the MIT-licensed project:

**LHB-Group/Civil-Work-Bidding-And-Investment-Helper**

The original MIT license is retained.

ConstructIQ subsequently adds the PySpark medallion pipeline, large-scale DataSF processing, ML/NLP V2 experimentation, temporal evaluation, Explainable AI, reproducible analytics, and the final ConstructIQ dashboard.

See `docs/ATTRIBUTION.md`.

## License

MIT License. See `LICENSE`.

## Academic Scope

**Real-world Data → Big Data → EDA → Machine Learning → NLP → Explainable AI → Interactive Dashboard**
