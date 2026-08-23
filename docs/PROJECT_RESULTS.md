# Final Project Results

## Data Engineering

- Raw DataSF source: ~666 MB
- Raw fields: 53
- Silver verified rows: 1,294,263
- Gold ML-ready rows: 1,231,991
- Gold columns: 29

## EDA

Construction cost is strongly right-skewed.

Selected Gold quantiles:

- median: approximately $7,000
- P90: approximately $80,000
- P95: approximately $165,000
- P99: approximately $510,000

## ML V2

Selected model:

**Structured + Description TF-IDF Linear Regression**

Validation:

- MAE: $34,906
- RMSE: $92,834
- R²: 0.2179
- Log-R²: 0.4785

Test:

- MAE: $41,718
- RMSE: $105,812
- R²: 0.2291
- Log-R²: 0.4641

Validation RMSE improvement versus median baseline: **15.87%**

## Text Ablation

Adding description NLP improved RMSE relative to the structured-only V2 model:

- Validation RMSE reduction: 11.97%
- Test RMSE reduction: 13.37%

## Explainable AI

Interpretable companion model:

- vocabulary: 1,500 terms
- Validation Log-R²: 0.4490
- Test Log-R²: 0.4301

Examples of higher-cost associations:

- remodel
- MEP
- stories
- kitchen
- crane
- shoring
- mechanical

## Final Application

The final ConstructIQ dashboard successfully performed live model inference.

Verified demonstration prediction:

**$71,057**

This is a model estimate, not a guaranteed final construction quotation.
