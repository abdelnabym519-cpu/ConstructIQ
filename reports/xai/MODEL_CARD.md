# ConstructIQ Model Card

## Production model

**Model:** Structured + Description TF-IDF Linear Regression

**Target:** revised construction cost, log10 transformed

**Supported cost domain:** $100 to $1,000,000

**Training period:** 2015-2023

**Validation period:** 2024

**Test period:** 2025+

## Production performance

Validation RMSE: $92,834.38

Validation R²: 0.2179

Validation Log-R²: 0.4785

Test RMSE: $105,812.16

Test R²: 0.2291

Test Log-R²: 0.4641

RMSE improvement over median baseline:
15.87%

## Explainability

The production model uses HashingTF for scalable text
representation. Hashing dimensions cannot be mapped reliably
to individual words because multiple terms can collide.

ConstructIQ therefore uses:

1. Structured-vs-text ablation analysis.
2. A separate CountVectorizer companion model with an explicit
   vocabulary for global text interpretation.

The companion model is explanatory only and is not used for
production predictions.

## Important limitations

- Predictions represent statistical estimates, not quotations.
- The model is trained on San Francisco permit data.
- Results should not be interpreted as causal relationships.
- Very unusual projects may have larger errors.
- Predictions are constrained to the declared training domain.
- Construction cost distributions are highly right-skewed.
- Historical permit valuation may differ from final delivered cost.
