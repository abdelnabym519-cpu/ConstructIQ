import csv
import json
import math
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
)
from pyspark.ml.regression import LinearRegression


GOLD = "/workspace/data/gold/construction_cost_ml"

ML_RESULTS = Path(
    "/workspace/reports/ml_v2/model_results_v2.json"
)

OUTPUT = Path("/workspace/reports/xai")

MIN_COST = 100.0
MAX_COST = 1_000_000.0

TRAIN_START = 2015
TRAIN_END = 2023
VALIDATION_YEAR = 2024
TEST_START = 2025

VOCAB_SIZE = 1500
MIN_DF = 20.0
TOP_N = 30


def metric(results, model, split):
    for row in results["metrics"]:
        if (
            row["model"] == model
            and row["split"] == split
        ):
            return row

    raise KeyError(
        f"Metric not found: {model}/{split}"
    )


def evaluate_log(predictions):
    row = predictions.agg(
        F.count("*").alias("rows"),

        F.sqrt(
            F.avg(
                F.pow(
                    F.col("target_cost_log10")
                    - F.col("prediction"),
                    2,
                )
            )
        ).alias("log_rmse"),

        F.sum(
            F.pow(
                F.col("target_cost_log10")
                - F.col("prediction"),
                2,
            )
        ).alias("sse"),

        F.sum(
            F.pow(
                F.col("target_cost_log10"),
                2,
            )
        ).alias("sum_y2"),

        F.avg(
            "target_cost_log10"
        ).alias("mean_y"),
    ).collect()[0]

    n = row["rows"]

    sst = (
        row["sum_y2"]
        - n * (row["mean_y"] ** 2)
    )

    r2 = (
        1.0 - row["sse"] / sst
        if sst and sst > 0
        else None
    )

    return {
        "rows": int(n),
        "log_rmse": float(row["log_rmse"]),
        "log_r2": float(r2),
    }


def write_csv(path, rows, fieldnames):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    spark = (
        SparkSession.builder
        .appName("ConstructIQ-Explainable-AI")
        .config(
            "spark.sql.shuffle.partitions",
            "8",
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("ConstructIQ — Explainable AI")
    print("=" * 96)

    # ------------------------------------------------------------
    # 1. Production-model ablation analysis
    # ------------------------------------------------------------

    with ML_RESULTS.open(
        encoding="utf-8"
    ) as f:
        results = json.load(f)

    structured_val = metric(
        results,
        "structured_lr_v2",
        "validation",
    )

    text_val = metric(
        results,
        "structured_text_lr_v2",
        "validation",
    )

    structured_test = metric(
        results,
        "structured_lr_v2",
        "test",
    )

    text_test = metric(
        results,
        "structured_text_lr_v2",
        "test",
    )

    ablation = {
        "validation": {
            "structured_rmse": structured_val["rmse"],
            "text_structured_rmse": text_val["rmse"],
            "rmse_reduction_percent": (
                (
                    structured_val["rmse"]
                    - text_val["rmse"]
                )
                / structured_val["rmse"]
                * 100.0
            ),
            "structured_r2": structured_val["r2"],
            "text_structured_r2": text_val["r2"],
            "structured_log_r2": structured_val["log_r2"],
            "text_structured_log_r2": text_val["log_r2"],
        },
        "test": {
            "structured_rmse": structured_test["rmse"],
            "text_structured_rmse": text_test["rmse"],
            "rmse_reduction_percent": (
                (
                    structured_test["rmse"]
                    - text_test["rmse"]
                )
                / structured_test["rmse"]
                * 100.0
            ),
            "structured_r2": structured_test["r2"],
            "text_structured_r2": text_test["r2"],
            "structured_log_r2": structured_test["log_r2"],
            "text_structured_log_r2": text_test["log_r2"],
        },
    }

    print()
    print("=== PRODUCTION MODEL ABLATION ===")

    for split in ["validation", "test"]:
        a = ablation[split]

        print(
            f"{split.upper():10} "
            f"| Structured RMSE "
            f"${a['structured_rmse']:>10,.0f} "
            f"| +Text RMSE "
            f"${a['text_structured_rmse']:>10,.0f} "
            f"| reduction "
            f"{a['rmse_reduction_percent']:>6.2f}%"
        )

        print(
            f"{'':10} "
            f"| R2 "
            f"{a['structured_r2']:.4f}"
            f" -> "
            f"{a['text_structured_r2']:.4f} "
            f"| LogR2 "
            f"{a['structured_log_r2']:.4f}"
            f" -> "
            f"{a['text_structured_log_r2']:.4f}"
        )

    # ------------------------------------------------------------
    # 2. Same scientific training population
    # ------------------------------------------------------------

    df = (
        spark.read.parquet(GOLD)
        .filter(
            (F.col("target_source") == "revised_cost")
            & (
                F.col("target_cost")
                >= MIN_COST
            )
            & (
                F.col("target_cost")
                <= MAX_COST
            )
            & (
                F.col("permit_creation_year")
                >= TRAIN_START
            )
            & F.col(
                "target_cost_log10"
            ).isNotNull()
        )
        .fillna(
            "",
            subset=["description"],
        )
    )

    train = df.filter(
        F.col(
            "permit_creation_year"
        ).between(
            TRAIN_START,
            TRAIN_END,
        )
    )

    validation = df.filter(
        F.col(
            "permit_creation_year"
        ) == VALIDATION_YEAR
    )

    test = df.filter(
        F.col(
            "permit_creation_year"
        ) >= TEST_START
    )

    print()
    print("=== INTERPRETABLE COMPANION DATA ===")
    print(f"Train      : {train.count():,}")
    print(f"Validation : {validation.count():,}")
    print(f"Test       : {test.count():,}")

    # ------------------------------------------------------------
    # 3. Interpretable vocabulary-based text model
    #
    # This is a companion explanatory model, NOT the production
    # prediction model. CountVectorizer provides an explicit
    # vocabulary so coefficients map back to real words.
    # ------------------------------------------------------------

    tokenizer = RegexTokenizer(
        inputCol="description",
        outputCol="tokens",
        pattern=r"\W+",
        gaps=True,
        minTokenLength=3,
        toLowercase=True,
    )

    remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="filtered_tokens",
    )

    vectorizer = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="term_counts",
        vocabSize=VOCAB_SIZE,
        minDF=MIN_DF,
        binary=True,
    )

    idf = IDF(
        inputCol="term_counts",
        outputCol="features",
        minDocFreq=5,
    )

    lr = LinearRegression(
        featuresCol="features",
        labelCol="target_cost_log10",
        predictionCol="prediction",
        solver="l-bfgs",
        maxIter=80,
        regParam=0.08,
        elasticNetParam=0.0,
        standardization=True,
    )

    pipeline = Pipeline(
        stages=[
            tokenizer,
            remover,
            vectorizer,
            idf,
            lr,
        ]
    )

    print()
    print(
        "Training interpretable "
        "CountVectorizer companion model..."
    )

    model = pipeline.fit(train)

    cv_model = model.stages[2]
    lr_model = model.stages[-1]

    vocabulary = cv_model.vocabulary
    coefficients = list(
        lr_model.coefficients
    )

    if len(vocabulary) != len(coefficients):
        raise RuntimeError(
            "Vocabulary/coefficient length mismatch"
        )

    validation_metrics = evaluate_log(
        model.transform(validation)
    )

    test_metrics = evaluate_log(
        model.transform(test)
    )

    # ------------------------------------------------------------
    # 4. Map coefficients to actual words
    # ------------------------------------------------------------

    terms = []

    for word, coefficient in zip(
        vocabulary,
        coefficients,
    ):
        multiplier = 10 ** coefficient

        terms.append({
            "term": word,
            "coefficient_log10": float(
                coefficient
            ),
            "prediction_multiplier": float(
                multiplier
            ),
            "direction": (
                "higher"
                if coefficient > 0
                else "lower"
            ),
        })

    positive = sorted(
        [
            row
            for row in terms
            if row["coefficient_log10"] > 0
        ],
        key=lambda x: x["coefficient_log10"],
        reverse=True,
    )[:TOP_N]

    negative = sorted(
        [
            row
            for row in terms
            if row["coefficient_log10"] < 0
        ],
        key=lambda x: x["coefficient_log10"],
    )[:TOP_N]

    fields = [
        "term",
        "coefficient_log10",
        "prediction_multiplier",
        "direction",
    ]

    write_csv(
        OUTPUT / "top_positive_terms.csv",
        positive,
        fields,
    )

    write_csv(
        OUTPUT / "top_negative_terms.csv",
        negative,
        fields,
    )

    # ------------------------------------------------------------
    # 5. Persist explainability summary
    # ------------------------------------------------------------

    summary = {
        "production_model": {
            "name": (
                "structured_text_lr_v2"
            ),
            "text_method": (
                "HashingTF + IDF"
            ),
            "explanation_limitation": (
                "HashingTF uses hashed dimensions, "
                "so exact word-to-coefficient mapping "
                "is not reliable because collisions "
                "can occur."
            ),
        },
        "ablation_analysis": ablation,
        "interpretable_companion_model": {
            "purpose": (
                "Global text explanation only; "
                "not the production predictor."
            ),
            "method": (
                "CountVectorizer + IDF + "
                "Linear Regression"
            ),
            "vocabulary_size": len(
                vocabulary
            ),
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "interpretation": (
            "Positive coefficients are associated "
            "with higher predicted log10 construction "
            "cost in the companion model. Negative "
            "coefficients are associated with lower "
            "predicted cost. These are predictive "
            "associations, not causal effects."
        ),
    }

    with (
        OUTPUT / "explainability_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # 6. Model Card
    # ------------------------------------------------------------

    best = results["selection"]

    model_card = f"""# ConstructIQ Model Card

## Production model

**Model:** Structured + Description TF-IDF Linear Regression

**Target:** revised construction cost, log10 transformed

**Supported cost domain:** ${MIN_COST:,.0f} to ${MAX_COST:,.0f}

**Training period:** {TRAIN_START}-{TRAIN_END}

**Validation period:** {VALIDATION_YEAR}

**Test period:** {TEST_START}+

## Production performance

Validation RMSE: ${text_val['rmse']:,.2f}

Validation R²: {text_val['r2']:.4f}

Validation Log-R²: {text_val['log_r2']:.4f}

Test RMSE: ${text_test['rmse']:,.2f}

Test R²: {text_test['r2']:.4f}

Test Log-R²: {text_test['log_r2']:.4f}

RMSE improvement over median baseline:
{best['rmse_improvement_vs_median_percent']:.2f}%

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
"""

    (
        OUTPUT / "MODEL_CARD.md"
    ).write_text(
        model_card,
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # 7. Console report
    # ------------------------------------------------------------

    print()
    print("=== COMPANION MODEL QUALITY ===")
    print(
        "Validation Log-R2 : "
        f"{validation_metrics['log_r2']:.4f}"
    )
    print(
        "Test Log-R2       : "
        f"{test_metrics['log_r2']:.4f}"
    )
    print(
        "Vocabulary size   : "
        f"{len(vocabulary):,}"
    )

    print()
    print("=== TOP TERMS — HIGHER COST ASSOCIATION ===")

    for row in positive[:15]:
        print(
            f"{row['term'][:25]:25} "
            f"| coef "
            f"{row['coefficient_log10']:>8.4f} "
            f"| x"
            f"{row['prediction_multiplier']:>6.2f}"
        )

    print()
    print("=== TOP TERMS — LOWER COST ASSOCIATION ===")

    for row in negative[:15]:
        print(
            f"{row['term'][:25]:25} "
            f"| coef "
            f"{row['coefficient_log10']:>8.4f} "
            f"| x"
            f"{row['prediction_multiplier']:>6.2f}"
        )

    print()
    print("=== OUTPUTS ===")
    print(
        " - reports/xai/"
        "explainability_summary.json"
    )
    print(
        " - reports/xai/"
        "top_positive_terms.csv"
    )
    print(
        " - reports/xai/"
        "top_negative_terms.csv"
    )
    print(
        " - reports/xai/MODEL_CARD.md"
    )

    print()
    print("XAI_EXPLAINABILITY_OK")
    print("=" * 96)

    spark.stop()


if __name__ == "__main__":
    main()
