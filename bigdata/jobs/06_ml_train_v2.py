import csv
import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import StorageLevel

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    Imputer,
    VectorAssembler,
    RegexTokenizer,
    StopWordsRemover,
    HashingTF,
    IDF,
)
from pyspark.ml.regression import LinearRegression


GOLD = "/workspace/data/gold/construction_cost_ml"

REPORT_DIR = Path("/workspace/reports/ml_v2")
MODEL_DIR = "/workspace/models/constructiq_spark_v2"

MIN_COST = 100.0
MAX_COST = 1_000_000.0

TRAIN_START = 2015
TRAIN_END = 2023
VALIDATION_YEAR = 2024
TEST_START = 2025

TEXT_FEATURES = 4096


def add_dollar_prediction(df):
    return (
        df
        .withColumn(
            "prediction_cost_raw",
            F.pow(F.lit(10.0), F.col("prediction")),
        )
        .withColumn(
            "prediction_cost",
            F.greatest(
                F.lit(MIN_COST),
                F.least(
                    F.lit(MAX_COST),
                    F.col("prediction_cost_raw"),
                ),
            ),
        )
    )


def evaluate(df, model_name, split_name):
    row = (
        df
        .select(
            F.col("target_cost").cast("double").alias("y"),
            F.col("prediction_cost").cast("double").alias("p"),
            F.col("label").cast("double").alias("log_y"),
            F.col("prediction").cast("double").alias("log_p"),
        )
        .agg(
            F.count("*").alias("rows"),

            F.avg(
                F.abs(F.col("y") - F.col("p"))
            ).alias("mae"),

            F.sqrt(
                F.avg(
                    F.pow(F.col("y") - F.col("p"), 2)
                )
            ).alias("rmse"),

            (
                F.avg(
                    F.abs(
                        (F.col("y") - F.col("p"))
                        / F.col("y")
                    )
                )
                * 100.0
            ).alias("mape"),

            F.sum(
                F.pow(F.col("y") - F.col("p"), 2)
            ).alias("sse"),

            F.sum(F.pow(F.col("y"), 2)).alias("sum_y2"),
            F.avg("y").alias("mean_y"),

            F.avg(
                F.abs(F.col("log_y") - F.col("log_p"))
            ).alias("log_mae"),

            F.sqrt(
                F.avg(
                    F.pow(F.col("log_y") - F.col("log_p"), 2)
                )
            ).alias("log_rmse"),

            F.sum(
                F.pow(F.col("log_y") - F.col("log_p"), 2)
            ).alias("log_sse"),

            F.sum(
                F.pow(F.col("log_y"), 2)
            ).alias("sum_log_y2"),

            F.avg("log_y").alias("mean_log_y"),
        )
        .collect()[0]
    )

    n = row["rows"]

    sst = row["sum_y2"] - n * (row["mean_y"] ** 2)
    log_sst = (
        row["sum_log_y2"]
        - n * (row["mean_log_y"] ** 2)
    )

    r2 = (
        1.0 - row["sse"] / sst
        if sst and sst > 0
        else None
    )

    log_r2 = (
        1.0 - row["log_sse"] / log_sst
        if log_sst and log_sst > 0
        else None
    )

    return {
        "model": model_name,
        "split": split_name,
        "rows": int(n),
        "mae": float(row["mae"]),
        "rmse": float(row["rmse"]),
        "r2": float(r2),
        "mape": float(row["mape"]),
        "log_mae": float(row["log_mae"]),
        "log_rmse": float(row["log_rmse"]),
        "log_r2": float(log_r2),
    }


def print_metric(m):
    print(
        f"{m['model']:28} "
        f"| rows {m['rows']:>7,} "
        f"| MAE ${m['mae']:>10,.0f} "
        f"| RMSE ${m['rmse']:>10,.0f} "
        f"| R2 {m['r2']:>7.4f} "
        f"| MAPE {m['mape']:>8.1f}% "
        f"| LogRMSE {m['log_rmse']:>6.4f} "
        f"| LogR2 {m['log_r2']:>7.4f}"
    )


def prepare(frame, categorical):
    return (
        frame
        .fillna("__MISSING__", subset=categorical)
        .fillna("", subset=["description"])
    )


def main():
    spark = (
        SparkSession.builder
        .appName("ConstructIQ-ML-V2-RevisedCost-NLP")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.maxResultSize", "1g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("ConstructIQ — ML V2: Revised Cost + Project Description NLP")
    print("=" * 110)

    gold = spark.read.parquet(GOLD)

    # ------------------------------------------------------------
    # 1. Revised-cost-only population
    # ------------------------------------------------------------

    eligible = (
        gold
        .filter(
            (F.col("target_source") == "revised_cost")
            & (F.col("target_cost") >= MIN_COST)
            & (F.col("target_cost") <= MAX_COST)
            & (F.col("permit_creation_year") >= TRAIN_START)
            & F.col("target_cost_log10").isNotNull()
        )
        .withColumn(
            "supervisor_district_cat",
            F.col("supervisor_district").cast("string"),
        )
    )

    train = eligible.filter(
        F.col("permit_creation_year").between(
            TRAIN_START,
            TRAIN_END,
        )
    )

    validation = eligible.filter(
        F.col("permit_creation_year") == VALIDATION_YEAR
    )

    test = eligible.filter(
        F.col("permit_creation_year") >= TEST_START
    )

    train_rows = train.count()
    validation_rows = validation.count()
    test_rows = test.count()

    print()
    print("=== V2 TRAINING POLICY ===")
    print("Target source : revised_cost only")
    print(f"Cost domain   : ${MIN_COST:,.0f} - ${MAX_COST:,.0f}")
    print(f"Train         : {TRAIN_START}-{TRAIN_END}")
    print(f"Validation    : {VALIDATION_YEAR}")
    print(f"Test          : {TEST_START}+")

    print()
    print("=== TEMPORAL SPLITS ===")
    print(f"Train rows      : {train_rows:,}")
    print(f"Validation rows : {validation_rows:,}")
    print(f"Test rows       : {test_rows:,}")

    if min(train_rows, validation_rows, test_rows) == 0:
        raise RuntimeError("Temporal split contains zero rows")

    # ------------------------------------------------------------
    # 2. Description coverage
    # ------------------------------------------------------------

    print()
    print("=== DESCRIPTION COVERAGE ===")

    for name, frame in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:
        row = frame.agg(
            F.count("*").alias("rows"),
            F.sum(
                F.when(
                    F.col("description").isNotNull()
                    & (F.length(F.trim(F.col("description"))) > 0),
                    1,
                ).otherwise(0)
            ).alias("with_description"),
        ).collect()[0]

        pct = row["with_description"] / row["rows"] * 100.0

        print(
            f"{name:10}: "
            f"{row['with_description']:>8,} / "
            f"{row['rows']:>8,} "
            f"({pct:6.2f}%)"
        )

    # ------------------------------------------------------------
    # 3. Structured features
    # ------------------------------------------------------------

    categorical = [
        "zipcode",
        "supervisor_district_cat",
        "neighborhoods_analysis_boundaries",
        "permit_type_definition",
        "application_submission_method",
        "existing_use",
        "proposed_use",
        "existing_construction_type_description",
        "proposed_construction_type_description",
    ]

    numeric_candidates = [
        "permit_creation_year",
        "permit_creation_month",
        "number_of_existing_stories",
        "number_of_proposed_stories",
        "story_change",
        "existing_units",
        "proposed_units",
        "unit_change",
        "is_adu",
        "is_site_permit",
        "is_reroof",
    ]

    categorical = [
        c for c in categorical
        if c in train.columns
    ]

    counts = (
        train
        .agg(
            *[
                F.count(F.col(c)).alias(c)
                for c in numeric_candidates
                if c in train.columns
            ]
        )
        .collect()[0]
        .asDict()
    )

    numeric = [
        c
        for c in numeric_candidates
        if c in train.columns
        and counts.get(c, 0) > 0
    ]

    train_ready = prepare(train, categorical)
    validation_ready = prepare(validation, categorical)
    test_ready = prepare(test, categorical)

    indexed = [f"{c}__idx" for c in categorical]
    encoded = [f"{c}__ohe" for c in categorical]
    imputed = [f"{c}__imp" for c in numeric]

    indexers = [
        StringIndexer(
            inputCol=c,
            outputCol=i,
            handleInvalid="keep",
            stringOrderType="frequencyDesc",
        )
        for c, i in zip(categorical, indexed)
    ]

    encoder = OneHotEncoder(
        inputCols=indexed,
        outputCols=encoded,
        handleInvalid="keep",
        dropLast=True,
    )

    imputer = Imputer(
        inputCols=numeric,
        outputCols=imputed,
        strategy="median",
    )

    # ------------------------------------------------------------
    # 4. Structured-only pipeline
    # ------------------------------------------------------------

    structured_assembler = VectorAssembler(
        inputCols=encoded + imputed,
        outputCol="features",
        handleInvalid="keep",
    )

    structured_lr = LinearRegression(
        featuresCol="features",
        labelCol="target_cost_log10",
        predictionCol="prediction",
        solver="l-bfgs",
        maxIter=80,
        regParam=0.05,
        elasticNetParam=0.0,
        standardization=True,
    )

    structured_pipeline = Pipeline(
        stages=indexers
        + [encoder, imputer, structured_assembler, structured_lr]
    )

    # ------------------------------------------------------------
    # 5. Structured + NLP pipeline
    # ------------------------------------------------------------

    tokenizer = RegexTokenizer(
        inputCol="description",
        outputCol="description_tokens",
        pattern=r"\W+",
        gaps=True,
        minTokenLength=2,
        toLowercase=True,
    )

    stopwords = StopWordsRemover(
        inputCol="description_tokens",
        outputCol="description_filtered",
    )

    hashing_tf = HashingTF(
        inputCol="description_filtered",
        outputCol="description_tf",
        numFeatures=TEXT_FEATURES,
        binary=True,
    )

    idf = IDF(
        inputCol="description_tf",
        outputCol="description_tfidf",
        minDocFreq=5,
    )

    text_assembler = VectorAssembler(
        inputCols=encoded + imputed + ["description_tfidf"],
        outputCol="features",
        handleInvalid="keep",
    )

    text_lr = LinearRegression(
        featuresCol="features",
        labelCol="target_cost_log10",
        predictionCol="prediction",
        solver="l-bfgs",
        maxIter=80,
        regParam=0.05,
        elasticNetParam=0.0,
        standardization=True,
    )

    text_pipeline = Pipeline(
        stages=indexers
        + [
            encoder,
            imputer,
            tokenizer,
            stopwords,
            hashing_tf,
            idf,
            text_assembler,
            text_lr,
        ]
    )

    # ------------------------------------------------------------
    # 6. Median baseline
    # ------------------------------------------------------------

    median_cost = train.approxQuantile(
        "target_cost",
        [0.5],
        0.001,
    )[0]

    median_log = float(__import__("math").log10(median_cost))

    metrics = []

    for split_name, frame in [
        ("validation", validation_ready),
        ("test", test_ready),
    ]:
        baseline = (
            frame
            .withColumn("label", F.col("target_cost_log10"))
            .withColumn("prediction", F.lit(median_log))
            .withColumn("prediction_cost", F.lit(median_cost))
        )

        metrics.append(
            evaluate(
                baseline,
                "median_baseline",
                split_name,
            )
        )

    print()
    print(f"Training median baseline: ${median_cost:,.2f}")

    # ------------------------------------------------------------
    # 7. Train structured model
    # ------------------------------------------------------------

    print()
    print("=" * 110)
    print("TRAINING V2: structured_linear_regression")
    print("=" * 110)

    structured_model = structured_pipeline.fit(train_ready)

    structured_validation = add_dollar_prediction(
        structured_model
        .transform(validation_ready)
        .withColumn("label", F.col("target_cost_log10"))
    ).persist(StorageLevel.MEMORY_AND_DISK)

    structured_test = add_dollar_prediction(
        structured_model
        .transform(test_ready)
        .withColumn("label", F.col("target_cost_log10"))
    ).persist(StorageLevel.MEMORY_AND_DISK)

    structured_val_metric = evaluate(
        structured_validation,
        "structured_lr_v2",
        "validation",
    )

    structured_test_metric = evaluate(
        structured_test,
        "structured_lr_v2",
        "test",
    )

    metrics.extend([
        structured_val_metric,
        structured_test_metric,
    ])

    print_metric(structured_val_metric)
    print_metric(structured_test_metric)

    structured_validation.unpersist()
    structured_test.unpersist()

    # ------------------------------------------------------------
    # 8. Train text + structured model
    # ------------------------------------------------------------

    print()
    print("=" * 110)
    print("TRAINING V2: structured_plus_description_tfidf")
    print("=" * 110)

    text_model = text_pipeline.fit(train_ready)

    text_validation = add_dollar_prediction(
        text_model
        .transform(validation_ready)
        .withColumn("label", F.col("target_cost_log10"))
    ).persist(StorageLevel.MEMORY_AND_DISK)

    text_test = add_dollar_prediction(
        text_model
        .transform(test_ready)
        .withColumn("label", F.col("target_cost_log10"))
    ).persist(StorageLevel.MEMORY_AND_DISK)

    text_val_metric = evaluate(
        text_validation,
        "structured_text_lr_v2",
        "validation",
    )

    text_test_metric = evaluate(
        text_test,
        "structured_text_lr_v2",
        "test",
    )

    metrics.extend([
        text_val_metric,
        text_test_metric,
    ])

    print_metric(text_val_metric)
    print_metric(text_test_metric)

    text_validation.unpersist()
    text_test.unpersist()

    # ------------------------------------------------------------
    # 9. Select winner by validation RMSE
    # ------------------------------------------------------------

    validation_metrics = [
        m for m in metrics
        if m["split"] == "validation"
    ]

    best = min(
        validation_metrics,
        key=lambda x: x["rmse"],
    )

    best_model_name = best["model"]

    if best_model_name == "structured_text_lr_v2":
        best_pipeline = text_model
    elif best_model_name == "structured_lr_v2":
        best_pipeline = structured_model
    else:
        best_pipeline = None

    if best_pipeline is not None:
        (
            best_pipeline
            .write()
            .overwrite()
            .save(f"{MODEL_DIR}/best_pipeline")
        )

    baseline = next(
        m for m in validation_metrics
        if m["model"] == "median_baseline"
    )

    improvement = (
        (baseline["rmse"] - best["rmse"])
        / baseline["rmse"]
        * 100.0
    )

    # ------------------------------------------------------------
    # 10. Persist reports
    # ------------------------------------------------------------

    csv_path = REPORT_DIR / "model_metrics_v2.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "split",
                "rows",
                "mae",
                "rmse",
                "r2",
                "mape",
                "log_mae",
                "log_rmse",
                "log_r2",
            ],
        )

        writer.writeheader()
        writer.writerows(metrics)

    metadata = {
        "version": "v2",
        "target_policy": {
            "source": "revised_cost only",
            "minimum_cost": MIN_COST,
            "maximum_cost": MAX_COST,
            "target_transform": "log10",
        },
        "temporal_split": {
            "train": f"{TRAIN_START}-{TRAIN_END}",
            "validation": VALIDATION_YEAR,
            "test": f"{TEST_START}+",
        },
        "split_rows": {
            "train": train_rows,
            "validation": validation_rows,
            "test": test_rows,
        },
        "text_features": {
            "column": "description",
            "method": "RegexTokenizer + StopWords + HashingTF + IDF",
            "hashing_dimensions": TEXT_FEATURES,
        },
        "selection": {
            "criterion": "validation dollar RMSE",
            "best_model": best_model_name,
            "validation_rmse": best["rmse"],
            "rmse_improvement_vs_median_percent": improvement,
        },
        "metrics": metrics,
    }

    with (
        REPORT_DIR / "model_results_v2.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------
    # 11. Final report
    # ------------------------------------------------------------

    print()
    print("=" * 110)
    print("V2 VALIDATION MODEL COMPARISON")
    print("=" * 110)

    for metric in sorted(
        validation_metrics,
        key=lambda x: x["rmse"],
    ):
        print_metric(metric)

    print()
    print("=" * 110)
    print("V2 FINAL TEST RESULTS")
    print("=" * 110)

    for metric in sorted(
        [
            m for m in metrics
            if m["split"] == "test"
        ],
        key=lambda x: x["rmse"],
    ):
        print_metric(metric)

    print()
    print("=== V2 MODEL SELECTION ===")
    print(f"Best model      : {best_model_name}")
    print(f"Validation RMSE : ${best['rmse']:,.2f}")
    print(f"Validation R²   : {best['r2']:.4f}")
    print(f"Log-space R²    : {best['log_r2']:.4f}")
    print(
        "RMSE improvement vs median baseline: "
        f"{improvement:.2f}%"
    )

    print()
    print("Outputs:")
    print(" - reports/ml_v2/model_metrics_v2.csv")
    print(" - reports/ml_v2/model_results_v2.json")

    if best_pipeline is not None:
        print(" - models/constructiq_spark_v2/best_pipeline")

    print()
    print("ML_V2_TRAINING_OK")
    print("=" * 110)

    spark.stop()


if __name__ == "__main__":
    main()
