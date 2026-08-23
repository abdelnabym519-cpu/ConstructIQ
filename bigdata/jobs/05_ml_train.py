import csv
import json
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.regression import (
    GBTRegressor,
    LinearRegression,
    RandomForestRegressor,
)


GOLD = "/workspace/data/gold/construction_cost_ml"
REPORT_DIR = Path("/workspace/reports/ml")
MODEL_DIR = "/workspace/models/constructiq_spark"

MIN_COST = 100.0
MAX_COST = 1_000_000.0

TRAIN_START = 2015
TRAIN_END = 2023
VALIDATION_YEAR = 2024
TEST_START = 2025


def add_cost_prediction(df):
    """Convert log10 prediction back to dollars and clip to model domain."""

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
    stats = (
        df
        .select(
            F.col("target_cost").cast("double").alias("y"),
            F.col("prediction_cost").cast("double").alias("p"),
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

            F.sum(
                F.pow(F.col("y"), 2)
            ).alias("sum_y2"),

            F.avg("y").alias("mean_y"),
        )
        .collect()[0]
    )

    rows = stats["rows"]

    if not rows:
        raise RuntimeError(
            f"No rows available for {model_name}/{split_name}"
        )

    sst = (
        stats["sum_y2"]
        - rows * (stats["mean_y"] ** 2)
    )

    r2 = (
        1.0 - (stats["sse"] / sst)
        if sst and sst > 0
        else None
    )

    return {
        "model": model_name,
        "split": split_name,
        "rows": int(rows),
        "mae": float(stats["mae"]),
        "rmse": float(stats["rmse"]),
        "r2": float(r2) if r2 is not None else None,
        "mape": float(stats["mape"]),
    }


def baseline_prediction(df, value):
    return df.withColumn(
        "prediction_cost",
        F.lit(float(value)),
    )


def print_metric(row):
    print(
        f"{row['model']:24} "
        f"| rows {row['rows']:>8,} "
        f"| MAE ${row['mae']:>11,.2f} "
        f"| RMSE ${row['rmse']:>11,.2f} "
        f"| R2 {row['r2']:>8.4f} "
        f"| MAPE {row['mape']:>8.2f}%"
    )


def main():
    spark = (
        SparkSession.builder
        .appName("ConstructIQ-ML-Model-Comparison")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.maxResultSize", "1g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("ConstructIQ — Machine Learning Training & Model Comparison")
    print("=" * 100)

    gold = spark.read.parquet(GOLD)

    # ------------------------------------------------------------------
    # 1. Scientifically defined training population
    # ------------------------------------------------------------------

    eligible = (
        gold
        .filter(
            (F.col("permit_creation_year") >= TRAIN_START)
            & (F.col("target_cost") >= MIN_COST)
            & (F.col("target_cost") <= MAX_COST)
            & F.col("target_cost_log10").isNotNull()
        )
        .withColumn(
            "supervisor_district_cat",
            F.col("supervisor_district").cast("string"),
        )
    )

    train = eligible.filter(
        (F.col("permit_creation_year") >= TRAIN_START)
        & (F.col("permit_creation_year") <= TRAIN_END)
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
    print("=== TRAINING POLICY ===")
    print(f"Cost domain : ${MIN_COST:,.0f} - ${MAX_COST:,.0f}")
    print(
        f"Train       : {TRAIN_START}-{TRAIN_END}"
    )
    print(
        f"Validation  : {VALIDATION_YEAR}"
    )
    print(
        f"Test        : {TEST_START}+"
    )

    print()
    print("=== TEMPORAL SPLITS ===")
    print(f"Train rows      : {train_rows:,}")
    print(f"Validation rows : {validation_rows:,}")
    print(f"Test rows       : {test_rows:,}")

    if min(train_rows, validation_rows, test_rows) == 0:
        raise RuntimeError(
            "One or more temporal ML splits contain zero rows"
        )

    # ------------------------------------------------------------------
    # 2. Feature policy
    # ------------------------------------------------------------------

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

    numeric_candidates = [
        c for c in numeric_candidates
        if c in train.columns
    ]

    # Prevent Imputer failures if any numeric feature is 100% null
    non_null_counts = (
        train
        .agg(
            *[
                F.count(F.col(c)).alias(c)
                for c in numeric_candidates
            ]
        )
        .collect()[0]
        .asDict()
    )

    numeric = [
        c
        for c in numeric_candidates
        if non_null_counts.get(c, 0) > 0
    ]

    dropped_numeric = sorted(
        set(numeric_candidates) - set(numeric)
    )

    print()
    print("=== FEATURE POLICY ===")
    print(f"Categorical features : {len(categorical)}")
    print(f"Numeric features     : {len(numeric)}")

    if dropped_numeric:
        print(
            "Dropped all-null numeric features:",
            ", ".join(dropped_numeric),
        )

    # Categorical nulls are explicit categories.
    train_ready = train.fillna(
        "__MISSING__",
        subset=categorical,
    )

    validation_ready = validation.fillna(
        "__MISSING__",
        subset=categorical,
    )

    test_ready = test.fillna(
        "__MISSING__",
        subset=categorical,
    )

    # ------------------------------------------------------------------
    # 3. Fit preprocessing ONLY on training data
    # ------------------------------------------------------------------

    indexed_columns = [
        f"{c}__idx"
        for c in categorical
    ]

    encoded_columns = [
        f"{c}__ohe"
        for c in categorical
    ]

    imputed_columns = [
        f"{c}__imputed"
        for c in numeric
    ]

    indexers = [
        StringIndexer(
            inputCol=input_col,
            outputCol=output_col,
            handleInvalid="keep",
            stringOrderType="frequencyDesc",
        )
        for input_col, output_col
        in zip(categorical, indexed_columns)
    ]

    encoder = OneHotEncoder(
        inputCols=indexed_columns,
        outputCols=encoded_columns,
        handleInvalid="keep",
        dropLast=True,
    )

    imputer = Imputer(
        inputCols=numeric,
        outputCols=imputed_columns,
        strategy="median",
    )

    assembler = VectorAssembler(
        inputCols=encoded_columns + imputed_columns,
        outputCol="features",
        handleInvalid="keep",
    )

    preprocessing = Pipeline(
        stages=indexers
        + [encoder, imputer, assembler]
    )

    print()
    print("Fitting preprocessing pipeline on TRAIN only...")

    preprocessing_model = preprocessing.fit(
        train_ready
    )

    (
        preprocessing_model
        .write()
        .overwrite()
        .save(f"{MODEL_DIR}/preprocessor")
    )

    def transform(frame):
        return (
            preprocessing_model
            .transform(frame)
            .select(
                "record_id",
                "permit_creation_year",
                "target_cost",
                F.col(
                    "target_cost_log10"
                ).alias("label"),
                "features",
            )
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

    train_ml = transform(train_ready)
    validation_ml = transform(validation_ready)
    test_ml = transform(test_ready)

    # Materialize caches
    train_ml.count()
    validation_ml.count()
    test_ml.count()

    vector_size = (
        train_ml
        .select("features")
        .first()["features"]
        .size
    )

    print(f"Final feature-vector size: {vector_size}")

    # ------------------------------------------------------------------
    # 4. Median baseline
    # ------------------------------------------------------------------

    median_cost = train.approxQuantile(
        "target_cost",
        [0.5],
        0.001,
    )[0]

    print()
    print(
        f"Training median baseline: "
        f"${median_cost:,.2f}"
    )

    metrics = []

    for split_name, frame in [
        ("validation", validation_ml),
        ("test", test_ml),
    ]:
        baseline = baseline_prediction(
            frame,
            median_cost,
        )

        metrics.append(
            evaluate(
                baseline,
                "median_baseline",
                split_name,
            )
        )

    # ------------------------------------------------------------------
    # 5. ML models
    # ------------------------------------------------------------------

    estimators = [
        (
            "linear_regression",
            LinearRegression(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction",
                maxIter=60,
                regParam=0.05,
                elasticNetParam=0.0,
                standardization=True,
            ),
        ),
        (
            "random_forest",
            RandomForestRegressor(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction",
                numTrees=35,
                maxDepth=10,
                maxBins=64,
                minInstancesPerNode=5,
                subsamplingRate=0.8,
                featureSubsetStrategy="sqrt",
                maxMemoryInMB=512,
                seed=42,
            ),
        ),
        (
            "gradient_boosted_trees",
            GBTRegressor(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction",
                maxIter=25,
                maxDepth=6,
                maxBins=64,
                stepSize=0.08,
                minInstancesPerNode=5,
                subsamplingRate=0.8,
                featureSubsetStrategy="sqrt",
                maxMemoryInMB=512,
                seed=42,
            ),
        ),
    ]

    fitted_models = {}

    for model_name, estimator in estimators:
        print()
        print("=" * 100)
        print(f"TRAINING: {model_name}")
        print("=" * 100)

        model = estimator.fit(train_ml)

        fitted_models[model_name] = model

        validation_prediction = add_cost_prediction(
            model.transform(validation_ml)
        )

        test_prediction = add_cost_prediction(
            model.transform(test_ml)
        )

        validation_metric = evaluate(
            validation_prediction,
            model_name,
            "validation",
        )

        test_metric = evaluate(
            test_prediction,
            model_name,
            "test",
        )

        metrics.extend(
            [
                validation_metric,
                test_metric,
            ]
        )

        print()
        print_metric(validation_metric)
        print_metric(test_metric)

    # ------------------------------------------------------------------
    # 6. Choose winner using validation RMSE only
    # ------------------------------------------------------------------

    ml_validation_metrics = [
        row
        for row in metrics
        if row["split"] == "validation"
        and row["model"] != "median_baseline"
    ]

    best_validation = min(
        ml_validation_metrics,
        key=lambda row: row["rmse"],
    )

    best_model_name = best_validation["model"]

    (
        fitted_models[best_model_name]
        .write()
        .overwrite()
        .save(f"{MODEL_DIR}/best_model")
    )

    baseline_validation = next(
        row
        for row in metrics
        if row["model"] == "median_baseline"
        and row["split"] == "validation"
    )

    improvement = (
        (
            baseline_validation["rmse"]
            - best_validation["rmse"]
        )
        / baseline_validation["rmse"]
        * 100.0
    )

    # ------------------------------------------------------------------
    # 7. Persist reproducible results
    # ------------------------------------------------------------------

    csv_path = REPORT_DIR / "model_metrics.csv"

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
            ],
        )

        writer.writeheader()
        writer.writerows(metrics)

    metadata = {
        "training_policy": {
            "target": "log10(target_cost)",
            "minimum_cost": MIN_COST,
            "maximum_cost": MAX_COST,
            "train_years": (
                f"{TRAIN_START}-{TRAIN_END}"
            ),
            "validation_year": VALIDATION_YEAR,
            "test_years": f"{TEST_START}+",
            "prediction_evaluation": (
                "Predictions inverse-transformed from "
                "log10 dollars and clipped to the "
                "declared model cost domain."
            ),
        },
        "split_rows": {
            "train": train_rows,
            "validation": validation_rows,
            "test": test_rows,
        },
        "features": {
            "categorical": categorical,
            "numeric": numeric,
            "vector_size": vector_size,
        },
        "baseline": {
            "training_median_cost": median_cost,
        },
        "selection": {
            "criterion": "validation RMSE",
            "best_model": best_model_name,
            "best_validation_rmse": (
                best_validation["rmse"]
            ),
            "rmse_improvement_vs_median_percent": (
                improvement
            ),
        },
        "metrics": metrics,
    }

    with (
        REPORT_DIR / "model_results.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # 8. Final console report
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 100)

    validation_metrics = sorted(
        [
            row
            for row in metrics
            if row["split"] == "validation"
        ],
        key=lambda row: row["rmse"],
    )

    for row in validation_metrics:
        print_metric(row)

    print()
    print("=" * 100)
    print("FINAL TEST RESULTS")
    print("=" * 100)

    test_metrics = sorted(
        [
            row
            for row in metrics
            if row["split"] == "test"
        ],
        key=lambda row: row["rmse"],
    )

    for row in test_metrics:
        print_metric(row)

    print()
    print("=== MODEL SELECTION ===")
    print(f"Best model          : {best_model_name}")
    print(
        f"Validation RMSE     : "
        f"${best_validation['rmse']:,.2f}"
    )
    print(
        f"RMSE improvement vs median baseline: "
        f"{improvement:.2f}%"
    )

    print()
    print("Outputs:")
    print(" - reports/ml/model_metrics.csv")
    print(" - reports/ml/model_results.json")
    print(
        " - models/constructiq_spark/preprocessor"
    )
    print(
        " - models/constructiq_spark/best_model"
    )

    print()
    print("ML_TRAINING_OK")
    print("=" * 100)

    train_ml.unpersist()
    validation_ml.unpersist()
    test_ml.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()
