from pyspark.sql import SparkSession
from pyspark.sql import functions as F


SILVER = "/workspace/data/silver/building_permits"
GOLD = "/workspace/data/gold/construction_cost_ml"


def yes_flag(column_name):
    value = F.lower(F.trim(F.col(column_name)))

    return (
        F.when(value.isin("y", "yes", "true", "1"), F.lit(1))
        .when(value.isin("n", "no", "false", "0"), F.lit(0))
        .otherwise(F.lit(None).cast("int"))
    )


def main():
    spark = (
        SparkSession.builder
        .appName("ConstructIQ-Gold-ML-Features")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("=" * 72)
    print("ConstructIQ — Gold Layer / ML Feature Engineering")
    print("=" * 72)

    df = spark.read.parquet(SILVER)

    silver_rows = df.count()

    print(f"Silver rows : {silver_rows:,}")

    # --------------------------------------------------
    # 1. Define ML target
    # --------------------------------------------------
    #
    # Revised Cost is preferred because it represents the
    # most recently revised project valuation.
    #
    # Estimated Cost is used only when Revised Cost is absent.
    #
    # Neither field is retained as an input feature,
    # preventing direct target leakage.
    # --------------------------------------------------

    df = (
        df
        .withColumn(
            "target_cost",
            F.when(
                F.col("dq_valid_revised_cost"),
                F.col("revised_cost"),
            ).when(
                F.col("dq_valid_estimated_cost"),
                F.col("estimated_cost"),
            ),
        )
        .withColumn(
            "target_source",
            F.when(
                F.col("dq_valid_revised_cost"),
                F.lit("revised_cost"),
            ).when(
                F.col("dq_valid_estimated_cost"),
                F.lit("estimated_cost"),
            ),
        )
    )

    target_df = df.filter(
        F.col("target_cost").isNotNull()
        & (F.col("target_cost") > 0)
    )

    positive_target_rows = target_df.count()

    print(f"Rows with positive target : {positive_target_rows:,}")

    # --------------------------------------------------
    # 2. Robust outlier bounds
    # --------------------------------------------------

    quantiles = target_df.approxQuantile(
        "target_cost",
        [0.01, 0.99],
        0.001,
    )

    if len(quantiles) != 2:
        raise RuntimeError("Unable to calculate target-cost quantiles")

    lower_cost, upper_cost = quantiles

    print(f"Target P01 : ${lower_cost:,.2f}")
    print(f"Target P99 : ${upper_cost:,.2f}")

    df = target_df.filter(
        (F.col("target_cost") >= F.lit(lower_cost))
        & (F.col("target_cost") <= F.lit(upper_cost))
    )

    # --------------------------------------------------
    # 3. Derived numerical features
    # --------------------------------------------------

    df = (
        df
        .withColumn(
            "target_cost_log10",
            F.log10(F.col("target_cost")),
        )
        .withColumn(
            "story_change",
            F.coalesce(
                F.col("number_of_proposed_stories"),
                F.lit(0.0),
            )
            - F.coalesce(
                F.col("number_of_existing_stories"),
                F.lit(0.0),
            ),
        )
        .withColumn(
            "unit_change",
            F.coalesce(
                F.col("proposed_units"),
                F.lit(0),
            )
            - F.coalesce(
                F.col("existing_units"),
                F.lit(0),
            ),
        )
    )

    # --------------------------------------------------
    # 4. Binary construction features
    # --------------------------------------------------

    if "adu" in df.columns:
        df = df.withColumn("is_adu", yes_flag("adu"))

    if "site_permit" in df.columns:
        df = df.withColumn(
            "is_site_permit",
            yes_flag("site_permit"),
        )

    if "reroof" in df.columns:
        df = df.withColumn(
            "is_reroof",
            yes_flag("reroof"),
        )

    # --------------------------------------------------
    # 5. ML-ready feature selection
    # --------------------------------------------------

    feature_columns = [
        # Identifiers retained for traceability, not training.
        "record_id",
        "permit_number",

        # Target
        "target_cost",
        "target_cost_log10",
        "target_source",

        # Temporal
        "permit_creation_year",
        "permit_creation_month",

        # Location
        "zipcode",
        "supervisor_district",
        "neighborhoods_analysis_boundaries",

        # Permit information
        "permit_type",
        "permit_type_definition",
        "application_submission_method",

        # Building use
        "existing_use",
        "proposed_use",

        # Scale
        "number_of_existing_stories",
        "number_of_proposed_stories",
        "story_change",

        "existing_units",
        "proposed_units",
        "unit_change",

        # Construction type
        "existing_construction_type",
        "proposed_construction_type",
        "existing_construction_type_description",
        "proposed_construction_type_description",

        # Project flags
        "is_adu",
        "is_site_permit",
        "is_reroof",
    ]

    feature_columns = [
        c for c in feature_columns
        if c in df.columns
    ]

    gold = df.select(*feature_columns)

    # --------------------------------------------------
    # 6. Final dataset-level validation
    # --------------------------------------------------

    gold = gold.dropDuplicates(["record_id"])

    gold_rows = gold.count()

    if gold_rows == 0:
        raise RuntimeError("Gold dataset contains zero rows")

    target_stats = gold.agg(
        F.min("target_cost").alias("min_cost"),
        F.avg("target_cost").alias("avg_cost"),
        F.expr(
            "percentile_approx(target_cost, 0.5)"
        ).alias("median_cost"),
        F.max("target_cost").alias("max_cost"),
    ).collect()[0]

    source_stats = (
        gold
        .groupBy("target_source")
        .count()
        .orderBy(F.desc("count"))
        .collect()
    )

    print()
    print("=== GOLD QUALITY ===")
    print(f"Gold rows       : {gold_rows:,}")
    print(f"Gold columns    : {len(gold.columns)}")
    print(
        f"Min target cost : "
        f"${target_stats['min_cost']:,.2f}"
    )
    print(
        f"Mean target cost: "
        f"${target_stats['avg_cost']:,.2f}"
    )
    print(
        f"Median cost     : "
        f"${target_stats['median_cost']:,.2f}"
    )
    print(
        f"Max target cost : "
        f"${target_stats['max_cost']:,.2f}"
    )

    print()
    print("Target source distribution:")

    for row in source_stats:
        print(
            f"  {row['target_source']}: "
            f"{row['count']:,}"
        )

    # --------------------------------------------------
    # 7. Persist ML-ready Gold dataset
    # --------------------------------------------------

    (
        gold
        .repartition(8)
        .write
        .mode("overwrite")
        .option("compression", "snappy")
        .parquet(GOLD)
    )

    # --------------------------------------------------
    # 8. Read-back verification
    # --------------------------------------------------

    verify = spark.read.parquet(GOLD)

    verified_rows = verify.count()

    if verified_rows != gold_rows:
        raise RuntimeError(
            "Gold verification failed: "
            f"expected={gold_rows}, actual={verified_rows}"
        )

    print()
    print("=== GOLD SCHEMA ===")
    verify.printSchema()

    print()
    print(f"Verified rows : {verified_rows:,}")
    print("GOLD_FEATURES_OK")
    print("=" * 72)

    spark.stop()


if __name__ == "__main__":
    main()
