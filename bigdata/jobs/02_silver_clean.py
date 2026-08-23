import re

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window


BRONZE = "/workspace/data/bronze/building_permits"
SILVER = "/workspace/data/silver/building_permits"


def snake_case(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def clean_strings(df):
    string_columns = [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, T.StringType)
    ]

    for column in string_columns:
        value = F.trim(F.col(column))

        df = df.withColumn(
            column,
            F.when(
                value.isNull()
                | (value == "")
                | (F.lower(value).isin("null", "none", "nan", "n/a")),
                F.lit(None),
            ).otherwise(value),
        )

    return df


def parse_date(column):
    return F.coalesce(
        # Current DataSF format, e.g. 2019/09/19 03:31:40 PM
        F.to_date(F.col(column), "yyyy/MM/dd hh:mm:ss a"),

        # Additional historical/defensive formats
        F.to_date(F.col(column), "yyyy/MM/dd HH:mm:ss"),
        F.to_date(F.col(column), "yyyy/MM/dd"),
        F.to_date(F.col(column), "MM/dd/yyyy hh:mm:ss a"),
        F.to_date(F.col(column), "MM/dd/yyyy HH:mm:ss"),
        F.to_date(F.col(column), "MM/dd/yyyy"),
        F.to_date(F.col(column), "yyyy-MM-dd"),

        # Last-resort date-prefix parsing
        F.to_date(F.substring(F.col(column), 1, 10), "yyyy/MM/dd"),
        F.to_date(F.substring(F.col(column), 1, 10), "MM/dd/yyyy"),
        F.to_date(F.substring(F.col(column), 1, 10), "yyyy-MM-dd"),
    )


def main():
    spark = (
        SparkSession.builder
        .appName("ConstructIQ-Silver-Building-Permits")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("=" * 72)
    print("ConstructIQ — Silver Layer")
    print("=" * 72)

    df = spark.read.parquet(BRONZE)

    bronze_rows = df.count()
    bronze_columns = len(df.columns)

    print(f"Bronze rows    : {bronze_rows:,}")
    print(f"Bronze columns : {bronze_columns}")

    # --------------------------------------------------
    # 1. Normalize column names
    # --------------------------------------------------

    renamed = [snake_case(c) for c in df.columns]

    if len(renamed) != len(set(renamed)):
        raise RuntimeError("Column-name normalization created duplicates")

    df = df.toDF(*renamed)

    # --------------------------------------------------
    # 2. Normalize empty/null string values
    # --------------------------------------------------

    df = clean_strings(df)

    # --------------------------------------------------
    # 3. Date normalization
    # --------------------------------------------------

    date_columns = [
        "permit_creation_date",
        "current_status_date",
        "filed_date",
        "issued_date",
        "completed_date",
        "first_construction_document_date",
        "approved_date",
        "structural_notification_date",
        "expiration_date",
        "last_permit_activity_date",
        "data_as_of",
        "data_loaded_at",
    ]

    for column in date_columns:
        if column in df.columns:
            df = df.withColumn(column, parse_date(column))

    # --------------------------------------------------
    # 4. Numeric normalization
    # --------------------------------------------------

    integer_columns = [
        "permit_type",
        "street_number",
        "existing_units",
        "proposed_units",
        "plansets",
        "existing_construction_type",
        "proposed_construction_type",
        "supervisor_district",
    ]

    double_columns = [
        "estimated_cost",
        "revised_cost",
        "number_of_existing_stories",
        "number_of_proposed_stories",
    ]

    for column in integer_columns:
        if column in df.columns:
            df = df.withColumn(
                column,
                F.regexp_replace(F.col(column), ",", "").cast("int"),
            )

    for column in double_columns:
        if column in df.columns:
            df = df.withColumn(
                column,
                F.regexp_replace(
                    F.regexp_replace(F.col(column), r"\$", ""),
                    ",",
                    "",
                ).cast("double"),
            )

    # Zip code remains string intentionally.
    if "zipcode" in df.columns:
        df = df.withColumn(
            "zipcode",
            F.regexp_extract(F.col("zipcode"), r"(\d{5})", 1),
        )
        df = df.withColumn(
            "zipcode",
            F.when(F.col("zipcode") == "", None).otherwise(F.col("zipcode")),
        )

    # --------------------------------------------------
    # 5. Core data-quality flags
    # --------------------------------------------------

    df = (
        df
        .withColumn(
            "dq_has_record_id",
            F.col("record_id").isNotNull(),
        )
        .withColumn(
            "dq_has_permit_number",
            F.col("permit_number").isNotNull(),
        )
        .withColumn(
            "dq_valid_estimated_cost",
            F.col("estimated_cost").isNotNull()
            & (F.col("estimated_cost") > 0),
        )
        .withColumn(
            "dq_valid_revised_cost",
            F.col("revised_cost").isNotNull()
            & (F.col("revised_cost") > 0),
        )
    )

    # --------------------------------------------------
    # 6. Deduplicate by DataSF Record ID
    # --------------------------------------------------

    valid_ids = df.filter(F.col("record_id").isNotNull())
    null_ids = df.filter(F.col("record_id").isNull())

    ordering = Window.partitionBy("record_id").orderBy(
        F.col("data_loaded_at").desc_nulls_last(),
        F.col("current_status_date").desc_nulls_last(),
        F.col("constructiq_ingested_at").desc_nulls_last(),
    )

    valid_ids = (
        valid_ids
        .withColumn("_row_rank", F.row_number().over(ordering))
        .filter(F.col("_row_rank") == 1)
        .drop("_row_rank")
    )

    df = valid_ids.unionByName(null_ids)

    # --------------------------------------------------
    # 7. Useful analytical fields
    # --------------------------------------------------

    if "permit_creation_date" in df.columns:
        df = (
            df
            .withColumn(
                "permit_creation_year",
                F.year("permit_creation_date"),
            )
            .withColumn(
                "permit_creation_month",
                F.month("permit_creation_date"),
            )
        )

    df = df.withColumn(
        "_constructiq_silver_processed_at",
        F.current_timestamp(),
    )

    # --------------------------------------------------
    # 8. Data-quality metrics
    # --------------------------------------------------

    silver_rows = df.count()
    duplicates_removed = bronze_rows - silver_rows

    metrics = df.agg(
        F.sum(
            F.when(~F.col("dq_has_record_id"), 1).otherwise(0)
        ).alias("missing_record_id"),

        F.sum(
            F.when(~F.col("dq_has_permit_number"), 1).otherwise(0)
        ).alias("missing_permit_number"),

        F.sum(
            F.when(F.col("dq_valid_estimated_cost"), 1).otherwise(0)
        ).alias("valid_estimated_cost"),

        F.sum(
            F.when(F.col("dq_valid_revised_cost"), 1).otherwise(0)
        ).alias("valid_revised_cost"),
    ).collect()[0]

    print()
    print("=== DATA QUALITY ===")
    print(f"Silver rows             : {silver_rows:,}")
    print(f"Silver columns          : {len(df.columns)}")
    print(f"Duplicates removed      : {duplicates_removed:,}")
    print(f"Missing Record ID       : {metrics['missing_record_id']:,}")
    print(f"Missing Permit Number   : {metrics['missing_permit_number']:,}")
    print(f"Valid Estimated Cost    : {metrics['valid_estimated_cost']:,}")
    print(f"Valid Revised Cost      : {metrics['valid_revised_cost']:,}")

    # --------------------------------------------------
    # 9. Persist Silver dataset
    # --------------------------------------------------

    (
        df.write
        .mode("overwrite")
        .option("compression", "snappy")
        .parquet(SILVER)
    )

    verify = spark.read.parquet(SILVER)

    verified_rows = verify.count()

    if verified_rows != silver_rows:
        raise RuntimeError(
            f"Silver verification failed: "
            f"expected={silver_rows}, actual={verified_rows}"
        )

    print()
    print("=== SILVER SCHEMA ===")
    verify.printSchema()

    print()
    print(f"Verified rows : {verified_rows:,}")
    print("SILVER_CLEAN_OK")
    print("=" * 72)

    spark.stop()


if __name__ == "__main__":
    main()
