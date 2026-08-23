from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


SOURCE = "/workspace/data/source/building_permits.csv"
BRONZE = "/workspace/data/bronze/building_permits"


def main():
    spark = (
        SparkSession.builder
        .appName("ConstructIQ-Bronze-Building-Permits")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("=" * 72)
    print("ConstructIQ — Bronze Layer")
    print("=" * 72)

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .option("multiLine", True)
        .option("escape", '"')
        .csv(SOURCE)
    )

    source_rows = df.count()
    source_columns = len(df.columns)

    print(f"Source rows    : {source_rows:,}")
    print(f"Source columns : {source_columns}")
    print(f"Partitions     : {df.rdd.getNumPartitions()}")

    if source_rows == 0:
        raise RuntimeError("Source dataset is empty")

    bronze = (
        df
        .withColumn("_constructiq_ingested_at", F.current_timestamp())
        .withColumn("_constructiq_source", F.lit("DataSF Building Permits"))
        .withColumn("_constructiq_dataset_id", F.lit("i98e-djp9"))
    )

    (
        bronze.write
        .mode("overwrite")
        .option("compression", "snappy")
        .parquet(BRONZE)
    )

    verify = spark.read.parquet(BRONZE)

    bronze_rows = verify.count()

    print()
    print(f"Bronze rows    : {bronze_rows:,}")
    print(f"Bronze columns : {len(verify.columns)}")

    if bronze_rows != source_rows:
        raise RuntimeError(
            f"Row-count mismatch: source={source_rows}, bronze={bronze_rows}"
        )

    print()
    print("Sample schema:")
    verify.printSchema()

    print()
    print("BRONZE_INGEST_OK")
    print("=" * 72)

    spark.stop()


if __name__ == "__main__":
    main()
