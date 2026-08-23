from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():
    spark = (
        SparkSession.builder
        .appName("ConstructIQ-BigData-Smoke-Test")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("=" * 70)
    print("ConstructIQ - Apache Spark Big Data Foundation")
    print("=" * 70)
    print(f"Spark version : {spark.version}")
    print(f"Master        : {spark.sparkContext.master}")
    print(f"Application   : {spark.sparkContext.appName}")

    df = (
        spark.range(0, 1_000_000, numPartitions=4)
        .withColumn("construction_metric", F.col("id") * 2)
        .withColumn("cost_signal", F.col("id") * F.col("id"))
    )

    result = (
        df.agg(
            F.count("*").alias("row_count"),
            F.sum("construction_metric").alias("metric_checksum"),
        )
        .collect()[0]
    )

    print(f"Partitions    : {df.rdd.getNumPartitions()}")
    print(f"Rows processed: {result['row_count']:,}")
    print(f"Checksum      : {result['metric_checksum']:,}")

    assert result["row_count"] == 1_000_000

    print()
    print("SPARK_FOUNDATION_OK")
    print("=" * 70)

    spark.stop()


if __name__ == "__main__":
    main()
