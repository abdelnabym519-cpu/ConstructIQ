from pyspark.sql import SparkSession
from pyspark.sql import functions as F

GOLD = "/workspace/data/gold/construction_cost_ml"

MIN_COST = 100.0
MAX_COST = 1_000_000.0


def summarize(name, df):
    row = df.agg(
        F.count("*").alias("rows"),
        F.avg("target_cost").alias("mean"),
        F.expr("percentile_approx(target_cost, 0.5)").alias("median"),
        F.expr("percentile_approx(target_cost, 0.75)").alias("p75"),
        F.expr("percentile_approx(target_cost, 0.90)").alias("p90"),
        F.expr("percentile_approx(target_cost, 0.95)").alias("p95"),
    ).collect()[0]

    print(
        f"{name:12} "
        f"| rows {row['rows']:>9,} "
        f"| mean ${row['mean']:>11,.0f} "
        f"| median ${row['median']:>9,.0f} "
        f"| P90 ${row['p90']:>10,.0f} "
        f"| P95 ${row['p95']:>10,.0f}"
    )


def main():
    spark = (
        SparkSession.builder
        .appName("ConstructIQ-ML-Diagnostic")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    df = (
        spark.read.parquet(GOLD)
        .filter(
            (F.col("target_cost") >= MIN_COST)
            & (F.col("target_cost") <= MAX_COST)
            & (F.col("permit_creation_year") >= 2015)
        )
    )

    train = df.filter(F.col("permit_creation_year").between(2015, 2023))
    validation = df.filter(F.col("permit_creation_year") == 2024)
    test = df.filter(F.col("permit_creation_year") >= 2025)

    print("=" * 100)
    print("CONSTRUCTIQ ML DIAGNOSTIC")
    print("=" * 100)

    print()
    print("=== TARGET DRIFT ===")
    summarize("TRAIN", train)
    summarize("VALIDATION", validation)
    summarize("TEST", test)

    print()
    print("=== TARGET SOURCE BY SPLIT ===")

    for name, frame in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:
        print()
        print(name)

        rows = (
            frame
            .groupBy("target_source")
            .agg(
                F.count("*").alias("rows"),
                F.avg("target_cost").alias("mean"),
                F.expr(
                    "percentile_approx(target_cost, 0.5)"
                ).alias("median"),
            )
            .orderBy(F.desc("rows"))
            .collect()
        )

        for row in rows:
            print(
                f"  {row['target_source']:15} "
                f"| {row['rows']:>9,} "
                f"| mean ${row['mean']:>11,.0f} "
                f"| median ${row['median']:>9,.0f}"
            )

    print()
    print("=== YEAR DISTRIBUTION ===")

    rows = (
        df.groupBy("permit_creation_year")
        .agg(
            F.count("*").alias("projects"),
            F.avg("target_cost").alias("mean_cost"),
            F.expr(
                "percentile_approx(target_cost, 0.5)"
            ).alias("median_cost"),
        )
        .orderBy("permit_creation_year")
        .collect()
    )

    for row in rows:
        print(
            f"{row['permit_creation_year']} "
            f"| {row['projects']:>9,} "
            f"| mean ${row['mean_cost']:>11,.0f} "
            f"| median ${row['median_cost']:>9,.0f}"
        )

    print()
    print("=== PERMIT TYPE DRIFT ===")

    for name, frame in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:
        print()
        print(name)

        rows = (
            frame
            .groupBy("permit_type_definition")
            .count()
            .orderBy(F.desc("count"))
            .limit(8)
            .collect()
        )

        total = frame.count()

        for row in rows:
            pct = row["count"] / total * 100

            print(
                f"  {(row['permit_type_definition'] or 'NULL')[:45]:45} "
                f"| {row['count']:>8,} "
                f"| {pct:>6.2f}%"
            )

    print()
    print("ML_DIAGNOSTIC_OK")
    print("=" * 100)

    spark.stop()


if __name__ == "__main__":
    main()
