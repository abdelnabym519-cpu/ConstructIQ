import csv
import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


GOLD = "/workspace/data/gold/construction_cost_ml"
OUTPUT = Path("/workspace/reports/eda")


def rows_to_dicts(rows):
    return [row.asDict(recursive=True) for row in rows]


def write_csv(filename, rows):
    rows = rows_to_dicts(rows)

    if not rows:
        return

    path = OUTPUT / filename

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    spark = (
        SparkSession.builder
        .appName("ConstructIQ-EDA-Business-Analysis")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("ConstructIQ — EDA & Business Analysis")
    print("=" * 76)

    df = spark.read.parquet(GOLD)

    total_rows = df.count()

    print(f"Gold rows : {total_rows:,}")
    print(f"Columns   : {len(df.columns)}")

    # --------------------------------------------------
    # 1. Cost distribution
    # --------------------------------------------------

    probs = [
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]

    quantiles = df.approxQuantile(
        "target_cost",
        probs,
        0.001,
    )

    quantile_summary = {
        f"p{int(p * 100):02d}": value
        for p, value in zip(probs, quantiles)
    }

    cost_stats = df.agg(
        F.min("target_cost").alias("min"),
        F.avg("target_cost").alias("mean"),
        F.stddev("target_cost").alias("stddev"),
        F.max("target_cost").alias("max"),
    ).collect()[0].asDict()

    # --------------------------------------------------
    # 2. Low-cost diagnostic
    # --------------------------------------------------

    thresholds = [
        10,
        100,
        500,
        1_000,
        5_000,
        10_000,
        50_000,
        100_000,
        500_000,
    ]

    low_cost = {}

    for threshold in thresholds:
        count = df.filter(
            F.col("target_cost") < threshold
        ).count()

        low_cost[str(threshold)] = {
            "rows_below": count,
            "percentage": round(
                (count / total_rows) * 100,
                4,
            ),
        }

    # --------------------------------------------------
    # 3. Target-source distribution
    # --------------------------------------------------

    target_source = (
        df.groupBy("target_source")
        .agg(
            F.count("*").alias("projects"),
            F.avg("target_cost").alias("mean_cost"),
            F.expr(
                "percentile_approx(target_cost, 0.5)"
            ).alias("median_cost"),
        )
        .orderBy(F.desc("projects"))
    )

    target_source_rows = target_source.collect()

    # --------------------------------------------------
    # 4. Permit-type analysis
    # --------------------------------------------------

    permit_types = (
        df.filter(
            F.col("permit_type_definition").isNotNull()
        )
        .groupBy(
            "permit_type",
            "permit_type_definition",
        )
        .agg(
            F.count("*").alias("projects"),
            F.avg("target_cost").alias("mean_cost"),
            F.expr(
                "percentile_approx(target_cost, 0.5)"
            ).alias("median_cost"),
        )
        .orderBy(F.desc("projects"))
        .limit(20)
    )

    permit_type_rows = permit_types.collect()

    # --------------------------------------------------
    # 5. Annual trend
    # --------------------------------------------------

    annual = (
        df.filter(
            F.col("permit_creation_year").isNotNull()
        )
        .groupBy("permit_creation_year")
        .agg(
            F.count("*").alias("projects"),
            F.avg("target_cost").alias("mean_cost"),
            F.expr(
                "percentile_approx(target_cost, 0.5)"
            ).alias("median_cost"),
        )
        .orderBy("permit_creation_year")
    )

    annual_rows = annual.collect()

    # --------------------------------------------------
    # 6. Neighborhood analysis
    # --------------------------------------------------

    neighborhoods = (
        df.filter(
            F.col(
                "neighborhoods_analysis_boundaries"
            ).isNotNull()
        )
        .groupBy(
            "neighborhoods_analysis_boundaries"
        )
        .agg(
            F.count("*").alias("projects"),
            F.avg("target_cost").alias("mean_cost"),
            F.expr(
                "percentile_approx(target_cost, 0.5)"
            ).alias("median_cost"),
        )
        .orderBy(F.desc("projects"))
        .limit(20)
    )

    neighborhood_rows = neighborhoods.collect()

    # --------------------------------------------------
    # 7. ZIP-code analysis
    # --------------------------------------------------

    zipcodes = (
        df.filter(F.col("zipcode").isNotNull())
        .groupBy("zipcode")
        .agg(
            F.count("*").alias("projects"),
            F.avg("target_cost").alias("mean_cost"),
            F.expr(
                "percentile_approx(target_cost, 0.5)"
            ).alias("median_cost"),
        )
        .orderBy(F.desc("projects"))
        .limit(20)
    )

    zipcode_rows = zipcodes.collect()

    # --------------------------------------------------
    # 8. Project-scale diagnostics
    # --------------------------------------------------

    scale_stats = df.agg(
        F.avg("number_of_existing_stories").alias(
            "avg_existing_stories"
        ),
        F.avg("number_of_proposed_stories").alias(
            "avg_proposed_stories"
        ),
        F.avg("story_change").alias(
            "avg_story_change"
        ),
        F.avg("existing_units").alias(
            "avg_existing_units"
        ),
        F.avg("proposed_units").alias(
            "avg_proposed_units"
        ),
        F.avg("unit_change").alias(
            "avg_unit_change"
        ),
    ).collect()[0].asDict()

    # --------------------------------------------------
    # 9. Null-rate analysis for ML candidates
    # --------------------------------------------------

    ml_columns = [
        "permit_creation_year",
        "permit_creation_month",
        "zipcode",
        "supervisor_district",
        "permit_type",
        "permit_type_definition",
        "application_submission_method",
        "existing_use",
        "proposed_use",
        "number_of_existing_stories",
        "number_of_proposed_stories",
        "existing_units",
        "proposed_units",
        "existing_construction_type",
        "proposed_construction_type",
        "is_adu",
        "is_site_permit",
        "is_reroof",
    ]

    null_exprs = [
        F.sum(
            F.when(F.col(c).isNull(), 1).otherwise(0)
        ).alias(c)
        for c in ml_columns
        if c in df.columns
    ]

    null_counts = df.agg(*null_exprs).collect()[0].asDict()

    null_rates = {
        column: {
            "missing_rows": int(count),
            "missing_percentage": round(
                count / total_rows * 100,
                4,
            ),
        }
        for column, count in null_counts.items()
    }

    # --------------------------------------------------
    # 10. Persist reproducible EDA outputs
    # --------------------------------------------------

    summary = {
        "dataset": {
            "rows": total_rows,
            "columns": len(df.columns),
        },
        "cost_statistics": cost_stats,
        "cost_quantiles": quantile_summary,
        "low_cost_diagnostic": low_cost,
        "project_scale": scale_stats,
        "null_rates": null_rates,
    }

    with (
        OUTPUT / "eda_summary.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            default=str,
        )

    write_csv(
        "target_source.csv",
        target_source_rows,
    )

    write_csv(
        "permit_types_top20.csv",
        permit_type_rows,
    )

    write_csv(
        "annual_trend.csv",
        annual_rows,
    )

    write_csv(
        "neighborhoods_top20.csv",
        neighborhood_rows,
    )

    write_csv(
        "zipcodes_top20.csv",
        zipcode_rows,
    )

    # --------------------------------------------------
    # Console summary
    # --------------------------------------------------

    print()
    print("=== COST DISTRIBUTION ===")

    for key, value in quantile_summary.items():
        print(f"{key.upper():>4} : ${value:,.2f}")

    print()
    print("=== LOW-COST DIAGNOSTIC ===")

    for threshold, values in low_cost.items():
        print(
            f"< ${int(threshold):>7,}: "
            f"{values['rows_below']:>10,} rows "
            f"({values['percentage']:>7.3f}%)"
        )

    print()
    print("=== TOP 10 PERMIT TYPES ===")

    for row in permit_type_rows[:10]:
        print(
            f"{row['permit_type_definition'][:45]:45} "
            f"| {row['projects']:>8,} "
            f"| median ${row['median_cost']:>12,.2f}"
        )

    print()
    print("=== TOP 10 NEIGHBORHOODS ===")

    for row in neighborhood_rows[:10]:
        print(
            f"{row['neighborhoods_analysis_boundaries'][:35]:35} "
            f"| {row['projects']:>8,} "
            f"| median ${row['median_cost']:>12,.2f}"
        )

    print()
    print("=== OUTPUT FILES ===")

    for path in sorted(OUTPUT.iterdir()):
        print(f" - {path.name}")

    print()
    print("EDA_ANALYSIS_OK")
    print("=" * 76)

    spark.stop()


if __name__ == "__main__":
    main()
