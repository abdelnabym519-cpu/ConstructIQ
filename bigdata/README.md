# ConstructIQ Big Data Layer

ConstructIQ uses Apache Spark / PySpark for scalable construction-data
processing.

## Medallion Architecture

Raw Source Data
    |
    v
Bronze Layer
    |
    v
Silver Layer
    |
    v
Gold Layer
    |
    v
EDA + Machine Learning + Dashboard

### Bronze

Immutable representation of source construction data.

### Silver

Validated and cleaned construction records.

### Gold

Analytics-ready and machine-learning-ready construction features.

## Spark Runtime

Apache Spark 3.5.9
PySpark
Java 17
Docker
Parquet
