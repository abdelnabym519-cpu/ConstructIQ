import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


ROOT = Path(__file__).resolve().parents[1]

EDA_DIR = ROOT / "reports" / "eda"
ML_DIR = ROOT / "reports" / "ml_v2"
XAI_DIR = ROOT / "reports" / "xai"

MODEL_PATH = (
    ROOT
    / "models"
    / "constructiq_spark_v2"
    / "best_pipeline"
)

MIN_COST = 100.0
MAX_COST = 1_000_000.0


st.set_page_config(
    page_title="ConstructIQ",
    page_icon="🏗️",
    layout="wide",
)


st.markdown(
    """
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}
.ciq-hero {
    padding: 28px 32px;
    border: 1px solid #2c3440;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        rgba(244,185,66,0.12),
        rgba(23,29,39,0.95)
    );
    margin-bottom: 24px;
}
.ciq-hero h1 {
    margin: 0;
    font-size: 2.6rem;
}
.ciq-sub {
    opacity: .78;
    margin-top: 8px;
    font-size: 1.03rem;
}
.ciq-note {
    padding: 14px 16px;
    border-left: 4px solid #F4B942;
    background: rgba(244,185,66,.08);
    border-radius: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_reports():
    eda = load_json(EDA_DIR / "eda_summary.json")
    ml = load_json(ML_DIR / "model_results_v2.json")
    xai = load_json(XAI_DIR / "explainability_summary.json")

    annual = pd.read_csv(EDA_DIR / "annual_trend.csv")
    permits = pd.read_csv(EDA_DIR / "permit_types_top20.csv")
    neighborhoods = pd.read_csv(
        EDA_DIR / "neighborhoods_top20.csv"
    )
    zipcodes = pd.read_csv(EDA_DIR / "zipcodes_top20.csv")

    ml_metrics = pd.read_csv(
        ML_DIR / "model_metrics_v2.csv"
    )

    positive = pd.read_csv(
        XAI_DIR / "top_positive_terms.csv"
    )
    negative = pd.read_csv(
        XAI_DIR / "top_negative_terms.csv"
    )

    return {
        "eda": eda,
        "ml": ml,
        "xai": xai,
        "annual": annual,
        "permits": permits,
        "neighborhoods": neighborhoods,
        "zipcodes": zipcodes,
        "ml_metrics": ml_metrics,
        "positive": positive,
        "negative": negative,
    }


reports = load_reports()

eda = reports["eda"]
ml = reports["ml"]
xai = reports["xai"]


@st.cache_resource(show_spinner=False)
def get_spark_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("ConstructIQ-Dashboard-Inference")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    model = PipelineModel.load(str(MODEL_PATH))

    return spark, model


PREDICTION_SCHEMA = StructType(
    [
        StructField("zipcode", StringType(), True),
        StructField(
            "supervisor_district_cat",
            StringType(),
            True,
        ),
        StructField(
            "neighborhoods_analysis_boundaries",
            StringType(),
            True,
        ),
        StructField(
            "permit_type_definition",
            StringType(),
            True,
        ),
        StructField(
            "application_submission_method",
            StringType(),
            True,
        ),
        StructField("existing_use", StringType(), True),
        StructField("proposed_use", StringType(), True),
        StructField(
            "existing_construction_type_description",
            StringType(),
            True,
        ),
        StructField(
            "proposed_construction_type_description",
            StringType(),
            True,
        ),
        StructField(
            "permit_creation_year",
            IntegerType(),
            False,
        ),
        StructField(
            "permit_creation_month",
            IntegerType(),
            False,
        ),
        StructField(
            "number_of_existing_stories",
            DoubleType(),
            True,
        ),
        StructField(
            "number_of_proposed_stories",
            DoubleType(),
            True,
        ),
        StructField("story_change", DoubleType(), True),
        StructField("existing_units", IntegerType(), True),
        StructField("proposed_units", IntegerType(), True),
        StructField("unit_change", IntegerType(), True),
        StructField("is_adu", IntegerType(), True),
        StructField("is_site_permit", IntegerType(), True),
        StructField("is_reroof", IntegerType(), True),
        StructField("description", StringType(), False),
    ]
)


def predict_cost(values):
    spark, model = get_spark_model()

    frame = spark.createDataFrame(
        [values],
        schema=PREDICTION_SCHEMA,
    )

    prediction = (
        model.transform(frame)
        .select("prediction")
        .first()["prediction"]
    )

    raw_cost = float(10 ** float(prediction))

    return float(
        np.clip(
            raw_cost,
            MIN_COST,
            MAX_COST,
        )
    )


st.markdown(
    """
<div class="ciq-hero">
<h1>🏗️ ConstructIQ</h1>
<div class="ciq-sub">
Big Data & AI Construction Cost Intelligence
</div>
</div>
""",
    unsafe_allow_html=True,
)


tabs = st.tabs(
    [
        "Executive Overview",
        "Construction Analytics",
        "AI Cost Intelligence",
        "Explainable AI",
        "Model Performance",
        "Architecture",
    ]
)


# ================================================================
# EXECUTIVE OVERVIEW
# ================================================================

with tabs[0]:
    gold_rows = int(
        eda["dataset"]["rows"]
    )

    selection = ml["selection"]

    test_metric = next(
        m
        for m in ml["metrics"]
        if (
            m["model"] == "structured_text_lr_v2"
            and m["split"] == "test"
        )
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "ML-ready projects",
        f"{gold_rows:,}",
    )

    c2.metric(
        "Gold features",
        eda["dataset"]["columns"],
    )

    c3.metric(
        "Test R²",
        f"{test_metric['r2']:.3f}",
    )

    c4.metric(
        "Test Log-R²",
        f"{test_metric['log_r2']:.3f}",
    )

    st.markdown("### End-to-End Data Pipeline")

    st.code(
        """
DataSF Building Permits
        │
        ▼
Bronze — Raw / Immutable
        │
        ▼
Silver — Cleaned / Validated / Typed
        │
        ▼
Gold — ML-ready Features
        │
        ├──► EDA & Business Analytics
        │
        ├──► Machine Learning
        │
        └──► Explainable AI
        │
        ▼
ConstructIQ Dashboard
""",
        language="text",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Production AI Model")

        st.write(
            "**Structured + Description TF-IDF "
            "Linear Regression**"
        )

        st.write(
            "Target: revised construction cost "
            "with log10 transformation."
        )

        st.write(
            f"Supported model domain: "
            f"**${MIN_COST:,.0f} – "
            f"${MAX_COST:,.0f}**"
        )

    with col2:
        st.markdown("### Scientific Evaluation")

        st.write(
            f"Validation RMSE: "
            f"**${selection['validation_rmse']:,.0f}**"
        )

        st.write(
            "Improvement over median baseline: "
            f"**{selection['rmse_improvement_vs_median_percent']:.2f}%**"
        )

        st.write(
            f"Final test RMSE: "
            f"**${test_metric['rmse']:,.0f}**"
        )


# ================================================================
# ANALYTICS
# ================================================================

with tabs[1]:
    st.header("Construction Analytics")

    annual = reports["annual"].copy()

    annual = annual[
        annual["permit_creation_year"] >= 1980
    ]

    fig = px.line(
        annual,
        x="permit_creation_year",
        y="median_cost",
        markers=True,
        title="Median Construction Cost by Permit Year",
        labels={
            "permit_creation_year": "Permit year",
            "median_cost": "Median cost ($)",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    left, right = st.columns(2)

    with left:
        permit_df = (
            reports["permits"]
            .head(10)
            .sort_values("projects")
        )

        fig = px.bar(
            permit_df,
            x="projects",
            y="permit_type_definition",
            orientation="h",
            title="Top Permit Types",
            labels={
                "projects": "Projects",
                "permit_type_definition": "",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        neighborhood_df = (
            reports["neighborhoods"]
            .head(10)
            .sort_values("projects")
        )

        fig = px.bar(
            neighborhood_df,
            x="projects",
            y="neighborhoods_analysis_boundaries",
            orientation="h",
            title="Top Neighborhoods",
            labels={
                "projects": "Projects",
                "neighborhoods_analysis_boundaries": "",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown("### Cost Distribution")

    quantiles = eda["cost_quantiles"]

    qdf = pd.DataFrame(
        {
            "Percentile": list(quantiles.keys()),
            "Cost": list(quantiles.values()),
        }
    )

    fig = px.bar(
        qdf,
        x="Percentile",
        y="Cost",
        title="Construction Cost Quantiles",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ================================================================
# AI COST INTELLIGENCE
# ================================================================

with tabs[2]:
    st.header("AI Construction Cost Estimate")

    st.markdown(
        """
<div class="ciq-note">
The model estimates revised permit construction cost.
It is an analytical estimate, not a contractor quotation
or guaranteed final delivered project cost.
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    permits = (
        reports["permits"][
            "permit_type_definition"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    neighborhoods = (
        reports["neighborhoods"][
            "neighborhoods_analysis_boundaries"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    zipcodes = (
        reports["zipcodes"]["zipcode"]
        .dropna()
        .astype(str)
        .tolist()
    )

    description = st.text_area(
        "Project description",
        value=(
            "Kitchen and bathroom remodel with "
            "electrical and mechanical upgrades"
        ),
        height=120,
    )

    a, b, c = st.columns(3)

    with a:
        permit_type = st.selectbox(
            "Permit type",
            permits,
        )

    with b:
        neighborhood = st.selectbox(
            "Neighborhood",
            ["__MISSING__"] + neighborhoods,
        )

    with c:
        zipcode = st.selectbox(
            "ZIP code",
            ["__MISSING__"] + zipcodes,
        )

    d, e, f = st.columns(3)

    with d:
        existing_stories = st.number_input(
            "Existing stories",
            min_value=0.0,
            max_value=100.0,
            value=1.0,
            step=1.0,
        )

    with e:
        proposed_stories = st.number_input(
            "Proposed stories",
            min_value=0.0,
            max_value=100.0,
            value=1.0,
            step=1.0,
        )

    with f:
        supervisor_district = st.number_input(
            "Supervisor district",
            min_value=1,
            max_value=11,
            value=1,
            step=1,
        )

    g, h = st.columns(2)

    with g:
        existing_units = st.number_input(
            "Existing units",
            min_value=0,
            max_value=10000,
            value=1,
            step=1,
        )

    with h:
        proposed_units = st.number_input(
            "Proposed units",
            min_value=0,
            max_value=10000,
            value=1,
            step=1,
        )

    flags = st.columns(3)

    is_adu = int(
        flags[0].checkbox("ADU")
    )

    is_site_permit = int(
        flags[1].checkbox("Site permit")
    )

    is_reroof = int(
        flags[2].checkbox("Reroof")
    )

    with st.expander("Advanced project attributes"):
        application_method = st.text_input(
            "Application submission method",
            value="__MISSING__",
        )

        existing_use = st.text_input(
            "Existing use",
            value="__MISSING__",
        )

        proposed_use = st.text_input(
            "Proposed use",
            value="__MISSING__",
        )

        existing_type = st.text_input(
            "Existing construction type",
            value="__MISSING__",
        )

        proposed_type = st.text_input(
            "Proposed construction type",
            value="__MISSING__",
        )

    if st.button(
        "Estimate Construction Cost",
        type="primary",
        use_container_width=True,
    ):
        values = {
            "zipcode": zipcode,
            "supervisor_district_cat": str(
                supervisor_district
            ),
            "neighborhoods_analysis_boundaries": (
                neighborhood
            ),
            "permit_type_definition": permit_type,
            "application_submission_method": (
                application_method
            ),
            "existing_use": existing_use,
            "proposed_use": proposed_use,
            "existing_construction_type_description": (
                existing_type
            ),
            "proposed_construction_type_description": (
                proposed_type
            ),
            "permit_creation_year": date.today().year,
            "permit_creation_month": date.today().month,
            "number_of_existing_stories": float(
                existing_stories
            ),
            "number_of_proposed_stories": float(
                proposed_stories
            ),
            "story_change": float(
                proposed_stories
                - existing_stories
            ),
            "existing_units": int(existing_units),
            "proposed_units": int(proposed_units),
            "unit_change": int(
                proposed_units
                - existing_units
            ),
            "is_adu": is_adu,
            "is_site_permit": is_site_permit,
            "is_reroof": is_reroof,
            "description": description or "",
        }

        try:
            with st.spinner(
                "Running ConstructIQ AI model..."
            ):
                estimated_cost = predict_cost(
                    values
                )

            st.success("Prediction complete")

            st.metric(
                "Estimated revised construction cost",
                f"${estimated_cost:,.0f}",
            )

            st.caption(
                "Model-supported range: "
                "$100 – $1,000,000. "
                "Typical validation MAE was "
                "approximately $34,906."
            )

        except FileNotFoundError:
            st.error(
                "The trained V2 model is not available "
                "inside models/constructiq_spark_v2/"
                "best_pipeline."
            )

        except Exception as exc:
            st.error(
                "Prediction failed."
            )
            st.exception(exc)


# ================================================================
# XAI
# ================================================================

with tabs[3]:
    st.header("Explainable AI")

    ablation = xai["ablation_analysis"]

    val = ablation["validation"]
    test = ablation["test"]

    x1, x2 = st.columns(2)

    x1.metric(
        "Validation RMSE reduction from text",
        f"{val['rmse_reduction_percent']:.2f}%",
    )

    x2.metric(
        "Test RMSE reduction from text",
        f"{test['rmse_reduction_percent']:.2f}%",
    )

    st.write(
        "Project descriptions materially improved "
        "prediction quality compared with the "
        "structured-only model."
    )

    positive = reports["positive"].copy()
    negative = reports["negative"].copy()

    # Hide numeric-only text noise from visual explanations.
    positive = positive[
        ~positive["term"]
        .astype(str)
        .str.fullmatch(r"\d+")
    ].head(12)

    negative = negative[
        ~negative["term"]
        .astype(str)
        .str.fullmatch(r"\d+")
    ].head(12)

    p1, p2 = st.columns(2)

    with p1:
        fig = px.bar(
            positive.sort_values(
                "coefficient_log10"
            ),
            x="coefficient_log10",
            y="term",
            orientation="h",
            title=(
                "Terms Associated with Higher "
                "Predicted Cost"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with p2:
        fig = px.bar(
            negative.sort_values(
                "coefficient_log10",
                ascending=False,
            ),
            x="coefficient_log10",
            y="term",
            orientation="h",
            title=(
                "Terms Associated with Lower "
                "Predicted Cost"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.info(
        "These terms represent predictive associations, "
        "not causal effects. The companion "
        "CountVectorizer model is used only for global "
        "interpretation."
    )


# ================================================================
# MODEL PERFORMANCE
# ================================================================

with tabs[4]:
    st.header("Model Performance")

    metrics = reports["ml_metrics"].copy()

    validation_metrics = metrics[
        metrics["split"] == "validation"
    ].copy()

    test_metrics = metrics[
        metrics["split"] == "test"
    ].copy()

    fig = px.bar(
        validation_metrics,
        x="model",
        y="rmse",
        title="Validation RMSE — Lower is Better",
        labels={
            "model": "",
            "rmse": "RMSE ($)",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    fig = px.bar(
        test_metrics,
        x="model",
        y="r2",
        title="Final Test R² — Higher is Better",
        labels={
            "model": "",
            "r2": "R²",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown("### Final V2 Metrics")

    st.dataframe(
        test_metrics[
            [
                "model",
                "mae",
                "rmse",
                "r2",
                "mape",
                "log_rmse",
                "log_r2",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


# ================================================================
# ARCHITECTURE
# ================================================================

with tabs[5]:
    st.header("System Architecture")

    st.code(
        """
┌──────────────────────────────┐
│ DataSF Building Permits      │
│ Real-world construction data │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Apache Spark / PySpark       │
│ Bronze → Silver → Gold       │
└──────────────┬───────────────┘
               │
       ┌───────┴─────────┐
       │                 │
       ▼                 ▼
┌───────────────┐   ┌──────────────────────┐
│ EDA / BI      │   │ ML Training          │
│ Trends / KPIs │   │ Temporal Evaluation  │
└───────────────┘   └──────────┬───────────┘
                               │
                               ▼
                     ┌──────────────────────┐
                     │ Structured + NLP     │
                     │ TF-IDF Regression    │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Explainable AI       │
                     │ Ablation + Vocabulary│
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ ConstructIQ          │
                     │ Streamlit Dashboard  │
                     └──────────────────────┘
""",
        language="text",
    )

    st.markdown("### Technology Stack")

    st.write(
        "**Big Data:** Apache Spark, PySpark, Parquet, "
        "Medallion Architecture"
    )

    st.write(
        "**AI/ML:** Spark MLlib, TF-IDF, "
        "Linear Regression, Random Forest, GBT"
    )

    st.write(
        "**Analytics:** Pandas, Plotly, Streamlit"
    )

    st.write(
        "**Engineering:** Docker, Git, GitHub"
    )
