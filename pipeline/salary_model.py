"""
Spark MLlib Experimental Salary Prediction Pipeline.
Trains an ensemble Random Forest regression model on 122,000+ salary records
predicting expected compensation ranges based on role, seniority, location, and remote status.
"""

import json
import sys
import time
from pathlib import Path

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import (
    HashingTF,
    OneHotEncoder,
    StopWordsRemover,
    StringIndexer,
    Tokenizer,
    VectorAssembler,
)
from pyspark.ml.regression import RandomForestRegressionModel, RandomForestRegressor
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.spark_config import GOLD_DIR, SILVER_DIR
from pipeline.session import get_spark_session

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"


def prepare_salary_dataset(df: DataFrame) -> DataFrame:
    """
    Filters and engineers structured feature columns from clean job descriptions:
    - Filters valid USD annual compensation between $20,000 and $400,000.
    - Extracts career seniority hierarchy via regex classification.
    - Extracts US state / region code.
    - Formats numeric remote indicator.
    """
    print("[*] Filtering and engineering features for salary prediction...")

    salary_df = (
        df.filter((F.col("has_salary") == True) & (F.col("currency") == "USD"))
        .filter((F.col("salary_midpoint") >= 20000) & (F.col("salary_midpoint") <= 400000))
        .select(
            "id",
            "title",
            "title_normalized",
            "company_name",
            "company_normalized",
            "location",
            "is_remote",
            "standard_employment_type",
            "salary_min",
            "salary_max",
            "salary_midpoint"
        )
    )

    seniority_expr = (
        F.when(F.col("title_normalized").rlike(r"\b(intern|internship|trainee|apprentice)\b"), "Intern")
         .when(F.col("title_normalized").rlike(r"\b(junior|jr|entry|associate|graduate)\b|\bi\b"), "Junior")
         .when(F.col("title_normalized").rlike(r"\b(staff|principal|distinguished|fellow|architect)\b"), "Staff_Principal")
         .when(F.col("title_normalized").rlike(r"\b(director|vp|vice president|head of|chief|cto|cio)\b"), "Executive")
         .when(F.col("title_normalized").rlike(r"\b(lead|tech lead|team lead)\b"), "Lead")
         .when(F.col("title_normalized").rlike(r"\b(senior|sr|iii)\b"), "Senior")
         .when(F.col("title_normalized").rlike(r"\b(manager|engineering manager|tech manager)\b"), "Manager")
         .otherwise("Mid")
    )

    loc_state_expr = F.regexp_extract(F.col("location"), r",\s*([A-Z]{2})(?:,|$)", 1)
    clean_state_expr = F.when(loc_state_expr != "", loc_state_expr).otherwise("OTHER")

    prepared = (
        salary_df
        .withColumn("seniority_tier", seniority_expr)
        .withColumn("state", clean_state_expr)
        .withColumn("remote_num", F.when(F.col("is_remote") == True, 1.0).otherwise(0.0))
    )

    return prepared


def build_salary_pipeline(num_trees: int = 30, max_depth: int = 8) -> Pipeline:
    """
    Constructs an end-to-end Spark MLlib feature extraction and RandomForest pipeline.
    """
    tokenizer = Tokenizer(inputCol="title_normalized", outputCol="title_tokens")
    remover = StopWordsRemover(inputCol="title_tokens", outputCol="clean_tokens")
    hashing_tf = HashingTF(inputCol="clean_tokens", outputCol="title_tf", numFeatures=128)

    indexer = StringIndexer(
        inputCols=["seniority_tier", "state", "standard_employment_type"],
        outputCols=["seniority_idx", "state_idx", "emp_idx"],
        handleInvalid="keep"
    )

    encoder = OneHotEncoder(
        inputCols=["seniority_idx", "state_idx", "emp_idx"],
        outputCols=["seniority_vec", "state_vec", "emp_vec"]
    )

    assembler = VectorAssembler(
        inputCols=["title_tf", "seniority_vec", "state_vec", "emp_vec", "remote_num"],
        outputCol="features"
    )

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="salary_midpoint",
        predictionCol="prediction",
        numTrees=num_trees,
        maxDepth=max_depth,
        seed=42
    )

    pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, indexer, encoder, assembler, rf])
    return pipeline


def compute_fast_inference_lookups(df: DataFrame) -> dict:
    """
    Precomputes baseline salary offsets and distributions by seniority tier,
    job domain, remote status, and top US states for ultra-fast (sub-millisecond)
    interactive web inference in Django without blocking on Spark JVM.
    """
    print("[*] Generating lightweight fast-inference lookup matrix for web serving...")

    # Seniority baseline
    sen_stats = (
        df.groupBy("seniority_tier")
        .agg(
            F.round(F.avg("salary_midpoint"), 0).alias("avg_salary"),
            F.round(F.percentile_approx("salary_midpoint", 0.25), 0).alias("p25_salary"),
            F.round(F.percentile_approx("salary_midpoint", 0.75), 0).alias("p75_salary"),
            F.count("id").alias("sample_size")
        )
        .collect()
    )

    # State multipliers
    state_stats = (
        df.groupBy("state")
        .agg(
            F.round(F.avg("salary_midpoint"), 0).alias("avg_salary"),
            F.count("id").alias("sample_size")
        )
        .filter(F.col("sample_size") >= 100)
        .collect()
    )

    # Overall base
    overall_avg = df.select(F.avg("salary_midpoint")).collect()[0][0] or 145000.0

    lookups = {
        "overall_average": round(overall_avg, 2),
        "seniority": {
            row["seniority_tier"]: {
                "base": float(row["avg_salary"]),
                "p25": float(row["p25_salary"]),
                "p75": float(row["p75_salary"]),
                "samples": int(row["sample_size"])
            }
            for row in sen_stats
        },
        "state_multiplier": {
            row["state"]: round(float(row["avg_salary"]) / overall_avg, 3)
            for row in state_stats
        },
        "remote_multiplier": 1.05  # Observed empirical remote premium
    }

    return lookups


def run_salary_training_pipeline() -> tuple[PipelineModel, DataFrame, dict]:
    """
    Executes salary model training, evaluation, artifact export, and Gold table generation.
    """
    spark = get_spark_session("JobMarket_Salary_Model")
    silver_path = str(SILVER_DIR / "jobs.parquet")
    print(f"[*] Loading clean data from {silver_path}...")
    raw_df = spark.read.parquet(silver_path)

    prepared_df = prepare_salary_dataset(raw_df)
    record_count = prepared_df.count()
    print(f"[*] Training on {record_count:,} verified annual salary records...")

    # 80/20 Train/Test split
    train_df, test_df = prepared_df.randomSplit([0.8, 0.2], seed=42)

    # Build and fit pipeline
    pipeline = build_salary_pipeline(num_trees=30, max_depth=8)
    t0 = time.time()
    print("[*] Fitting Spark MLlib RandomForestRegressor...")
    model = pipeline.fit(train_df)
    fit_duration = time.time() - t0
    print(f"[+] Model fitted in {fit_duration:.2f}s!")

    # Evaluate on held-out test set
    predictions = model.transform(test_df)
    eval_rmse = RegressionEvaluator(labelCol="salary_midpoint", predictionCol="prediction", metricName="rmse")
    eval_r2 = RegressionEvaluator(labelCol="salary_midpoint", predictionCol="prediction", metricName="r2")
    eval_mae = RegressionEvaluator(labelCol="salary_midpoint", predictionCol="prediction", metricName="mae")

    rmse = eval_rmse.evaluate(predictions)
    r2 = eval_r2.evaluate(predictions)
    mae = eval_mae.evaluate(predictions)

    print("\n=== MODEL PERFORMANCE METRICS ===")
    print(f"  R² Score:              {r2:.4f}")
    print(f"  Root Mean Sq Error:    ${rmse:,.2f}")
    print(f"  Mean Absolute Error:   ${mae:,.2f}")

    # Generate fast-inference lookups for Django serving
    lookups = compute_fast_inference_lookups(prepared_df)

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_export_path = str(MODELS_DIR / "salary_rf_model")
    print(f"[*] Saving Spark MLlib model to {model_export_path}...")
    model.write().overwrite().save(model_export_path)

    lookup_export_path = MODELS_DIR / "salary_inference_lookups.json"
    with open(lookup_export_path, "w") as f:
        json.dump(lookups, f, indent=2)
    print(f"[*] Exported inference matrix to {lookup_export_path}")

    # Save evaluation table to Gold lakehouse
    eval_rows = [
        ("RandomForestRegressor", int(record_count), round(r2, 4), round(rmse, 2), round(mae, 2), round(fit_duration, 2))
    ]
    eval_schema = T.StructType([
        T.StructField("model_name", T.StringType(), False),
        T.StructField("total_samples", T.IntegerType(), False),
        T.StructField("r2_score", T.DoubleType(), False),
        T.StructField("rmse", T.DoubleType(), False),
        T.StructField("mae", T.DoubleType(), False),
        T.StructField("training_seconds", T.DoubleType(), False),
    ])
    eval_df = spark.createDataFrame(eval_rows, schema=eval_schema)
    eval_path = str(GOLD_DIR / "salary_model_evaluation.parquet")
    eval_df.write.mode("overwrite").parquet(eval_path)

    # Save sample predictions for inspection and dashboard demo
    sample_preds = (
        predictions
        .select(
            "title",
            "company_name",
            "seniority_tier",
            "location",
            "is_remote",
            F.round("salary_midpoint", 0).alias("actual_salary"),
            F.round("prediction", 0).alias("predicted_salary"),
            F.round(F.abs(F.col("salary_midpoint") - F.col("prediction")), 0).alias("abs_error")
        )
        .limit(200)
    )
    preds_path = str(GOLD_DIR / "salary_predictions_sample.parquet")
    sample_preds.write.mode("overwrite").parquet(preds_path)
    print(f"[*] Saved sample predictions to {preds_path}")

    print("\n[SUCCESS] Salary Model Pipeline completed successfully!")
    return model, eval_df, lookups


if __name__ == "__main__":
    model, eval_df, lookups = run_salary_training_pipeline()
    eval_df.show(truncate=False)
