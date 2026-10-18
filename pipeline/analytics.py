"""
Spark SQL Market Intelligence & Velocity Engine.
Computes company hiring momentum, remote work adoption, seniority distribution,
and source efficiency matrices using advanced Spark SQL Window Functions.
"""

import sys
import time
from pathlib import Path
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.spark_config import GOLD_DIR, SILVER_DIR
from pipeline.session import get_spark_session


def compute_company_hiring_velocity(df: DataFrame) -> DataFrame:
    """
    Computes week-over-week hiring velocity and momentum per company using Spark SQL Window Functions.

    Formulas:
    - weekly_jobs: COUNT(id) partitioned by company and truncated week.
    - prev_week_jobs: LAG(weekly_jobs, 1) OVER (PARTITION BY company ORDER BY posting_week).
    - jobs_delta: weekly_jobs - prev_week_jobs.
    - growth_rate_pct: ((weekly_jobs - prev_week_jobs) / prev_week_jobs) * 100.
    - weekly_rank: DENSE_RANK() OVER (PARTITION BY posting_week ORDER BY weekly_jobs DESC).
    """
    print("[*] Computing Company Hiring Velocity & Momentum...")

    # Filter out empty companies
    valid_companies = df.filter(
        F.col("company_normalized").isNotNull() & (F.length("company_normalized") > 1)
    )

    # 1. Truncate scraped_at timestamp to start of calendar week (Monday)
    df_weekly = (
        valid_companies
        .withColumn("posting_week", F.date_trunc("week", F.col("scraped_at")))
        .groupBy("company_name", "company_normalized", "posting_week")
        .agg(
            F.count("id").alias("weekly_jobs"),
            F.sum(F.when(F.col("is_remote") == True, 1).otherwise(0)).alias("weekly_remote_jobs"),
            F.round(F.avg("salary_midpoint"), 0).alias("weekly_avg_salary")
        )
    )

    # 2. Define Window specifications
    w_company = Window.partitionBy("company_normalized").orderBy("posting_week")
    w_week_rank = Window.partitionBy("posting_week").orderBy(F.col("weekly_jobs").desc())

    # 3. Apply Window Functions (LAG, DENSE_RANK)
    velocity_df = (
        df_weekly
        .withColumn("prev_week_jobs", F.lag("weekly_jobs", 1).over(w_company))
        .withColumn("jobs_delta", F.col("weekly_jobs") - F.col("prev_week_jobs"))
        .withColumn(
            "growth_rate_pct",
            F.when(
                F.col("prev_week_jobs") > 0,
                F.round(((F.col("weekly_jobs") - F.col("prev_week_jobs")) / F.col("prev_week_jobs")) * 100.0, 1)
            ).otherwise(None)
        )
        .withColumn("weekly_rank", F.dense_rank().over(w_week_rank))
        .withColumn("prev_weekly_rank", F.lag("weekly_rank", 1).over(w_company))
        .withColumn("rank_momentum", F.col("prev_weekly_rank") - F.col("weekly_rank"))
        .orderBy("posting_week", "weekly_rank")
    )

    return velocity_df


def compute_seniority_dynamics(df: DataFrame) -> DataFrame:
    """
    Classifies standardized career seniority tiers from normalized job titles
    and aggregates seniority distributions across companies, employment types, and portals.
    """
    print("[*] Classifying and aggregating Seniority Dynamics...")

    seniority_expr = (
        F.when(F.col("title_normalized").rlike(r"\b(intern|internship|trainee|apprentice)\b"), "Intern")
         .when(F.col("title_normalized").rlike(r"\b(junior|jr|entry|associate|graduate)\b|\bi\b"), "Junior")
         .when(F.col("title_normalized").rlike(r"\b(staff|principal|distinguished|fellow|architect)\b"), "Staff / Principal")
         .when(F.col("title_normalized").rlike(r"\b(director|vp|vice president|head of|chief|cto|cio|ciso)\b"), "Executive / Director")
         .when(F.col("title_normalized").rlike(r"\b(lead|tech lead|team lead)\b"), "Lead")
         .when(F.col("title_normalized").rlike(r"\b(senior|sr|iii)\b"), "Senior")
         .when(F.col("title_normalized").rlike(r"\b(manager|engineering manager|tech manager)\b"), "Manager")
         .otherwise("Mid-Level")
    )

    df_seniority = df.withColumn("seniority_tier", seniority_expr)

    # Aggregate by seniority tier and month
    seniority_summary = (
        df_seniority
        .groupBy("seniority_tier", "scraped_year_month")
        .agg(
            F.count("id").alias("postings_count"),
            F.round(F.avg(F.when(F.col("is_remote") == True, 1.0).otherwise(0.0)) * 100.0, 1).alias("remote_pct"),
            F.round(F.avg("salary_midpoint"), 0).alias("avg_salary_midpoint")
        )
        .orderBy("seniority_tier", "scraped_year_month")
    )

    return seniority_summary


def compute_remote_market_dynamics(df: DataFrame) -> DataFrame:
    """
    Analyzes temporal shifts in remote work adoption, compensation differentials,
    and geographic tech hub concentration.
    """
    print("[*] Computing Remote vs On-Site Dynamics & Geographic Trends...")

    # Overall remote trend by month
    remote_trends = (
        df.groupBy("scraped_year_month")
        .agg(
            F.count("id").alias("total_jobs"),
            F.sum(F.when(F.col("is_remote") == True, 1).otherwise(0)).alias("remote_jobs"),
            F.sum(F.when(F.col("is_remote") == False, 1).otherwise(0)).alias("onsite_jobs"),
            F.round(F.avg(F.when(F.col("is_remote") == True, F.col("salary_midpoint"))), 0).alias("avg_remote_salary"),
            F.round(F.avg(F.when(F.col("is_remote") == False, F.col("salary_midpoint"))), 0).alias("avg_onsite_salary")
        )
        .withColumn("remote_adoption_pct", F.round((F.col("remote_jobs") / F.col("total_jobs")) * 100.0, 2))
        .orderBy("scraped_year_month")
    )

    return remote_trends


def compute_source_efficiency_matrix(df: DataFrame, dup_stats_df: DataFrame) -> DataFrame:
    """
    Combines ingestion health, salary transparency, remote friendliness,
    and cross-portal duplication rates to produce a single portal scorecard.
    """
    print("[*] Computing Source Efficiency & Transparency Matrix...")

    portal_summary = (
        df.groupBy("source")
        .agg(
            F.count("id").alias("total_jobs"),
            F.round(F.avg(F.when(F.col("has_salary") == True, 1.0).otherwise(0.0)) * 100.0, 1).alias("salary_disclosure_pct"),
            F.round(F.avg(F.when(F.col("is_remote") == True, 1.0).otherwise(0.0)) * 100.0, 1).alias("remote_pct"),
            F.round(F.avg("desc_length"), 0).alias("avg_desc_chars")
        )
    )

    # Join with portal duplication stats from deduplication stage
    efficiency_matrix = (
        portal_summary.join(
            dup_stats_df.select("source", "duplicate_postings", "duplication_rate_pct"),
            on="source",
            how="left"
        )
        .fillna(0, subset=["duplicate_postings", "duplication_rate_pct"])
        .withColumn("net_unique_jobs", F.col("total_jobs") - F.col("duplicate_postings"))
        .orderBy(F.col("total_jobs").desc())
    )

    return efficiency_matrix


def run_market_analytics_pipeline():
    """
    Executes the Spark SQL analytical pipeline and persists Gold tables to disk.
    """
    spark = get_spark_session("JobMarket_Analytics_Engine")
    silver_path = str(SILVER_DIR / "jobs.parquet")
    print(f"[*] Reading Silver dataset from {silver_path}...")
    df = spark.read.parquet(silver_path)

    dup_stats_path = str(GOLD_DIR / "portal_duplication_stats.parquet")
    print(f"[*] Reading Portal Duplication Stats from {dup_stats_path}...")
    dup_stats_df = spark.read.parquet(dup_stats_path)

    t0 = time.time()

    # 1. Company Hiring Velocity
    velocity_df = compute_company_hiring_velocity(df)
    velocity_path = str(GOLD_DIR / "company_hiring_velocity.parquet")
    print(f"[*] Persisting {velocity_path}...")
    velocity_df.write.mode("overwrite").parquet(velocity_path)

    # 2. Seniority Dynamics
    seniority_df = compute_seniority_dynamics(df)
    seniority_path = str(GOLD_DIR / "seniority_dynamics.parquet")
    print(f"[*] Persisting {seniority_path}...")
    seniority_df.write.mode("overwrite").parquet(seniority_path)

    # 3. Remote Market Dynamics
    remote_df = compute_remote_market_dynamics(df)
    remote_path = str(GOLD_DIR / "remote_market_dynamics.parquet")
    print(f"[*] Persisting {remote_path}...")
    remote_df.write.mode("overwrite").parquet(remote_path)

    # 4. Source Efficiency Matrix
    efficiency_df = compute_source_efficiency_matrix(df, dup_stats_df)
    efficiency_path = str(GOLD_DIR / "source_efficiency_matrix.parquet")
    print(f"[*] Persisting {efficiency_path}...")
    efficiency_df.write.mode("overwrite").parquet(efficiency_path)

    print(f"\n[SUCCESS] Market Analytics Pipeline completed in {time.time()-t0:.2f}s!")
    return velocity_df, seniority_df, remote_df, efficiency_df


if __name__ == "__main__":
    v_df, s_df, r_df, e_df = run_market_analytics_pipeline()

    print("\n=== TOP 10 HIRING VELOCITY COMPANIES (MOST RECENT WEEK) ===")
    recent_week = v_df.select(F.max("posting_week")).collect()[0][0]
    v_df.filter(F.col("posting_week") == recent_week).orderBy("weekly_rank").show(10, truncate=False)

    print("\n=== SENIORITY DISTRIBUTION (RECENT MONTH) ===")
    s_df.filter(F.col("scraped_year_month") == "2026-10").show(10, truncate=False)

    print("\n=== REMOTE WORK ADOPTION TRENDS ===")
    r_df.show(10, truncate=False)

    print("\n=== SOURCE EFFICIENCY & REDUNDANCY SCORECARD ===")
    e_df.show(20, truncate=False)
