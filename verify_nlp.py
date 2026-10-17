"""
NLP Pipeline Verification & Execution Runner:
Extracts skills across all jobs, computes market rankings and technology co-occurrence matrix,
and persists Gold layer analytical outputs.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.spark_config import GOLD_DIR, SILVER_DIR
from pipeline.nlp_skills import (
    compute_skill_cooccurrence,
    compute_skill_rankings,
    extract_skills_pipeline,
)
from pipeline.session import get_spark_session
import pyspark.sql.functions as F


def main() -> None:
    print("=" * 70)
    print(" NLP & SKILL INTELLIGENCE PIPELINE (428K RECORDS)")
    print("=" * 70)

    start_time = time.time()
    spark = get_spark_session("SkillIntelligencePipeline")

    # 1. Load Cleaned Dataset from Silver Parquet Lakehouse
    silver_path = str(SILVER_DIR / "jobs.parquet")
    print(f"\n[1/4] Loading cleaned dataset from Silver Parquet: {silver_path}...")
    jobs_df = spark.read.parquet(silver_path)
    total_records = jobs_df.count()
    print(f" -> Successfully loaded {total_records:,} records in {round(time.time() - start_time, 2)}s")

    # 2. Extract Skills using Broadcast Regex Engine
    print("\n[2/4] Running Broadcast-Accelerated Skill Extraction...")
    extract_start = time.time()
    enriched_df = extract_skills_pipeline(spark, jobs_df)
    # Cache enriched dataframe because we will use it for both rankings and co-occurrence
    enriched_df.cache()

    # Trigger action to evaluate
    jobs_with_skills_count = enriched_df.filter(F.col("skill_count") > 0).count()
    extract_duration = round(time.time() - extract_start, 2)
    print(f" -> Extraction complete in {extract_duration}s ({round(total_records / extract_duration):,} jobs/sec)")
    print(f" -> Jobs with at least 1 identified skill: {jobs_with_skills_count:,} ({round(jobs_with_skills_count / total_records * 100, 1)}%)")

    # 3. Compute Skill Market Rankings
    print("\n[3/4] Aggregating Skill Demand Rankings...")
    rankings_df = compute_skill_rankings(enriched_df)
    rankings_df.cache()

    print("\nTop 15 Most In-Demand Technologies:")
    print("-" * 65)
    print(f"{'Rank':<5} | {'Skill':<22} | {'Category':<15} | {'Jobs Count':<10} | {'Demand %':<8}")
    print("-" * 65)
    top_15 = rankings_df.take(15)
    for idx, row in enumerate(top_15, 1):
        print(f"{idx:<5} | {row['skill']:<22} | {row['category']:<15} | {row['job_count']:<10,d} | {row['demand_percentage']:<8}%")
    print("-" * 65)

    # 4. Compute Technology Co-occurrence Matrix
    print("\n[4/4] Computing Technology Co-occurrence Matrix (Skill Pairings)...")
    cooccur_start = time.time()
    cooccurrence_df = compute_skill_cooccurrence(enriched_df, min_job_count=100)
    cooccurrence_df.cache()
    cooccur_duration = round(time.time() - cooccur_start, 2)
    print(f" -> Co-occurrence matrix generated in {cooccur_duration}s")

    print("\nTop 10 Most Frequently Paired Technologies:")
    print("-" * 55)
    print(f"{'Skill A':<18} + {'Skill B':<18} | {'Together in Jobs'}")
    print("-" * 55)
    top_pairs = cooccurrence_df.take(10)
    for row in top_pairs:
        print(f"{row['skill_a']:<18} + {row['skill_b']:<18} | {row['cooccurrence_count']:,} jobs")
    print("-" * 55)

    # 5. Persist to Gold Analytical Layer
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    rankings_output = str(GOLD_DIR / "skills_ranking.parquet")
    cooccur_output = str(GOLD_DIR / "skill_cooccurrence.parquet")

    print("\nSaving Gold Analytical Tables...")
    rankings_df.write.mode("overwrite").parquet(rankings_output)
    cooccurrence_df.write.mode("overwrite").parquet(cooccur_output)
    print(f" -> Saved: {rankings_output}")
    print(f" -> Saved: {cooccur_output}")

    total_time = round(time.time() - start_time, 2)
    print(f"\n[*] Full NLP Pipeline completed in {total_time}s")
    spark.stop()


if __name__ == "__main__":
    main()
