"""
Verification script: Tests SparkSession initialization, parallel JDBC ingestion,
and Catalyst cleaning transformations on our local PostgreSQL database.
"""

import time
from pipeline.session import get_spark_session
from pipeline.ingest import ingest_jobs_dataframe
from pipeline.clean import clean_and_normalize_jobs
import pyspark.sql.functions as F


def main() -> None:
    print("=" * 70)
    print(" STEP 1: INITIALIZING TUNED SPARK SESSION")
    print("=" * 70)

    start_time = time.time()
    spark = get_spark_session("SparkIngestVerification")
    print(f"[*] Spark Version: {spark.version}")
    print(f"[*] Spark Master: {spark.conf.get('spark.master')}")
    print(f"[*] Shuffle Partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
    print(f"[*] Driver Memory: {spark.conf.get('spark.driver.memory')}")
    print(f"[*] Spark Web UI URL: http://localhost:4040")

    print("\n" + "=" * 70)
    print(" STEP 2: PARALLEL PARTITIONED JDBC INGESTION")
    print("=" * 70)

    # Ingest using our 16 parallel slices
    raw_df = ingest_jobs_dataframe(spark, num_partitions=16)
    num_partitions = raw_df.rdd.getNumPartitions()
    print(f"[*] DataFrame RDD Partitions: {num_partitions} parallel slices")
    print(f"[*] Schema columns: {len(raw_df.columns)} fields")

    print("\n" + "=" * 70)
    print(" STEP 3: CATALYST OPTIMIZED CLEANING & TRANSFORMATION")
    print("=" * 70)

    clean_df = clean_and_normalize_jobs(raw_df)

    print("[*] Displaying Catalyst Physical Execution Plan:")
    print("-" * 50)
    clean_df.explain(mode="simple")
    print("-" * 50)

    print("\n" + "=" * 70)
    print(" STEP 4: TRIGGERING FIRST ACTION (EVALUATING SAMPLE DATA)")
    print("=" * 70)

    sample_rows = (
        clean_df.select(
            "title_normalized",
            "company_normalized",
            "location",
            "source",
            "standard_employment_type",
            "is_remote",
            "has_salary",
            "desc_length"
        )
        .limit(5)
        .collect()
    )

    print("\nSample Cleaned Records:")
    for idx, row in enumerate(sample_rows, 1):
        print(f" [{idx}] {row['title_normalized']} @ {row['company_normalized']}")
        print(f"     Location: {row['location']} | Source: {row['source']} | Remote: {row['is_remote']}")
        print(f"     Type: {row['standard_employment_type']} | Has Salary: {row['has_salary']} | Desc Length: {row['desc_length']} chars")
        print()

    elapsed = round(time.time() - start_time, 2)
    print(f"[*] Verification complete in {elapsed}s")
    spark.stop()


if __name__ == "__main__":
    main()
