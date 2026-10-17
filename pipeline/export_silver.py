"""
Silver Layer Exporter: Ingests, cleans, and saves the 493K job records to a partitioned Parquet Lakehouse.
Includes performance benchmarks and partition pruning demonstration.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.spark_config import SILVER_DIR
from pipeline.clean import clean_and_normalize_jobs
from pipeline.ingest import ingest_jobs_dataframe
from pipeline.session import get_spark_session
import pyspark.sql.functions as F


def export_to_silver_parquet() -> None:
    """
    Executes the full ingestion and cleaning pipeline, saving the output
    as an optimized partitioned Parquet dataset in data/silver/jobs.parquet.
    """
    total_start = time.time()
    spark = get_spark_session("SilverLakehouseExporter")
    print("=" * 70)
    print(" EXPORTING 493K JOBS TO SILVER PARQUET LAKEHOUSE")
    print("=" * 70)

    # 1. Parallel Ingestion from Postgres
    print("\n[1/4] Ingesting all rows via parallel JDBC (16 partitions)...")
    ingest_start = time.time()
    raw_df = ingest_jobs_dataframe(spark, num_partitions=16)

    # 2. Catalyst Cleaning (HTML & Markdown stripped)
    print("[2/4] Applying Catalyst expressions (HTML & Markdown sanitization)...")
    clean_df = clean_and_normalize_jobs(raw_df)

    # 3. Write to Parquet partitioned by scraped_year_month
    output_path = str(SILVER_DIR / "jobs.parquet")
    print(f"[3/4] Writing to partitioned Parquet lakehouse at: {output_path}...")
    write_start = time.time()
    (
        clean_df.write
        .mode("overwrite")
        .partitionBy("scraped_year_month")
        .parquet(output_path)
    )
    write_duration = round(time.time() - write_start, 2)
    print(f" -> Parquet write completed in {write_duration}s")

    # 4. Measure Size and Verify by Reading back
    print("\n[4/4] Verifying Silver Lakehouse and Benchmarking Columnar Reads...")
    read_start = time.time()
    silver_df = spark.read.parquet(output_path)
    record_count = silver_df.count()
    read_duration = round(time.time() - read_start, 2)

    # Calculate on-disk size
    total_bytes = sum(f.stat().st_size for f in Path(output_path).rglob("*.parquet"))
    size_mb = round(total_bytes / (1024 * 1024), 2)

    print(f" -> Verified Record Count: {record_count:,} rows")
    print(f" -> Total Parquet Lakehouse Size on disk: {size_mb} MB (compressed with Snappy)")
    print(f" -> Full scan read time: {read_duration}s ({round(record_count / max(read_duration, 0.01)):,} rows/sec)")

    # Demonstrate Partition Pruning
    print("\n" + "-" * 50)
    print("DEMO: PARTITION PRUNING IN SPARK SQL")
    print("-" * 50)
    october_count = silver_df.filter(F.col("scraped_year_month") == "2026-10").count()
    print(f" -> October 2026 jobs: {october_count:,} rows (Spark pruned all other months without reading them)")

    total_time = round(time.time() - total_start, 2)
    print(f"\n[*] Complete pipeline finished in {total_time}s")
    spark.stop()


if __name__ == "__main__":
    export_to_silver_parquet()
