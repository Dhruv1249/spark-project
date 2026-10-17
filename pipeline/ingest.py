"""
Parallel JDBC ingestion module for extracting relational data into Spark DataFrames.
"""

import psycopg
from pyspark.sql import DataFrame, SparkSession
from config.spark_config import (
    JDBC_NUM_PARTITIONS,
    POSTGRES_DRIVER,
    POSTGRES_JDBC_URL,
    POSTGRES_PASSWORD,
    POSTGRES_USER,
)


def get_epoch_bounds() -> tuple[int, int]:
    """
    Queries PostgreSQL to retrieve the minimum and maximum epoch timestamps of jobs.
    These bounds define the partitioning range for Spark's parallel JDBC workers.
    """
    connection_string = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:5432/jobcruiser"
    with psycopg.connect(connection_string) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    COALESCE(EXTRACT(EPOCH FROM MIN(scraped_at))::bigint, 0),
                    COALESCE(EXTRACT(EPOCH FROM MAX(scraped_at))::bigint, 0)
                FROM jobs;
            """)
            min_epoch, max_epoch = cursor.fetchone()
            return int(min_epoch), int(max_epoch)


def ingest_jobs_dataframe(spark: SparkSession, num_partitions: int = JDBC_NUM_PARTITIONS) -> DataFrame:
    """
    Extracts 490k+ jobs from PostgreSQL into a partitioned Spark DataFrame.

    Spark Core Concept - Parallel JDBC Slicing:
    -------------------------------------------
    By default, spark.read.jdbc() runs on a SINGLE core using 1 task, creating
    a severe bottleneck on large tables. By supplying `partitionColumn`, `lowerBound`,
    `upperBound`, and `numPartitions`, Spark dynamically splits the table into N disjoint
    ranges:
        Partition 0: WHERE partition_epoch < stride_1
        Partition 1: WHERE partition_epoch >= stride_1 AND partition_epoch < stride_2
        ...
        Partition N: WHERE partition_epoch >= stride_N
    Each slice is ingested concurrently by an executor task, maximizing network and disk I/O.
    """
    min_epoch, max_epoch = get_epoch_bounds()

    # Wrap the table and join companies to enrich company_name directly in the source query
    source_query = """
    (
        SELECT 
            j.id,
            j.company_id,
            COALESCE(c.name, 'Unknown Company') AS company_name,
            j.title,
            COALESCE(j.location, 'Unspecified') AS location,
            j.salary_min,
            j.salary_max,
            COALESCE(j.currency, 'USD') AS currency,
            j.job_type,
            j.employment_type,
            COALESCE(j.is_remote, false) AS is_remote,
            COALESCE(j.is_easy_apply, false) AS is_easy_apply,
            j.source,
            j.url,
            j.raw_desc,
            j.scraped_at,
            EXTRACT(EPOCH FROM j.scraped_at)::bigint AS partition_epoch
        FROM jobs j
        LEFT JOIN companies c ON j.company_id = c.id
    ) AS jobs_partitioned_source
    """

    dataframe = (
        spark.read.format("jdbc")
        .option("url", POSTGRES_JDBC_URL)
        .option("dbtable", source_query)
        .option("user", POSTGRES_USER)
        .option("password", POSTGRES_PASSWORD)
        .option("driver", POSTGRES_DRIVER)
        .option("partitionColumn", "partition_epoch")
        .option("lowerBound", str(min_epoch))
        .option("upperBound", str(max_epoch))
        .option("numPartitions", str(num_partitions))
        .option("fetchsize", "10000")
        .load()
    )

    return dataframe
