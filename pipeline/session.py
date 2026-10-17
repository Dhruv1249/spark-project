"""
SparkSession factory module with optimized cluster and local memory settings.
"""

import os
import sys
from pyspark.sql import SparkSession

# Ensure Spark workers use the exact same virtualenv Python interpreter
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from config.spark_config import (
    POSTGRES_JDBC_JAR,
    SPARK_APP_NAME,
    SPARK_DRIVER_MEMORY,
    SPARK_MASTER,
    SPARK_SQL_SHUFFLE_PARTITIONS,
)


def get_spark_session(app_name: str = SPARK_APP_NAME) -> SparkSession:
    """
    Initializes and returns a tuned local PySpark SparkSession.

    Architecture Details:
    ---------------------
    - spark.master: Set to 'local[*]' to utilize all available physical and logical CPU cores.
    - spark.driver.memory: Allocated 6GB heap to store metadata, broadcast variables, and query plans.
    - spark.jars: Registers the PostgreSQL JDBC driver jar for parallel relational ingestion.
    - spark.sql.shuffle.partitions: Optimized to 20 partitions to match CPU parallelism and avoid
      the default 200 partitions which creates small-file overhead on local execution.
    - spark.sql.adaptive.enabled: Enables Adaptive Query Execution (AQE) in Spark to dynamically
      coalesce shuffle partitions and optimize joins at runtime.
    """
    builder = (
        SparkSession.builder.appName(app_name)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", str(SPARK_SQL_SHUFFLE_PARTITIONS))
        .config("spark.jars", POSTGRES_JDBC_JAR)
        .config("spark.driver.extraClassPath", POSTGRES_JDBC_JAR)
        .config("spark.executor.extraClassPath", POSTGRES_JDBC_JAR)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .config("spark.ui.enabled", "true")
        .config("spark.ui.port", "4040")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
