"""
Configuration constants and database connection settings for the Spark pipeline.
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
JARS_DIR = BASE_DIR / "jars"
DATA_DIR = BASE_DIR / "data"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
CONFIG_DIR = BASE_DIR / "config"

# Spark driver / executor tuning
SPARK_APP_NAME = "JobMarketIntelligence"
SPARK_MASTER = "local[*]"
SPARK_DRIVER_MEMORY = "6g"
SPARK_SQL_SHUFFLE_PARTITIONS = 20
POSTGRES_JDBC_JAR = str(JARS_DIR / "postgresql-42.7.3.jar")

# Local Postgres JDBC settings
POSTGRES_JDBC_URL = "jdbc:postgresql://localhost:5432/jobcruiser"
POSTGRES_USER = "jobcruiser"
POSTGRES_PASSWORD = "jobcruiser"
POSTGRES_DRIVER = "org.postgresql.Driver"

# Parallel partition settings for JDBC ingestion
JDBC_NUM_PARTITIONS = 16
