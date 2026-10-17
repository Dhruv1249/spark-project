"""
Skill & Technology Stack Extraction Engine using Spark MLlib patterns,
Broadcast Variables, and Co-occurrence Matrix Aggregation.
"""

import json
import re
import sys
from pathlib import Path
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.spark_config import CONFIG_DIR, GOLD_DIR, SILVER_DIR


def load_skills_taxonomy() -> tuple[dict[str, list[str]], dict[str, str], dict[str, str], re.Pattern]:
    """
    Loads taxonomy and builds canonical synonym mappings, category lookup,
    and a compiled boundary-aware regex pattern.

    Design Note:
    ------------
    Regex preserves specialized programming language characters like '+' in 'c++',
    '#' in 'c#', and '.' in '.net' and 'node.js', while matching only whole words.
    """
    taxonomy_file = CONFIG_DIR / "skills_taxonomy.json"
    with open(taxonomy_file, "r", encoding="utf-8") as f:
        taxonomy: dict[str, list[str]] = json.load(f)

    all_skills: set[str] = set()
    canonical_map: dict[str, str] = {}
    skill_to_category: dict[str, str] = {}

    for category, skills in taxonomy.items():
        for skill in skills:
            clean_s = skill.lower().strip()
            all_skills.add(clean_s)
            skill_to_category[clean_s] = category

            # Synonym mappings to canonical names
            if clean_s == "k8s":
                canonical_map["k8s"] = "kubernetes"
            elif clean_s in ("postgres", "postgresql"):
                canonical_map[clean_s] = "postgresql"
            elif clean_s in ("golang", "go"):
                canonical_map[clean_s] = "go"
            elif clean_s in ("nodejs", "node.js"):
                canonical_map[clean_s] = "node.js"
            elif clean_s in ("react", "react.js"):
                canonical_map[clean_s] = "react"
            elif clean_s in ("vue", "vue.js"):
                canonical_map[clean_s] = "vue"
            elif clean_s in ("express", "express.js"):
                canonical_map[clean_s] = "express"
            elif clean_s in ("amazon web services", "aws"):
                canonical_map[clean_s] = "aws"
            elif clean_s in ("google cloud", "gcp"):
                canonical_map[clean_s] = "gcp"
            elif clean_s in ("ruby on rails", "rails"):
                canonical_map[clean_s] = "ruby on rails"
            elif clean_s in ("spring", "spring boot"):
                canonical_map[clean_s] = "spring boot"
            elif clean_s in (".net", "asp.net", ".net core"):
                canonical_map[clean_s] = ".net"
            else:
                canonical_map[clean_s] = clean_s

    # Sort descending by length so multi-word tokens match first (e.g. "spring boot" before "spring")
    sorted_skills = sorted(all_skills, key=len, reverse=True)
    escaped = [re.escape(s) for s in sorted_skills]
    # Boundary check: ensure not preceded or followed by alphanumeric or symbol tokens
    pattern_str = r"(?<![a-zA-Z0-9_#+])(" + "|".join(escaped) + r")(?![a-zA-Z0-9_#+])"
    pattern = re.compile(pattern_str, re.IGNORECASE)

    return taxonomy, canonical_map, skill_to_category, pattern


def extract_skills_pipeline(spark: SparkSession, jobs_df: DataFrame) -> DataFrame:
    """
    Extracts tech skills from job descriptions using Broadcast Variables and Arrow UDF.

    Spark Concept - Broadcast Variables:
    ------------------------------------
    Instead of serializing our regex pattern and synonym map with every task (16 tasks x 428k rows),
    spark.sparkContext.broadcast() ships the read-only dictionary ONCE to each worker machine.
    All cores on that executor share the exact same in-memory object, saving memory and CPU cycles.
    """
    _, canonical_map, skill_to_category, pattern = load_skills_taxonomy()

    # Broadcast dictionaries to all cluster worker nodes
    b_pattern = spark.sparkContext.broadcast(pattern)
    b_map = spark.sparkContext.broadcast(canonical_map)

    @F.udf(returnType=T.ArrayType(T.StringType()))
    def extract_skills(text: str | None) -> list[str]:
        if not text:
            return []
        matches = b_pattern.value.findall(text.lower())
        unique_canonical = {b_map.value.get(m.lower(), m.lower()) for m in matches}
        return sorted(list(unique_canonical))

    enriched_df = (
        jobs_df
        .withColumn("skills", extract_skills(F.col("clean_desc")))
        .withColumn("skill_count", F.size(F.col("skills")))
    )

    return enriched_df


def compute_skill_rankings(enriched_df: DataFrame) -> DataFrame:
    """
    Aggregates skill demand rankings overall and by category across all job listings.
    """
    _, _, skill_to_category, _ = load_skills_taxonomy()
    total_jobs = enriched_df.count()

    # Explode the skills array into 1 row per skill
    exploded_df = (
        enriched_df
        .select("id", "source", "scraped_year_month", F.explode("skills").alias("skill"))
    )

    # Broadcast category lookup mapping
    category_map_expr = F.create_map([F.lit(x) for item in skill_to_category.items() for x in item])

    rankings_df = (
        exploded_df
        .groupBy("skill")
        .agg(
            F.count("id").alias("job_count"),
            F.countDistinct("source").alias("sources_count")
        )
        .withColumn("category", F.coalesce(category_map_expr[F.col("skill")], F.lit("other")))
        .withColumn("demand_percentage", F.round((F.col("job_count") / F.lit(total_jobs)) * 100, 2))
        .orderBy(F.col("job_count").desc())
    )

    return rankings_df


def compute_skill_cooccurrence(enriched_df: DataFrame, min_job_count: int = 50) -> DataFrame:
    """
    Computes technology co-occurrence pairings across all job postings.
    Generates the top paired skills (e.g. Docker + Kubernetes, Python + PyTorch).

    Spark Concept - Array Combinations & Pairwise Aggregation:
    ----------------------------------------------------------
    For each job with N skills, we create distinct pairs (skill_a, skill_b) where skill_a < skill_b.
    This avoids generating duplicate symmetric pairs (A, B) and (B, A).
    """
    # Filter jobs having at least 2 skills
    multi_skill_jobs = enriched_df.filter(F.col("skill_count") >= 2).select("id", "skills")

    # Generate pairs using array self-combination via Spark SQL expressions
    pairs_df = multi_skill_jobs.select(
        F.col("id"),
        F.expr("""
            flatten(
                transform(skills, (x, i) -> 
                    transform(slice(skills, i + 2, size(skills)), y -> named_struct('skill_a', x, 'skill_b', y))
                )
            )
        """).alias("pairs")
    ).select(F.col("id"), F.explode("pairs").alias("pair"))

    cooccurrence_df = (
        pairs_df
        .select("pair.skill_a", "pair.skill_b")
        .groupBy("skill_a", "skill_b")
        .agg(F.count("*").alias("cooccurrence_count"))
        .filter(F.col("cooccurrence_count") >= min_job_count)
        .orderBy(F.col("cooccurrence_count").desc())
    )

    return cooccurrence_df
