"""
Cross-Source Fuzzy Deduplication Engine using Character Shingling, Spark MLlib MinHash LSH,
and Graph-Based Disjoint Set Union (Connected Components) Clustering.
Finds duplicate and syndicated job postings across portals (Indeed, LinkedIn, Greenhouse, Ashby, Lever) in sub-quadratic time.
"""

import sys
import time
from pathlib import Path
from pyspark.ml.feature import CountVectorizer, MinHashLSH, NGram
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.spark_config import GOLD_DIR, SILVER_DIR
from pipeline.session import get_spark_session


class DisjointSet:
    """
    High-performance Disjoint Set Union (Union-Find) with path compression
    and union by rank for clustering duplicate job postings into canonical clusters.
    Runs in near O(1) amortized time per operation: O(M * alpha(N)).
    """
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                root_x, root_y = root_y, root_x
            self.parent[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1


def build_job_signatures(df: DataFrame) -> DataFrame:
    """
    Constructs a unified signature string for each job by combining
    normalized company, normalized title, and standardized location.
    """
    clean_loc = F.lower(F.trim(F.regexp_replace(F.col("location"), "[^a-zA-Z0-9\\s]", " ")))
    clean_loc = F.regexp_replace(clean_loc, "\\s+", " ")

    # Signature: company + title + location
    signature = F.concat_ws(" ", F.col("company_normalized"), F.col("title_normalized"), clean_loc)

    return df.withColumn("signature", signature)


def create_minhash_lsh_pipeline(
    df: DataFrame,
    vocab_size: int = 25000,
    min_df: float = 2.0,
    num_hash_tables: int = 3
) -> tuple[DataFrame, CountVectorizer, MinHashLSH]:
    """
    Builds the MLlib feature extraction pipeline:
    1. Splits signature string into character sequences
    2. NGram: generates character 3-grams (shingles)
    3. CountVectorizer: converts shingles to sparse term-frequency vectors
    4. MinHashLSH: fits locality-sensitive hash tables for sub-quadratic nearest neighbors
    """
    # 1. Split signature into individual characters
    char_df = df.withColumn(
        "chars",
        F.split(F.regexp_replace(F.col("signature"), "[^a-zA-Z0-9]", ""), "")
    ).filter(F.size("chars") >= 4)

    # 2. Character 3-gram shingling
    ngram = NGram(n=3, inputCol="chars", outputCol="shingles")
    shingled_df = ngram.transform(char_df)

    # 3. Vectorize shingles
    vectorizer = CountVectorizer(
        inputCol="shingles",
        outputCol="features",
        vocabSize=vocab_size,
        minDF=min_df,
        binary=True
    )
    cv_model = vectorizer.fit(shingled_df)
    vectorized_df = cv_model.transform(shingled_df)

    # 4. Fit MinHash LSH Model
    minhash = MinHashLSH(
        inputCol="features",
        outputCol="lsh_hashes",
        numHashTables=num_hash_tables,
        seed=42
    )
    lsh_model = minhash.fit(vectorized_df)
    hashed_df = lsh_model.transform(vectorized_df)

    return hashed_df, cv_model, lsh_model


def find_cross_source_duplicates(
    hashed_df: DataFrame,
    lsh_model: MinHashLSH,
    jaccard_distance_threshold: float = 0.25,
    ats_only_match: bool = True
) -> DataFrame:
    """
    Executes an approximate similarity join on LSH hash buckets to find duplicate pairs
    with Jaccard Distance <= threshold (i.e. Jaccard Similarity >= 1 - threshold).

    Big Data Strategy:
    ------------------
    - If ats_only_match is True: Joins primary ATS postings (Greenhouse, Lever, Ashby, etc.)
      against aggregator listings (LinkedIn, Indeed) to discover cross-portal syndication.
    - If ats_only_match is False: Joins across all records with id_a < id_b.
    """
    ats_sources = ["greenhouse", "ashby", "lever", "smartrecruiters", "oracle_cloud"]

    if ats_only_match:
        subset_a = hashed_df.filter(F.col("source").isin(ats_sources)).select(
            F.col("id").alias("id_a"),
            F.col("title").alias("title_a"),
            F.col("company_name").alias("company_a"),
            F.col("source").alias("source_a"),
            F.col("location").alias("location_a"),
            F.col("features")
        )
        subset_b = hashed_df.filter(~F.col("source").isin(ats_sources)).select(
            F.col("id").alias("id_b"),
            F.col("title").alias("title_b"),
            F.col("company_name").alias("company_b"),
            F.col("source").alias("source_b"),
            F.col("location").alias("location_b"),
            F.col("features")
        )
    else:
        subset_a = hashed_df.select(
            F.col("id").alias("id_a"),
            F.col("title").alias("title_a"),
            F.col("company_name").alias("company_a"),
            F.col("source").alias("source_a"),
            F.col("location").alias("location_a"),
            F.col("features")
        )
        subset_b = hashed_df.select(
            F.col("id").alias("id_b"),
            F.col("title").alias("title_b"),
            F.col("company_name").alias("company_b"),
            F.col("source").alias("source_b"),
            F.col("location").alias("location_b"),
            F.col("features")
        )

    print(f"[*] Executing MinHash approxSimilarityJoin (max Jaccard distance = {jaccard_distance_threshold})...")
    raw_pairs = lsh_model.approxSimilarityJoin(
        subset_a,
        subset_b,
        threshold=jaccard_distance_threshold,
        distCol="jaccard_distance"
    )

    duplicate_pairs = (
        raw_pairs
        .filter(F.col("datasetA.id_a") != F.col("datasetB.id_b"))
        .filter(F.col("datasetA.id_a") < F.col("datasetB.id_b") if not ats_only_match else F.lit(True))
        .select(
            F.col("datasetA.id_a").alias("job_id_a"),
            F.col("datasetA.title_a").alias("title_a"),
            F.col("datasetA.company_a").alias("company_a"),
            F.col("datasetA.source_a").alias("source_a"),
            F.col("datasetA.location_a").alias("location_a"),
            F.col("datasetB.id_b").alias("job_id_b"),
            F.col("datasetB.title_b").alias("title_b"),
            F.col("datasetB.company_b").alias("company_b"),
            F.col("datasetB.source_b").alias("source_b"),
            F.col("datasetB.location_b").alias("location_b"),
            F.round(1.0 - F.col("jaccard_distance"), 4).alias("similarity_score"),
            F.when(F.col("datasetA.source_a") != F.col("datasetB.source_b"), True).otherwise(False).alias("is_cross_source")
        )
        .orderBy(F.col("similarity_score").desc())
    )

    return duplicate_pairs


def cluster_duplicate_pairs(
    spark: SparkSession,
    duplicate_pairs_df: DataFrame,
    raw_jobs_df: DataFrame
) -> DataFrame:
    """
    Applies Connected Components via Disjoint Set Union (Union-Find) to group
    pairwise duplicate matches into transitive equivalence classes (clusters).
    Example: (A ~ B) and (B ~ C) -> Cluster {A, B, C}.
    """
    print("[*] Grouping pairwise duplicates into transitive clusters via Disjoint Set...")
    pair_edges = duplicate_pairs_df.select("job_id_a", "job_id_b").collect()

    ds = DisjointSet()
    for row in pair_edges:
        ds.union(row["job_id_a"], row["job_id_b"])

    # Collect members per root cluster
    clusters_map: dict[str, list[str]] = {}
    for node in ds.parent.keys():
        root = ds.find(node)
        clusters_map.setdefault(root, []).append(node)

    cluster_rows = []
    for idx, (root, members) in enumerate(clusters_map.items(), start=1):
        cid = f"CLUST_{idx:05d}"
        for member_id in members:
            cluster_rows.append((member_id, cid, len(members), member_id == root))

    schema = T.StructType([
        T.StructField("job_id", T.StringType(), False),
        T.StructField("cluster_id", T.StringType(), False),
        T.StructField("cluster_size", T.IntegerType(), False),
        T.StructField("is_canonical", T.BooleanType(), False)
    ])

    clusters_df = spark.createDataFrame(cluster_rows, schema=schema)

    # Join with jobs metadata to enrich cluster records
    enriched_clusters = (
        clusters_df.join(
            raw_jobs_df.select("id", "title", "company_name", "source", "location", "employment_type"),
            clusters_df["job_id"] == raw_jobs_df["id"],
            how="inner"
        )
        .drop("id")
        .orderBy("cluster_id", F.col("is_canonical").desc(), "source")
    )

    return enriched_clusters


def compute_portal_duplication_stats(
    raw_jobs_df: DataFrame,
    clusters_df: DataFrame
) -> DataFrame:
    """
    Calculates duplication metrics per portal:
    - Total listings per portal
    - Count of duplicate listings in clusters
    - Percentage duplication rate
    """
    total_by_source = raw_jobs_df.groupBy("source").agg(F.count("id").alias("total_postings"))

    dup_jobs = clusters_df.select("job_id", "source").distinct()
    dup_by_source = dup_jobs.groupBy("source").agg(F.count("job_id").alias("duplicate_postings"))

    stats_df = (
        total_by_source.join(dup_by_source, on="source", how="left")
        .fillna(0, subset=["duplicate_postings"])
        .withColumn(
            "duplication_rate_pct",
            F.round((F.col("duplicate_postings") / F.col("total_postings")) * 100.0, 2)
        )
        .orderBy(F.col("duplicate_postings").desc(), F.col("total_postings").desc())
    )

    return stats_df


def run_deduplication_pipeline(sample_fraction: float | None = None) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Executes end-to-end deduplication pipeline:
    1. Loads clean Silver dataset
    2. Builds signatures & shingles
    3. Fits MinHash LSH model
    4. Evaluates cross-portal approximate similarity join
    5. Disjoint Set Union clustering
    6. Writes Gold Parquet tables
    """
    spark = get_spark_session("JobMarket_MinHash_Deduplication")
    silver_path = str(SILVER_DIR / "jobs.parquet")
    print(f"[*] Reading Silver dataset from {silver_path}...")
    df = spark.read.parquet(silver_path)

    if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        print(f"[*] Subsampling dataset with fraction={sample_fraction}...")
        df = df.sample(fraction=sample_fraction, seed=42)

    total_records = df.count()
    print(f"[*] Total records to analyze: {total_records:,}")

    # Build signatures
    t0 = time.time()
    sig_df = build_job_signatures(df)

    # Build LSH model
    print("[*] Building character 3-grams, CountVectorizer, and MinHash LSH model...")
    hashed_df, cv_model, lsh_model = create_minhash_lsh_pipeline(
        sig_df,
        vocab_size=25000,
        min_df=2.0,
        num_hash_tables=3
    )
    print(f"[+] LSH Pipeline trained with vocabulary size {len(cv_model.vocabulary):,} in {time.time()-t0:.2f}s")

    # Similarity Join & Checkpoint-to-Lakehouse Pattern
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    pairs_path = str(GOLD_DIR / "duplicate_pairs.parquet")
    clusters_path = str(GOLD_DIR / "duplicate_clusters.parquet")
    stats_path = str(GOLD_DIR / "portal_duplication_stats.parquet")

    t_join = time.time()
    raw_dup_pairs = find_cross_source_duplicates(
        hashed_df,
        lsh_model,
        jaccard_distance_threshold=0.25,
        ats_only_match=True
    )
    print(f"[*] Persisting duplicate pairs to {pairs_path}...")
    raw_dup_pairs.write.mode("overwrite").parquet(pairs_path)
    print(f"[+] Wrote duplicate pairs to Parquet in {time.time()-t_join:.2f}s!")

    # Read back persisted pairs to avoid re-evaluating the heavy LSH join DAG
    duplicate_pairs = spark.read.parquet(pairs_path)
    pair_count = duplicate_pairs.count()
    print(f"[+] Total verified duplicate pairs: {pair_count:,}")

    # Graph Clustering via Disjoint Set Union
    t_clust = time.time()
    clusters_df = cluster_duplicate_pairs(spark, duplicate_pairs, df)
    cluster_count = clusters_df.select("cluster_id").distinct().count()
    print(f"[+] Formed {cluster_count:,} distinct duplicate clusters in {time.time()-t_clust:.2f}s!")

    # Portal Duplication Stats
    stats_df = compute_portal_duplication_stats(df, clusters_df)

    print(f"[*] Saving duplicate clusters to {clusters_path}...")
    clusters_df.write.mode("overwrite").parquet(clusters_path)

    print(f"[*] Saving portal stats to {stats_path}...")
    stats_df.write.mode("overwrite").parquet(stats_path)

    print(f"[SUCCESS] Deduplication Pipeline finished in {time.time()-t0:.2f}s total.")
    return duplicate_pairs, clusters_df, stats_df


if __name__ == "__main__":
    pairs, clusters, stats = run_deduplication_pipeline()
    print("\n=== TOP CROSS-PORTAL DUPLICATE PAIRS ===")
    pairs.select("source_a", "company_a", "title_a", "source_b", "title_b", "similarity_score").show(10, truncate=False)

    print("\n=== SAMPLE DUPLICATE CLUSTERS ===")
    clusters.select("cluster_id", "source", "company_name", "title", "is_canonical").show(15, truncate=False)

    print("\n=== PORTAL DUPLICATION METRICS ===")
    stats.show(20, truncate=False)

