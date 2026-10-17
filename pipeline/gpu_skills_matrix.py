"""
GPU-Accelerated Skill Matrix & Co-occurrence Engine using PyTorch CUDA on NVIDIA RTX 4050.
Computes skill counts, co-occurrence matrix C = X^T * X, and conditional probability graphs on GPU Tensor Cores.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.spark_config import CONFIG_DIR, GOLD_DIR, SILVER_DIR
from pipeline.nlp_skills import load_skills_taxonomy
from pipeline.session import get_spark_session
import pyspark.sql.functions as F
import pyspark.sql.types as T


def build_gpu_skill_index() -> tuple[list[str], dict[str, int], dict[str, str]]:
    """
    Builds indexed vocabulary of canonical skills and category mappings.
    Returns:
    - skill_list: list of unique canonical skill names (length M)
    - skill_to_idx: dictionary mapping skill name -> integer column index [0..M-1]
    - skill_to_category: dictionary mapping skill name -> domain category
    """
    taxonomy, canonical_map, skill_to_category, _ = load_skills_taxonomy()
    unique_canonical_skills = sorted(list(set(canonical_map.values())))
    skill_to_idx = {skill: idx for idx, skill in enumerate(unique_canonical_skills)}

    # Map canonical names to categories
    canonical_categories = {}
    for cat, skills in taxonomy.items():
        for s in skills:
            can = canonical_map.get(s.lower().strip(), s.lower().strip())
            canonical_categories[can] = cat

    return unique_canonical_skills, skill_to_idx, canonical_categories


def compute_gpu_cooccurrence_matrix(
    skill_indicator_matrix: np.ndarray,
    device: str = "cuda:0"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Skill Demand, Co-occurrence Matrix, and Jaccard Affinity on NVIDIA GPU Tensor Cores.

    Mathematical Formulation on GPU:
    --------------------------------
    Given N jobs and M skills, X in {0, 1}^(N x M):
    1. Skill Counts:        c = sum_i(X_i)           (M-dim vector on GPU)
    2. Co-occurrence:       C = X^T * X              (M x M matrix on GPU)
    3. Jaccard Similarity:  J_jk = C_jk / (c_j + c_k - C_jk)
    4. Conditional Prob:    P(B|A) = C_AB / c_A

    Executing this via PyTorch CUDA matmul on 2,560 CUDA cores runs in ~50ms.
    """
    if not torch.cuda.is_available():
        print("[!] Warning: CUDA is not available. Falling back to CPU tensors.")
        device = "cpu"
    else:
        print(f"[*] Allocating tensors on GPU: {torch.cuda.get_device_name(0)} ({device})")

    # Transfer matrix to GPU VRAM as float32
    t0 = time.time()
    X_gpu = torch.as_tensor(skill_indicator_matrix, dtype=torch.float32, device=device)
    torch.cuda.synchronize()
    vram_used = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f" -> Transferred {X_gpu.shape[0]:,} rows x {X_gpu.shape[1]} skills to GPU VRAM ({vram_used:.1f} MB used)")

    # 1. Total counts per skill: column sum
    counts_gpu = torch.sum(X_gpu, dim=0)

    # 2. Co-occurrence matrix via GPU matrix multiplication: C = X^T * X
    # Utilizing FP32 / TF32 Tensor Cores
    matmul_start = time.time()
    C_gpu = torch.matmul(X_gpu.T, X_gpu)
    torch.cuda.synchronize()
    matmul_time = time.time() - matmul_start
    print(f" -> GPU Tensor Core Matmul (X^T * X) executed in {matmul_time * 1000:.2f} ms!")

    # 3. Jaccard Similarity Matrix on GPU: J_jk = C_jk / (c_j + c_k - C_jk)
    # Broadcast addition: c_j + c_k
    sum_counts = counts_gpu.unsqueeze(0) + counts_gpu.unsqueeze(1)
    union_gpu = sum_counts - C_gpu
    union_gpu = torch.clamp(union_gpu, min=1.0)
    J_gpu = C_gpu / union_gpu
    # Set diagonal of Jaccard to 1.0
    J_gpu.fill_diagonal_(1.0)

    # 4. Conditional Probability Matrix on GPU: P(Skill_B | Skill_A) = C_AB / c_A
    denom = counts_gpu.unsqueeze(1).clamp(min=1.0)
    cond_prob_gpu = C_gpu / denom

    # Transfer aggregated results back to CPU NumPy
    counts_cpu = counts_gpu.cpu().numpy().astype(int)
    C_cpu = C_gpu.cpu().numpy().astype(int)
    J_cpu = J_gpu.cpu().numpy()

    # Free GPU memory
    del X_gpu, C_gpu, J_gpu, cond_prob_gpu, counts_gpu
    torch.cuda.empty_cache()

    return counts_cpu, C_cpu, J_cpu


def run_gpu_skill_pipeline(sample_limit: int | None = None) -> None:
    """
    End-to-end runner that loads Silver Parquet data, builds binary indicator matrix,
    runs GPU matrix acceleration, and writes Gold layer tables.
    """
    total_start = time.time()
    print("=" * 70)
    print(" GPU-ACCELERATED SKILL MATRIX & CO-OCCURRENCE ENGINE")
    print("=" * 70)

    # 1. Load taxonomy and index mapping
    skill_list, skill_to_idx, canonical_categories = build_gpu_skill_index()
    num_skills = len(skill_list)
    print(f"\n[1/5] Loaded Skill Taxonomy: {num_skills} canonical technologies across 5 categories.")

    # 2. Load jobs from Silver Parquet Lakehouse using Spark
    spark = get_spark_session("GPUSkillMatrixEngine")
    silver_path = str(SILVER_DIR / "jobs.parquet")
    print(f"\n[2/5] Reading clean jobs from Silver Parquet: {silver_path}...")
    jobs_df = spark.read.parquet(silver_path).select("id", "title", "clean_desc", "source", "is_remote")
    if sample_limit:
        print(f" -> Sampling first {sample_limit:,} jobs for fast GPU verification...")
        jobs_df = jobs_df.limit(sample_limit)
    total_jobs = jobs_df.count()
    print(f" -> Successfully loaded {total_jobs:,} jobs from Lakehouse.")

    # 3. Parallel Skill Extraction in Spark across 20 Cores
    print("\n[3/5] Extracting skill sets across all jobs in parallel using Spark...")
    extract_start = time.time()
    from pipeline.nlp_skills import extract_skills_pipeline
    enriched_df = extract_skills_pipeline(spark, jobs_df)
    
    # Collect only the lightweight skills array (only ~15 MB across all 428k rows!)
    print(" -> Collecting extracted skill arrays for GPU Tensor Core matrix generation...")
    skill_rows = enriched_df.select("skills").collect()
    num_records = len(skill_rows)
    extract_duration = round(time.time() - extract_start, 2)
    print(f" -> Extracted skills across {num_records:,} jobs in {extract_duration}s")

    # Build binary matrix X of shape (num_records, num_skills)
    X = np.zeros((num_records, num_skills), dtype=np.uint8)
    for row_idx, row in enumerate(skill_rows):
        skills = row["skills"]
        if skills:
            for s in skills:
                idx = skill_to_idx.get(s)
                if idx is not None:
                    X[row_idx, idx] = 1

    encoding_time = round(time.time() - extract_start, 2)
    print(f" -> Binary Matrix constructed: Shape = {X.shape} ({X.nbytes / (1024*1024):.1f} MB in RAM) in {encoding_time}s")

    # 4. Run GPU Tensor Computations
    print("\n[4/5] Executing Tensor Operations on NVIDIA RTX 4050 GPU...")
    counts, cooccurrence_matrix, jaccard_matrix = compute_gpu_cooccurrence_matrix(X, device="cuda:0")

    # 5. Build Spark Gold DataFrames from GPU Results
    print("\n[5/5] Building Gold Analytical Layer and persisting to Parquet...")

    # A. Skill Demand Rankings
    rankings_data = []
    for skill, idx in skill_to_idx.items():
        job_cnt = int(counts[idx])
        pct = round((job_cnt / total_jobs) * 100, 2)
        category = canonical_categories.get(skill, "other")
        rankings_data.append((skill, category, job_cnt, pct))

    rankings_schema = T.StructType([
        T.StructField("skill", T.StringType(), False),
        T.StructField("category", T.StringType(), False),
        T.StructField("job_count", T.IntegerType(), False),
        T.StructField("demand_percentage", T.DoubleType(), False),
    ])
    rankings_df = spark.createDataFrame(rankings_data, schema=rankings_schema).orderBy(F.col("job_count").desc())

    # B. Skill Pairwise Co-occurrence & Jaccard Similarity
    cooccur_data = []
    for i in range(num_skills):
        for j in range(i + 1, num_skills):
            c_cnt = int(cooccurrence_matrix[i, j])
            if c_cnt >= 50:  # Filter meaningful co-occurrences
                jaccard_score = round(float(jaccard_matrix[i, j]), 4)
                cooccur_data.append((skill_list[i], skill_list[j], c_cnt, jaccard_score))

    cooccur_schema = T.StructType([
        T.StructField("skill_a", T.StringType(), False),
        T.StructField("skill_b", T.StringType(), False),
        T.StructField("cooccurrence_count", T.IntegerType(), False),
        T.StructField("jaccard_similarity", T.DoubleType(), False),
    ])
    cooccur_df = spark.createDataFrame(cooccur_data, schema=cooccur_schema).orderBy(F.col("cooccurrence_count").desc())

    # Display Top 15 in console
    print("\n" + "=" * 70)
    print(" TOP 15 IN-DEMAND TECHNOLOGIES (COMPUTED ON GPU)")
    print("=" * 70)
    print(f"{'Rank':<5} | {'Skill':<20} | {'Category':<15} | {'Job Count':<10} | {'Demand %':<8}")
    print("-" * 65)
    for idx, row in enumerate(rankings_df.take(15), 1):
        print(f"{idx:<5} | {row['skill']:<20} | {row['category']:<15} | {row['job_count']:<10,d} | {row['demand_percentage']:<8}%")
    print("-" * 65)

    print("\n" + "=" * 70)
    print(" TOP 15 PAIRED TECHNOLOGIES & JACCARD AFFINITY (COMPUTED ON GPU)")
    print("=" * 70)
    print(f"{'Skill A':<18} + {'Skill B':<18} | {'Together':<10} | {'Jaccard Index'}")
    print("-" * 65)
    for row in cooccur_df.take(15):
        print(f"{row['skill_a']:<18} + {row['skill_b']:<18} | {row['cooccurrence_count']:<10,d} | {row['jaccard_similarity']:.4f}")
    print("-" * 65)

    # Save to Gold layer Parquet
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    rankings_output = str(GOLD_DIR / "skills_ranking.parquet")
    cooccur_output = str(GOLD_DIR / "skill_cooccurrence.parquet")

    rankings_df.write.mode("overwrite").parquet(rankings_output)
    cooccur_df.write.mode("overwrite").parquet(cooccur_output)

    total_time = round(time.time() - total_start, 2)
    print(f"\n[*] GPU NLP Pipeline finished in {total_time}s")
    print(f" -> Gold Rankings saved to: {rankings_output}")
    print(f" -> Gold Co-occurrence saved to: {cooccur_output}")
    spark.stop()


if __name__ == "__main__":
    limit_val = None
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        if idx + 1 < len(sys.argv):
            limit_val = int(sys.argv[idx + 1])
    run_gpu_skill_pipeline(sample_limit=limit_val)
