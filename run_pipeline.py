#!/usr/bin/env python3
"""
End-to-End CLI Pipeline Orchestrator for Apache Spark Job Market Intelligence.
Allows running all or individual pipeline stages:
1. Ingest: Parallel partitioned JDBC extraction from PostgreSQL 17 into Spark
2. Clean: Catalyst JVM regex HTML & Markdown stripping, entity preservation
3. NLP: Vectorized Technology Skill Extraction & GPU tensor co-occurrence matrix (RTX 4050)
4. Dedupe: Cross-source fuzzy deduplication via MinHash LSH and Disjoint Set clustering
5. Analytics: Spark SQL Window Functions for company hiring velocity & market trends
6. Salary: Experimental Spark MLlib RandomForest regression model
7. Serve: Launch the Django interactive analytical dashboard
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_banner(stage_name: str):
    width = 75
    print("\n" + "=" * width)
    print(f"🚀 [STAGE] {stage_name.upper()}")
    print("=" * width + "\n")


def run_stage_ingest_and_clean():
    print_banner("1. Ingestion & Catalyst Data Cleaning")
    from pipeline.export_silver import export_clean_silver_lakehouse
    t0 = time.time()
    export_clean_silver_lakehouse()
    print(f"\n[✓] Ingest & Cleaning completed in {time.time()-t0:.2f}s")


def run_stage_nlp_gpu():
    print_banner("2. GPU-Accelerated Skill Matrix & Co-occurrence")
    from pipeline.gpu_skills_matrix import run_gpu_skills_pipeline
    t0 = time.time()
    run_gpu_skills_pipeline()
    print(f"\n[✓] GPU Skills Pipeline completed in {time.time()-t0:.2f}s")


def run_stage_deduplication(sample_fraction: float | None = None):
    print_banner("3. MinHash LSH Cross-Portal Deduplication")
    from pipeline.deduplication import run_deduplication_pipeline
    t0 = time.time()
    run_deduplication_pipeline(sample_fraction=sample_fraction)
    print(f"\n[✓] Deduplication Pipeline completed in {time.time()-t0:.2f}s")


def run_stage_analytics():
    print_banner("4. Spark SQL Window Analytics (Velocity & Trends)")
    from pipeline.analytics import run_market_analytics_pipeline
    t0 = time.time()
    run_market_analytics_pipeline()
    print(f"\n[✓] Market Analytics Pipeline completed in {time.time()-t0:.2f}s")


def run_stage_salary_model():
    print_banner("5. Spark MLlib Salary Prediction Model")
    from pipeline.salary_model import run_salary_training_pipeline
    t0 = time.time()
    run_salary_training_pipeline()
    print(f"\n[✓] Salary Model Pipeline completed in {time.time()-t0:.2f}s")


def run_stage_serve(host: str = "127.0.0.1", port: int = 8000):
    print_banner("6. Django Interactive Dashboard Server")
    manage_py = PROJECT_ROOT / "dashboard" / "manage.py"
    python_bin = sys.executable

    print(f"[*] Starting Django server at http://{host}:{port}/ ...")
    print("[*] Press Ctrl+C to stop server.\n")
    try:
        subprocess.run([python_bin, str(manage_py), "runserver", f"{host}:{port}"])
    except KeyboardInterrupt:
        print("\n[!] Dashboard server stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Spark Job Market Intelligence & Deduplication Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "ingest", "nlp", "dedupe", "analytics", "salary", "serve"],
        default="all",
        help="Pipeline stage to execute (default: all)"
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Optional sampling fraction (e.g. 0.1 for 10%%) for quick deduplication runs"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for Django server")
    parser.add_argument("--port", type=int, default=8000, help="Port for Django server")

    args = parser.parse_args()
    total_start = time.time()

    if args.stage == "all":
        print("\n" + "#" * 75)
        print("⚡ EXECUTING COMPLETE END-TO-END BIG DATA PIPELINE (428K+ JOBS)")
        print("#" * 75)

        run_stage_ingest_and_clean()
        run_stage_nlp_gpu()
        run_stage_deduplication(sample_fraction=args.sample)
        run_stage_analytics()
        run_stage_salary_model()

        print("\n" + "#" * 75)
        print(f"🎉 ALL PIPELINE STAGES COMPLETED IN {time.time()-total_start:.2f}s TOTAL!")
        print("#" * 75)

    elif args.stage == "ingest":
        run_stage_ingest_and_clean()
    elif args.stage == "nlp":
        run_stage_nlp_gpu()
    elif args.stage == "dedupe":
        run_stage_deduplication(sample_fraction=args.sample)
    elif args.stage == "analytics":
        run_stage_analytics()
    elif args.stage == "salary":
        run_stage_salary_model()
    elif args.stage == "serve":
        run_stage_serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
