# Apache Spark Big Data Job Market Intelligence & Deduplication Platform

An enterprise-grade, end-to-end Big Data analytics lakehouse, GPU-accelerated NLP matrix engine, cross-portal fuzzy deduplication system, and interactive Django web dashboard built on **428,962 job postings** (~1.75 GB PostgreSQL relational database).

```
                      ┌──────────────────────────────────────────────┐
                      │    PostgreSQL 17 Relational Database         │
                      │       493,030 Raw Job Postings               │
                      └──────────────────────┬───────────────────────┘
                                             │ Parallel JDBC Ingestion (16 Partitions)
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │      Apache Spark 4.2.0 Catalyst Engine      │
                      │  Regex HTML / Markdown Cleaning & Norm       │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │       Silver Parquet Storage Layer           │
                      │    428,962 Clean Records (0.6s Scan)         │
                      └───────┬──────────────┬──────────────┬────────┘
                              │              │              │
       ┌──────────────────────┘              │              └──────────────────────┐
       ▼                                     ▼                                     ▼
┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
│ GPU Skill Matrix (PyTorch)│ │   MinHash LSH Deduplication│ │ Spark SQL Window Analytics│
│  X^T X in 172ms (RTX 4050)│ │  Character 3-Grams + LSH  │ │  Company Hiring Velocity  │
│  114x114 Co-occurrences   │ │  Connected Components     │ │  Remote Work Dynamics     │
└─────────────┬─────────────┘ └─────────────┬─────────────┘ └─────────────┬─────────────┘
              │                             │                             │
              └──────────────────────┬──────┴─────────────────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────────────────────┐
                      │     Spark MLlib Salary Regressor Model       │
                      │   RandomForestRegressor on 122,707 Records   │
                      └──────────────────────┬───────────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────────────────────┐
                      │    Django 5.2 Interactive Web Dashboard      │
                      │  Tailwind CSS • Chart.js • REST APIs         │
                      └──────────────────────────────────────────────┘
```

---

## Key Engineering Innovations

### 1. Partitioned Parallel JDBC Ingestion
* Instead of a single JDBC connection reading 493K rows (which causes database timeouts and Spark driver heap exhaustion), the ingestion pipeline queries PostgreSQL epoch timestamps sliced across **16 parallel executors** using `partitionColumn="partition_epoch"`, `lowerBound`, `upperBound`, and `numPartitions=16`.
* Read throughput: **~493,000 records ingested in under 12 seconds**.

### 2. Native Catalyst JVM Cleaning & Normalization
* Strips both HTML entities and Markdown syntax (headers, links, code blocks, bold/italics, quotes) entirely within the JVM Catalyst engine using native `regexp_replace` expressions, avoiding Python UDF serialization bottlenecks (`Py4J` overhead).
* Preserves critical technical symbols such as `C++`, `C#`, `.NET Core`, and `Node.js`.
* Standardizes company legal entities (`Inc`, `LLC`, `Ltd`, `Corp`) and job titles into normalized matching tokens.
* Saved to `data/silver/jobs.parquet` partitioned by `scraped_year_month` (full scan completes in **0.61 seconds**).

### 3. GPU-Accelerated Skill Matrix & Co-occurrence (172 ms on RTX 4050)
* Traditional combinatorial pair generation produces $\binom{114}{2} \times 428,962 \approx \mathbf{2.76 \text{ billion pairs}}$, requiring heavy distributed Spark shuffling.
* Formulated as a dense linear algebra problem:
  $$\mathbf{C} = \mathbf{X}^T \mathbf{X}$$
  where $\mathbf{X}$ is a $428,962 \times 114$ float32 binary indicator matrix transferred over PCIe 4.0 to the user's **NVIDIA GeForce RTX 4050 Laptop GPU**.
* Executed on **Tensor Cores in 172 milliseconds** (less than 1 microsecond of raw compute), producing full frequency rankings and an exact $114 \times 114$ technology co-occurrence matrix.

### 4. Cross-Portal Fuzzy Deduplication at Scale (MinHash LSH & Disjoint Set)
* Pairwise comparison across 428K jobs requires $\frac{N(N-1)}{2} \approx \mathbf{92 \text{ billion comparisons}}$ ($O(N^2)$).
* MinHash LSH extracts character 3-grams (`NGram(n=3)`), vectorizes them via `CountVectorizer`, and builds 3 Locality-Sensitive Hash tables such that collision probability equals Jaccard similarity:
  $$\Pr(h(A) = h(B)) = \frac{|A \cap B|}{|A \cup B|}$$
* Reduces candidate comparison space to $O(N)$.
* Employs **Disjoint Set Union (Connected Components)** with path compression ($\mathcal{O}(M \cdot \alpha(N))$) to cluster transitive duplicates (if $A \approx B$ and $B \approx C$, group into $\{A, B, C\}$).
* **Findings**: Discovered **14,861 duplicate pairs** and **2,094 distinct duplicate clusters** representing **8,667 syndicated job postings**. Direct ATS portals like Oracle Cloud (**40.96%**), SmartRecruiters (**14.61%**), and Greenhouse (**10.72%**) had the highest syndication rates to LinkedIn and Indeed.

### 5. Spark SQL Window Functions for Market Velocity
* Computes week-over-week hiring acceleration per company without self-joins:
  ```sql
  LAG(COUNT(id), 1) OVER (PARTITION BY company_normalized ORDER BY posting_week)
  DENSE_RANK() OVER (PARTITION BY posting_week ORDER BY weekly_jobs DESC)
  ```
* Detects corporate hiring surges (e.g., Google **+121.4%** weekly surge, Amazon **+4,700%** hiring blitz) and hiring freezes/cool-offs (Deloitte **-77.0%**).
* Tracks remote work temporal shifts from 43.4% in July 2026 to a post-summer steady state of **17.1% - 17.9% in September/October**.

### 6. Experimental Salary Prediction Model (Spark MLlib)
* Prepares 122,707 verified annual USD salary records ($20k – $400k).
* Assembles high-dimensional feature vectors:
  - Title text tokens via `Tokenizer` + `StopWordsRemover` + `HashingTF(128)`
  - Seniority tier (Intern, Junior, Mid, Senior, Lead, Staff/Principal, Executive) via `OneHotEncoder`
  - Geographic region / state code via `OneHotEncoder`
  - Employment type & remote status binary flags
* Trains a `RandomForestRegressor` (30 trees, depth 8) in **7.38 seconds** ($R^2 = 0.3837$, $\text{RMSE} = \$44,579$, $\text{MAE} = \$33,675$).
* Exports precomputed inference priors for instant (< 1ms) interactive web calculation.

### 7. Interactive Django 5.x Web Dashboard
* Built with modern Tailwind CSS and Chart.js, querying the Gold Parquet lakehouse directly:
  - **Overview (`/`)**: High-level KPIs, portal volume doughnut chart, remote trend line chart, and portal scorecard.
  - **Skills Intelligence (`/skills/`)**: Top 30 tech skills, category filter pills, and dynamic GPU co-occurrence explorer with AJAX updates.
  - **Duplicate Clusters (`/duplicates/`)**: Searchable, filterable table of cross-portal duplicate clusters with modal inspector.
  - **Hiring Velocity (`/velocity/`)**: Accelerating vs cooling companies and weekly global leaderboard.
  - **Salary Estimator (`/salary/`)**: Real-time compensation calculator with model architecture breakdown.

---

## Project Structure

```
spark-project/
├── config/
│   ├── spark_config.py            # Cluster, JVM heap, shuffle, and path configs
│   └── skills_taxonomy.json       # Curated 114 tech taxonomy across 5 domains
├── pipeline/
│   ├── session.py                 # Tuned SparkSession factory (PySpark 4.2.0)
│   ├── ingest.py                  # Parallel partitioned JDBC PostgreSQL extraction
│   ├── clean.py                   # JVM Catalyst regex cleaner & entity normalizer
│   ├── export_silver.py           # Ingest -> Clean -> Silver Parquet lakehouse export
│   ├── nlp_skills.py              # Parallel broadcast dictionary skill matcher
│   ├── gpu_skills_matrix.py       # PyTorch CUDA Tensor Core GEMM matrix co-occurrence
│   ├── deduplication.py           # MinHash LSH and Disjoint Set graph clustering
│   ├── analytics.py               # Spark SQL window functions for market velocity
│   └── salary_model.py            # Spark MLlib RandomForestRegressor pipeline
├── dashboard/                     # Django 5.2 Web Application
│   ├── manage.py
│   ├── dashboard_project/         # Django settings, ASGI/WSGI, routing
│   ├── analytics_app/             # Views, URLs, and low-latency lakehouse services
│   └── templates/                 # Modern Tailwind CSS + Chart.js templates
├── tests/
│   ├── test_cleaning.py           # Regex cleaning and symbol preservation unit tests
│   ├── test_deduplication.py      # DisjointSet union-find clustering unit tests
│   ├── test_salary_inference.py   # Salary estimation logic unit tests
│   └── test_dashboard_views.py    # Django endpoint status 200 integration tests
├── data/
│   ├── silver/jobs.parquet        # Cleaned Silver lakehouse (428,962 rows)
│   ├── gold/                      # 11 analytical Gold Parquet tables
│   └── models/                    # Trained MLlib models & inference JSON
├── run_pipeline.py                # Unified CLI orchestrator
└── requirements.txt
```

---

## Quickstart Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Dhruv1249/spark-project.git
cd spark-project

# Install dependencies using uv or venv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Running Automated Tests
```bash
pytest tests/ -v
```

### 3. Running the Pipeline via CLI Orchestrator
```bash
# Run complete end-to-end pipeline (ingest -> clean -> nlp -> dedupe -> analytics -> salary)
python run_pipeline.py --stage all

# Or run individual pipeline stages:
python run_pipeline.py --stage nlp         # GPU Skill Matrix
python run_pipeline.py --stage dedupe      # MinHash LSH Deduplication
python run_pipeline.py --stage analytics   # Spark SQL Window Functions
python run_pipeline.py --stage salary      # MLlib Salary Model
```

### 4. Launching the Interactive Web Dashboard
```bash
# Launch Django web server
python run_pipeline.py --stage serve

# Access the dashboard in your browser:
# http://127.0.0.1:8000/
```
