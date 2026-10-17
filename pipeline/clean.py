"""
Data cleaning and feature normalization module leveraging Spark Catalyst native expressions.
"""

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T


def clean_and_normalize_jobs(raw_dataframe: DataFrame) -> DataFrame:
    """
    Transforms raw scraped job records into a clean, normalized analytical schema.

    Spark Core Concept - Catalyst Optimization vs Python UDF:
    ---------------------------------------------------------
    Using Python UDFs (e.g. re.sub in Python) forces Spark to serialize each row from the
    JVM to a Python worker process via Py4J pipes, causing massive CPU context switching.
    Instead, we use native Spark functions (F.regexp_replace, F.lower, F.trim), which
    are compiled directly into JVM bytecode by the Tungsten execution engine.
    """
    # 1. Clean Markdown and HTML formatting while preserving language tokens (C++, C#, .NET, Node.js)
    # Strip markdown images: ![alt](url) -> space
    desc_cleaned = F.regexp_replace(F.col("raw_desc"), "!\\[[^\\]]*\\]\\([^)]+\\)", " ")
    # Replace markdown links: [Anchor Text](URL) -> Anchor Text
    desc_cleaned = F.regexp_replace(desc_cleaned, "\\[([^\\]]+)\\]\\([^)]+\\)", "$1")
    # Strip raw URLs (http:// or https://)
    desc_cleaned = F.regexp_replace(desc_cleaned, "https?://\\S+", " ")
    # Strip HTML tags: <div ...>, <p>, </div>
    desc_cleaned = F.regexp_replace(desc_cleaned, "<[^>]+>", " ")
    # Decode / strip HTML entities: &nbsp;, &amp;
    desc_cleaned = F.regexp_replace(desc_cleaned, "&[a-zA-Z0-9#]+;", " ")
    # Strip Markdown headers: ### Title -> Title
    desc_cleaned = F.regexp_replace(desc_cleaned, "(^|\\s)#{1,6}\\s+", "$1")
    # Strip Markdown bold and italic markers: **bold**, *italic*, __bold__, _italic_
    desc_cleaned = F.regexp_replace(desc_cleaned, "(\\*{1,2}|_{1,2})", " ")
    # Strip Markdown code block ticks: ``` or `
    desc_cleaned = F.regexp_replace(desc_cleaned, "```|`", " ")
    # Strip Markdown blockquote arrows: > quote
    desc_cleaned = F.regexp_replace(desc_cleaned, "(^|\\s)>+\\s*", "$1")
    # Strip Markdown horizontal rules: --- or ***
    desc_cleaned = F.regexp_replace(desc_cleaned, "[-*]{3,}", " ")
    # Collapse multiple whitespace and trim
    clean_desc = F.trim(F.regexp_replace(desc_cleaned, "\\s+", " "))

    # 2. Normalize company names for fuzzy clustering
    company_cleaned = F.lower(F.trim(F.col("company_name")))
    company_normalized = F.trim(
        F.regexp_replace(company_cleaned, "\\b(inc|llc|ltd|corp|corporation|gmbh|co|services|technologies)\\b[.]?", "")
    )
    company_normalized = F.regexp_replace(company_normalized, "[^a-zA-Z0-9\\s]", "")

    # 3. Normalize job title for fuzzy clustering
    title_cleaned = F.lower(F.trim(F.col("title")))
    title_normalized = F.trim(F.regexp_replace(title_cleaned, "[^a-zA-Z0-9\\s]", " "))
    title_normalized = F.regexp_replace(title_normalized, "\\s+", " ")

    # 4. Standardize employment types
    standard_employment_type = (
        F.when(F.col("employment_type").isin("full_time", "Full-time") | F.col("job_type").isin("Full-time", "full_time"), "full_time")
        .when(F.col("employment_type").isin("intern", "intern_ppo", "Internship") | F.col("job_type").isin("Internship", "intern"), "internship")
        .when(F.col("employment_type").isin("contract", "Contract") | F.col("job_type").isin("Contract", "contract"), "contract")
        .when(F.col("employment_type").isin("part_time", "Part-time"), "part_time")
        .otherwise("unspecified")
    )

    # 5. Salary metrics and flags
    has_salary_flag = F.col("salary_min").isNotNull() | F.col("salary_max").isNotNull()
    salary_midpoint = (
        F.when(F.col("salary_min").isNotNull() & F.col("salary_max").isNotNull(), (F.col("salary_min") + F.col("salary_max")) / 2.0)
        .otherwise(F.coalesce(F.col("salary_min"), F.col("salary_max")).cast(T.DoubleType()))
    )

    cleaned_dataframe = (
        raw_dataframe
        .withColumn("clean_desc", clean_desc)
        .withColumn("desc_length", F.length(F.col("clean_desc")))
        .withColumn("company_normalized", F.when(F.length(company_normalized) > 0, company_normalized).otherwise(company_cleaned))
        .withColumn("title_normalized", title_normalized)
        .withColumn("standard_employment_type", standard_employment_type)
        .withColumn("has_salary", has_salary_flag)
        .withColumn("salary_midpoint", salary_midpoint)
        .withColumn("scraped_year_month", F.date_format(F.col("scraped_at"), "yyyy-MM"))
        .filter(F.col("clean_desc").isNotNull() & (F.length(F.col("clean_desc")) >= 20))
    )

    return cleaned_dataframe
