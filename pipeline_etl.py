"""
SCOR Interview Asset: End-to-End Automated P&C Reinsurance ETL Pipeline Engine
Author: Monsef Djamel Eddine BOUDALIA
Description: Production script for automated asset ingestion, caching, and
             relational database cleaning for P&C reinsurance claim portfolios.
Usage: python pipeline_etl.py
"""

import os
import time
import sqlite3
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from ydata_profiling import ProfileReport


# ======================================================================
# STEP 1: PORTFOLIO INGESTION WITH EXPLICIT LOCAL CACHE GATE
# ======================================================================
def ingest_portfolio_with_cache(dataset_id=42571, filename="allstate_raw_claims.csv"):
    """
    Streams the Allstate claims portfolio from the OpenML API on first run,
    then serves subsequent runs from a local CSV cache to eliminate redundant
    API calls and enable fully offline execution.

    Args:
        dataset_id (int): OpenML dataset identifier. Default: 42571 (Allstate).
        filename (str): Local cache path for the CSV snapshot.

    Returns:
        pd.DataFrame: Raw claims portfolio with 188,318 rows and 131 features.

    Raises:
        ConnectionError: If the OpenML API is unreachable on a cache miss.
    """
    # --- Cache Hit: serve from disk, skip the network entirely ---
    if os.path.exists(filename):
        print(f"📦 Cache hit. Loading portfolio from disk: '{filename}'...")
        df = pd.read_csv(filename, index_col=0)
        # Note: 'index_col=0' restores the original row identifier as the DataFrame index.
        # 'id' is a sequential row number assigned during CSV creation, not a business key.
        print(f"   ✓ {len(df):,} rows loaded from local cache.")
        return df

    # --- Cache Miss: stream from OpenML and persist locally ---
    print(f"🌐 Cache miss. Streaming dataset ID {dataset_id} from OpenML API...")
    try:
        start_time = time.time()
        openml_payload = fetch_openml(data_id=dataset_id, as_frame=True, parser='pandas')
        df = openml_payload.frame
        df.to_csv(filename)
        elapsed = time.time() - start_time
        print(f"💾 API stream complete. {len(df):,} rows cached to '{filename}' in {elapsed:.2f}s.")
        return df
    except Exception as e:
        raise ConnectionError(f"❌ OpenML API unreachable. Cannot proceed without data. Details: {e}")


# ======================================================================
# STEP 2: AUTOMATED DATA QUALITY GATE AUDIT
# ======================================================================
def generate_quality_gate_audit(df, output_path="production_portfolio_report.html"):
    """
    Generates a standalone HTML data profiling report on a 10,000-row sample
    to surface skewness, missing values, and feature correlations without
    loading the full 188K-row portfolio into the profiler.

    Args:
        df (pd.DataFrame): The full raw claims DataFrame.
        output_path (str): File path for the output HTML audit report.
    """
    print("🔍 Generating data quality audit report...")

    # Sample for performance — profiling the full 188K rows is unnecessarily slow.
    # drop=True prevents the old index from leaking in as a spurious extra column.
    sample_df = df.sample(n=10000, random_state=42).copy().reset_index(drop=True)

    profile = ProfileReport(sample_df, title="SCOR Portfolio Integrity Audit — Allstate Claims")
    profile.to_file(output_path)
    print(f"   ✓ Audit report saved to: '{output_path}'")


# ======================================================================
# STEP 3: BULK STAGING — RAW DATAFRAME → SQLITE
# ======================================================================
def stage_raw_portfolio(df, db_name="scor_portfolio_management.db"):
    """
    Bulk-loads the raw claims DataFrame into the SQLite staging table
    'clean_allstate_claims'. This table acts as the immutable source layer
    before any transformation is applied.

    Args:
        df (pd.DataFrame): Raw claims portfolio.
        db_name (str): SQLite database file path.
    """
    print(f"💾 Staging raw portfolio into '{db_name}' → table 'clean_allstate_claims'...")
    conn = sqlite3.connect(db_name)

    # if_exists="replace" ensures a clean slate on every pipeline run,
    # preventing stale data from a previous execution contaminating results.
    df.to_sql("clean_allstate_claims", conn, if_exists="replace", index=True, index_label="id")

    conn.close()
    print(f"   ✓ {len(df):,} rows staged successfully.")


# ======================================================================
# STEP 4: SQL TRANSFORMATION ENGINE — CAP, NORMALIZE, INDEX
# ======================================================================
def execute_relational_cleaning_pipeline(db_name="scor_portfolio_management.db"):
    """
    Executes three sequential transformation operations on the staged claims:

      1. Catastrophe Cap   — clips losses above the 99.5th percentile to
                             separate the volatile catastrophe layer from the
                             stable working layer (mirrors a reinsurance treaty).
      2. Log Normalization — applies log1p to the capped loss column to convert
                             the heavy-tailed distribution into a model-friendly
                             bell curve.
      3. Index Injection   — creates a read-path index on log_loss to prevent
                             query latency during downstream model runs.

    Args:
        db_name (str): SQLite database file path.
    """
    print(f"🧹 Executing relational cleaning pipeline on '{db_name}'...")
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # ------------------------------------------------------------------
    # 4A: Compute the 99.5th percentile catastrophe cap inside the DB
    # ------------------------------------------------------------------
    # Calculated directly in SQL to avoid pulling the full loss column
    # into Python memory. OFFSET gives the row at the 99.5th percentile
    # when the table is sorted ascending.
    # ------------------------------------------------------------------
    cursor.execute("""
        SELECT loss
        FROM clean_allstate_claims
        ORDER BY loss ASC
        LIMIT 1 OFFSET (
            SELECT CAST(COUNT(*) * 0.995 AS INT)
            FROM clean_allstate_claims
        );
    """)
    cat_threshold = cursor.fetchone()[0]
    print(f"   🎯 Catastrophe cap (99.5th percentile): €{cat_threshold:,.2f}")

    # ------------------------------------------------------------------
    # 4B: Build the cleaned analytics table with outliers capped
    # ------------------------------------------------------------------
    # PARAMETERIZED QUERY: cat_threshold is passed as a bound parameter (?)
    # rather than interpolated via an f-string. This prevents SQL injection
    # and follows production-safe database engineering practices.
    # ------------------------------------------------------------------
    cursor.execute("DROP TABLE IF EXISTS analytics_portfolio_ready;")
    cursor.execute("""
        CREATE TABLE analytics_portfolio_ready AS
        SELECT
            id,
            cat1, cat2, cat3, cat4, cat5,
            cont1, cont2, cont3, cont4, cont5,
            loss                                    AS raw_loss,
            CASE
                WHEN loss > ? THEN ?
                ELSE loss
            END                                     AS cleaned_loss
        FROM clean_allstate_claims;
    """, (cat_threshold, cat_threshold))  # ← values bound safely, never interpolated
    conn.commit()

    # ------------------------------------------------------------------
    # 4C: Pull back the table and apply log1p normalization in numpy
    # ------------------------------------------------------------------
    # log1p(x) = log(x + 1) — the +1 guard prevents log(0) errors on
    # any zero-value claims that survive the cleaning step.
    # ------------------------------------------------------------------
    df_analytics = pd.read_sql_query("SELECT * FROM analytics_portfolio_ready", conn)
    df_analytics['log_loss'] = np.log1p(df_analytics['cleaned_loss'])
    df_analytics.to_sql("analytics_portfolio_ready", conn, if_exists="replace", index=False)

    # ------------------------------------------------------------------
    # 4D: Inject a read-path index on log_loss
    # ------------------------------------------------------------------
    # Ensures downstream model queries on log_loss don't trigger full
    # table scans, which would bottleneck on a 188K-row portfolio.
    # ------------------------------------------------------------------
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_log_loss
        ON analytics_portfolio_ready (log_loss);
    """)

    # ------------------------------------------------------------------
    # 4E: Final validation — verify row count and summary statistics
    # ------------------------------------------------------------------
    cursor.execute("""
        SELECT
            COUNT(*)                    AS total_rows,
            ROUND(AVG(raw_loss), 2)     AS avg_raw_loss,
            ROUND(AVG(cleaned_loss), 2) AS avg_capped_loss,
            ROUND(AVG(log_loss), 4)     AS avg_log_loss
        FROM analytics_portfolio_ready;
    """)
    total_rows, avg_raw, avg_capped, avg_log = cursor.fetchone()
    conn.close()

    print("\n   ✅ Pipeline validation report:")
    print(f"      • Total rows indexed     : {total_rows:,}")
    print(f"      • Avg raw loss           : €{avg_raw:,}")
    print(f"      • Avg capped loss        : €{avg_capped:,}")
    print(f"      • Log-normalized mean    : {avg_log}")


# ======================================================================
# MAIN EXECUTION ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    print("\n🚀 SCOR P&C REINSURANCE ETL PIPELINE — STARTING...\n")
    pipeline_start = time.time()

    # Step 1: Ingest with cache gate
    raw_dataframe = ingest_portfolio_with_cache()

    # Step 2: Quality audit on raw portfolio
    generate_quality_gate_audit(raw_dataframe)

    # Step 3: Stage raw data into SQLite
    stage_raw_portfolio(raw_dataframe)

    # Step 4: Transform, normalize, and index
    execute_relational_cleaning_pipeline()

    elapsed = time.time() - pipeline_start
    print(f"\n🎉 Pipeline complete. Total runtime: {elapsed:.2f} seconds.")