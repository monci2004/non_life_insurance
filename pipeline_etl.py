"""
SCOR Interview Asset: End-to-End Automated P&C Reinsurance ETL Pipeline Engine
Author: Monsef Djamel Eddine BOUDALIA
Description: Production script for automated asset ingestion, caching, relational 
             database cleaning, unsupervised clustering, and supervised pricing regression.
Usage: python pipeline_etl.py
"""

import os
import time
import sqlite3
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from ydata_profiling import ProfileReport
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ======================================================================
# PHASE 1: PORTFOLIO INGESTION WITH EXPLICIT LOCAL CACHE GATE
# ======================================================================
def ingest_portfolio_with_cache(dataset_id=42571, filename="allstate_raw_claims.csv"):
    """Streams data from OpenML API on first run, then serves from a local CSV cache."""
    if os.path.exists(filename):
        print(f"📦 Cache hit. Loading portfolio from disk: '{filename}'...")
        df = pd.read_csv(filename, index_col=0)
        print(f"   ✓ {len(df):,} rows loaded from local cache.")
        return df

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
        raise ConnectionError(f"❌ OpenML API unreachable. Cannot proceed. Details: {e}")


# ======================================================================
# PHASE 1 (AUDIT): AUTOMATED DATA QUALITY GATE AUDIT
# ======================================================================
def generate_quality_gate_audit(df, output_path="production_portfolio_report.html"):
    """Generates an HTML data profiling report on a sample to surface statistical trends."""
    print("🔍 Generating data quality audit report...")
    sample_df = df.sample(n=10000, random_state=42).copy().reset_index(drop=True)
    profile = ProfileReport(sample_df, title="SCOR Portfolio Integrity Audit — Allstate Claims")
    profile.to_file(output_path)
    print(f"   ✓ Audit report saved to: '{output_path}'")


# ======================================================================
# PHASE 2: BULK STAGING — RAW DATAFRAME → SQLITE
# ======================================================================
def stage_raw_portfolio(df, db_name="scor_portfolio_management.db"):
    """Bulk-loads the raw claims DataFrame into the SQLite staging table."""
    print(f"💾 Staging raw portfolio into '{db_name}' → table 'clean_allstate_claims'...")
    conn = sqlite3.connect(db_name)
    df.to_sql("clean_allstate_claims", conn, if_exists="replace", index=True, index_label="id")
    conn.close()
    print(f"   ✓ {len(df):,} rows staged successfully.")


# ======================================================================
# PHASE 2 (ENGINE): SQL TRANSFORMATION ENGINE — CAP, NORMALIZE, INDEX
# ======================================================================
def execute_relational_cleaning_pipeline(db_name="scor_portfolio_management.db"):
    """Executes database anomaly truncation, log normalization, and indexing maps."""
    print(f"🧹 Executing relational cleaning pipeline on '{db_name}'...")
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Compute 99.5th percentile catastrophe cap directly inside SQL
    cursor.execute("""
        SELECT loss FROM clean_allstate_claims
        ORDER BY loss ASC
        LIMIT 1 OFFSET (SELECT CAST(COUNT(*) * 0.995 AS INT) FROM clean_allstate_claims);
    """)
    cat_threshold = cursor.fetchone()[0]
    print(f"   🎯 Catastrophe cap (99.5th percentile): €{cat_threshold:,.2f}")

    # Build the cleaned analytics table with outliers capped safely via bound parameters
    cursor.execute("DROP TABLE IF EXISTS analytics_portfolio_ready;")
    cursor.execute("""
        CREATE TABLE analytics_portfolio_ready AS
        SELECT
            id, cat1, cat2, cat3, cat4, cat5, cont1, cont2, cont3, cont4, cont5,
            loss AS raw_loss,
            CASE WHEN loss > ? THEN ? ELSE loss END AS cleaned_loss
        FROM clean_allstate_claims;
    """, (cat_threshold, cat_threshold))
    conn.commit()

    # Apply mathematical log1p normalization using NumPy vectors
    df_analytics = pd.read_sql_query("SELECT * FROM analytics_portfolio_ready", conn)
    df_analytics['log_loss'] = np.log1p(df_analytics['cleaned_loss'])
    df_analytics.to_sql("analytics_portfolio_ready", conn, if_exists="replace", index=False)

    # Inject a read-path database index over our new target variable
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_log_loss ON analytics_portfolio_ready (log_loss);")

    # Run structural audit summary stats
    cursor.execute("""
        SELECT COUNT(*), ROUND(AVG(raw_loss), 2), ROUND(AVG(cleaned_loss), 2), ROUND(AVG(log_loss), 4)
        FROM analytics_portfolio_ready;
    """)
    total_rows, avg_raw, avg_capped, avg_log = cursor.fetchone()
    conn.close()

    print("\n   ✅ Pipeline validation report:")
    print(f"      • Total rows indexed     : {total_rows:,}")
    print(f"      • Avg raw loss            : €{avg_raw:,}")
    print(f"      • Avg capped loss         : €{avg_capped:,}")
    print(f"      • Log-normalized mean     : {avg_log}")


# ======================================================================
# PHASE 3: UNSUPERVISED RISK SEGMENTATION ENGINE (K-MEANS)
# ======================================================================
def run_unsupervised_segmentation(db_name="scor_portfolio_management.db"):
    """Pulls clean tables, standardizes continuous exposure vectors, and tags K-Means clusters."""
    print(f"🧩 Production Analytics: Executing K-Means clustering track on '{db_name}'...")
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query("SELECT * FROM analytics_portfolio_ready", conn)
    
    # Isolate exposure metrics
    exposure_cols = [col for col in df.columns if col.startswith('cont')]
    X = df[exposure_cols]
    
    # Standardize scale distribution to protect geometric calculations
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Partition portfolio into 3 operational risk groups
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df['RiskSegmentID'] = kmeans.fit_predict(X_scaled)
    
    # Re-save matrix back to SQLite table
    df.to_sql("analytics_portfolio_ready", conn, if_exists="replace", index=False)
    
    # Verify cluster counts
    cursor = conn.cursor()
    cursor.execute("SELECT RiskSegmentID, COUNT(*) FROM analytics_portfolio_ready GROUP BY RiskSegmentID;")
    counts = cursor.fetchall()
    conn.close()
    
    print("   ✓ Unsupervised Segmentation Locked. Distribution map verified:")
    for cluster_id, count in counts:
        print(f"      • Cluster {cluster_id} → {count:,} records.")


# ======================================================================
# PHASE 4: SUPERVISED PREDICTIVE PRICING ENGINE (RANDOM FOREST)
# ======================================================================
def run_supervised_pricing_engine(db_name="scor_portfolio_management.db"):
    """Encodes categorical strings, maps train/test frameworks, and trains pricing ensemble trees."""
    print(f"\n🔮 Production Analytics: Training Supervised Predictive Pricing Engine on '{db_name}'...")
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query("SELECT * FROM analytics_portfolio_ready", conn)
    conn.close()
    
    # Isolate feature vectors and target metrics
    feature_cols = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'RiskSegmentID']
    X = df[feature_cols].copy()
    y = df['log_loss']
    
    # One-Hot Encoding transformation
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Train the Ensemble Forest Model using parallel threading processing
    model = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predict and calculate validation analytics metrics
    y_pred_log = model.predict(X_test)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
    r2 = r2_score(y_test, y_pred_log)
    
    # Actuarial back-transformation out of logarithmic mapping into Euro premiums
    y_test_euros = np.expm1(y_test)
    y_pred_euros = np.expm1(y_pred_log)
    rmse_euros = np.sqrt(mean_squared_error(y_test_euros, y_pred_euros))
    
    print("======================================================================")
    print("📈📈📈 FINAL ARCHITECTURE PRODUCTION SUCCESS REPORT:")
    print("======================================================================")
    print(f"   • Log-Scale Validation RMSE   : {rmse_log:.4f}")
    print(f"   • Underwriting R² Score        : {r2 * 100:.2f}%")
    print(f"   • Absolute Financial Error     : €{rmse_euros:,.2f}")
    print("======================================================================")


# ======================================================================
# MAIN SYSTEM INVOCATION
# ======================================================================
if __name__ == "__main__":
    print("\n🚀 SCOR P&C REINSURANCE AUTOMATED SYSTEM PIPELINE — INITIATING...\n")
    pipeline_start = time.time()

    # Step 1 & 2: Ingestion & Profiling Check
    raw_dataframe = ingest_portfolio_with_cache()
    generate_quality_gate_audit(raw_dataframe)

    # Step 3 & 4: Relational Layer Ingestion & Cleaning Queries
    stage_raw_portfolio(raw_dataframe)
    execute_relational_cleaning_pipeline()
    
    # Step 5: Unsupervised Risk Archetype Segmentation Feature Store
    run_unsupervised_segmentation()

    # Step 6: Supervised Machine Learning Ensemble Pricing Execution
    run_supervised_pricing_engine()

    elapsed = time.time() - pipeline_start
    print(f"\n🎉 Complete End-to-End Pipeline Automated Run Concluded. Total Runtime: {elapsed:.2f} seconds.")