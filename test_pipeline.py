"""
Quick smoke test to make sure the full pipeline works before launching the dashboard.
Run from project root: python test_pipeline.py
"""
import sys
import time

print("=" * 60)
print("RiskLens Pipeline Smoke Test")
print("=" * 60)

# step 1: load data
print("\n[1/6] Loading Kaggle dataset...")
t0 = time.time()
from src.data_loader import load_kaggle_data, generate_upi_data, init_database
df = load_kaggle_data()
print(f"  -> Loaded {len(df)} transactions in {time.time()-t0:.1f}s")
print(f"  -> Columns: {list(df.columns)}")
print(f"  -> Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
print(f"  -> Unique users: {df['user_id'].nunique()}")

# step 2: init database
print("\n[2/6] Initializing SQLite database...")
t0 = time.time()
init_database()
print(f"  -> Done in {time.time()-t0:.1f}s")

# step 3: sql queries
print("\n[3/6] Running SQL queries...")
t0 = time.time()
from src.sql_queries import fraud_by_time_bucket, fraud_by_amount_bucket, upi_failure_analysis
from src.utils import get_connection
conn = get_connection()
time_df = fraud_by_time_bucket(conn)
print(f"  -> Fraud by time bucket:")
print(time_df.to_string(index=False))

amt_df, medians = fraud_by_amount_bucket(conn)
print(f"\n  -> Fraud by amount bucket:")
print(amt_df.to_string(index=False))
print(f"\n  -> Median amounts:")
print(medians.to_string(index=False))

upi_df = upi_failure_analysis(conn)
print(f"\n  -> UPI failure analysis:")
print(upi_df.to_string(index=False))
conn.close()
print(f"  -> SQL queries done in {time.time()-t0:.1f}s")

# step 4: risk scoring (only first weight combo to save time)
print("\n[4/6] Computing user baselines and scoring transactions...")
t0 = time.time()
from src.risk_engine import compute_user_baselines, score_transactions, evaluate_scoring
baselines = compute_user_baselines(df)
print(f"  -> Baselines computed for {len(baselines)} users in {time.time()-t0:.1f}s")

t0 = time.time()
scored_df = score_transactions(df, baselines)
print(f"  -> Scoring done in {time.time()-t0:.1f}s")
print(f"  -> Risk distribution:")
print(scored_df['risk_bucket'].value_counts().to_string())

eval_results = evaluate_scoring(scored_df)

# step 5: threshold simulator
print("\n[5/6] Running threshold simulator...")
from src.threshold_simulator import precompute_all_thresholds, compute_cost_tradeoff
thresh_df = precompute_all_thresholds(scored_df)
thresh_df = compute_cost_tradeoff(thresh_df)
print(thresh_df[['threshold', 'flagged_count', 'true_positives', 'false_positives', 'precision', 'recall', 'net_benefit']].to_string(index=False))

# step 6: drift detection
print("\n[6/6] Computing risk trajectories...")
from src.drift_detector import compute_risk_trajectory, compute_drift_slopes, flag_drifting_users
trajectory = compute_risk_trajectory(scored_df)
print(f"  -> Trajectory rows: {len(trajectory)}")
drift = compute_drift_slopes(trajectory)
print(f"  -> Users with drift data: {len(drift)}")
drifting = flag_drifting_users(drift, slope_threshold=0.1)
print(f"  -> Drifting users flagged: {len(drifting)}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED - OK")
print("=" * 60)
