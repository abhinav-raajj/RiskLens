"""Quick test: compare hand-weighted vs LR vs LR+PCA."""
import time
from src.data_loader import load_kaggle_data
from src.risk_engine import (compute_user_baselines, score_transactions,
                              tune_weights, compare_all_approaches)

print("Loading data...")
df = load_kaggle_data()
print(f"Loaded {len(df):,} transactions, {df['Class'].sum()} fraud")

print("\nComputing baselines...")
baselines = compute_user_baselines(df)

print("Scoring with hand-weighted rules...")
t0 = time.time()
best_weights, scored_df = tune_weights(df, baselines)
print(f"Done in {time.time()-t0:.1f}s")

print("\n" + "=" * 70)
print("RUNNING 3-WAY COMPARISON")
print("=" * 70)
comparison = compare_all_approaches(scored_df)

# print detailed metrics for each approach
for name, data in comparison.items():
    if not isinstance(data, dict) or 'label' not in data:
        continue
    print(f"\n--- {data['label']} ---")
    print(data['metrics'].to_string(index=False))

