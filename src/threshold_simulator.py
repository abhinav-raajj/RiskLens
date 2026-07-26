import pandas as pd

def simulate_threshold(scored_df, threshold):
    """
    Simulates business outcomes at a specific risk score threshold.
    """
    flagged = scored_df['risk_score'] >= threshold
    
    total_transactions = len(scored_df)
    flagged_count = flagged.sum()
    
    tp = (flagged & (scored_df['Class'] == 1)).sum()
    fp = (flagged & (scored_df['Class'] == 0)).sum()
    fn = (~flagged & (scored_df['Class'] == 1)).sum()
    tn = (~flagged & (scored_df['Class'] == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    total_fraud = tp + fn
    pct_flagged = flagged_count / total_transactions if total_transactions > 0 else 0.0
    
    return {
        'threshold': threshold,
        'flagged_count': flagged_count,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'precision': precision,
        'recall': recall,
        'false_positive_rate': fpr,
        'total_fraud': total_fraud,
        'pct_flagged': pct_flagged
    }

def precompute_all_thresholds(scored_df, max_threshold=10):
    """
    Computes metrics for a range of thresholds to power a UI slider.
    """
    results = []
    for t in range(max_threshold + 1):
        metrics = simulate_threshold(scored_df, t)
        results.append(metrics)
        
    return pd.DataFrame(results)

def compute_cost_tradeoff(threshold_df, fraud_cost_per_txn=500, friction_cost_per_txn=15):
    """
    Translates ML metrics into dollar amounts (ROI / business case).
    """
    df = threshold_df.copy()
    
    df['fraud_loss_prevented'] = df['true_positives'] * fraud_cost_per_txn
    df['friction_cost'] = df['false_positives'] * friction_cost_per_txn
    df['net_benefit'] = df['fraud_loss_prevented'] - df['friction_cost']
    
    return df
