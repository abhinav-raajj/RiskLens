import pandas as pd

def simulate_threshold(scored_df, threshold):
    """
    Simulates business outcomes at a specific risk score threshold.
    Now also captures the actual dollar amounts of caught fraud.
    """
    flagged = scored_df['risk_score'] >= threshold

    total_transactions = len(scored_df)
    flagged_count = flagged.sum()

    tp_mask = flagged & (scored_df['Class'] == 1)
    fp_mask = flagged & (scored_df['Class'] == 0)
    fn_mask = ~flagged & (scored_df['Class'] == 1)
    tn_mask = ~flagged & (scored_df['Class'] == 0)

    tp = tp_mask.sum()
    fp = fp_mask.sum()
    fn = fn_mask.sum()
    tn = tn_mask.sum()

    # actual dollar amounts — not assumptions, straight from the data
    tp_amount = scored_df.loc[tp_mask, 'Amount'].sum()
    fn_amount = scored_df.loc[fn_mask, 'Amount'].sum()
    total_fraud_amount = tp_amount + fn_amount

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
        'tp_amount': tp_amount,         # actual $ of fraud we caught
        'fn_amount': fn_amount,         # actual $ of fraud we missed
        'total_fraud_amount': total_fraud_amount,
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

def compute_cost_tradeoff(threshold_df, review_cost_per_txn=15, incident_cost_multiplier=1.0):
    """
    Translates detection metrics into dollar impact.

    Key assumptions (all stated explicitly):

    1. fraud_loss_prevented = actual sum of fraud transaction amounts caught,
       multiplied by incident_cost_multiplier.
       - At 1x: just the raw transaction amount (from the data)
       - At 3-5x: includes chargeback fees, card replacement, account
         remediation, regulatory reporting. Industry estimates put total
         fraud incident cost at 3-5x the transaction amount.

    2. review_cost = number of flagged transactions * review_cost_per_txn
       - Default $15: ~10 min of analyst time at ~$90/hr loaded cost.
       - This covers ALL flagged transactions (both TP and FP), since the
         reviewer doesn't know which is which before investigating.

    3. net_benefit = fraud_loss_prevented - review_cost
    """
    df = threshold_df.copy()

    # fraud prevented = actual dollar amounts * incident multiplier
    df['fraud_loss_prevented'] = df['tp_amount'] * incident_cost_multiplier

    # review cost = assumption (stated explicitly)
    # cost applies to ALL flagged txns, not just false positives,
    # because every flag requires investigation regardless of outcome
    df['review_cost'] = df['flagged_count'] * review_cost_per_txn

    # net benefit = savings minus review cost
    df['net_benefit'] = df['fraud_loss_prevented'] - df['review_cost']

    # context metrics
    df['avg_fraud_amount_caught'] = df['tp_amount'] / df['true_positives'].replace(0, 1)

    return df
