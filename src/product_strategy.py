import pandas as pd
import numpy as np


def segment_analysis(scored_df):
    """
    Product insight: the optimal fraud threshold is DIFFERENT for different
    transaction segments. A $2 test charge and a $5,000 purchase shouldn't
    have the same risk tolerance.

    Splits transactions into 3 segments and finds the optimal threshold
    for each based on net financial benefit.
    """
    segments = {
        'Micro ($0-10)': scored_df[scored_df['Amount'] <= 10],
        'Standard ($10-500)': scored_df[(scored_df['Amount'] > 10) & (scored_df['Amount'] <= 500)],
        'High-Value ($500+)': scored_df[scored_df['Amount'] > 500],
    }

    results = []
    for seg_name, seg_df in segments.items():
        total = len(seg_df)
        fraud = int(seg_df['Class'].sum())
        fraud_rate = fraud / total * 100 if total > 0 else 0
        avg_amount = seg_df['Amount'].mean()
        total_fraud_amount = seg_df.loc[seg_df['Class'] == 1, 'Amount'].sum()

        # find the threshold that maximizes net benefit for this segment
        best_threshold = 0
        best_net = -float('inf')
        best_details = {}

        for t in range(0, 10):
            flagged = seg_df['risk_score'] >= t
            tp = (flagged & (seg_df['Class'] == 1)).sum()
            fp = (flagged & (seg_df['Class'] == 0)).sum()
            tp_amount = seg_df.loc[flagged & (seg_df['Class'] == 1), 'Amount'].sum()
            review_cost = int(flagged.sum()) * 15
            net = tp_amount * 3 - review_cost  # 3x incident cost multiplier

            if net > best_net:
                best_net = net
                best_threshold = t
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / fraud if fraud > 0 else 0
                best_details = {'precision': round(prec, 4), 'recall': round(rec, 4),
                                'flagged': int(flagged.sum()), 'tp': int(tp)}

        results.append({
            'segment': seg_name,
            'total_txns': total,
            'fraud_cases': fraud,
            'fraud_rate_pct': round(fraud_rate, 3),
            'avg_amount': round(avg_amount, 2),
            'total_fraud_amount': round(total_fraud_amount, 2),
            'optimal_threshold': best_threshold,
            'net_benefit': round(best_net, 2),
            **best_details
        })

    return pd.DataFrame(results)


def stakeholder_impact(threshold_df, threshold):
    """
    Multi-stakeholder impact analysis at a given threshold.
    Translates technical metrics into language each stakeholder cares about.

    Dataset covers ~2 days → daily_factor = 0.5 for extrapolation.
    """
    row = threshold_df[threshold_df['threshold'] == threshold]
    if row.empty:
        return None
    row = row.iloc[0]

    daily = 0.5  # dataset is ~2 days
    tp = max(row['true_positives'], 1)
    fp = row['false_positives']

    return {
        'customer': {
            'label': 'Customer Experience',
            'blocked_per_day': int(fp * daily),
            'friction_ratio': round(fp / tp, 1),
            'summary': f"{int(fp * daily):,} legitimate customers blocked per day "
                       f"({round(fp / tp, 1)} innocents per real fraud catch)"
        },
        'operations': {
            'label': 'Operations Load',
            'analyst_hours_per_day': round(row['flagged_count'] * daily * 10 / 60, 1),
            'cases_per_day': int(row['flagged_count'] * daily),
            'summary': f"{round(row['flagged_count'] * daily * 10 / 60, 1)} analyst-hours/day "
                       f"to review {int(row['flagged_count'] * daily):,} flagged cases"
        },
        'finance': {
            'label': 'Financial Impact',
            'fraud_saved_per_day': round(row.get('fraud_loss_prevented', 0) * daily, 2),
            'review_cost_per_day': round(row.get('review_cost', 0) * daily, 2),
            'net_per_day': round(row.get('net_benefit', 0) * daily, 2),
            'cost_per_catch': round(row.get('review_cost', 0) / tp, 2),
            'summary': f"${round(row.get('net_benefit', 0) * daily, 0):,.0f}/day net "
                       f"(${round(row.get('review_cost', 0) / tp, 0):,.0f} cost per fraud caught)"
        },
    }


def get_recommendations():
    """
    Pre-built product recommendation cards for different business contexts.
    Shows that the 'right' threshold depends on who you are, not just the data.
    """
    return [
        {
            'context': 'Premium Card Issuer',
            'example': 'e.g., Amex',
            'icon': '💳',
            'recommended_threshold': 3,
            'reasoning': (
                'Premium cardholders expect zero friction. A false negative '
                '(missed fraud on a $10K card) costs more in brand damage '
                'than a false positive (a quick verification call). Use a '
                'lower threshold and absorb higher review costs as a service investment.'
            ),
            'priority': 'Minimize missed fraud',
            'color': '#667eea'
        },
        {
            'context': 'High-Volume Payment Platform',
            'example': 'e.g., PhonePe / Navi',
            'icon': '📱',
            'recommended_threshold': 5,
            'reasoning': (
                'At millions of daily transactions, even 0.1% false positive rate '
                'means thousands of blocked payments and support tickets. Use a '
                'higher threshold for auto-blocking, supplement with post-transaction '
                'monitoring and ML-based queue prioritization.'
            ),
            'priority': 'Minimize customer friction',
            'color': '#48bb78'
        },
        {
            'context': 'Traditional Retail Bank',
            'example': 'e.g., SBI / HDFC',
            'icon': '🏦',
            'recommended_threshold': 4,
            'reasoning': (
                'Balanced approach: catch enough fraud to satisfy regulators '
                'without overwhelming the operations team. Use threshold 4 as '
                'default, with dynamic adjustment — lower for high-value '
                'transactions, higher for micro-payments.'
            ),
            'priority': 'Balance detection and experience',
            'color': '#ecc94b'
        }
    ]
