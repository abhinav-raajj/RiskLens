import pandas as pd
import numpy as np

def compute_user_baselines(df):
    """
    Computes baseline stats for each user — their average spending,
    standard deviation, and what hours they typically transact in.
    This becomes the 'normal behavior profile' we compare against.
    """
    # compute numeric baselines with groupby — much faster than looping
    stats = df.groupby('user_id').agg(
        avg_amount=('Amount', 'mean'),
        std_amount=('Amount', 'std'),
        transaction_count=('Amount', 'count'),
        min_time=('Time', 'min'),
        max_time=('Time', 'max')
    ).fillna(0)  # std is NaN for single-transaction users

    # for the historical hours, we need to collect sets per user
    hour_sets = df.groupby('user_id')['hour'].apply(set).to_dict()

    baselines = {}
    for user_id, row in stats.iterrows():
        baselines[user_id] = {
            'avg_amount': row['avg_amount'],
            'std_amount': row['std_amount'],
            'transaction_count': int(row['transaction_count']),
            'historical_hours': hour_sets.get(user_id, set()),
            'min_time': row['min_time'],
            'max_time': row['max_time']
        }

    return baselines


def score_transactions(df, baselines, weights=None):
    """
    Scores each transaction using 5 rule-based signals.
    Fully vectorized for performance — no row-by-row loops on 284K rows.
    """
    if weights is None:
        weights = {
            'amount_deviation': 2,
            'category_rarity': 1,
            'time_anomaly': 1,
            'velocity': 3,
            'round_number_pattern': 2
        }

    scored_df = df.copy()
    scored_df = scored_df.sort_values(['user_id', 'Time']).reset_index(drop=True)

    # build lookup arrays from baselines for vectorized access
    user_avg = scored_df['user_id'].map(lambda u: baselines.get(u, {}).get('avg_amount', 0))
    user_std = scored_df['user_id'].map(lambda u: baselines.get(u, {}).get('std_amount', 0))
    user_hours = scored_df['user_id'].map(lambda u: baselines.get(u, {}).get('historical_hours', set()))

    global_std = scored_df['Amount'].std()

    # replace zero std with global std (users with 1 transaction)
    user_std = user_std.where(user_std > 0, global_std)

    # ---- Signal 1: Amount Deviation ----
    # two-pronged check:
    # a) unusually LARGE amounts vs the user's history (stolen card spending spree)
    # b) very small amounts under $2 (card-testing pattern — median fraud is $9.25)
    # the z-score low-end check doesn't work here because std >> avg for most users,
    # so user_avg - 2*std goes negative. using a fixed $2 threshold instead.
    high_flag = scored_df['Amount'] > (user_avg + 3 * user_std)
    test_charge_flag = scored_df['Amount'] <= 2.0
    high_value_flag = scored_df['Amount'] >= 500  # $500+ bucket has 0.37% fraud rate (2x baseline)
    scored_df['amount_deviation'] = (high_flag | test_charge_flag | high_value_flag).astype(int)

    # ---- Signal 2: Time Anomaly ----
    # data finding: fraud rate during hours 0-7 is 0.44% avg, roughly 4x the
    # daytime rate of ~0.13%. flagging this window is more useful than checking
    # user-specific hour history (which is noise with synthetic user IDs)
    scored_df['time_anomaly'] = scored_df['hour'].isin(range(0, 8)).astype(int)

    # ---- Signal 3: Velocity ----
    # 3+ transactions by same user within a 5-minute window
    # use a rolling count approach per user group
    scored_df['velocity'] = 0
    for user_id, group in scored_df.groupby('user_id'):
        if len(group) < 3:
            continue
        times = group['Time'].values
        indices = group.index.values
        # for each transaction, count how many others fall within ±150 seconds (2.5 min)
        for i in range(len(times)):
            t = times[i]
            count = np.sum((times >= t - 150) & (times <= t + 150))
            if count >= 3:
                scored_df.loc[indices[i], 'velocity'] = 1

    # ---- Signal 4: Round Number Pattern ----
    # small round-number test charge followed by a large purchase within 2 minutes
    scored_df['round_number_pattern'] = 0
    test_amounts = {1, 2, 5, 10, 20, 50, 100}

    for user_id, group in scored_df.groupby('user_id'):
        if len(group) < 2:
            continue
        amounts = group['Amount'].values
        times = group['Time'].values
        indices = group.index.values

        for i in range(len(amounts) - 1):
            amt = amounts[i]
            # check if it's a round test amount
            if amt % 1 == 0 and int(amt) in test_amounts:
                next_amt = amounts[i + 1]
                time_gap = times[i + 1] - times[i]
                # followed by a large charge within 2 minutes
                if next_amt > 100 and time_gap <= 120:
                    scored_df.loc[indices[i], 'round_number_pattern'] = 1

    # ---- Signal 5: Category Rarity ----
    # V14 is the PCA component with the strongest separation between
    # normal and anomalous transactions. Values below -5 indicate
    # transaction patterns significantly different from the norm.
    # this is our best proxy for "unusual merchant category" since
    # actual categories are anonymized.
    scored_df['category_rarity'] = (scored_df['V14'] < -5).astype(int)

    # ---- Compute weighted risk score ----
    scored_df['risk_score'] = (
        scored_df['amount_deviation'] * weights['amount_deviation'] +
        scored_df['category_rarity'] * weights['category_rarity'] +
        scored_df['time_anomaly'] * weights['time_anomaly'] +
        scored_df['velocity'] * weights['velocity'] +
        scored_df['round_number_pattern'] * weights['round_number_pattern']
    )

    # bucket into risk levels
    scored_df['risk_bucket'] = pd.cut(
        scored_df['risk_score'],
        bins=[-1, 1, 4, float('inf')],
        labels=['Low', 'Medium', 'High']
    )

    return scored_df


def evaluate_scoring(scored_df):
    """
    Checks how well our rule-based scoring identifies actual fraud.
    Reports precision/recall honestly — we're not trying to compete with ML here,
    just showing that the rules capture meaningful signal.
    """
    total_fraud = scored_df['Class'].sum()

    # high bucket performance
    high_mask = (scored_df['risk_bucket'] == 'High')
    high_tp = (high_mask & (scored_df['Class'] == 1)).sum()
    high_fp = (high_mask & (scored_df['Class'] == 0)).sum()

    high_prec = high_tp / (high_tp + high_fp) if (high_tp + high_fp) > 0 else 0
    high_rec = high_tp / total_fraud if total_fraud > 0 else 0
    high_f1 = 2 * (high_prec * high_rec) / (high_prec + high_rec) if (high_prec + high_rec) > 0 else 0

    # medium + high bucket (more permissive)
    mh_mask = scored_df['risk_bucket'].isin(['Medium', 'High'])
    mh_tp = (mh_mask & (scored_df['Class'] == 1)).sum()
    mh_fp = (mh_mask & (scored_df['Class'] == 0)).sum()

    mh_prec = mh_tp / (mh_tp + mh_fp) if (mh_tp + mh_fp) > 0 else 0
    mh_rec = mh_tp / total_fraud if total_fraud > 0 else 0
    mh_f1 = 2 * (mh_prec * mh_rec) / (mh_prec + mh_rec) if (mh_prec + mh_rec) > 0 else 0

    print("\n--- Scoring Evaluation ---")
    print(f"High Bucket   - Precision: {high_prec:.4f}, Recall: {high_rec:.4f}, F1: {high_f1:.4f}")
    print(f"Med+High      - Precision: {mh_prec:.4f}, Recall: {mh_rec:.4f}, F1: {mh_f1:.4f}")
    print(f"(Total fraud in dataset: {total_fraud})")

    return {
        'high_precision': high_prec,
        'high_recall': high_rec,
        'high_f1': high_f1,
        'mh_precision': mh_prec,
        'mh_recall': mh_rec,
        'mh_f1': mh_f1,
        'total_fraud': int(total_fraud)
    }


def tune_weights(df, baselines):
    """
    Tests a handful of weight combinations and picks the one that best
    separates fraud from non-fraud. Not a grid search — just some sensible
    variations I tried manually to see what moves the needle.
    """
    weight_combos = [
        # original from spec
        {'amount_deviation': 2, 'category_rarity': 1, 'time_anomaly': 1, 'velocity': 3, 'round_number_pattern': 2},
        # bump category_rarity since V14 is the strongest fraud signal in PCA
        {'amount_deviation': 2, 'category_rarity': 2, 'time_anomaly': 1, 'velocity': 3, 'round_number_pattern': 2},
        # category-heavy — see if the PCA signal alone carries most of the weight
        {'amount_deviation': 2, 'category_rarity': 3, 'time_anomaly': 1, 'velocity': 2, 'round_number_pattern': 2},
        # category-first approach
        {'amount_deviation': 1, 'category_rarity': 3, 'time_anomaly': 1, 'velocity': 3, 'round_number_pattern': 2},
        # balanced across all signals
        {'amount_deviation': 2, 'category_rarity': 2, 'time_anomaly': 2, 'velocity': 2, 'round_number_pattern': 2},
        # strong category + strong amount (catch both PCA anomaly and test charges)
        {'amount_deviation': 3, 'category_rarity': 3, 'time_anomaly': 1, 'velocity': 2, 'round_number_pattern': 2},
    ]

    best_score = -1
    best_weights = None
    best_scored_df = None

    print("\n--- Weight Tuning ---")
    print(f"{'Weights':<55} | {'Prec':>6} | {'Rec':>6} | {'Avg':>6}")
    print("-" * 82)

    for w in weight_combos:
        scored = score_transactions(df, baselines, weights=w)

        high_mask = (scored['risk_bucket'] == 'High')
        tp = (high_mask & (scored['Class'] == 1)).sum()
        fp = (high_mask & (scored['Class'] == 0)).sum()
        total_fraud = scored['Class'].sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / total_fraud if total_fraud > 0 else 0
        avg = (prec + rec) / 2

        weight_str = f"a={w['amount_deviation']} c={w['category_rarity']} t={w['time_anomaly']} v={w['velocity']} r={w['round_number_pattern']}"
        print(f"{weight_str:<55} | {prec:>6.4f} | {rec:>6.4f} | {avg:>6.4f}")

        if avg > best_score:
            best_score = avg
            best_weights = w
            best_scored_df = scored

    print("-" * 82)
    print(f"Best average metric: {best_score:.4f}")
    print(f"Best weights: {best_weights}")

    return best_weights, best_scored_df


# ============================================================
# LOGISTIC REGRESSION — fitted weights instead of guessed ones
# ============================================================

SIGNAL_COLUMNS = ['amount_deviation', 'category_rarity', 'time_anomaly',
                  'velocity', 'round_number_pattern']


def _compute_signal_frequency(scored_df, features=None):
    """
    Stable alternative to odds ratios: for each signal, compute
    what % of fraud vs legit transactions trigger it.
    This sidesteps quasi-complete-separation issues entirely.
    """
    if features is None:
        features = SIGNAL_COLUMNS
    fraud = scored_df[scored_df['Class'] == 1]
    legit = scored_df[scored_df['Class'] == 0]
    rows = []
    for feat in features:
        # what fraction of fraud/legit transactions have this flag = 1?
        fraud_rate = fraud[feat].mean() if len(fraud) > 0 else 0
        legit_rate = legit[feat].mean() if len(legit) > 0 else 0
        # how much more common is this flag in fraud vs legit?
        lift = round(fraud_rate / legit_rate, 1) if legit_rate > 0 else float('inf')
        rows.append({
            'signal': feat,
            'pct_fraud_flagged': round(fraud_rate * 100, 1),
            'pct_legit_flagged': round(legit_rate * 100, 1),
            'lift': lift
        })
    return pd.DataFrame(rows)


def fit_logistic_model(scored_df, features=None):
    """
    Fit LR on interpretable signals — data-fitted weights, still explainable.

    Uses strong regularization (C=0.001) because category_rarity (V14 < -5)
    causes quasi-complete separation — without it, LR pushes that coefficient
    toward infinity, producing misleading 2000x+ odds ratios. With regularization,
    we get stable, reproducible coefficients that reflect *relative* importance
    across features rather than an inflated artifact of near-perfect separation.
    """
    from sklearn.linear_model import LogisticRegression
    if features is None:
        features = SIGNAL_COLUMNS
    X = scored_df[features].values
    y = scored_df['Class'].values
    model = LogisticRegression(C=0.001, class_weight='balanced', max_iter=1000,
                                random_state=42)
    model.fit(X, y)

    # compute relative importance: normalize absolute coefficients to sum to 100%
    abs_coefs = np.abs(model.coef_[0])
    total = abs_coefs.sum()
    rel_importance = abs_coefs / total * 100 if total > 0 else abs_coefs

    coefficients = {}
    for feat, coef, imp in zip(features, model.coef_[0], rel_importance):
        coefficients[feat] = {
            'coefficient': round(coef, 4),
            'relative_importance_pct': round(imp, 1),
            'direction': 'increases fraud risk' if coef > 0 else 'decreases fraud risk'
        }
    coefficients['intercept'] = round(model.intercept_[0], 4)
    probs = model.predict_proba(X)[:, 1]
    result_df = scored_df.copy()
    result_df['fraud_probability'] = probs
    return model, coefficients, result_df


def evaluate_logistic_model(scored_df, prob_column='fraud_probability'):
    """Evaluate LR at several probability thresholds."""
    total_fraud = scored_df['Class'].sum()
    total = len(scored_df)
    results = []
    for t in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
        flagged = scored_df[prob_column] >= t
        tp = (flagged & (scored_df['Class'] == 1)).sum()
        fp = (flagged & (scored_df['Class'] == 0)).sum()
        tp_amount = scored_df.loc[flagged & (scored_df['Class'] == 1), 'Amount'].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_fraud if total_fraud > 0 else 0
        results.append({
            'prob_threshold': t, 'flagged': int(flagged.sum()),
            'flagged_pct': round(flagged.sum() / total * 100, 2),
            'true_positives': int(tp), 'false_positives': int(fp),
            'precision': round(precision, 4), 'recall': round(recall, 4),
            'tp_amount': round(tp_amount, 2)
        })
    return pd.DataFrame(results)


def compare_all_approaches(scored_df):
    """Side-by-side: hand-weighted vs LR (5 signals) vs LR+PCA."""
    comparison = {}
    total_fraud = scored_df['Class'].sum()

    # Approach 1: Hand-weighted rules
    hand_metrics = []
    for threshold in [1, 2, 3, 4, 5]:
        flagged = scored_df['risk_score'] >= threshold
        tp = (flagged & (scored_df['Class'] == 1)).sum()
        fp = (flagged & (scored_df['Class'] == 0)).sum()
        tp_amt = scored_df.loc[flagged & (scored_df['Class'] == 1), 'Amount'].sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / total_fraud if total_fraud > 0 else 0
        hand_metrics.append({
            'threshold': f'score>={threshold}', 'flagged': int(flagged.sum()),
            'flagged_pct': round(flagged.sum() / len(scored_df) * 100, 2),
            'true_positives': int(tp), 'precision': round(prec, 4),
            'recall': round(rec, 4), 'tp_amount': round(tp_amt, 2)
        })
    comparison['hand_weighted'] = {
        'metrics': pd.DataFrame(hand_metrics), 'label': 'Hand-Weighted Rules'
    }

    # --- Signal Frequency Analysis (stable, interview-safe) ---
    print("\n--- Signal Frequency Analysis ---")
    print("  (What % of fraud vs legit transactions trigger each signal)")
    freq_df = _compute_signal_frequency(scored_df)
    for _, row in freq_df.iterrows():
        print(f"    {row['signal']:25s}  fraud={row['pct_fraud_flagged']:5.1f}%  "
              f"legit={row['pct_legit_flagged']:5.1f}%  lift={row['lift']:.1f}x")
    comparison['signal_frequency'] = freq_df

    # Approach 2: LR on 5 interpretable signals
    print("\nFitting LR on 5 interpretable signals...")
    model_5, coefs_5, df_5 = fit_logistic_model(scored_df)
    lr5_metrics = evaluate_logistic_model(df_5)
    comparison['lr_interpretable'] = {
        'model': model_5, 'coefficients': coefs_5, 'metrics': lr5_metrics,
        'scored_df': df_5, 'features': list(SIGNAL_COLUMNS),
        'label': 'LR (5 Interpretable Signals)'
    }
    print("  Feature importance (relative contribution to model):")
    # sort by importance descending
    sorted_feats = sorted(
        [(f, v) for f, v in coefs_5.items() if f != 'intercept'],
        key=lambda x: x[1]['relative_importance_pct'], reverse=True
    )
    for rank, (feat, vals) in enumerate(sorted_feats, 1):
        print(f"    #{rank}  {feat:25s}  {vals['relative_importance_pct']:5.1f}%  "
              f"({vals['direction']})")

    # Approach 3: LR + top PCA features
    pca_cols = ['V14', 'V17', 'V12', 'V10']
    boosted = SIGNAL_COLUMNS + pca_cols
    print(f"\nFitting LR on 5 signals + PCA {pca_cols}...")
    model_b, coefs_b, df_b = fit_logistic_model(scored_df, features=boosted)
    lrb_metrics = evaluate_logistic_model(df_b)
    comparison['lr_boosted'] = {
        'model': model_b, 'coefficients': coefs_b, 'metrics': lrb_metrics,
        'scored_df': df_b, 'features': list(boosted),
        'label': 'LR (5 Signals + PCA)'
    }
    print("  Feature importance (relative contribution to model):")
    sorted_b = sorted(
        [(f, v) for f, v in coefs_b.items() if f != 'intercept'],
        key=lambda x: x[1]['relative_importance_pct'], reverse=True
    )
    for rank, (feat, vals) in enumerate(sorted_b, 1):
        tag = " [PCA]" if feat in pca_cols else ""
        print(f"    #{rank}  {feat:25s}  {vals['relative_importance_pct']:5.1f}%  "
              f"({vals['direction']}){tag}")

    # Summary comparison at ~1% flag rate
    print("\n" + "=" * 70)
    print("COMPARISON AT ~1% FLAG RATE")
    print("=" * 70)
    for name, data in comparison.items():
        if name == 'signal_frequency':
            continue
        df_m = data['metrics']
        if 'flagged_pct' in df_m.columns:
            idx = (df_m['flagged_pct'] - 1.0).abs().idxmin()
        else:
            idx = len(df_m) // 2
        best = df_m.iloc[idx]
        print(f"  {data['label']:30s}: prec={best['precision']:.4f}  "
              f"recall={best['recall']:.4f}  flagged={int(best['flagged']):,}")

    return comparison
