import pandas as pd
import numpy as np

def compute_risk_trajectory(scored_df, n_periods=8):
    """
    Divides time into n_periods and computes user risk stats per period.
    Dataset spans ~48 hours (172800 seconds).
    """
    max_time = 172800
    actual_max = scored_df['Time'].max()
    period_length = max(actual_max, max_time) / n_periods
    
    scored_df['period'] = np.clip((scored_df['Time'] // period_length).astype(int), 0, n_periods - 1)
    
    grouped = scored_df.groupby(['user_id', 'period'])
    
    trajectory = grouped.agg(
        avg_risk_score=('risk_score', 'mean'),
        max_risk_score=('risk_score', 'max'),
        txn_count=('risk_score', 'count')
    ).reset_index()
    
    return trajectory

def compute_drift_slopes(trajectory_df):
    """
    Computes linear regression slope of avg_risk_score over time for users with 3+ periods.
    """
    results = []
    
    for user_id, group in trajectory_df.groupby('user_id'):
        n_periods_active = len(group)
        if n_periods_active < 3:
            continue
            
        group = group.sort_values('period')
        x = group['period'].values
        y = group['avg_risk_score'].values
        
        slope, _ = np.polyfit(x, y, 1)
        
        first_val = y[0]
        last_val = y[-1]
        
        if first_val > 0:
            pct_change = (last_val - first_val) / first_val
        else:
            pct_change = last_val
            
        latest_risk_score = last_val
        
        if latest_risk_score <= 1:
            latest_bucket = 'Low'
        elif latest_risk_score <= 4:
            latest_bucket = 'Medium'
        else:
            latest_bucket = 'High'
            
        results.append({
            'user_id': user_id,
            'slope': slope,
            'pct_change': pct_change,
            'latest_risk_score': latest_risk_score,
            'latest_bucket': latest_bucket,
            'n_periods_active': n_periods_active
        })
        
    return pd.DataFrame(results)

def flag_drifting_users(drift_df, slope_threshold=0.3):
    """
    Flags users whose risk is trending up but aren't already considered high risk.
    """
    if drift_df.empty:
        return pd.DataFrame()
        
    flagged = drift_df[(drift_df['slope'] > slope_threshold) & (drift_df['latest_bucket'] != 'High')]
    return flagged.sort_values(by='slope', ascending=False)

def get_interesting_users(trajectory_df, drift_df, n=6):
    """
    Selects interesting user profiles for dashboard examples.
    """
    interesting_users = []
    
    if not drift_df.empty:
        drifting = drift_df[drift_df['slope'] > 0.1].sort_values('slope', ascending=False)
        interesting_users.extend(drifting['user_id'].head(2).tolist())
        
    if not drift_df.empty:
        stable_low = drift_df[(drift_df['latest_bucket'] == 'Low') & (drift_df['slope'].abs() < 0.05)]
        stable_low = stable_low[~stable_low['user_id'].isin(interesting_users)]
        interesting_users.extend(stable_low['user_id'].head(2).tolist())
        
    spike_candidates = trajectory_df[trajectory_df['max_risk_score'] >= 5]['user_id'].unique()
    if not drift_df.empty:
        spike_users = drift_df[(drift_df['user_id'].isin(spike_candidates)) & (drift_df['slope'] < 0.2)]
        spike_users = spike_users[~spike_users['user_id'].isin(interesting_users)]
        if not spike_users.empty:
            interesting_users.append(spike_users['user_id'].iloc[0])
            
    if not drift_df.empty:
        high_risk = drift_df[(drift_df['latest_bucket'] == 'High') & (drift_df['slope'].abs() < 0.2)]
        high_risk = high_risk[~high_risk['user_id'].isin(interesting_users)]
        if not high_risk.empty:
            interesting_users.append(high_risk['user_id'].iloc[0])
            
    all_users = drift_df['user_id'].unique() if not drift_df.empty else trajectory_df['user_id'].unique()
    for u in all_users:
        if len(interesting_users) >= n:
            break
        if u not in interesting_users:
            interesting_users.append(u)
            
    return interesting_users[:n]
