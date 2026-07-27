import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .utils import get_csv_path, get_db_path, get_connection

def load_kaggle_data(csv_path=None):
    """
    Loads the Kaggle Credit Card Fraud dataset and enriches it with new features.
    """
    if csv_path is None:
        csv_path = get_csv_path()
        
    print(f"Loading Kaggle data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found. Returning empty dataframe for testing.")
        return pd.DataFrame(columns=['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)])

    # Add temporal features based on the 'Time' column (seconds since first transaction)
    df['hour'] = (df['Time'] % 86400) // 3600
    df['hour'] = df['hour'].astype(int)
    
    df['minute_of_day'] = (df['Time'] % 86400) // 60
    df['minute_of_day'] = df['minute_of_day'].astype(int)
    
    # Assign synthetic user IDs — the Kaggle dataset doesn't include real user IDs,
    # so we create 5000 synthetic users (~57 transactions each) to demonstrate
    # per-user analysis capabilities (velocity detection, risk trajectories).
    # With real user IDs, velocity and drift signals would be significantly stronger.
    # The signals that drive most of our detection power (category_rarity,
    # amount_deviation, time_anomaly) don't depend on user identity.
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(df))
    df['user_id'] = (shuffled_indices % 5000) + 1
    
    return df

def generate_upi_data(n_rows=1500):
    """
    Generates a synthetic dataset of UPI transaction failures reflecting realistic scenarios.
    Note: This is a synthetic dataset designed to demonstrate domain knowledge
    about Indian payment systems. Distribution parameters are informed by
    RBI's published UPI failure category data and industry benchmarks.
    """
    print(f"Generating {n_rows} synthetic UPI failure records...")
    np.random.seed(42)
    
    user_ids = [f"UPI_{i:03d}" for i in range(1, 201)]
    
    # Distribution of common failure types
    categories = [
        'timeout', 'insufficient_balance', 'bank_server_down', 
        'wrong_vpa', 'daily_limit_breach', 'account_frozen'
    ]
    probs = [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]
    
    failure_category = np.random.choice(categories, size=n_rows, p=probs)
    amounts = np.random.uniform(50, 50000, size=n_rows).round(2)
    users = np.random.choice(user_ids, size=n_rows)
    
    # Distribute events over the last 90 days
    base_time = datetime.now()
    time_offsets = np.random.uniform(0, 90 * 24 * 3600, size=n_rows)
    timestamps = [base_time - timedelta(seconds=float(offset)) for offset in time_offsets]
    
    df = pd.DataFrame({
        'transaction_id': [f"TXN_UPI_{i}" for i in range(10000, 10000 + n_rows)],
        'user_id': users,
        'amount': amounts,
        'failure_category': failure_category,
        'timestamp': timestamps
    })
    
    # Model realistic resolution times and outcomes based on the failure category
    def get_resolution_metrics(cat):
        if cat == 'bank_server_down':
            res_hours = max(0.5, np.random.normal(48, 12)) 
            resolved = np.random.random() < 0.75         
        elif cat == 'wrong_vpa':
            res_hours = max(0.1, np.random.normal(2, 1))
            resolved = np.random.random() < 0.95
        elif cat == 'timeout':
            res_hours = max(0.2, np.random.normal(4, 2))
            resolved = np.random.random() < 0.90
        elif cat == 'insufficient_balance':
            res_hours = max(0.1, np.random.normal(1, 0.5))
            resolved = True 
        elif cat == 'account_frozen':
            res_hours = max(24, np.random.normal(72, 24))
            resolved = np.random.random() < 0.85
        else: # daily_limit_breach
            res_hours = max(0.1, np.random.normal(1, 0.5))
            resolved = True
            
        disputed = False
        if not resolved:
            disputed = np.random.random() < 0.30 
            
        return pd.Series([round(res_hours, 2), int(resolved), int(disputed)])
        
    df[['resolution_time_hours', 'resolved_flag', 'dispute_filed']] = df['failure_category'].apply(get_resolution_metrics)
    
    return df

def init_database(csv_path=None, db_path=None):
    """
    Initializes the SQLite database with Kaggle credit card data and synthetic UPI data.
    """
    if db_path is None:
        db_path = get_db_path()
        
    print(f"Initializing database at: {db_path}")
    conn = get_connection(db_path)
    
    df_cc = load_kaggle_data(csv_path)
    if not df_cc.empty:
        df_cc.to_sql('transactions', conn, if_exists='replace', index=False)
        print(f"Saved {len(df_cc)} credit card transactions to database.")
    
    df_upi = generate_upi_data()
    df_upi.to_sql('upi_failures', conn, if_exists='replace', index=False)
    print(f"Saved {len(df_upi)} UPI failure records to database.")
    
    conn.close()
    return db_path
