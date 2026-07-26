import pandas as pd

def fraud_by_time_bucket(conn):
    """
    Analyzes fraud frequency across 4-hour time blocks.
    """
    query = """
    WITH Buckets AS (
        SELECT 
            CASE 
                WHEN hour >= 0 AND hour < 4 THEN '0-3'
                WHEN hour >= 4 AND hour < 8 THEN '4-7'
                WHEN hour >= 8 AND hour < 12 THEN '8-11'
                WHEN hour >= 12 AND hour < 16 THEN '12-15'
                WHEN hour >= 16 AND hour < 20 THEN '16-19'
                ELSE '20-23'
            END as time_bucket,
            Class
        FROM transactions
    )
    SELECT 
        time_bucket,
        COUNT(*) as total_txns,
        SUM(Class) as fraud_txns,
        ROUND(CAST(SUM(Class) AS FLOAT) / COUNT(*) * 100, 4) as fraud_rate
    FROM Buckets
    GROUP BY time_bucket
    ORDER BY time_bucket;
    """
    return pd.read_sql_query(query, conn)

def fraud_by_amount_bucket(conn):
    """
    Analyzes fraud distribution across transaction amount ranges, and also computes medians.
    """
    bucket_query = """
    WITH Buckets AS (
        SELECT 
            CASE 
                WHEN Amount >= 0 AND Amount < 10 THEN '0-10'
                WHEN Amount >= 10 AND Amount < 50 THEN '10-50'
                WHEN Amount >= 50 AND Amount < 100 THEN '50-100'
                WHEN Amount >= 100 AND Amount < 500 THEN '100-500'
                ELSE '500+'
            END as amount_bucket,
            Class
        FROM transactions
    )
    SELECT 
        amount_bucket,
        COUNT(*) as total_txns,
        SUM(Class) as fraud_txns,
        ROUND(CAST(SUM(Class) AS FLOAT) / COUNT(*) * 100, 4) as fraud_rate
    FROM Buckets
    GROUP BY amount_bucket
    ORDER BY 
        CASE amount_bucket
            WHEN '0-10' THEN 1
            WHEN '10-50' THEN 2
            WHEN '50-100' THEN 3
            WHEN '100-500' THEN 4
            ELSE 5
        END;
    """
    buckets_df = pd.read_sql_query(bucket_query, conn)
    
    # Compute median by using a window function approach since SQLite lacks MEDIAN()
    median_query = """
    WITH OrderedData AS (
        SELECT 
            Class, 
            Amount,
            ROW_NUMBER() OVER (PARTITION BY Class ORDER BY Amount) as rn,
            COUNT(*) OVER (PARTITION BY Class) as cnt
        FROM transactions
    )
    SELECT 
        CASE WHEN Class = 1 THEN 'Fraud' ELSE 'Legit' END as txn_type,
        AVG(Amount) as median_amount
    FROM OrderedData
    WHERE rn IN (cnt/2, (cnt/2) + 1)
    GROUP BY Class;
    """
    medians_df = pd.read_sql_query(median_query, conn)
    
    return buckets_df, medians_df

def lag_spike_detection(conn):
    """
    Detects sudden spikes where a transaction amount is > 5x the user's previous transaction.
    """
    query = """
    WITH UserTxns AS (
        SELECT 
            user_id,
            Time,
            Amount,
            Class,
            LAG(Amount) OVER (PARTITION BY user_id ORDER BY Time) as previous_amount
        FROM transactions
    )
    SELECT 
        user_id,
        Time,
        Amount as current_amount,
        previous_amount,
        Class
    FROM UserTxns
    WHERE previous_amount > 0 
      AND Amount > 5 * previous_amount
    ORDER BY (Amount / previous_amount) DESC
    LIMIT 50;
    """
    return pd.read_sql_query(query, conn)

def rank_users_by_fraud(conn):
    """
    Ranks the top users by their absolute number of fraudulent transactions.
    """
    query = """
    WITH UserFraud AS (
        SELECT 
            user_id,
            SUM(Class) as total_fraud_txns
        FROM transactions
        GROUP BY user_id
        HAVING SUM(Class) > 0
    ),
    RankedUsers AS (
        SELECT 
            user_id,
            total_fraud_txns,
            DENSE_RANK() OVER (ORDER BY total_fraud_txns DESC) as rank
        FROM UserFraud
    )
    SELECT * 
    FROM RankedUsers 
    WHERE rank <= 20
    ORDER BY rank;
    """
    return pd.read_sql_query(query, conn)

def upi_failure_analysis(conn):
    """
    Examines UPI failures, their typical resolution times, and dispute frequencies.
    """
    query = """
    SELECT 
        failure_category,
        COUNT(*) as total_cases,
        ROUND(AVG(resolution_time_hours), 2) as avg_resolution_time_hours,
        ROUND(CAST(SUM(resolved_flag) AS FLOAT) / COUNT(*) * 100, 2) as pct_resolved,
        ROUND(CAST(SUM(dispute_filed) AS FLOAT) / COUNT(*) * 100, 2) as pct_disputed
    FROM upi_failures
    GROUP BY failure_category
    ORDER BY total_cases DESC;
    """
    return pd.read_sql_query(query, conn)

def precision_validation(conn, risk_table='scored_transactions'):
    """
    Validates scoring against actual outcomes to track model performance.
    """
    query = f"""
    WITH Stats AS (
        SELECT 
            risk_bucket,
            SUM(CASE WHEN predicted_fraud = 1 AND actual_class = 1 THEN 1 ELSE 0 END) as TP,
            SUM(CASE WHEN predicted_fraud = 1 AND actual_class = 0 THEN 1 ELSE 0 END) as FP,
            SUM(CASE WHEN predicted_fraud = 0 AND actual_class = 1 THEN 1 ELSE 0 END) as FN,
            SUM(CASE WHEN predicted_fraud = 0 AND actual_class = 0 THEN 1 ELSE 0 END) as TN
        FROM {risk_table}
        GROUP BY risk_bucket
    )
    SELECT 
        risk_bucket,
        TP, FP, FN, TN,
        ROUND(CAST(TP AS FLOAT) / NULLIF(TP + FP, 0), 4) as precision,
        ROUND(CAST(TP AS FLOAT) / NULLIF(TP + FN, 0), 4) as recall
    FROM Stats
    ORDER BY 
        CASE risk_bucket
            WHEN 'High' THEN 1
            WHEN 'Medium' THEN 2
            WHEN 'Low' THEN 3
            ELSE 4
        END;
    """
    try:
        return pd.read_sql_query(query, conn)
    except pd.io.sql.DatabaseError:
        print(f"Note: Table '{risk_table}' not found. Skipping validation.")
        return pd.DataFrame()

def run_all_queries(conn):
    """
    Executes the analytical pipeline and prints top-level findings.
    """
    results = {}
    print("\n--- Running RiskLens Analytics ---")
    
    results['time_bucket'] = fraud_by_time_bucket(conn)
    if not results['time_bucket'].empty:
        worst_time = results['time_bucket'].sort_values('fraud_rate', ascending=False).iloc[0]
        print(f"Insight: {worst_time['fraud_rate']}% of fraud occurs between hours {worst_time['time_bucket']}.")
    
    buckets_df, medians_df = fraud_by_amount_bucket(conn)
    results['amount_bucket'] = buckets_df
    results['medians'] = medians_df
    
    results['spikes'] = lag_spike_detection(conn)
    
    results['top_users'] = rank_users_by_fraud(conn)
    
    results['upi'] = upi_failure_analysis(conn)
    if not results['upi'].empty:
        lowest_res = results['upi'].sort_values('pct_resolved').iloc[0]
        print(f"Insight: '{lowest_res['failure_category']}' has the lowest resolution rate ({lowest_res['pct_resolved']}%).")
    
    return results
