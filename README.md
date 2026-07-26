# RiskLens

An interactive risk-decision system that tracks customers drifting toward risk over time and simulates the business tradeoff of different intervention policies.

## What This Is
This is a project I built to explore how fraud detection works in practice, focusing on the product and business decisions rather than just building a black-box ML model. It uses real anonymized credit card transaction data from Kaggle (about 284K European transactions) along with some synthetic UPI failure data I created to model a different payment context.

## Key Features
- **Rule-based risk scoring** (no ML) — fully explainable, 5 signals combined into a weighted score
- **Risk trajectory tracking** — catch users drifting toward risk BEFORE they become high-risk
- **Threshold tradeoff simulator** — slide a threshold and see real-time impact on fraud caught vs customer friction
- **SQL analysis layer** — 6 core analytical queries with window functions (LAG, DENSE_RANK)
- **UPI failure taxonomy** — synthetic Indian payments failure data with resolution tracking
- **AI triage** (Gemini API) — automated complaint categorization and response drafting

## Architecture
```text
creditcard.csv (Kaggle) --> data_loader.py --> SQLite DB --> sql_queries.py
                                                  |               |
                                                  v               v
                                            risk_engine.py    SQL Insights
                                                  |
                                    +-------------+-------------+
                                    |             |             |
                              drift_detector  threshold_sim  ai_triage
                                    |             |             |
                                    +-------------+-------------+
                                                  |
                                               app.py (Streamlit Dashboard)
```

## Setup
```bash
git clone https://github.com/yourusername/risklens.git
cd risklens
pip install -r requirements.txt

# Download dataset from Kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
# Or manually download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# Run the dashboard
streamlit run app.py
```

## Project Structure
```
risklens/
├── README.md
├── product_writeup.md         # one-page product analysis
├── resume_bullets.md          # resume-ready bullet points
├── requirements.txt
├── app.py                     # Streamlit dashboard
├── test_pipeline.py           # end-to-end smoke test
├── data/
│   ├── creditcard.csv         # Kaggle dataset (not in repo, download separately)
│   └── risklens.db            # SQLite database (auto-generated)
└── src/
    ├── __init__.py
    ├── utils.py               # shared helpers
    ├── data_loader.py         # CSV loading + synthetic UPI generation
    ├── sql_queries.py         # 6 analytical SQL queries
    ├── risk_engine.py         # 5-signal risk scoring + weight tuning
    ├── drift_detector.py      # risk trajectory + drift flagging
    ├── threshold_simulator.py # tradeoff calculator
    └── ai_triage.py           # Gemini API complaint triage
```

## Key Findings
- Fraud concentrates during hours 0-3 at 0.48% rate — roughly 4x the daytime baseline of 0.13%
- Median fraud amount: $9.25 (vs $22.00 for legit transactions)
- Transactions over $500 have the highest fraud rate at 0.37%, more than 2x the overall 0.17%
- At threshold 1: 86% recall — catches 424 out of 492 fraud cases with just 5 explainable rules
- At optimal threshold (4): 9.9% precision, 13.4% recall, +$24K net benefit — flags only 0.23% of transactions
- Medium+High risk buckets capture 59% of all fraud
- 30 users flagged as drifting toward risk before reaching High status
- Bank server failures have the lowest UPI resolution rate (73.4%) and highest dispute rate (6.9%)

## Tech Stack
Python, pandas, NumPy, SQLite, Streamlit, Plotly, Google Gemini API

## Data Attribution
Primary dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by the Machine Learning Group at ULB.
Secondary: synthetic UPI failure data generated to demonstrate domain knowledge in Indian payment systems.
