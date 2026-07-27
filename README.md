# RiskLens

An interactive risk-decision system that demonstrates product thinking: where should you set the threshold, how does the optimal strategy differ by customer segment, and what's the real business cost of being wrong?

## What This Is
This is a project I built to explore how fraud detection works in practice, focusing on the product and business decisions rather than just building a black-box ML model. It uses real anonymized credit card transaction data from Kaggle (about 284K European transactions) along with some synthetic UPI failure data I created to model a different payment context.

## Product Thinking
- This isn't a model-accuracy project — it's a decision-framework project
- Every fraud system faces the same tradeoff: catch more fraud vs block more customers
- The project answers this with real numbers, segment analysis, and stakeholder impact

## Key Features
- **Signal deep-dive with overlap analysis** (why 5 signals, not 1)
- **Customer segment analysis** (different optimal thresholds per segment)
- **Product strategy recommendations** (premium issuer vs high-volume platform)
- **Stakeholder impact matrix** (customer/ops/finance impact per threshold)
- **Rule-based risk scoring** (no ML) — fully explainable, 5 signals combined into a weighted score
- **Risk trajectory tracking** — catch users drifting toward risk BEFORE they become high-risk
- **Threshold tradeoff simulator** — slide a threshold and see real-time impact on fraud caught vs customer friction
- **SQL analysis layer** — 6 core analytical queries with window functions (LAG, DENSE_RANK)
- **UPI failure taxonomy** — synthetic Indian payments failure data with resolution tracking
- **AI triage** (Gemini API) — automated complaint categorization and response drafting

## 3-Model Iteration
- **Version 1:** Hand-weighted rules (5 signals, manual weights, best: 54% precision at score≥5)
- **Version 2:** Logistic regression on same 5 signals (data-fitted weights, 70% recall)
- **Version 3:** LR + PCA features (88% recall but loses explainability)
- **Key finding:** hand-weighted rules outperform LR at practical thresholds

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
git clone https://github.com/abhinav-raajj/RiskLens.git
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
    ├── ai_triage.py           # Gemini API complaint triage
    └── product_strategy.py    # product strategy recommendations
```

## Key Findings
- Fraud rate hours 0-3: 0.48% (4x daytime baseline of 0.13%)
- Median fraud amount: $9.25 vs $22.00 legit
- $500+ fraud rate: 0.37% (2x overall 0.17%)
- Hand-weighted rules at score≥5: 54% precision, 34% recall (305 flagged)
- Hand-weighted at score≥4: 24% precision, 44% recall (902 flagged)
- LR 5-signal at p≥0.5: 70% recall, 4.3% precision
- LR+PCA at p≥0.5: 88% recall, 6.7% precision
- Top signal: category_rarity — 66% of fraud flagged, 67% of LR model weight
- 30 users drifting toward risk
- UPI: bank_server_down has 73.4% resolution, 6.9% disputes

## Tech Stack
Python, pandas, NumPy, SQLite, Streamlit, Plotly, Google Gemini API

## Data Attribution
Primary dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by the Machine Learning Group at ULB.
Secondary: synthetic UPI failure data generated to demonstrate domain knowledge in Indian payment systems.
