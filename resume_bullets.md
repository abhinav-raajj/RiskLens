# Resume Bullets - RiskLens

## For Amex (Card Fraud / Product Angle)
- Built **RiskLens**, a fraud risk-decision system on **284K real transactions** — iterated from hand-weighted rules → logistic regression → PCA-boosted LR, improving recall from **34% → 70% → 88%** while maintaining explainability at each stage
- Designed **threshold tradeoff simulator** quantifying that aggressive fraud blocking (86% recall) flags **31% of all transactions** vs. optimal threshold catching **44% of fraud** while flagging only **0.32%** — framing detection as a **business cost decision**, not a classification problem
- Identified **category_rarity** as the #1 fraud signal via signal-frequency analysis: triggered in **66% of fraud** vs **0.1% of legit** transactions (862x lift) — validated with regularized LR showing **67% relative feature importance**

## For Navi (UPI / Lending Angle)
- Built **RiskLens**, an interactive risk-decision system with **UPI failure taxonomy** covering 1,500 transactions across 6 failure categories — identified **bank_server_down** as highest-impact category (**73% resolved**, **7% dispute rate**, **~50hr avg resolution**)
- Developed **risk drift detection** flagging **30 users** whose risk profiles were accelerating before traditional scoring caught them — directly applicable to lending portfolio monitoring for **early NPA detection**
- Prototyped **AI triage layer** (Google Gemini) categorizing customer complaints into fraud/technical_failure/user_error with draft responses, reducing manual triage time
