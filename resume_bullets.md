# Resume Bullets - RiskLens

## For Amex (Card Fraud / Product Angle)
- Built RiskLens, a rule-based risk scoring system analyzing 284K real credit card transactions — 5 explainable signals, no ML, achieving 86% recall and 9.9% precision at optimal thresholds
- Designed a threshold tradeoff simulator showing that threshold=4 yields +$24K net benefit by catching 13% of fraud while flagging just 0.23% of all transactions — framing fraud detection as a business decision, not a classification problem
- Developed risk trajectory tracking that flagged 30 users drifting toward high risk before traditional scoring caught them — enabling proactive outreach instead of reactive blocking
- Wrote SQL analytical layer (LAG, DENSE_RANK) uncovering that fraud rate during hours 0-3 is 0.48% (4x daytime rate) and median fraud amount is $9.25 vs $22 legit

## For Navi (UPI / Lending Angle)
- Built RiskLens, an interactive risk-decision system with a UPI failure taxonomy covering 1,500 transactions across 6 failure categories with resolution time and dispute tracking
- Identified that bank_server_down failures have the lowest resolution rate (73.4%) and highest dispute rate (6.9%) — recommended automated server-side resolution to cut support ticket volume
- Prototyped an AI triage layer (Google Gemini) categorizing 20 customer complaints into fraud/technical_failure/user_error with draft first-responses, reducing manual triage time
- Created drift detection logic identifying users whose risk profiles are accelerating, directly applicable to lending portfolio monitoring for early NPA detection
