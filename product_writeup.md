# RiskLens - Product Analysis

## The Problem

Fraud detection systems typically work as binary classifiers: flag or don't flag. But the real product question isn't "is this fraud?" - it's "what's the cost of being wrong, and in which direction?"

Every false positive creates customer friction - a blocked card, a failed payment, a support call. Every false negative is fraud loss. The optimal threshold depends on the business context: a premium card issuer (Amex) has different friction tolerance than a UPI payment platform (Navi).

## What I Found

### Fraud Patterns (from 284,807 real European transactions)

- **Time matters:** Fraud rate during hours 0-3 is 0.48% - roughly 4x the daytime rate of ~0.13%. This aligns with card-not-present fraud peaking when cardholders are asleep and can't notice alerts.
- **Small amounts are suspicious:** Median fraud transaction is $9.25 vs $22.00 for legit. This matches the "card testing" pattern - stolen cards get validated with small purchases before larger ones.
- **Large amounts carry high risk:** Transactions over $500 have a 0.37% fraud rate - more than 2x the overall 0.17%.
- **Half of all fraud is under $10:** The $0-10 bucket contains 249 out of 492 total fraud cases (50.6%).

### The Scoring Engine

I built a 5-signal rule-based scoring engine. No ML - every flag has a clear, explainable reason.

| Signal | What It Checks | Weight | Why This Signal |
|--------|---------------|--------|-----------------|
| Amount deviation | Amount <= $2 (test charge) OR >= $500 (high-risk bucket) | 2 | Data shows both extremes carry elevated fraud risk |
| Category rarity | PCA feature V14 < -5 (strongest anomaly separator) | 3 | V14 has the best statistical separation between fraud/legit in the anonymized features |
| Time anomaly | Transaction during hours 0-7 (high-risk window) | 1 | Nighttime fraud rate is 4x daytime - a pattern I found in the data |
| Velocity | 3+ transactions by same user in 5 minutes | 2 | Fraudsters rush to use stolen cards before they're blocked |
| Round number pattern | Small round charge ($1, $5, $10) followed by large one within 2 min | 2 | Classic card-testing behavior seen in industry reports |

Results:
- **At threshold 1:** 86% recall - catches 424 out of 492 fraud cases using just these 5 rules
- **At threshold 4 (optimal):** 9.9% precision, 13.4% recall, flags only 668 transactions out of 284K
- **Medium+High risk buckets:** capture 59% of all fraud

These numbers are modest compared to ML models, and that's intentional. The point isn't to compete with XGBoost - it's to show that even simple, explainable rules capture meaningful signal. More importantly, the REAL question is where to set the threshold.

### The Tradeoff

The threshold simulator shows the business impact clearly:

| Threshold | Transactions Flagged | Fraud Caught | Precision | Net Financial Impact |
|-----------|---------------------|-------------|-----------|---------------------|
| 1 | 88,670 (31.1%) | 424 (86.2%) | 0.5% | -$1.1M (too many false positives) |
| 3 | 9,810 (3.4%) | 187 (38.0%) | 1.9% | -$51K (still negative) |
| **4** | **668 (0.23%)** | **66 (13.4%)** | **9.9%** | **+$24K (profitable)** |
| 6 | 36 (0.01%) | 2 (0.4%) | 5.6% | +$490 (barely catches anything) |

**Threshold 4 is the sweet spot.** It's the only threshold where the company makes more money from catching fraud ($33K saved) than it spends on reviewing false positives ($9K cost). For every fraud caught, about 9 legit customers are briefly inconvenienced.

### Risk Drift

30 users were flagged as "drifting toward risk" - their average risk scores were increasing across time periods even though they hadn't yet triggered a High risk alert. These are users whose behavior is gradually shifting - more late-night transactions, more unusual amounts, more anomalous patterns.

This is the early-warning angle. Rather than waiting for someone to cross a threshold, you can proactively reach out: "We noticed some unusual patterns on your account - is everything okay?" That's better customer experience than an abrupt card block.

### UPI Failure Taxonomy

From 1,500 synthetic UPI failure records:

| Category | Cases | Avg Resolution (hrs) | Resolved (%) | Disputed (%) |
|----------|-------|---------------------|-------------|-------------|
| Timeout | 533 | 4.1 | 88.6% | 2.6% |
| Insufficient balance | 354 | 1.0 | 100.0% | 0.0% |
| **Bank server down** | **233** | **49.9** | **73.4%** | **6.9%** |
| Wrong VPA | 172 | 2.0 | 94.2% | 1.7% |
| Daily limit breach | 131 | 0.9 | 100.0% | 0.0% |
| **Account frozen** | **77** | **70.4** | **79.2%** | **6.5%** |

Bank server failures are the worst category - nearly 50 hours average resolution, only 73.4% get resolved, and 6.9% become formal disputes. These create the most customer frustration because the failure isn't their fault.

## Recommendations

### For Amex (Card Issuer)

1. **Implement time-based step-up authentication** - require additional verification for transactions during hours 0-7, especially for amounts under $10 (card testing range) or over $500 (high-value risk). This targets the 4x elevated fraud window without adding friction during normal business hours.

2. **Use drift detection for premium customers** - instead of blocking a card when a single transaction looks suspicious, track whether the customer's risk profile is trending upward. Proactive outreach ("We noticed some unusual patterns - is everything okay?") preserves the premium experience that Amex cardholders expect.

3. **Set the default review threshold at 4** - this maximizes net benefit (+$24K), catching 13.4% of fraud while flagging just 0.23% of transactions. For premium/high-value cards, consider a lower threshold and absorb the higher false-positive cost as a service investment.

### For Navi (UPI / Lending)

1. **Prioritize bank_server_down resolution** - this category has the lowest resolution rate (73.4%) and highest dispute rate (6.9%). Building automated server health monitoring and instant refund processing for confirmed server failures could cut dispute volume by an estimated 30-40%.

2. **Implement front-end VPA validation** - wrong VPA errors are user errors that create unnecessary failed transactions. Real-time VPA format validation and a "recent payees" list would prevent most of these before the user even submits.

3. **Apply drift detection to lending portfolio** - the same trajectory tracking logic can monitor borrower risk profiles. Flag borrowers whose payment behavior is deteriorating (increasing late payments, smaller amounts, longer gaps) before they become NPAs. Early intervention is worth significantly more per case in lending than in fraud.
