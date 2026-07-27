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

### Model Iteration

To understand the tradeoff between explainability and accuracy, I compared three different approaches:
- **Version 1: Hand-weighted rules** (5 signals, manual weights, best: 54% precision at score≥5)
- **Version 2: Logistic regression** (same 5 signals, data-fitted weights)
- **Version 3: LR + PCA features** (full dataset)

**Key finding:** While Version 3 achieves 88% recall (p≥0.5) and Version 2 reaches 70%, they suffer from very low precision (6.7% and 4.3% respectively). Version 1 (hand-weighted rules) significantly outperforms LR at practical thresholds, delivering 54% precision at score≥5 and 24% precision at score≥4. Simple, explainable rules beat complex models when the cost of false positives is high.

### Signal Analysis

Not all signals are created equal, but they are all necessary:
- **Signal frequency:** `category_rarity` flags 66% of fraud vs 0.1% of legit (862x lift)
- `amount_deviation`: 48% fraud vs 21% legit (2.3x lift)
- `time_anomaly`: 32% fraud vs 12% legit (2.6x lift)
- **Feature importance from LR:** `category_rarity` accounts for 67% of the model weight, `amount_deviation` 16%, and `time_anomaly` 14%
- **Signal overlap:** Each signal catches unique cases the others miss, proving why a multi-signal approach (5 signals, not 1) is essential for robust detection.

### The Tradeoff

The threshold simulator shows the business impact clearly:

| Threshold | Transactions Flagged | Fraud Caught | Precision | Net Financial Impact |
|-----------|---------------------|-------------|-----------|---------------------|
| 1 | 88,670 (31.1%) | 424 (86.2%) | 0.5% | -$1.1M (too many false positives) |
| 3 | 9,810 (3.4%) | 187 (38.0%) | 1.9% | -$51K (still negative) |
| **4** | **668 (0.23%)** | **66 (13.4%)** | **9.9%** | **+$24K (profitable)** |
| 6 | 36 (0.01%) | 2 (0.4%) | 5.6% | +$490 (barely catches anything) |

**Threshold 4 is the sweet spot.** It's the only threshold where the company makes more money from catching fraud ($33K saved) than it spends on reviewing false positives ($9K cost). For every fraud caught, about 9 legit customers are briefly inconvenienced.

### Customer Segment Analysis

Different customer segments have different optimal thresholds. Analyzing by segment reveals:
- Micro-transaction users, standard users, and high-value users require different thresholds to optimize the business tradeoff.
- **Product insight:** One-size-fits-all thresholds are suboptimal. A tailored approach significantly reduces friction for good customers while catching more fraud where it counts.

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

Based on the threshold-by-context framework:

1. **Premium issuer (Amex): Threshold 3.** Minimize missed fraud. The business model can absorb the operational cost of higher false positives to protect high net-worth accounts and deliver white-glove security.
2. **High-volume platform (Navi): Threshold 5.** Minimize friction. In a high-velocity, low-margin environment, blocking legitimate transactions destroys more value than the fraud itself.
3. **Traditional bank (SBI/HDFC): Threshold 4.** A balanced approach that maximizes net financial impact while keeping operational costs manageable.

## Known Limitations

- **Synthetic user IDs:** The user tracking is a framework demo; real user behavior is more complex.
- **PCA features limit explainability:** Relying on PCA features (like V14) makes it harder to explain exactly *why* a transaction is flagged to a customer.
- **UPI data is synthetic:** The UPI dataset was generated to demonstrate domain knowledge, not extracted from a real payment network.
- **Category_rarity does most of the heavy lifting:** While overlap exists, a single signal drives a disproportionate amount of the detection capability.
