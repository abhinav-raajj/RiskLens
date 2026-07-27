# RiskLens - Interview Prep Guide

## The 60-Second Pitch (When They Ask "Tell Me About a Project")

> "I built RiskLens — an interactive risk-decision system using 284,000 real credit card transactions from Kaggle.
>
> Instead of throwing this into a black-box model and reporting accuracy, I took an iterative approach. I started with 5 explainable rule-based signals, then fitted logistic regression to learn the optimal weights from data, and finally tested adding PCA features as a 'boost' to see the interpretability-vs-accuracy tradeoff.
>
> The real product insight isn't the model — it's the threshold. I built a simulator that shows the business cost of being wrong in each direction: every false positive is a blocked card, every false negative is fraud loss. At the optimal threshold, the system generates a net benefit of ~$24K."

**Keep it under 60 seconds. Then STOP and let them ask questions.**

---

## Common Questions and How to Answer Them

### Q: "Why didn't you just use XGBoost/Random Forest?"

> "I wanted to focus on the product question, not the ML question. The data is PCA-anonymized — V1 through V28 have no business meaning. An ML model would get great accuracy, but when a customer calls asking 'why was my card blocked?', you can't say 'because V14 was -7.2.'
>
> I started with fully explainable rules — 5 signals where each flag has a plain-English reason. Then I fitted logistic regression on those same signals to let the data pick the weights instead of me guessing. That's a middle ground: still explainable (I can rank which signals matter most), but data-fitted instead of hand-tuned.
>
> Finally I tested adding PCA features back in as a boosted model. Recall jumped from 70% to 88% — but now I can't explain *why* V14 matters in business terms. That's a real product tradeoff, and being able to articulate it is more valuable than just reporting the best F1 score."

### Q: "Walk me through the iteration"

> "**Version 1: Hand-weighted rules.** I assigned weights manually — amount_deviation=2, category_rarity=3, etc. — tested 6 combinations and picked the best. At the optimal threshold, 54% precision and 34% recall. Decent, but I was essentially guessing relative importance.
>
> **Version 2: Logistic regression on the same 5 signals.** Instead of guessing weights, I let the data learn them. The model confirmed category_rarity (unusual transaction patterns in V14) is the strongest signal — 67% of the model's explanatory power — followed by amount_deviation at 16% and time_anomaly at 14%. Recall improved to 70% at a ~3% flag rate.
>
> **Version 3: LR + PCA features (V14, V12, V17, V10).** Adding raw PCA components back in pushed recall to 88% at a 2.3% flag rate. The tradeoff: those PCA features aren't explainable to a customer, so this model would need a human-in-the-loop review process rather than auto-blocking.
>
> The point of showing all three isn't to say 'Version 3 is best' — it's to show how each iteration changes the product requirements. Version 1 can auto-block with a clear reason. Version 3 needs analyst review. That's a staffing and process decision, not just a model decision."

### Q: "How did you validate the feature importance?"

> "I deliberately *avoided* reporting odds ratios, even though logistic regression gives them. Here's why: one of my signals — category_rarity, based on V14 < -5 — nearly perfectly separates fraud from legit in this dataset. That causes quasi-complete separation, which means the LR coefficient gets pushed toward infinity. The raw odds ratio comes out as something absurd like 2,000x, which isn't a stable or meaningful number.
>
> Instead, I used two stable metrics:
>
> **Signal frequency analysis** — what percentage of fraud vs legit transactions trigger each flag:
> - Category rarity: triggered in 66% of fraud vs 0.1% of legit (862x lift)
> - Amount deviation: 48% of fraud vs 21% of legit (2.3x lift)
> - Time anomaly: 32% of fraud vs 12% of legit (2.6x lift)
>
> **Relative importance** — normalizing the absolute LR coefficients to sum to 100%:
> - Category rarity: 67%, Amount deviation: 16%, Time anomaly: 14%
>
> These tell the same story without relying on unstable coefficient estimates."

### Q: "What were your key data findings?"

> "Three things stood out:
> 1. **Fraud concentrates at night** — 0.48% fraud rate during hours 0-3, versus 0.13% during the day. That's a 4x difference. This makes sense because card-not-present fraud peaks when the cardholder is asleep.
> 2. **Median fraud is $9.25** versus $22 for legitimate transactions. Half of all fraud is under $10. This confirms the card-testing hypothesis — fraudsters validate stolen cards with small amounts first.
> 3. **The $500+ bucket has the highest fraud rate** at 0.37%, more than double the overall rate. So both very small and very large transactions carry elevated risk."

### Q: "What's the threshold simulator?"

> "This is the part I'm most proud of. Most fraud systems just give you a binary flag. But the real product question is: how aggressive should you be?
>
> If you set the threshold low, you catch 86% of fraud — but you're also flagging 88,000 legitimate transactions. That's a terrible customer experience.
>
> If you set it high, you barely flag anyone — but fraud slips through.
>
> The simulator shows that threshold 4 is the sweet spot for hand-weighted rules. You flag only 902 transactions out of 284,000 — that's 0.32% — with 24% precision and 44% recall. The net financial impact is positive because the fraud you prevent is worth more than the cost of reviewing false positives.
>
> For Amex, you might want a lower threshold and absorb more false positives for premium cards. For a high-volume platform like Navi, you'd optimize for minimum friction. The slider lets the product team make that call."

### Q: "Tell me about the UPI failure analysis"

(Use this for Navi specifically)

> "I created a synthetic dataset of 1,500 UPI payment failures across 6 categories because I wanted to show domain thinking beyond the Kaggle dataset.
>
> The key finding: bank server failures are the worst category. They take an average of 50 hours to resolve, only 73% actually get resolved, and 7% turn into formal disputes. And the customer didn't do anything wrong — the bank's server went down.
>
> My recommendation would be to build automated refund processing for confirmed server failures. If the bank's health monitoring confirms the server was down during the transaction window, issue an instant refund instead of making the customer wait 50 hours and file a dispute. That alone could cut dispute volume significantly."

### Q: "What would you improve with more time?"

> "Three things:
> 1. **Real merchant categories** — the PCA features limit what I can do. With actual merchant codes, I could build much stronger signals around category-specific fraud patterns.
> 2. **Network analysis** — I'd look at connections between users. If User A is flagged as fraud and shares a merchant with User B, that's a signal.
> 3. **A/B testing framework** — right now the threshold simulator is analytical. In production, you'd want to A/B test different thresholds on real traffic and measure the impact on both fraud loss and customer satisfaction (NPS, churn)."

---

## Numbers to Memorize (Flash Card Style)

| Question | Answer |
|----------|--------|
| Total transactions? | 284,807 |
| Fraud cases? | 492 (0.17%) |
| Hand-weighted best precision? | 54% at threshold 5 (score≥5) |
| Hand-weighted best recall? | 86% at threshold 1 |
| LR (5 signals) recall? | 70% at 3% flag rate |
| LR + PCA recall? | 88% at 2.3% flag rate |
| #1 signal by importance? | Category rarity — 67% of model weight, 66% of fraud flagged vs 0.1% legit |
| #2 signal? | Amount deviation — 16% of model weight, 48% fraud vs 21% legit |
| #3 signal? | Time anomaly — 14% of model weight, 32% fraud vs 12% legit |
| Peak fraud hours? | 0-3 AM, 0.48% rate (4x daytime) |
| Median fraud amount? | $9.25 |
| Median legit amount? | $22.00 |
| Drifting users? | 30 flagged |
| Worst UPI category? | bank_server_down (73% resolved, 7% disputes, ~50hr avg resolution) |

---

## Traps to Watch For

### If they ask about odds ratios:
> "I computed them but chose not to report them. Category_rarity causes quasi-complete separation in the data — 66% of fraud triggers it but only 0.1% of legit — which makes the LR coefficient unstable. The odds ratio comes out as 2,000x+, which isn't meaningful. I report signal frequency and relative importance instead — they tell the same story without the numerical instability."

### If they ask about the 862x lift on category_rarity:
> "That's the raw frequency ratio — 66% of fraud flagged vs 0.1% of legit. It's real, but it's also a reflection of how V14 was constructed in the PCA. In production with real merchant data, you'd expect a weaker but broader signal. The important thing is the *ranking* — this is our strongest signal, followed by amount and time — not the exact multiple."

### If they challenge precision numbers:
> "Rule-based scoring on PCA-anonymized data will always have modest precision — we're working with proxy signals, not real merchant categories or device fingerprints. The point isn't to compete with a production ML pipeline; it's to demonstrate the decision framework: even with a perfect model, WHERE do you set the threshold? The simulator and cost analysis work regardless of the model quality."

---

## Body Language Tips

- When showing the dashboard, **let THEM play with the slider**. Hand them the mouse.
- When they challenge a number, don't get defensive. Say "That's a fair point" and explain.
- If they ask something you don't know, say "I haven't explored that yet, but here's how I'd approach it..."
- Show genuine curiosity — "I was actually surprised that median fraud was only $9.25. I expected higher."

---

## Red Flags to Avoid

- DON'T say "AI built this" or "I used a tool"
- DON'T memorize answers word-for-word — use the structure above but say it in YOUR words
- DON'T cite odds ratios or unstable coefficient numbers — use signal frequency and relative importance
- DON'T oversell the precision — say "rule-based has limits, and I chose explainability over accuracy"
- DON'T skip the UPI section for Navi — that's your domain knowledge differentiator
