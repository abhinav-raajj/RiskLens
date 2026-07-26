# RiskLens - Interview Prep Guide

## The 60-Second Pitch (When They Ask "Tell Me About a Project")

> "I built a project called RiskLens - it's an interactive risk-decision system. I used the Kaggle credit card fraud dataset, which has about 284,000 real European bank transactions with actual fraud labels.
>
> Most people just throw this into an ML model and report accuracy. I took a different approach - I built a rule-based scoring engine with 5 explainable signals and focused on the PRODUCT question: where should you set the threshold? Because every false positive is a blocked card and an angry customer, and every false negative is fraud loss.
>
> I built a threshold simulator that lets you slide between 'catch more fraud' and 'reduce customer friction' and see the business impact in real-time. At the optimal threshold, the system generates a net benefit of $24,000."

**Keep it under 60 seconds. Then STOP and let them ask questions.**

---

## Common Questions and How to Answer Them

### Q: "Why didn't you use Machine Learning?"

> "Two reasons. First, this is PCA-anonymized data - the features are already transformed, so an ML model would just be fitting to mathematical artifacts, not learning real patterns. Second, and more importantly, explainability matters in fintech. When a customer calls and asks 'why was my card blocked?', I can say 'you made 3 transactions in 2 minutes at 3 AM from an unusual merchant type.' An ML model can't give that answer. For a product team, knowing WHY something was flagged is more valuable than a marginally higher accuracy number."

### Q: "Walk me through your scoring logic"

> "I check 5 things for every transaction:
> 1. **Amount deviation** - is this a tiny test charge under $2, or a large purchase over $500? Both carry elevated fraud risk in the data.
> 2. **Category rarity** - using the PCA features, does this transaction look statistically unusual? I used the V14 component which has the strongest separation between fraud and legitimate transactions.
> 3. **Time anomaly** - is this happening between midnight and 7 AM? I found that fraud rate during those hours is 0.48%, which is 4 times the daytime rate.
> 4. **Velocity** - did this user make 3 or more transactions within 5 minutes? Fraudsters rush to use stolen cards before they're cancelled.
> 5. **Round number pattern** - is this a small round charge like $1 or $5 followed immediately by a big purchase? That's a classic card-testing pattern.
>
> Each signal gets a weight based on importance, and they add up to a risk score. I tested 6 different weight combinations and picked the one that best separates fraud from legitimate transactions."

### Q: "What were your key findings from the data?"

> "Three things stood out:
> 1. **Fraud concentrates at night** - 0.48% fraud rate during hours 0-3, versus 0.13% during the day. That's a 4x difference. This makes sense because card-not-present fraud peaks when the cardholder is asleep.
> 2. **Median fraud is $9.25** versus $22 for legitimate transactions. Half of all fraud is under $10. This confirms the card-testing hypothesis - fraudsters validate stolen cards with small amounts first.
> 3. **The $500+ bucket has the highest fraud rate** at 0.37%, more than double the overall rate. So both very small and very large transactions carry elevated risk."

### Q: "What's this threshold simulator?"

> "This is the part I'm most proud of. Most fraud systems just give you a binary flag. But the real product question is: how aggressive should you be?
>
> If you set the threshold low, you catch 86% of fraud - but you're also flagging 88,000 legitimate transactions. That's a terrible customer experience.
>
> If you set it high, you barely flag anyone - but fraud slips through.
>
> The simulator shows that threshold 4 is the sweet spot. You flag only 668 transactions out of 284,000 - that's 0.23% - and you catch 66 fraud cases. The net financial impact is positive $24,000 because the fraud you prevent is worth more than the cost of reviewing false positives.
>
> For a company like Amex, you might want to be more aggressive with premium cards and absorb more false positives. For a high-volume platform like Navi, you'd probably optimize for minimum friction. The slider lets the product team make that call."

### Q: "What's the drift detection about?"

> "Instead of scoring each transaction independently, I track how a user's risk score changes over time. I divided the data into 8 time periods and computed the average risk score per user per period.
>
> Then I ran linear regression on each user's scores to find the slope - are they trending up, down, or stable?
>
> I found 30 users whose scores were steadily increasing but hadn't yet reached the High risk threshold. These are users you'd normally miss because no single transaction looks bad enough to flag. But looking at the trajectory, something is changing.
>
> For a product team, this means proactive outreach: 'We noticed some unusual patterns on your account - is everything okay?' That's much better than suddenly blocking someone's card."

### Q: "Tell me about the UPI failure analysis"

(Use this for Navi specifically)

> "I created a synthetic dataset of 1,500 UPI payment failures across 6 categories because I wanted to show domain thinking beyond the Kaggle dataset.
>
> The key finding: bank server failures are the worst category. They take an average of 50 hours to resolve, only 73% actually get resolved, and 7% turn into formal disputes. And the customer didn't do anything wrong - the bank's server went down.
>
> My recommendation would be to build automated refund processing for confirmed server failures. If the bank's health monitoring confirms the server was down during the transaction window, issue an instant refund instead of making the customer wait 50 hours and file a dispute. That alone could cut dispute volume significantly."

### Q: "What would you improve with more time?"

> "Three things:
> 1. **Real merchant categories** - the PCA features limit what I can do. With actual merchant codes, I could build much stronger signals around category-specific fraud patterns.
> 2. **Network analysis** - I'd look at connections between users. If User A is flagged as fraud and shares a merchant with User B, that's a signal.
> 3. **A/B testing framework** - right now the threshold simulator is analytical. In production, you'd want to A/B test different thresholds on real traffic and measure the impact on both fraud loss and customer satisfaction (NPS, churn)."

---

## Numbers to Memorize (Flash Card Style)

| Question | Answer |
|----------|--------|
| Total transactions? | 284,807 |
| Fraud cases? | 492 (0.17%) |
| Recall at threshold 1? | 86% - catches 424 out of 492 |
| Best precision? | 9.9% at threshold 4 |
| Net benefit? | +$24,000 at threshold 4 |
| Peak fraud hours? | 0-3 AM, 0.48% rate (4x daytime) |
| Median fraud amount? | $9.25 |
| Median legit amount? | $22.00 |
| Highest fraud rate bucket? | $500+ at 0.37% |
| Drifting users? | 30 flagged |
| Worst UPI category? | bank_server_down (73% resolved, 7% disputes) |
| Risk signals? | 5 (amount, category, time, velocity, round number) |

---

## Body Language Tips

- When showing the dashboard, **let THEM play with the slider**. Hand them the mouse.
- When they challenge a number, don't get defensive. Say "That's a fair point" and explain.
- If they ask something you don't know, say "I haven't explored that yet, but here's how I'd approach it..."
- Show genuine curiosity - "I was actually surprised that median fraud was only $9.25. I expected higher."

---

## Red Flags to Avoid

- DON'T say "AI built this" or "I used a tool"
- DON'T memorize answers word-for-word - use the structure above but say it in YOUR words
- DON'T oversell the precision - say "rule-based has limits, and I chose explainability over accuracy"
- DON'T skip the UPI section for Navi - that's your domain knowledge differentiator
