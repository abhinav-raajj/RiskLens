# Resume Bullets — RiskLens (Product Intern Focus)

Use these bullet points on your resume or LinkedIn. They are written in the **CAR (Context, Action, Result)** format and highlight product thinking, business impact, and decision-making over raw coding.

## Core Product Management Bullets
*Use 2-3 of these for general PM applications.*

- **Designed an Interactive Risk-Decision System:** Built RiskLens to analyze 284K credit card transactions, creating a threshold simulator that quantified the tradeoff between fraud prevention ($ caught) and customer friction (false positives), enabling data-driven policy adjustments.
- **Conducted Customer Segment Analysis:** Segmented transactions by monetary value (Micro, Standard, High-Value) to prove that one-size-fits-all policies fail; demonstrated that dynamic thresholds prevent over-blocking on small purchases while maximizing ROI on $500+ transactions.
- **Evaluated Model Complexity vs. Business Value:** Iterated from explainable hand-weighted rules to Logistic Regression, discovering that simpler rules optimized for precision (54% at operational flag rates) outperformed LR; recommended rules for auto-blocking and LR for manual review queueing. 
- **Analyzed Signal Redundancy for Defense-in-Depth:** Performed overlap analysis on 5 fraud signals to prove that while the primary signal caught 66% of fraud, the remaining signals caught distinct edge cases, justifying the cost of maintaining a multi-signal engine.

## Specialized for Amex (Premium Card Issuer Focus)
*Swap these in when applying to Amex or premium financial services.*

- **Optimized for Premium Customer Experience:** Modeled stakeholder impact matrices for premium cardholders, demonstrating that the brand cost of false positives (card declines) outweighs minor fraud losses, justifying a lower-friction threshold policy.
- **Prototyped AI Customer Support Triage:** Built and evaluated an automated triage layer using the Google Gemini API to categorize customer complaints (fraud vs. technical failure), complete with accuracy measurement against ground-truth labels.

## Specialized for Navi (High-Volume UPI / Lending Focus)
*Swap these in when applying to Navi, PhonePe, or high-volume fintechs.*

- **Modeled High-Volume Operations Impact:** Showcased how a 0.1% false-positive rate at scale generates massive support loads; designed recommendation engine suggesting higher auto-block thresholds supplemented by post-transaction ML monitoring for high-volume platforms.
- **Mapped UPI Failure Taxonomy:** Generated a synthetic UPI failure dataset mirroring RBI benchmarks to track resolution times and dispute rates; identified `bank_server_down` as the highest-friction category (73% resolution) to prioritize engineering fixes and automated refunds.
- **Developed Risk Trajectory Tracking:** Built drift detection logic to flag users whose risk profiles were accelerating across time periods before reaching critical thresholds — a framework directly applicable to early NPA (Non-Performing Asset) detection in lending portfolios.
