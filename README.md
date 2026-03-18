# Petrol Attendants — January Performance Analysis

Statistical analysis of 9 petrol station attendants' January sales performance. Covers descriptive statistics, correlation analysis, OLS regression, Z-score outlier detection, Pareto analysis, and a 9-panel visualisation dashboard.

---

## Overview

This script takes raw sales (ZAR) and transaction count data for a single month, engineers performance KPIs, runs five statistical tests, and produces a full ranked analysis with actionable findings — including a quantified estimate of potential revenue uplift from training.

---

## Dataset

9 attendants, January sales period:

| Name | Jan Sales (ZAR) | Transactions | Avg Sale/Txn | Tier |
|---|---|---|---|---|
| ANDY | R18,736.92 | 853 | R21.96 | TOP |
| RETSI | R18,701.00 | 663 | R28.21 | TOP |
| SEBONGILE | R15,740.89 | 783 | R20.10 | TOP |
| LERATO | R15,673.00 | 724 | R21.65 | TOP |
| AGNESS | R15,061.41 | 634 | R23.76 | TOP |
| BROWN | R13,394.59 | 607 | R22.07 | MID |
| KENNY | R13,384.67 | 560 | R23.90 | MID |
| JAMES | R1,807.16 | 85 | R21.26 | LOW |
| GEORGE | R1,702.29 | 74 | R23.00 | LOW |

**Total station sales: ~R114,201 | Total transactions: 4,981**

---

## KPIs Computed

- `Avg_Sale_Value` — ZAR per transaction
- `Sales_Rank` / `Txn_Rank` — ordinal ranking
- `Sales_Share_Pct` / `Txn_Share_Pct` — % of station totals
- `Sales_Zscore` / `Txn_Zscore` — standard deviations from mean
- `Performance_Tier` — TOP (≥R15k) / MID (≥R13k) / LOW (<R13k)
- `Efficiency_Score` — composite normalised score
- `Cum_Sales_Pct` — cumulative % for Pareto chart

---

## Statistical Tests

**[1] Pearson Correlation — Sales vs Transactions**
Near-perfect positive correlation. Transactions volume is the dominant driver of sales, not average sale value.

**[2] Spearman Rank Correlation — Sales vs Transactions**
Confirms rank ordering is highly consistent — the best sellers are also the highest transaction processors.

**[3] Z-Score Outlier Detection**
JAMES and GEORGE both fall below −1.5σ on sales AND transactions. They are not just underperformers — they are statistical outliers requiring management attention.

**[4] One-Sample T-Test — Is avg sales significantly > R10,000?**
Tests H₀: μ = R10,000. Result printed at runtime.

**[5] OLS Linear Regression — Transactions → Sales**
```
Sales = intercept + slope × Transactions
```
R² and residuals printed per attendant. Identifies who is over/under-performing relative to their transaction volume.

---

## Visualisations (9-panel dashboard)

Saved as `attendants_analysis.png`:

1. Sales bar chart ranked, coloured by performance tier
2. Transaction count bar chart
3. Average sale value per attendant
4. Scatter: Sales vs Transactions + OLS regression line
5. Sales share pie chart
6. Pareto chart — cumulative sales contribution + 80% line
7. Composite efficiency score (horizontal bar)
8. Z-score chart — Sales and Transaction z-scores side by side
9. Transaction share % vs Sales share % scatter (effort vs output)

---

## Key Findings

**Finding 1 — Top vs Bottom gap is extreme**
ANDY leads at ~R18,737. GEORGE is last at ~R1,702. That is a ~11x gap. JAMES and GEORGE are outliers, not just poor performers.

**Finding 2 — Transaction volume drives everything**
Pearson r is near-perfect. The station earns more from attendants who serve more customers, not from those who achieve higher per-transaction values.

**Finding 3 — RETSI has the lowest avg sale value among top performers**
Despite being #2 overall, RETSI's per-transaction average is the lowest in the top group. There is upsell opportunity.

**Finding 4 — Pareto: top 4 attendants generate ~80% of revenue**
ANDY, RETSI, SEBONGILE, and LERATO carry the station. The bottom two together contribute under 4% of total sales.

**Finding 5 — JAMES and GEORGE need urgent attention**
Combined 159 transactions for the entire month. Both have z-scores below −1.5 on sales and transactions simultaneously.

**Finding 6 — Quantified revenue uplift from training**
If every attendant matched ANDY's avg sale value per transaction, the station would earn significantly more per month without increasing foot traffic. Exact figure printed at runtime.

---

## Requirements

```bash
pip install numpy pandas matplotlib scipy
```

```bash
python petrol_attendants.py
```

**Output:** `attendants_analysis.png` (160 DPI, saved to working directory)

---

## Structure

```
petrol_attendants.py
├── Section 1: Data + KPI engineering
├── Section 2: Descriptive statistics (full distributional summary)
├── Section 3: Performance tier breakdown with inline bar charts
├── Section 4: Five statistical tests + OLS residual table
├── Section 5: Efficiency scoring + Pareto cumulative calc
├── Section 6: 9-panel matplotlib dashboard
└── Section 7: Written findings with dynamic values
```
Author:
Retshidistswe sebekedi 
