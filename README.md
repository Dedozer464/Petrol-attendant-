# Petrol Attendants — January Performance Analysis

![Python](https://img.shields.io/badge/Python-100%25-blue) ![License](https://img.shields.io/badge/License-MIT-green)

Comprehensive statistical analysis of 9 petrol station attendants' January sales performance. Includes descriptive statistics, correlation analysis, OLS regression, Z-score outlier detection, Pareto analysis, efficiency scoring, and a 9-panel visualization dashboard with actionable business insights.

---

## Quick Start

```bash
git clone https://github.com/Dedozer464/Petrol-attendant-.git
cd Petrol-attendant-
pip install -r requirements.txt
python petrol_attendants.py
```

**Output:** `attendants_analysis.png` — a 3×3 dashboard with sales rankings, transaction volumes, statistical distributions, outlier detection, and Pareto contribution analysis (160 DPI).

---

## Overview

This script transforms raw sales (ZAR) and transaction count data into performance KPIs, runs five statistical tests, and generates a full ranked analysis with actionable findings:

- **Identify top performers** and revenue drivers
- **Detect statistical outliers** requiring immediate intervention
- **Quantify training ROI** based on per-transaction improvements
- **Visualize effort vs. output** across the team
- **Benchmark performance** against station averages and Pareto rules

---

## File Structure

```
.
├── petrol_attendants.py      # Main analysis script
├── requirements.txt           # Python dependencies
├── attendants_analysis.png    # Output: 9-panel dashboard (generated)
└── README.md                  # This file
```

---

## Installation

**Requirements:** Python 3.7+ with dependencies listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `numpy` — numerical operations
- `pandas` — data manipulation
- `matplotlib` — visualization
- `scipy` — statistical tests

---

## Dataset

**Period:** January | **Sample:** 9 petrol station attendants | **Total station sales:** ~R114,201 | **Total transactions:** 4,981

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

**Business Context:** This analysis identifies top performers, detects statistical outliers requiring intervention, and quantifies the revenue impact of improved per-transaction selling techniques.

---

## KPIs Computed

- `Avg_Sale_Value` — ZAR per transaction
- `Sales_Rank` / `Txn_Rank` — ordinal ranking (1 = highest)
- `Sales_Share_Pct` / `Txn_Share_Pct` — % of station totals
- `Sales_Zscore` / `Txn_Zscore` — standard deviations from mean
- `Performance_Tier` — TOP (≥R15k) / MID (≥R13k) / LOW (<R13k)
- `Efficiency_Score` — composite normalised score (0–100 scale)
- `Cum_Sales_Pct` — cumulative % for Pareto analysis

---

## Statistical Tests

**[1] Pearson Correlation — Sales vs Transactions**  
Measures linear relationship strength. Near-perfect positive correlation indicates transaction volume is the dominant driver of sales, not average sale value.

**[2] Spearman Rank Correlation — Sales vs Transactions**  
Rank-based correlation confirms ordering consistency — best sellers are also highest transaction processors.

**[3] Z-Score Outlier Detection**  
Z-scores below −1.5σ (approximately 7th percentile) flag statistical outliers. JAMES and GEORGE fall below this threshold on **both** sales and transactions, indicating systemic performance issues, not random variation.

**[4] One-Sample T-Test — Is average sales significantly > R10,000?**  
Tests H₀: μ = R10,000 (null hypothesis: station average ≤ R10k). Result printed at runtime with p-value and interpretation.

**[5] OLS Linear Regression — Transactions → Sales**  
```
Sales = intercept + slope × Transactions
```
Fits a line through (Transactions, Sales) pairs. R² and residuals printed per attendant. Identifies who is over/under-performing relative to their transaction volume (positive residual = outperformer, negative = underperformer).

---

## Visualisations (9-panel Dashboard)

Saved as `attendants_analysis.png`:

1. **Sales bar chart** — ranked, coloured by performance tier
2. **Transaction count bar chart** — volume by attendant
3. **Average sale value per attendant** — upsell opportunities highlighted
4. **Scatter plot** — Sales vs Transactions + OLS regression line
5. **Sales share pie chart** — revenue concentration
6. **Pareto chart** — cumulative sales contribution + 80% reference line
7. **Efficiency score** — horizontal bar chart (0–100 scale)
8. **Z-score chart** — Sales and Transaction z-scores side by side (outliers flagged)
9. **Effort vs Output scatter** — Transaction share % vs Sales share % (identifies efficiency gaps)

---

## Key Findings

**Finding 1 — Top vs Bottom gap is extreme**  
ANDY leads at ~R18,737. GEORGE ranks last at ~R1,702. That is an **11× performance gap**. JAMES and GEORGE are statistical outliers, not simply poor performers.

**Finding 2 — Transaction volume drives everything**  
Pearson r is near-perfect (close to 1.0). The station earns more from attendants who serve more customers, not from those who achieve higher per-transaction values. **Volume >> pricing power.**

**Finding 3 — RETSI has upsell opportunity**  
Despite ranking #2 overall (R18,701), RETSI's per-transaction average (R28.21) is the lowest in the top tier. Compared to AGNESS (R23.76 average) and SEBONGILE (R20.10), there is clear headroom for cross-sell and bundling training.

**Finding 4 — Pareto concentration**  
Top 4 attendants (ANDY, RETSI, SEBONGILE, LERATO) generate approximately **80% of station revenue**. Bottom 2 (JAMES, GEORGE) contribute under 4% despite being on payroll.

**Finding 5 — JAMES and GEORGE need urgent intervention**  
Combined: 159 transactions for the entire month (~5 per day each). Both have z-scores below −1.5 on sales AND transactions simultaneously, indicating this is not a data anomaly but a systemic capability or engagement issue.

**Finding 6 — Quantified revenue uplift from training**  
If every attendant matched ANDY's average sale value per transaction (R21.96), the station would realize incremental revenue without increasing foot traffic. Exact uplift figure printed at runtime based on current transaction volumes.

---

## Recommended Actions

| Priority | Action | Owner | Metric |
|----------|--------|-------|--------|
| **Immediate** | Interview JAMES and GEORGE; conduct performance diagnostics | Management | Root cause of low transaction count |
| **Urgent** | Enroll JAMES/GEORGE in upsell and customer engagement training | Training | Lift to MID tier (≥R13k/month) |
| **Short-term** | Peer coaching: pair RETSI with high-average-value performers | Management | Improve R28.21 → R22+ baseline |
| **Ongoing** | Monthly performance dashboard; set 10% uplift targets | Analytics | Track month-on-month growth |
| **Long-term** | Incentive realignment tied to efficiency score (not just sales) | HR | Encourage volume + quality balance |

---

## Structure

The script is organized into 7 sections:

```
petrol_attendants.py
├── Section 1: Data engineering + KPI computation
├── Section 2: Descriptive statistics (full distributional summary)
├── Section 3: Performance tier breakdown + inline summaries
├── Section 4: Five statistical tests + OLS regression table
├── Section 5: Efficiency scoring + Pareto cumulative calculation
├── Section 6: 9-panel matplotlib dashboard + export
└── Section 7: Written findings + dynamic value interpolation
```

Each section is independently runnable; modify thresholds or data as needed.

---

## Usage Notes

- **Data format:** Expects attendant name, Jan sales (ZAR), and transaction count as inputs. Edit Section 1 to swap in new data.
- **Z-score threshold:** Currently set to −1.5σ for outlier flagging. Adjust in Section 3 to tighten/relax sensitivity.
- **Pareto rule:** Fixed at 80% for visualization. Edit Section 5 to use 70% or 90% if preferred.
- **Output DPI:** Saved at 160 DPI (high-resolution for print). Adjust in Section 6 if needed.

---

## Requirements (pip)

```bash
numpy>=1.19
pandas>=1.1
matplotlib>=3.3
scipy>=1.5
```

Or install all at once:

```bash
pip install -r requirements.txt
```

---

## Contributing

Found a bug or have a feature request? Open an issue or submit a pull request.

---

## License

MIT License — see LICENSE file for details.

---

**Author:** Retshidistswe Sebekedi  
**Last Updated:** January 2024  
**Contact:** [Your contact info or GitHub profile link]
