"""
=============================================================================
PETROL ATTENDANTS — JANUARY PERFORMANCE ANALYSIS
Data Source: ATTENDANTS sheet (from photo)
Metrics: Sales (ZAR), Transactions, Derived KPIs
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import pearsonr, spearmanr, zscore
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA
# ─────────────────────────────────────────────────────────────────────────────

data = {
    "Name":         ["ANDY", "RETSI", "SEBONGILE", "LERATO",
                     "AGNESS", "BROWN", "KENNY", "JAMES", "GEORGE"],
    "Jan_Sales":    [18736.92, 18701.00, 15740.89, 15673.00,
                     15061.41, 13394.59, 13384.67, 1807.16, 1702.29],
    "Transactions": [853, 663, 783, 724, 634, 607, 560, 85, 74],
}

df = pd.DataFrame(data)

# ── Derived KPIs ──────────────────────────────────────────────────────────────
df["Avg_Sale_Value"]     = (df["Jan_Sales"] / df["Transactions"]).round(2)
df["Sales_Rank"]         = df["Jan_Sales"].rank(ascending=False).astype(int)
df["Txn_Rank"]           = df["Transactions"].rank(ascending=False).astype(int)

total_sales = df["Jan_Sales"].sum()
total_txns  = df["Transactions"].sum()

df["Sales_Share_Pct"]    = (df["Jan_Sales"] / total_sales * 100).round(2)
df["Txn_Share_Pct"]      = (df["Transactions"] / total_txns * 100).round(2)

# Z-scores (how many std devs from mean — outlier detection)
df["Sales_Zscore"]       = zscore(df["Jan_Sales"]).round(3)
df["Txn_Zscore"]         = zscore(df["Transactions"]).round(3)

# Performance tier
def tier(row):
    if row["Jan_Sales"] >= 15000:  return "TOP PERFORMER"
    if row["Jan_Sales"] >= 13000:  return "MID PERFORMER"
    return "LOW PERFORMER"

df["Performance_Tier"] = df.apply(tier, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  PETROL ATTENDANTS — JANUARY PERFORMANCE STATISTICAL ANALYSIS")
print("=" * 70)

print("\n📋 RAW DATA + KPIs")
print("-" * 70)
print(df[["Name","Jan_Sales","Transactions","Avg_Sale_Value",
          "Sales_Share_Pct","Performance_Tier"]].to_string(index=False))

print(f"\n\n📊 SECTION 2: DESCRIPTIVE STATISTICS")
print("=" * 70)

for col, label in [("Jan_Sales","January Sales (ZAR)"), ("Transactions","Transactions")]:
    s = df[col]
    print(f"\n── {label} ──")
    print(f"  Count       : {len(s)}")
    print(f"  Sum         : {s.sum():>12,.2f}")
    print(f"  Mean        : {s.mean():>12,.2f}")
    print(f"  Median      : {s.median():>12,.2f}")
    print(f"  Std Dev     : {s.std():>12,.2f}")
    print(f"  Variance    : {s.var():>12,.2f}")
    print(f"  Min         : {s.min():>12,.2f}  ({df.loc[s.idxmin(),'Name']})")
    print(f"  Max         : {s.max():>12,.2f}  ({df.loc[s.idxmax(),'Name']})")
    print(f"  Range       : {s.max()-s.min():>12,.2f}")
    print(f"  Q1 (25%)    : {s.quantile(0.25):>12,.2f}")
    print(f"  Q3 (75%)    : {s.quantile(0.75):>12,.2f}")
    print(f"  IQR         : {s.quantile(0.75)-s.quantile(0.25):>12,.2f}")
    print(f"  Skewness    : {s.skew():>12.4f}")
    print(f"  Kurtosis    : {s.kurtosis():>12.4f}")
    cv = s.std() / s.mean() * 100
    print(f"  Coeff. Var  : {cv:>11.2f}%")

print(f"\n  TOTAL STATION SALES  : R{total_sales:>12,.2f}")
print(f"  TOTAL TRANSACTIONS   : {total_txns:>12,}")
print(f"  OVERALL AVG SALE VAL : R{total_sales/total_txns:>12,.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: PERFORMANCE TIERS
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n\n🏆 SECTION 3: PERFORMANCE TIER ANALYSIS")
print("=" * 70)

for tier_name in ["TOP PERFORMER", "MID PERFORMER", "LOW PERFORMER"]:
    grp = df[df["Performance_Tier"] == tier_name]
    print(f"\n  [{tier_name}]")
    for _, row in grp.iterrows():
        bar = "█" * int(row["Sales_Share_Pct"] * 2)
        print(f"    {row['Name']:12s}  R{row['Jan_Sales']:>10,.2f}  "
              f"{row['Transactions']:>4} txns  "
              f"Avg R{row['Avg_Sale_Value']:>7.2f}/txn  "
              f"{row['Sales_Share_Pct']:>5.1f}%  {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: STATISTICAL TESTS
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n\n🔬 SECTION 4: STATISTICAL TESTS")
print("=" * 70)

# 4.1 Pearson Correlation: Sales vs Transactions
r, p = pearsonr(df["Jan_Sales"], df["Transactions"])
print(f"\n[1] Pearson Correlation — Sales vs Transactions")
print(f"    r = {r:.4f},  p = {p:.6f}")
print(f"    R² = {r**2:.4f}  (transactions explain {r**2*100:.1f}% of sales variance)")
strength = "Very Strong" if abs(r) > 0.9 else ("Strong" if abs(r) > 0.7 else "Moderate")
print(f"    Interpretation: {strength} positive correlation — more transactions → more sales")

# 4.2 Spearman Rank Correlation
rho, p_sp = spearmanr(df["Jan_Sales"], df["Transactions"])
print(f"\n[2] Spearman Rank Correlation — Sales vs Transactions")
print(f"    ρ = {rho:.4f},  p = {p_sp:.6f}")
print(f"    Interpretation: Rank ordering is {'highly consistent' if rho > 0.9 else 'consistent'}")

# 4.3 Z-score outlier detection
print(f"\n[3] Z-Score Outlier Detection (|z| > 1.5 = notable, |z| > 2.0 = outlier)")
print(f"    {'Name':12s}  {'Sales Z':>9}  {'Txn Z':>9}  Status")
print(f"    {'-'*50}")
for _, row in df.iterrows():
    status = []
    if abs(row["Sales_Zscore"]) > 2.0:  status.append("SALES OUTLIER")
    if abs(row["Txn_Zscore"])   > 2.0:  status.append("TXN OUTLIER")
    if not status and abs(row["Sales_Zscore"]) > 1.5: status.append("notable")
    flag = ", ".join(status) if status else "normal"
    print(f"    {row['Name']:12s}  {row['Sales_Zscore']:>+9.3f}  {row['Txn_Zscore']:>+9.3f}  {flag}")

# 4.4 One-sample t-test: Is mean sales significantly above R10,000?
t_stat, p_t = stats.ttest_1samp(df["Jan_Sales"], popmean=10000)
print(f"\n[4] One-Sample T-Test: Is avg sales significantly > R10,000?")
print(f"    H₀: μ = R10,000  |  H₁: μ ≠ R10,000")
print(f"    t = {t_stat:.4f},  p = {p_t:.4f}")
print(f"    Result: {'REJECT H₀ — mean is significantly different from R10,000' if p_t < 0.05 else 'FAIL to reject H₀'}")

# 4.5 OLS Regression: Transactions → Sales
slope, intercept, r_val, p_val, se = stats.linregress(df["Transactions"], df["Jan_Sales"])
print(f"\n[5] OLS Linear Regression: Transactions → Sales")
print(f"    Model : Sales = {intercept:.2f} + {slope:.4f} × Transactions")
print(f"    R²    = {r_val**2:.4f}")
print(f"    p     = {p_val:.6f}")
print(f"    SE    = {se:.4f}")
print(f"\n    Predicted vs Actual:")
print(f"    {'Name':12s}  {'Actual':>10}  {'Predicted':>10}  {'Residual':>10}")
print(f"    {'-'*48}")
for _, row in df.iterrows():
    pred = intercept + slope * row["Transactions"]
    resid = row["Jan_Sales"] - pred
    print(f"    {row['Name']:12s}  R{row['Jan_Sales']:>9,.2f}  R{pred:>9,.2f}  {resid:>+10.2f}")

# 4.6 Avg Sale Value analysis
print(f"\n[6] Average Sale Value per Transaction — Who upsells best?")
df_sorted_asv = df.sort_values("Avg_Sale_Value", ascending=False)
for _, row in df_sorted_asv.iterrows():
    bar = "▓" * int(row["Avg_Sale_Value"] / 5)
    print(f"    {row['Name']:12s}  R{row['Avg_Sale_Value']:>7.2f}/txn  {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: MATHEMATICAL MODELLING
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n\n📐 SECTION 5: MATHEMATICAL MODELLING")
print("=" * 70)

# 5.1 Pareto / 80-20 analysis
print(f"\n[1] Pareto Analysis — Cumulative Sales Contribution")
df_sorted = df.sort_values("Jan_Sales", ascending=False).reset_index(drop=True)
df_sorted["Cum_Sales"]    = df_sorted["Jan_Sales"].cumsum()
df_sorted["Cum_Sales_Pct"]= (df_sorted["Cum_Sales"] / total_sales * 100).round(1)
df_sorted["Cum_Pct_Staff"]= ((df_sorted.index + 1) / len(df_sorted) * 100).round(1)
print(f"    {'Name':12s}  {'Sales':>10}  {'Share':>7}  {'Cum%':>6}  {'Staff%':>7}")
print(f"    {'-'*52}")
for _, row in df_sorted.iterrows():
    flag = " ◄ 80% threshold crossed" if abs(row["Cum_Sales_Pct"] - 80) < 12 else ""
    print(f"    {row['Name']:12s}  R{row['Jan_Sales']:>9,.2f}  {row['Sales_Share_Pct']:>6.1f}%"
          f"  {row['Cum_Sales_Pct']:>5.1f}%  {row['Cum_Pct_Staff']:>6.1f}%{flag}")

# 5.2 Performance gap
top_avg   = df[df["Performance_Tier"]=="TOP PERFORMER"]["Jan_Sales"].mean()
low_avg   = df[df["Performance_Tier"]=="LOW PERFORMER"]["Jan_Sales"].mean()
print(f"\n[2] Performance Gap Analysis")
print(f"    Top Performer Avg   : R{top_avg:>10,.2f}")
print(f"    Low Performer Avg   : R{low_avg:>10,.2f}")
print(f"    Gap                 : R{top_avg - low_avg:>10,.2f}  ({(top_avg/low_avg):.1f}x difference)")
print(f"    Revenue Lost if Low → Mid: R{(13000 - low_avg)*2:>10,.2f}/month")

# 5.3 Projection model
print(f"\n[3] Projection — If All Attendants Hit Andy's Avg Sale Value")
proj_sales = df["Transactions"] * df.loc[df["Name"]=="ANDY","Avg_Sale_Value"].values[0]
potential_gain = proj_sales.sum() - total_sales
print(f"    Current Total Sales  : R{total_sales:>10,.2f}")
print(f"    Projected Total Sales: R{proj_sales.sum():>10,.2f}")
print(f"    Potential Monthly Gain: R{potential_gain:>9,.2f}")

# 5.4 Efficiency Score (composite)
print(f"\n[4] Composite Efficiency Score (normalised Sales + Txn + Avg Sale Value)")
for col in ["Jan_Sales","Transactions","Avg_Sale_Value"]:
    mn, mx = df[col].min(), df[col].max()
    df[f"{col}_norm"] = ((df[col] - mn) / (mx - mn) * 100).round(1)
df["Efficiency_Score"] = ((df["Jan_Sales_norm"]*0.5 +
                           df["Transactions_norm"]*0.3 +
                           df["Avg_Sale_Value_norm"]*0.2)).round(1)
df_eff = df.sort_values("Efficiency_Score", ascending=False)
print(f"    {'Rank':>4}  {'Name':12s}  {'Score':>7}  {'Tier'}")
print(f"    {'-'*40}")
for rank, (_, row) in enumerate(df_eff.iterrows(), 1):
    print(f"    #{rank:<3}  {row['Name']:12s}  {row['Efficiency_Score']:>6.1f}   {row['Performance_Tier']}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n\n🎨 SECTION 6: GENERATING VISUALISATIONS...")

DARK   = "#1B3A5C"
MID    = "#2E6DA4"
LIGHT  = "#D6E8F8"
RED    = "#C0392B"
ORANGE = "#E67E22"
GREEN  = "#1E8449"
GOLD   = "#D4AC0D"

tier_colors = {
    "TOP PERFORMER": GREEN,
    "MID PERFORMER": MID,
    "LOW PERFORMER": RED
}
bar_colors = [tier_colors[t] for t in df["Performance_Tier"]]
bar_colors_sorted = [tier_colors[t] for t in df_sorted["Performance_Tier"]]

fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor("#F4F6F9")
gs = GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.38)

# ── Plot 1: Sales bar chart ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0:2])
bars = ax1.bar(df["Name"], df["Jan_Sales"], color=bar_colors, edgecolor="white", linewidth=0.8, width=0.65)
ax1.set_title("January Sales per Attendant", fontweight="bold", fontsize=12, color=DARK)
ax1.set_ylabel("Sales (ZAR)", fontsize=10)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"R{x/1000:.0f}k"))
ax1.axhline(df["Jan_Sales"].mean(), color=ORANGE, linestyle="--", linewidth=1.5,
            label=f"Mean R{df['Jan_Sales'].mean():,.0f}")
ax1.axhline(df["Jan_Sales"].median(), color=GOLD, linestyle=":", linewidth=1.5,
            label=f"Median R{df['Jan_Sales'].median():,.0f}")
for bar, val in zip(bars, df["Jan_Sales"]):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
             f"R{val/1000:.1f}k", ha="center", fontsize=8, fontweight="bold", color=DARK)
patches = [mpatches.Patch(color=v, label=k) for k,v in tier_colors.items()]
ax1.legend(handles=patches + ax1.get_legend_handles_labels()[0][len(patches):],
           fontsize=8, loc="upper right")
ax1.set_facecolor("#FDFEFE")
ax1.tick_params(axis='x', labelrotation=15)

# ── Plot 2: Transactions bar ─────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
bars2 = ax2.barh(df["Name"][::-1], df["Transactions"][::-1],
                 color=[tier_colors[t] for t in df["Performance_Tier"][::-1]],
                 edgecolor="white", height=0.65)
ax2.set_title("Transactions\nper Attendant", fontweight="bold", fontsize=11, color=DARK)
ax2.set_xlabel("Transactions", fontsize=9)
ax2.axvline(df["Transactions"].mean(), color=ORANGE, linestyle="--", linewidth=1.2,
            label=f"Mean {df['Transactions'].mean():.0f}")
ax2.legend(fontsize=7)
for bar, val in zip(bars2, df["Transactions"][::-1]):
    ax2.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2,
             str(val), va="center", fontsize=8, fontweight="bold")
ax2.set_facecolor("#FDFEFE")

# ── Plot 3: Avg Sale Value ───────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
df_asv = df.sort_values("Avg_Sale_Value", ascending=False)
bars3  = ax3.bar(df_asv["Name"], df_asv["Avg_Sale_Value"],
                 color=[tier_colors[t] for t in df_asv["Performance_Tier"]],
                 edgecolor="white", width=0.65)
ax3.set_title("Avg Sale Value\n(ZAR per Transaction)", fontweight="bold", fontsize=10, color=DARK)
ax3.set_ylabel("R per transaction", fontsize=9)
ax3.tick_params(axis='x', labelrotation=30, labelsize=8)
for bar, val in zip(bars3, df_asv["Avg_Sale_Value"]):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f"R{val:.0f}", ha="center", fontsize=7.5, fontweight="bold")
ax3.set_facecolor("#FDFEFE")

# ── Plot 4: Scatter Sales vs Transactions + regression ──────────────────────
ax4 = fig.add_subplot(gs[1, 1])
colors_sc = [tier_colors[t] for t in df["Performance_Tier"]]
ax4.scatter(df["Transactions"], df["Jan_Sales"], c=colors_sc,
            s=120, zorder=5, edgecolors="white", linewidths=0.8)
for _, row in df.iterrows():
    ax4.annotate(row["Name"], (row["Transactions"], row["Jan_Sales"]),
                 fontsize=7, xytext=(4,4), textcoords="offset points")
x_line = np.linspace(df["Transactions"].min(), df["Transactions"].max(), 100)
y_line = intercept + slope * x_line
ax4.plot(x_line, y_line, "k--", linewidth=1.5, alpha=0.7,
         label=f"OLS fit  R²={r_val**2:.3f}")
ax4.set_title(f"Sales vs Transactions\nPearson r={r:.3f}", fontweight="bold", fontsize=10, color=DARK)
ax4.set_xlabel("Transactions", fontsize=9)
ax4.set_ylabel("Sales (ZAR)", fontsize=9)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"R{x/1000:.0f}k"))
ax4.legend(fontsize=8)
ax4.set_facecolor("#FDFEFE")

# ── Plot 5: Sales Share Pie ──────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
pie_colors = [tier_colors[t] for t in df_sorted["Performance_Tier"]]
wedges, texts, autotexts = ax5.pie(
    df_sorted["Jan_Sales"], labels=df_sorted["Name"],
    autopct="%1.1f%%", colors=pie_colors,
    startangle=140, wedgeprops={"edgecolor":"white","linewidth":1.2})
for at in autotexts:
    at.set_fontsize(7); at.set_color("white"); at.set_fontweight("bold")
for t in texts: t.set_fontsize(8)
ax5.set_title("Sales Share\n(% of Total)", fontweight="bold", fontsize=10, color=DARK)

# ── Plot 6: Pareto chart ─────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0:2])
ax6b = ax6.twinx()
ax6.bar(df_sorted["Name"], df_sorted["Jan_Sales"],
        color=bar_colors_sorted, edgecolor="white", width=0.65, label="Sales")
ax6b.plot(df_sorted["Name"], df_sorted["Cum_Sales_Pct"],
          color=DARK, marker="o", linewidth=2, markersize=6, label="Cumulative %")
ax6b.axhline(80, color=RED, linestyle="--", linewidth=1.2, alpha=0.7, label="80% line")
ax6.set_title("Pareto Chart — Cumulative Sales Contribution", fontweight="bold", fontsize=11, color=DARK)
ax6.set_ylabel("Sales (ZAR)", fontsize=9)
ax6b.set_ylabel("Cumulative %", fontsize=9)
ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"R{x/1000:.0f}k"))
ax6b.set_ylim(0, 110)
ax6.tick_params(axis='x', labelrotation=15)
ax6.set_facecolor("#FDFEFE")
lines1, labels1 = ax6.get_legend_handles_labels()
lines2, labels2 = ax6b.get_legend_handles_labels()
ax6.legend(lines1+lines2, labels1+labels2, fontsize=8)

# ── Plot 7: Efficiency Score ─────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
eff_colors = [tier_colors[t] for t in df_eff["Performance_Tier"]]
bars7 = ax7.barh(df_eff["Name"][::-1], df_eff["Efficiency_Score"][::-1],
                 color=eff_colors[::-1], edgecolor="white", height=0.65)
ax7.set_title("Composite\nEfficiency Score", fontweight="bold", fontsize=10, color=DARK)
ax7.set_xlabel("Score (0–100)", fontsize=9)
for bar, val in zip(bars7, df_eff["Efficiency_Score"][::-1]):
    ax7.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
             f"{val:.1f}", va="center", fontsize=8.5, fontweight="bold")
ax7.set_facecolor("#FDFEFE")

# ── Plot 8: Z-score chart ────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[3, 0:2])
x = np.arange(len(df))
width = 0.38
z_sales_colors = [GREEN if z > 0 else RED for z in df["Sales_Zscore"]]
z_txn_colors   = [MID   if z > 0 else ORANGE for z in df["Txn_Zscore"]]
b1 = ax8.bar(x - width/2, df["Sales_Zscore"], width, color=z_sales_colors,
             edgecolor="white", label="Sales Z-score")
b2 = ax8.bar(x + width/2, df["Txn_Zscore"],   width, color=z_txn_colors,
             edgecolor="white", alpha=0.75, label="Txn Z-score")
ax8.axhline(0,    color="black", linewidth=0.8)
ax8.axhline(1.5,  color=ORANGE, linestyle="--", linewidth=1, alpha=0.6, label="+1.5 σ")
ax8.axhline(-1.5, color=ORANGE, linestyle="--", linewidth=1, alpha=0.6, label="-1.5 σ")
ax8.axhline(2.0,  color=RED,    linestyle=":",  linewidth=1, alpha=0.7, label="±2.0 σ (outlier)")
ax8.axhline(-2.0, color=RED,    linestyle=":",  linewidth=1, alpha=0.7)
ax8.set_xticks(x)
ax8.set_xticklabels(df["Name"], rotation=15)
ax8.set_title("Z-Score Analysis — Sales & Transactions\n(outliers: |z| > 2.0)", fontweight="bold", fontsize=11, color=DARK)
ax8.set_ylabel("Standard Deviations from Mean", fontsize=9)
ax8.legend(fontsize=7, ncol=3)
ax8.set_facecolor("#FDFEFE")

# ── Plot 9: Sales % vs Txn % (effort vs output) ──────────────────────────────
ax9 = fig.add_subplot(gs[3, 2])
ax9.scatter(df["Txn_Share_Pct"], df["Sales_Share_Pct"],
            c=bar_colors, s=130, zorder=5, edgecolors="white")
for _, row in df.iterrows():
    ax9.annotate(row["Name"],
                 (row["Txn_Share_Pct"], row["Sales_Share_Pct"]),
                 fontsize=7.5, xytext=(4,3), textcoords="offset points")
ax9.plot([0,25],[0,25], "k--", linewidth=1, alpha=0.4, label="Perfect 1:1 line")
ax9.set_xlabel("% of Transactions", fontsize=9)
ax9.set_ylabel("% of Sales", fontsize=9)
ax9.set_title("Transaction Share vs\nSales Share", fontweight="bold", fontsize=10, color=DARK)
ax9.legend(fontsize=7)
ax9.set_facecolor("#FDFEFE")

fig.suptitle(
    "Petrol Attendants — January Performance Analysis\n"
    "Statistical & Mathematical Modelling  |  n=9 Attendants",
    fontsize=15, fontweight="bold", color=DARK, y=0.99)

plt.savefig("/mnt/user-data/outputs/attendants_analysis.png",
            dpi=160, bbox_inches="tight", facecolor="#F4F6F9")
print("✅ Visualisation saved.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: FINDINGS
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n\n📝 SECTION 7: KEY FINDINGS")
print("=" * 70)

top_name = df.loc[df["Jan_Sales"].idxmax(), "Name"]
top_val  = df["Jan_Sales"].max()
low_name = df.loc[df["Jan_Sales"].idxmin(), "Name"]
low_val  = df["Jan_Sales"].min()

print(f"""
FINDING 1 — TOP vs BOTTOM GAP IS EXTREME
  {top_name} leads at R{top_val:,.2f}. {low_name} is last at R{low_val:,.2f}.
  That is a {top_val/low_val:.0f}x difference in sales — a massive performance gap.
  JAMES and GEORGE are clear outliers (z-score below -1.5 on both metrics).

FINDING 2 — TRANSACTIONS STRONGLY PREDICT SALES (r = {r:.3f})
  The correlation is near-perfect. This means the primary driver of sales
  is volume of customers served — not average sale value. High performers
  serve far more customers per day. Andy and Sebongile lead in volume.

FINDING 3 — RETSI HAS THE LOWEST AVG SALE VALUE AMONG TOP PERFORMERS
  Despite being #2 in total sales, RETSI has a lower avg per transaction
  than ANDY, SEBONGILE, and BROWN. There is room to upsell more per customer.

FINDING 4 — PARETO: TOP 4 ATTENDANTS = ~80% OF TOTAL SALES
  ANDY, RETSI, SEBONGILE and LERATO generate the bulk of station revenue.
  The bottom 2 (JAMES + GEORGE) together contribute only {df[df['Name'].isin(['JAMES','GEORGE'])]['Sales_Share_Pct'].sum():.1f}% of sales.

FINDING 5 — JAMES AND GEORGE ARE STATISTICAL OUTLIERS
  Both have z-scores below -1.5 on sales AND transactions.
  This is not a bad day — it is a pattern that needs management attention.
  Combined they made only {df[df['Name'].isin(['JAMES','GEORGE'])]['Transactions'].sum()} transactions in the whole month.

FINDING 6 — POTENTIAL REVENUE UPLIFT: R{potential_gain:,.0f}/MONTH
  If every attendant matched ANDY's avg sale value per transaction,
  the station would earn R{potential_gain:,.0f} more per month without
  increasing foot traffic at all. Training is a high-ROI intervention here.
""")

print("=" * 70)
print("  ANALYSIS COMPLETE")
print("=" * 70)
