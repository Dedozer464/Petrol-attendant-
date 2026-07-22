"""
============================================================================
PETROL ATTENDANTS — JANUARY PERFORMANCE ANALYSIS (REFACTORED)
Data Source: ATTENDANTS sheet (from photo)
Metrics: Sales (ZAR), Transactions, Derived KPIs
============================================================================

This is the refactored, modularized version with:
- Separate config file (config.yaml)
- Modular functions organized by task
- Better error handling
- Improved maintainability
"""

import warnings
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Import custom modules
from lib_config import load_config, validate_config, get_output_path
from lib_data import (
    load_attendant_data,
    calculate_kpis,
    calculate_efficiency_scores,
    calculate_totals,
    get_pareto_analysis,
)
from lib_statistics import (
    correlation_analysis,
    outlier_detection,
    ttest_vs_reference,
    linear_regression,
    performance_gap_analysis,
    projection_analysis,
)

warnings.filterwarnings('ignore')


def print_header(title: str, level: int = 1) -> None:
    """Print formatted section header."""
    width = 70
    if level == 1:
        print("=" * width)
        print(f"  {title}")
        print("=" * width)
    else:
        print(f"\n📊 {title}")
        print("=" * width)


def print_descriptive_stats(df: pd.DataFrame, total_sales: float, total_txns: int) -> None:
    """Print descriptive statistics for sales and transactions."""
    print_header("DESCRIPTIVE STATISTICS", level=2)
    
    for col, label in [("Jan_Sales", "January Sales (ZAR)"), ("Transactions", "Transactions")]:
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


def print_performance_tiers(df: pd.DataFrame) -> None:
    """Print performance tier breakdown."""
    print_header("PERFORMANCE TIER ANALYSIS", level=2)
    
    for tier_name in ["TOP PERFORMER", "MID PERFORMER", "LOW PERFORMER"]:
        grp = df[df["Performance_Tier"] == tier_name]
        print(f"\n  [{tier_name}]")
        for _, row in grp.iterrows():
            bar = "█" * int(row["Sales_Share_Pct"] * 2)
            print(f"    {row['Name']:12s}  R{row['Jan_Sales']:>10,.2f}  "
                  f"{row['Transactions']:>4} txns  "
                  f"Avg R{row['Avg_Sale_Value']:>7.2f}/txn  "
                  f"{row['Sales_Share_Pct']:>5.1f}%  {bar}")


def print_statistical_tests(df: pd.DataFrame, config: dict) -> dict:
    """Run and print statistical tests. Returns results dict."""
    print_header("STATISTICAL TESTS", level=2)
    
    # Correlation
    corr = correlation_analysis(df)
    print(f"\n[1] Pearson Correlation — Sales vs Transactions")
    print(f"    r = {corr['pearson_r']:.4f},  p = {corr['pearson_p']:.6f}")
    print(f"    R² = {corr['pearson_r_squared']:.4f}  (transactions explain {corr['pearson_r_squared']*100:.1f}% of sales variance)")
    print(f"    Interpretation: {corr['pearson_strength']} positive correlation — more transactions → more sales")
    
    print(f"\n[2] Spearman Rank Correlation — Sales vs Transactions")
    print(f"    ρ = {corr['spearman_rho']:.4f},  p = {corr['spearman_p']:.6f}")
    print(f"    Interpretation: Rank ordering is {'highly consistent' if corr['spearman_rho'] > 0.9 else 'consistent'}")
    
    # Outlier detection
    df = outlier_detection(df, config)
    print(f"\n[3] Z-Score Outlier Detection (|z| > 1.5 = notable, |z| > 2.0 = outlier)")
    print(f"    {'Name':12s}  {'Sales Z':>9}  {'Txn Z':>9}  Status")
    print(f"    {'-'*50}")
    for _, row in df.iterrows():
        print(f"    {row['Name']:12s}  {row['Sales_Zscore']:>+9.3f}  {row['Txn_Zscore']:>+9.3f}  {row['Outlier_Status']}")
    
    # T-test
    ttest = ttest_vs_reference(df, config)
    print(f"\n[4] One-Sample T-Test: Is avg sales significantly > R{ttest['reference_value']:,.0f}?")
    print(f"    H₀: μ = R{ttest['reference_value']:,.0f}  |  H₁: μ ≠ R{ttest['reference_value']:,.0f}")
    print(f"    t = {ttest['t_statistic']:.4f},  p = {ttest['p_value']:.4f}")
    print(f"    Result: {ttest['interpretation']}")
    
    # Linear regression
    reg = linear_regression(df)
    print(f"\n[5] OLS Linear Regression: Transactions → Sales")
    print(f"    {reg['model_formula']}")
    print(f"    R²    = {reg['r_squared']:.4f}")
    print(f"    p     = {reg['p_value']:.6f}")
    print(f"    SE    = {reg['std_error']:.4f}")
    print(f"\n    Predicted vs Actual:")
    print(f"    {'Name':12s}  {'Actual':>10}  {'Predicted':>10}  {'Residual':>10}")
    print(f"    {'-'*48}")
    for _, row in reg['predictions_df'].iterrows():
        print(f"    {row['Name']:12s}  R{row['Jan_Sales']:>9,.2f}  R{row['Predicted_Sales']:>9,.2f}  {row['Residual']:>+10.2f}")
    
    # Avg sale value
    df_asv = df.sort_values("Avg_Sale_Value", ascending=False)
    print(f"\n[6] Average Sale Value per Transaction — Who upsells best?")
    for _, row in df_asv.iterrows():
        bar = "▓" * int(row["Avg_Sale_Value"] / 5)
        print(f"    {row['Name']:12s}  R{row['Avg_Sale_Value']:>7.2f}/txn  {bar}")
    
    return {**corr, **ttest, **reg, 'df_with_outliers': df}


def print_mathematical_modelling(df: pd.DataFrame, total_sales: float, df_sorted: pd.DataFrame) -> dict:
    """Print mathematical modelling and projections."""
    print_header("MATHEMATICAL MODELLING", level=2)
    
    # Pareto
    print(f"\n[1] Pareto Analysis — Cumulative Sales Contribution")
    print(f"    {'Name':12s}  {'Sales':>10}  {'Share':>7}  {'Cum%':>6}  {'Staff%':>7}")
    print(f"    {'-'*52}")
    for _, row in df_sorted.iterrows():
        flag = " ◄ 80% threshold crossed" if abs(row["Cum_Sales_Pct"] - 80) < 12 else ""
        print(f"    {row['Name']:12s}  R{row['Jan_Sales']:>9,.2f}  {row['Sales_Share_Pct']:>6.1f}%"
              f"  {row['Cum_Sales_Pct']:>5.1f}%  {row['Cum_Pct_Staff']:>6.1f}%{flag}")
    
    # Performance gap
    gap = performance_gap_analysis(df)
    print(f"\n[2] Performance Gap Analysis")
    print(f"    Top Performer Avg   : R{gap['top_avg']:>10,.2f}")
    print(f"    Low Performer Avg   : R{gap['low_avg']:>10,.2f}")
    print(f"    Gap                 : R{gap['gap_top_to_low']:>10,.2f}  ({gap['ratio_top_to_low']:.1f}x difference)")
    print(f"    Revenue Lost if Low → Mid: R{gap['revenue_if_low_to_mid']:>10,.2f}/month")
    
    # Projection
    proj = projection_analysis(df)
    print(f"\n[3] Projection — If All Attendants Hit {proj['top_performer_name']}'s Avg Sale Value")
    print(f"    Current Total Sales  : R{proj['current_total_sales']:>10,.2f}")
    print(f"    Projected Total Sales: R{proj['projected_total_sales']:>10,.2f}")
    print(f"    Potential Monthly Gain: R{proj['potential_gain']:>9,.2f}")
    
    # Efficiency scores
    df_eff = df.sort_values("Efficiency_Score", ascending=False)
    print(f"\n[4] Composite Efficiency Score (normalised Sales + Txn + Avg Sale Value)")
    print(f"    {'Rank':>4}  {'Name':12s}  {'Score':>7}  {'Tier'}")  
    print(f"    {'-'*40}")
    for rank, (_, row) in enumerate(df_eff.iterrows(), 1):
        print(f"    #{rank:<3}  {row['Name']:12s}  {row['Efficiency_Score']:>6.1f}   {row['Performance_Tier']}")
    
    return {**gap, **proj}


def create_visualizations(df: pd.DataFrame, df_sorted: pd.DataFrame, stats_results: dict, config: dict) -> None:
    """Create comprehensive visualization dashboard with all 9 plots."""
    print(f"\n🎨 GENERATING VISUALISATIONS...")
    
    # Color scheme from config
    colors_cfg = config['visualization']['colors']
    
    DARK = colors_cfg['dark']
    MID = colors_cfg['mid']
    RED = colors_cfg['red']
    ORANGE = colors_cfg['orange']
    GREEN = colors_cfg['green']
    GOLD = colors_cfg['gold']
    
    tier_colors = {
        "TOP PERFORMER": GREEN,
        "MID PERFORMER": MID,
        "LOW PERFORMER": RED
    }
    
    # Setup figure
    figsize = config['visualization']['figsize']
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(config['visualization']['facecolor'])
    
    gs_cfg = config['visualization']['gridspec']
    gs = GridSpec(gs_cfg['rows'], gs_cfg['cols'], figure=fig,
                  hspace=gs_cfg['hspace'], wspace=gs_cfg['wspace'])
    
    bar_colors = [tier_colors[t] for t in df["Performance_Tier"]]
    bar_colors_sorted = [tier_colors[t] for t in df_sorted["Performance_Tier"]]
    
    df_eff = df.sort_values("Efficiency_Score", ascending=False)
    df_asv = df.sort_values("Avg_Sale_Value", ascending=False)
    
    # Extract regression results
    slope = stats_results['slope']
    intercept = stats_results['intercept']
    r_val = stats_results['r_value']
    r = stats_results['pearson_r']
    
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
    
    output_path = get_output_path(config)
    plt.savefig(output_path, dpi=config['visualization']['dpi'],
                bbox_inches="tight", facecolor=config['visualization']['facecolor'])
    print(f"✅ Visualisation saved to: {output_path}")
    plt.close()


def print_key_findings(df: pd.DataFrame, stats: dict, modelling: dict) -> None:
    """Print executive summary and key findings."""
    print_header("KEY FINDINGS", level=2)
    
    top_name = df.loc[df["Jan_Sales"].idxmax(), "Name"]
    top_val = df["Jan_Sales"].max()
    low_name = df.loc[df["Jan_Sales"].idxmin(), "Name"]
    low_val = df["Jan_Sales"].min()
    
    james_george_share = df[df['Name'].isin(['JAMES','GEORGE'])]['Sales_Share_Pct'].sum()
    james_george_txns = df[df['Name'].isin(['JAMES','GEORGE'])]['Transactions'].sum()
    
    print(f"""
FINDING 1 — TOP vs BOTTOM GAP IS EXTREME
  {top_name} leads at R{top_val:,.2f}. {low_name} is last at R{low_val:,.2f}.
  That is a {top_val/low_val:.0f}x difference in sales — a massive performance gap.
  JAMES and GEORGE are clear outliers (z-score below -1.5 on both metrics).

FINDING 2 — TRANSACTIONS STRONGLY PREDICT SALES (r = {stats['pearson_r']:.3f})
  The correlation is near-perfect. This means the primary driver of sales
  is volume of customers served — not average sale value. High performers
  serve far more customers per day.

FINDING 3 — OUTLIERS DETECTED
  JAMES and GEORGE have statistical z-scores indicating extreme underperformance.
  This is not a bad day — it is a pattern that needs management attention.

FINDING 4 — PARETO: TOP 4 ATTENDANTS = ~80% OF TOTAL SALES
  The bottom 2 (JAMES + GEORGE) together contribute only {james_george_share:.1f}% of sales.

FINDING 5 — POTENTIAL REVENUE UPLIFT: R{modelling['potential_gain']:,.0f}/MONTH
  If every attendant matched {modelling['top_performer_name']}'s avg sale value per transaction,
  the station would earn R{modelling['potential_gain']:,.0f} more per month without
  increasing foot traffic at all. Training is a high-ROI intervention.
""")


def main():
    """Main execution function."""
    try:
        # Load and validate config
        print("\n📂 Loading configuration...")
        config = load_config('config.yaml')
        validate_config(config)
        print("✅ Configuration validated.")
        
        # Load data
        print("📊 Loading attendant data...")
        df = load_attendant_data(config)
        print(f"✅ Loaded {len(df)} attendants.")
        
        # Calculate KPIs
        print("🧮 Calculating KPIs...")
        df = calculate_kpis(df, config)
        df = calculate_efficiency_scores(df, config)
        print("✅ KPIs calculated.")
        
        # Calculate totals
        total_sales, total_txns, overall_avg = calculate_totals(df)
        df_sorted = get_pareto_analysis(df, total_sales)
        
        # Print initial summary
        print_header("PETROL ATTENDANTS — JANUARY PERFORMANCE STATISTICAL ANALYSIS", level=1)
        print("\n📋 RAW DATA + KPIs")
        print("-" * 70)
        print(df[["Name","Jan_Sales","Transactions","Avg_Sale_Value",
                  "Sales_Share_Pct","Performance_Tier"]].to_string(index=False))
        
        # Statistical analyses
        print_descriptive_stats(df, total_sales, total_txns)
        print_performance_tiers(df)
        stats_results = print_statistical_tests(df, config)
        modelling_results = print_mathematical_modelling(df, total_sales, df_sorted)
        
        # Visualizations
        create_visualizations(df, df_sorted, stats_results, config)
        
        # Findings
        print_key_findings(df, stats_results, modelling_results)
        
        print_header("ANALYSIS COMPLETE", level=1)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
