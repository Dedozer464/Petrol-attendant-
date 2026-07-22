"""
Statistical analysis functions for Petrol Attendant Analysis
"""

import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Tuple, Any


def correlation_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform Pearson and Spearman correlation between sales and transactions.
    
    Args:
        df: DataFrame with Jan_Sales and Transactions columns.
    
    Returns:
        Dictionary with correlation results.
    """
    r, p_pearson = pearsonr(df['Jan_Sales'], df['Transactions'])
    rho, p_spearman = spearmanr(df['Jan_Sales'], df['Transactions'])
    
    # Interpret correlation strength
    abs_r = abs(r)
    if abs_r > 0.9:
        strength = "Very Strong"
    elif abs_r > 0.7:
        strength = "Strong"
    else:
        strength = "Moderate"
    
    return {
        'pearson_r': r,
        'pearson_p': p_pearson,
        'pearson_r_squared': r ** 2,
        'pearson_strength': strength,
        'spearman_rho': rho,
        'spearman_p': p_spearman,
    }


def outlier_detection(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Flag outliers based on z-scores.
    
    Args:
        df: DataFrame with Sales_Zscore and Txn_Zscore columns.
        config: Configuration with outlier thresholds.
    
    Returns:
        DataFrame with Outlier_Status column added.
    """
    zscore_notable = config['statistics']['zscore_notable']
    zscore_outlier = config['statistics']['zscore_outlier']
    
    statuses = []
    
    for _, row in df.iterrows():
        status = []
        
        if abs(row['Sales_Zscore']) > zscore_outlier:
            status.append("SALES_OUTLIER")
        if abs(row['Txn_Zscore']) > zscore_outlier:
            status.append("TXN_OUTLIER")
        
        if not status and abs(row['Sales_Zscore']) > zscore_notable:
            status.append("notable")
        
        statuses.append(", ".join(status) if status else "normal")
    
    df['Outlier_Status'] = statuses
    return df


def ttest_vs_reference(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    One-sample t-test: Is mean sales significantly different from reference value?
    
    Args:
        df: DataFrame with Jan_Sales column.
        config: Configuration with ttest_reference and significance_level.
    
    Returns:
        Dictionary with t-test results.
    """
    reference = config['statistics']['ttest_reference']
    alpha = config['statistics']['significance_level']
    
    t_stat, p_value = stats.ttest_1samp(df['Jan_Sales'], popmean=reference)
    
    is_significant = p_value < alpha
    interpretation = (
        f"REJECT H₀ — mean is significantly different from R{reference:,.0f}"
        if is_significant
        else f"FAIL to reject H₀ — mean is NOT significantly different from R{reference:,.0f}"
    )
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'reference_value': reference,
        'significance_level': alpha,
        'is_significant': is_significant,
        'interpretation': interpretation,
    }


def linear_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """
    OLS Linear Regression: Transactions → Sales
    
    Args:
        df: DataFrame with Jan_Sales and Transactions columns.
    
    Returns:
        Dictionary with regression parameters and predictions.
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['Transactions'], df['Jan_Sales']
    )
    
    # Calculate predictions and residuals
    df_copy = df.copy()
    df_copy['Predicted_Sales'] = intercept + slope * df_copy['Transactions']
    df_copy['Residual'] = df_copy['Jan_Sales'] - df_copy['Predicted_Sales']
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_error': std_err,
        'model_formula': f"Sales = {intercept:.2f} + {slope:.4f} × Transactions",
        'predictions_df': df_copy,
    }


def performance_gap_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze performance gap between tiers.
    
    Args:
        df: DataFrame with Performance_Tier and Jan_Sales columns.
    
    Returns:
        Dictionary with gap analysis.
    """
    top_avg = df[df['Performance_Tier'] == 'TOP PERFORMER']['Jan_Sales'].mean()
    mid_avg = df[df['Performance_Tier'] == 'MID PERFORMER']['Jan_Sales'].mean()
    low_avg = df[df['Performance_Tier'] == 'LOW PERFORMER']['Jan_Sales'].mean()
    
    return {
        'top_avg': top_avg,
        'mid_avg': mid_avg,
        'low_avg': low_avg,
        'gap_top_to_low': top_avg - low_avg,
        'ratio_top_to_low': top_avg / low_avg if low_avg > 0 else 0,
        'revenue_if_low_to_mid': (13000 - low_avg) * len(df[df['Performance_Tier'] == 'LOW PERFORMER']),
    }


def projection_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Projection: What if all attendants matched top performer's avg sale value?
    
    Args:
        df: DataFrame with Transactions and Avg_Sale_Value columns.
    
    Returns:
        Dictionary with projection results.
    """
    current_total = df['Jan_Sales'].sum()
    top_performer_asv = df.loc[df['Jan_Sales'].idxmax(), 'Avg_Sale_Value']
    
    projected_sales = (df['Transactions'] * top_performer_asv).sum()
    potential_gain = projected_sales - current_total
    
    return {
        'current_total_sales': current_total,
        'projected_total_sales': projected_sales,
        'potential_gain': potential_gain,
        'top_performer_name': df.loc[df['Jan_Sales'].idxmax(), 'Name'],
        'top_performer_asv': top_performer_asv,
    }
