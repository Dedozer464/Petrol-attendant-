"""
Data processing functions for Petrol Attendant Analysis
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore
from typing import Dict, Any, Tuple


def load_attendant_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load attendant data from configuration and create DataFrame.
    
    Args:
        config: Configuration dictionary containing attendant data.
    
    Returns:
        DataFrame with raw attendant data.
    
    Raises:
        ValueError: If data is invalid or incomplete.
    """
    attendants = config['data']['attendants']
    
    if not attendants or len(attendants) < 2:
        raise ValueError("Need at least 2 attendants for analysis")
    
    data = {
        'Name': [a['name'] for a in attendants],
        'Jan_Sales': [a['sales'] for a in attendants],
        'Transactions': [a['transactions'] for a in attendants],
    }
    
    df = pd.DataFrame(data)
    
    # Validate data
    if (df[['Jan_Sales', 'Transactions']] < 0).any().any():
        raise ValueError("Negative values detected in data")
    
    if (df['Transactions'] == 0).any():
        raise ValueError("Zero transactions detected (division by zero risk)")
    
    return df


def calculate_kpis(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate derived KPIs and performance metrics.
    
    Args:
        df: DataFrame with raw attendant data.
        config: Configuration dictionary.
    
    Returns:
        DataFrame with added KPI columns.
    """
    # Basic derived metrics
    df['Avg_Sale_Value'] = (df['Jan_Sales'] / df['Transactions']).round(2)
    df['Sales_Rank'] = df['Jan_Sales'].rank(ascending=False).astype(int)
    df['Txn_Rank'] = df['Transactions'].rank(ascending=False).astype(int)
    
    # Percentage shares
    total_sales = df['Jan_Sales'].sum()
    total_txns = df['Transactions'].sum()
    
    df['Sales_Share_Pct'] = (df['Jan_Sales'] / total_sales * 100).round(2)
    df['Txn_Share_Pct'] = (df['Transactions'] / total_txns * 100).round(2)
    
    # Z-scores for outlier detection
    df['Sales_Zscore'] = zscore(df['Jan_Sales']).round(3)
    df['Txn_Zscore'] = zscore(df['Transactions']).round(3)
    
    # Performance tier
    tier_top = config['thresholds']['tier_top']
    tier_mid = config['thresholds']['tier_mid']
    
    def assign_tier(sales):
        if sales >= tier_top:
            return "TOP PERFORMER"
        elif sales >= tier_mid:
            return "MID PERFORMER"
        else:
            return "LOW PERFORMER"
    
    df['Performance_Tier'] = df['Jan_Sales'].apply(assign_tier)
    
    return df


def calculate_efficiency_scores(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate composite efficiency scores.
    
    Args:
        df: DataFrame with KPIs.
        config: Configuration dictionary.
    
    Returns:
        DataFrame with efficiency score columns added.
    """
    weights = config['efficiency']['weights']
    
    # Normalize each metric to 0-100
    for col in ['Jan_Sales', 'Transactions', 'Avg_Sale_Value']:
        min_val = df[col].min()
        max_val = df[col].max()
        
        if max_val == min_val:
            df[f'{col}_norm'] = 50.0  # Default if all values are same
        else:
            df[f'{col}_norm'] = (
                (df[col] - min_val) / (max_val - min_val) * 100
            ).round(1)
    
    # Weighted composite score
    df['Efficiency_Score'] = (
        df['Jan_Sales_norm'] * weights['sales'] +
        df['Transactions_norm'] * weights['transactions'] +
        df['Avg_Sale_Value_norm'] * weights['avg_sale_value']
    ).round(1)
    
    return df


def calculate_totals(df: pd.DataFrame) -> Tuple[float, int, float]:
    """
    Calculate station-wide totals.
    
    Args:
        df: DataFrame with attendant data.
    
    Returns:
        Tuple of (total_sales, total_transactions, overall_avg_sale_value)
    """
    total_sales = df['Jan_Sales'].sum()
    total_txns = df['Transactions'].sum()
    overall_avg = total_sales / total_txns if total_txns > 0 else 0
    
    return total_sales, total_txns, overall_avg


def get_pareto_analysis(df: pd.DataFrame, total_sales: float) -> pd.DataFrame:
    """
    Perform Pareto analysis - sort by sales and add cumulative metrics.
    
    Args:
        df: DataFrame with attendant data.
        total_sales: Total station sales.
    
    Returns:
        DataFrame sorted by sales descending with cumulative columns.
    """
    df_sorted = df.sort_values('Jan_Sales', ascending=False).reset_index(drop=True)
    
    df_sorted['Cum_Sales'] = df_sorted['Jan_Sales'].cumsum()
    df_sorted['Cum_Sales_Pct'] = (
        df_sorted['Cum_Sales'] / total_sales * 100
    ).round(1)
    df_sorted['Cum_Pct_Staff'] = (
        (df_sorted.index + 1) / len(df_sorted) * 100
    ).round(1)
    
    return df_sorted
