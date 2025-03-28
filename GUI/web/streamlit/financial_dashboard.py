import os
import sys
import time
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from PIL import Image
import io
import base64
import tempfile
import shutil
import argparse

# Force standalone mode if run directly
STANDALONE_MODE = os.environ.get('STREAMLIT_RUN_MODE', '') == '' or __name__ == "__main__"

# Set page title and favicon
st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Attempt to display a logo
try:
    # Try multiple paths to locate the logo
    possible_logo_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../static/images/icone_1.png"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../static/images/icone_1.png"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../Logos/icone_1.png"),
        os.path.join(os.path.abspath("../../"), "Logos/icone_1.png"),
        os.path.join(os.path.abspath("../../../"), "Logos/icone_1.png")
    ]
    
    # Instead of displaying the logo and path, just add a simple title
    st.sidebar.markdown("# 💰 Financial Dashboard")
    
except Exception as e:
    st.sidebar.markdown("# 💰 Financial Dashboard")

# Add title to main area
st.title("Financial Dashboard")

# Function to load transaction data
def load_transaction_data(session_id):
    """Load transaction data from CSV file - ONLY use actual uploaded transaction data"""
    # Check environment variables for paths (set by webapp.py)
    csv_folder = os.environ.get('CSV_FOLDER')
    output_folder = os.environ.get('OUTPUT_FOLDER')
    
    # Force refresh every time this function is called
    try:
        # Clear ALL Streamlit caches completely
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Add a timestamp to ensure we're not using cached data
        cache_buster = time.time()
    except Exception as e:
        st.sidebar.warning(f"Cache clearing error: {str(e)}")
    
    # Define the primary paths where transaction data might be stored
    possible_paths = []
    
    # Add paths from environment variables first (most reliable)
    if output_folder:
        possible_paths.append(os.path.join(output_folder, session_id, f'all_transactions.csv'))
    if csv_folder:
        possible_paths.append(os.path.join(csv_folder, session_id, f'all_transactions.csv'))
    
    # Add fallback relative paths
    possible_paths.extend([
        os.path.join(os.path.abspath("../../GUI/web/output"), session_id, 'all_transactions.csv'),
        os.path.join(os.path.abspath("../../CSV Files"), session_id, 'all_transactions.csv'),
        # Try parent directories as well
        os.path.join(os.path.abspath("../output"), session_id, 'all_transactions.csv'),
        os.path.join(os.path.abspath("../../../CSV Files"), session_id, 'all_transactions.csv'),
    ])
    
    # Debug information in sidebar
    with st.sidebar:
        st.markdown("### Data Source")
        st.write(f"Session ID: {session_id}")
        st.write(f"Cache timestamp: {datetime.fromtimestamp(cache_buster).strftime('%H:%M:%S')}")
        
        # Remove the technical details expander that shows all file paths
        # Replace with a simple indicator of data status
        csv_path_found = False
        output_path_found = False
        
        for path in possible_paths:
            if os.path.exists(path):
                if 'CSV Files' in path:
                    csv_path_found = True
                if 'output' in path:
                    output_path_found = True
        
        if csv_path_found:
            st.success("✅ CSV data files found")
        if output_path_found:
            st.success("✅ Output data files found")
    
    # Try each path
    data_path_used = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                # Explicitly read with no caching
                df = pd.read_csv(path)
                # Verify we have actual data
                if len(df) > 0:
                    data_path_used = path
                    break
            except Exception as e:
                st.sidebar.error(f"Error reading {path}: {str(e)}")
                continue
    
    # Show which file was actually loaded
    if data_path_used:
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(data_path_used)).strftime('%H:%M:%S')
        st.sidebar.success(f"✅ Data loaded from: {os.path.basename(os.path.dirname(data_path_used))}")
        st.sidebar.info(f"File last modified at: {file_mod_time}")
        st.sidebar.info(f"Loaded {len(df)} transactions")
        
        # Add refresh button to force reload data
        if st.sidebar.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
            
        return df
    
    # If we're still looking for sample data for demo purposes
    sample_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_transactions.csv")
    if os.path.exists(sample_data_path):
        st.sidebar.warning("⚠️ Using SAMPLE data (not your actual transactions)")
        return pd.read_csv(sample_data_path)
    
    # If no data found, provide clear message and refresh button
    st.error("No transaction data found. This dashboard requires you to upload bank statements for processing first.")
    st.info("Please return to the main page and upload your bank statement PDFs.")
    
    if st.button("🔄 Try Again"):
        st.cache_data.clear()
        st.rerun()
    
    st.stop()  # Stop execution of the app
    return pd.DataFrame()  # This won't be reached due to st.stop()

def format_currency(amount):
    """Format amount as currency with thousands separator and 2 decimal places"""
    return f"${amount:,.2f}"

def prepare_data(df):
    """Clean and prepare the data for analysis"""
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Ensure amount is numeric
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Add or fix type column
    if 'type' not in df.columns:
        df['type'] = df.apply(lambda row: 'credit' if row['amount'] >= 0 else 'debit', axis=1)
    
    # Add month column for grouping
    df['month'] = df['date'].dt.strftime('%Y-%m')
    df['month_name'] = df['date'].dt.strftime('%b %Y')
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.day_name()
    
    # Add expense category based on description (simplified)
    def categorize(desc):
        desc = str(desc).lower()
        if any(word in desc for word in ['salary', 'deposit', 'income', 'bonus', 'interest', 'dividend', 'refund']):
            return 'Income'
        elif any(word in desc for word in ['grocery', 'food', 'market', 'supermarket']):
            return 'Groceries'
        elif any(word in desc for word in ['restaurant', 'dining', 'cafe', 'coffee', 'takeout']):
            return 'Dining'
        elif any(word in desc for word in ['rent', 'mortgage', 'hoa', 'housing']):
            return 'Housing'
        elif any(word in desc for word in ['gas', 'fuel', 'transit', 'uber', 'lyft', 'transport', 'subway', 'train', 'toll']):
            return 'Transportation'
        elif any(word in desc for word in ['utility', 'electric', 'water', 'internet', 'phone', 'gas bill', 'sewage']):
            return 'Utilities'
        elif any(word in desc for word in ['amazon', 'walmart', 'target', 'shop', 'store', 'purchase']):
            return 'Shopping'
        elif any(word in desc for word in ['doctor', 'medical', 'pharmacy', 'healthcare', 'dental', 'hospital']):
            return 'Healthcare'
        elif any(word in desc for word in ['entertain', 'movie', 'theater', 'music', 'concert', 'netflix', 'spotify']):
            return 'Entertainment'
        elif any(word in desc for word in ['travel', 'hotel', 'airbnb', 'flight', 'vacation']):
            return 'Travel'
        elif any(word in desc for word in ['insurance', 'premium']):
            return 'Insurance'
        elif any(word in desc for word in ['education', 'tuition', 'school', 'book', 'course']):
            return 'Education'
        elif any(word in desc for word in ['subscription', 'membership']):
            return 'Subscriptions'
        elif any(word in desc for word in ['tax', 'irs']):
            return 'Taxes'
        elif any(word in desc for word in ['saving', 'investment', '401k', 'ira']):
            return 'Savings & Investments'
        else:
            return 'Other'
    
    df['category'] = df['description'].astype(str).apply(categorize)
    
    return df

def calculate_financial_ratios(df):
    """Calculate key financial metrics and ratios"""
    # Income-related calculations
    income_df = df[df['amount'] > 0]
    expense_df = df[df['amount'] < 0]
    
    total_income = income_df['amount'].sum() if not income_df.empty else 0
    total_expenses = abs(expense_df['amount'].sum()) if not expense_df.empty else 0
    net_profit = total_income - total_expenses
    
    # Key financial ratios
    ratios = {
        'income_expense_ratio': total_income / total_expenses if total_expenses > 0 else 0,
        'profit_margin': net_profit / total_income if total_income > 0 else 0,
        'expense_ratio': total_expenses / total_income if total_income > 0 else 0,
        'savings_rate': (total_income - total_expenses) / total_income if total_income > 0 else 0
    }
    
    # Monthly averages
    monthly_data = df.groupby('month').agg(
        total_amount=('amount', 'sum'),
        income=('amount', lambda x: sum(i for i in x if i > 0)),
        expenses=('amount', lambda x: sum(i for i in x if i < 0)),
        transactions=('amount', 'count')
    ).reset_index()
    monthly_data['expenses'] = monthly_data['expenses'].abs()
    
    avg_monthly_income = total_income / len(monthly_data) if len(monthly_data) > 0 else 0
    avg_monthly_expenses = total_expenses / len(monthly_data) if len(monthly_data) > 0 else 0
    avg_monthly_savings = avg_monthly_income - avg_monthly_expenses
    
    # Category insights
    category_expenses = expense_df.groupby('category').agg(
        amount=('amount', lambda x: abs(sum(x))),
        count=('amount', 'count')
    ).reset_index()
    
    # Top expense categories
    if not category_expenses.empty:
        top_expenses = category_expenses.sort_values('amount', ascending=False).head(3)
        top_expense_categories = top_expenses['category'].tolist()
        top_expense_amounts = top_expenses['amount'].tolist()
    else:
        top_expense_categories = []
        top_expense_amounts = []
    
    # Income sources
    income_sources = income_df.groupby('category').agg(
        amount=('amount', 'sum'),
        count=('amount', 'count')
    ).reset_index()
    
    # Find extreme months
    if not monthly_data.empty:
        highest_income_month = monthly_data.loc[monthly_data['income'].idxmax()]
        highest_expense_month = monthly_data.loc[monthly_data['expenses'].idxmax()]
        lowest_income_month = monthly_data.loc[monthly_data['income'].idxmin()]
        lowest_expense_month = monthly_data.loc[monthly_data['expenses'].idxmin()]
    else:
        # Create empty Series with necessary fields to avoid errors
        highest_income_month = pd.Series({'month': 'N/A', 'income': 0})
        highest_expense_month = pd.Series({'month': 'N/A', 'expenses': 0})
        lowest_income_month = pd.Series({'month': 'N/A', 'income': 0})
        lowest_expense_month = pd.Series({'month': 'N/A', 'expenses': 0})
    
    # Volatility measures
    if len(monthly_data) > 1:
        income_volatility = monthly_data['income'].std() / monthly_data['income'].mean() if monthly_data['income'].mean() > 0 else 0
        expense_volatility = monthly_data['expenses'].std() / monthly_data['expenses'].mean() if monthly_data['expenses'].mean() > 0 else 0
    else:
        income_volatility = 0
        expense_volatility = 0
    
    # Calculate trends
    if len(monthly_data) > 1:
        # Create numeric X for regression
        monthly_data = monthly_data.sort_values('month')
        monthly_data['period'] = range(1, len(monthly_data) + 1)
        
        # Income trend
        income_model = LinearRegression()
        X = monthly_data['period'].values.reshape(-1, 1)
        y = monthly_data['income'].values
        income_model.fit(X, y)
        income_trend = income_model.coef_[0]  # Positive = upward trend, Negative = downward trend
        
        # Expense trend
        expense_model = LinearRegression()
        y = monthly_data['expenses'].values
        expense_model.fit(X, y)
        expense_trend = expense_model.coef_[0]
        
        # Net trend
        net_trend = income_trend - expense_trend
    else:
        income_trend = 0
        expense_trend = 0
        net_trend = 0
    
    # Combine all metrics
    metrics = {
        'total_income': total_income,
        'total_expenses': total_expenses,
        'net_profit': net_profit,
        'income_expense_ratio': ratios['income_expense_ratio'],
        'profit_margin': ratios['profit_margin'],
        'expense_ratio': ratios['expense_ratio'],
        'savings_rate': ratios['savings_rate'],
        'avg_monthly_income': avg_monthly_income,
        'avg_monthly_expenses': avg_monthly_expenses,
        'avg_monthly_savings': avg_monthly_savings,
        'top_expense_categories': top_expense_categories,
        'top_expense_amounts': top_expense_amounts,
        'income_sources': income_sources,
        'highest_income_month': highest_income_month,
        'highest_expense_month': highest_expense_month,
        'lowest_income_month': lowest_income_month,
        'lowest_expense_month': lowest_expense_month,
        'income_volatility': income_volatility,
        'expense_volatility': expense_volatility,
        'income_trend': income_trend,
        'expense_trend': expense_trend,
        'net_trend': net_trend
    }
    
    return metrics

def detect_seasonal_patterns(monthly_data):
    """
    Analyze monthly data to identify seasonal patterns in income and expenses
    
    Args:
        monthly_data: DataFrame with monthly income and expense data
        
    Returns:
        dict: Dictionary with detected seasonal factors for income and expenses
    """
    # Default seasonal factors (equal weighting for all months)
    default_factors = {
        'income': {i: 1.0 for i in range(1, 13)},
        'expenses': {i: 1.0 for i in range(1, 13)}
    }
    
    # Need at least 4 months of data to detect meaningful seasonal patterns
    if len(monthly_data) < 4:
        return default_factors
        
    try:
        # Group monthly data by month number and calculate average income and expenses
        # This will identify patterns like "December always has higher expenses"
        monthly_avgs = monthly_data.copy()
        
        # Check if index needs to be reset or if 'month' is already a column
        if 'month' not in monthly_avgs.columns and monthly_avgs.index.name != 'month':
            # If index is the month information, reset it to make it a column
            monthly_avgs = monthly_avgs.reset_index()
        
        # Now convert the month strings to datetime and extract month number
        if 'month' in monthly_avgs.columns:
            # Ensure month is in the YYYY-MM format and add day to make it parseable
            monthly_avgs['month_num'] = pd.to_datetime(monthly_avgs['month'] + '-01').dt.month
        else:
            # If somehow we don't have a 'month' column, create default month numbers
            monthly_avgs['month_num'] = range(1, len(monthly_avgs) + 1)
        
        # Calculate average income and expenses by month number
        month_patterns = monthly_avgs.groupby('month_num').agg({
            'income': 'mean',
            'expenses': 'mean'
        })
        
        # Get overall means
        income_mean = monthly_avgs['income'].mean()
        expenses_mean = monthly_avgs['expenses'].mean()
        
        # Calculate seasonal factors as percentage deviations from overall mean
        income_factors = {}
        expense_factors = {}
        
        for month_num, row in month_patterns.iterrows():
            # Prevent division by zero
            if income_mean > 0:
                income_factors[month_num] = row['income'] / income_mean
            else:
                income_factors[month_num] = 1.0
                
            if expenses_mean > 0:
                expense_factors[month_num] = row['expenses'] / expenses_mean
            else:
                expense_factors[month_num] = 1.0
        
        # Fill in any missing months with default value of 1.0
        for month_num in range(1, 13):
            if month_num not in income_factors:
                income_factors[month_num] = 1.0
            if month_num not in expense_factors:
                expense_factors[month_num] = 1.0
                
        return {
            'income': income_factors,
            'expenses': expense_factors
        }
    
    except Exception as e:
        # If anything goes wrong, return default factors
        return default_factors

def forecast_financials(df, forecast_periods=6):
    """
    Generate financial forecasts based on historical data
    
    Args:
        df: DataFrame with transaction data
        forecast_periods: Number of months to forecast
    
    Returns:
        tuple: (historical monthly data, forecast data)
    """
    # Create monthly aggregations
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Ensure date is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Add month column for grouping
    df['month'] = df['date'].dt.strftime('%Y-%m')
    
    # Group by month
    monthly_data = df.groupby('month').agg(
        income=('amount', lambda x: sum(i for i in x if i > 0)),
        expenses=('amount', lambda x: abs(sum(i for i in x if i < 0))),
        net=('amount', 'sum'),
        transactions=('amount', 'count')
    ).sort_index()
    
    # Calculate volatility (standard deviation / mean)
    if len(monthly_data) >= 3:
        income_volatility = monthly_data['income'].std() / monthly_data['income'].mean() if monthly_data['income'].mean() > 0 else 0.1
        expense_volatility = monthly_data['expenses'].std() / monthly_data['expenses'].mean() if monthly_data['expenses'].mean() > 0 else 0.1
    else:
        # Default volatility for insufficient data
        income_volatility = 0.1
        expense_volatility = 0.1
    
    # If less than 3 months of data, return historical only without forecast
    if len(monthly_data) < 3:
        return monthly_data, pd.DataFrame()
        
    # Detect seasonal patterns from historical data
    seasonal_factors = detect_seasonal_patterns(monthly_data)
    
    # Create a DataFrame for the forecast
    try:
        # Handle both string index and DataFrame index cases
        if isinstance(monthly_data.index[-1], str):
            last_date_str = monthly_data.index[-1]
            last_date = pd.to_datetime(last_date_str + '-01')
        else:
            # If the index is not a string (might be a reset index DataFrame)
            if 'month' in monthly_data.columns:
                last_date_str = monthly_data['month'].iloc[-1]
                last_date = pd.to_datetime(last_date_str + '-01')
            else:
                # Fallback to current date if no valid date found
                last_date = pd.to_datetime('today')
    except Exception as e:
        # Fallback to current date if any errors occur
        print(f"Error parsing last date: {e}")
        last_date = pd.to_datetime('today')
        
    # Create forecast months
    forecast_months = [(last_date + pd.DateOffset(months=i+1)).strftime('%Y-%m') 
                     for i in range(forecast_periods)]
    forecast_month_nums = [(last_date + pd.DateOffset(months=i+1)).month 
                        for i in range(forecast_periods)]
    
    # Calculate trends
    if len(monthly_data) >= 3:
        # Simple trend calculation based on average month-to-month change
        income_trend = monthly_data['income'].diff().mean() / monthly_data['income'].mean() \
            if monthly_data['income'].mean() > 0 else 0.01
        expense_trend = monthly_data['expenses'].diff().mean() / monthly_data['expenses'].mean() \
            if monthly_data['expenses'].mean() > 0 else 0.01
            
        # Cap trends to reasonable values
        income_trend = max(min(income_trend, 0.10), -0.10)  # Cap at ±10% 
        expense_trend = max(min(expense_trend, 0.10), -0.10)  # Cap at ±10%
    else:
        # Default trend values for insufficient data
        income_trend = 0.01  # 1% growth
        expense_trend = 0.01  # 1% growth
    
    # Use average of last 3 months (or all if less) as base for forecasting
    lookback = min(3, len(monthly_data))
    base_income = monthly_data['income'].iloc[-lookback:].mean()
    base_expenses = monthly_data['expenses'].iloc[-lookback:].mean()
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame(index=forecast_months)
    
    # Add the month column explicitly to ensure it's available
    forecast_df['month'] = forecast_months
    
    # Generate forecast values with trends, seasonality, and some randomness
    for i, (month, month_num) in enumerate(zip(forecast_months, forecast_month_nums)):
        # Apply trend (compound growth)
        trend_factor_income = (1 + income_trend) ** (i + 1)
        trend_factor_expenses = (1 + expense_trend) ** (i + 1)
        
        # Apply seasonality
        season_factor_income = seasonal_factors['income'].get(month_num, 1.0)
        season_factor_expenses = seasonal_factors['expenses'].get(month_num, 1.0)
        
        # Calculate expected values
        expected_income = base_income * trend_factor_income * season_factor_income
        expected_expenses = base_expenses * trend_factor_expenses * season_factor_expenses
        
        # Add best and worst case scenarios based on volatility
        best_case_income = expected_income * (1 + income_volatility)
        worst_case_income = expected_income * (1 - income_volatility)
        
        best_case_expenses = expected_expenses * (1 - expense_volatility)
        worst_case_expenses = expected_expenses * (1 + expense_volatility)
        
        # Store in forecast dataframe
        forecast_df.loc[month, 'income'] = expected_income
        forecast_df.loc[month, 'expenses'] = expected_expenses
        forecast_df.loc[month, 'net'] = expected_income - expected_expenses
        
        # Add best and worst case values
        forecast_df.loc[month, 'income_best'] = best_case_income
        forecast_df.loc[month, 'income_worst'] = worst_case_income
        forecast_df.loc[month, 'expenses_best'] = best_case_expenses
        forecast_df.loc[month, 'expenses_worst'] = worst_case_expenses
        forecast_df.loc[month, 'net_best'] = best_case_income - best_case_expenses
        forecast_df.loc[month, 'net_worst'] = worst_case_income - worst_case_expenses
    
    # Return both historical and forecast data for continuity in charts
    return monthly_data, forecast_df

def display_dashboard(df, metrics):
    """Display the main dashboard with all the charts and metrics"""
    # DASHBOARD LAYOUT
    # Tabs for navigation
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Income Analysis", "Expense Analysis", "Financial Forecasting"])

    # OVERVIEW TAB
    with tab1:
        st.header("Financial Overview")
        
        # Top KPI Cards - 1st row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Income", f"${metrics['total_income']:,.2f}")
        col2.metric("Total Expenses", f"${metrics['total_expenses']:,.2f}")
        col3.metric("Net Profit", f"${metrics['net_profit']:,.2f}")
        col4.metric("Income/Expense Ratio", f"{metrics['income_expense_ratio']:.2f}")
        
        # Top KPI Cards - 2nd row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Monthly Income (Avg)", f"${metrics['avg_monthly_income']:,.2f}")
        col2.metric("Monthly Expenses (Avg)", f"${metrics['avg_monthly_expenses']:,.2f}")
        col3.metric("Monthly Savings (Avg)", f"${metrics['avg_monthly_savings']:,.2f}")
        col4.metric("Savings Rate", f"{metrics['savings_rate']:.1%}")

        # Income vs Expenses Trends
        st.subheader("Monthly Income vs Expenses")
        
        # Group by month for trend analysis
        monthly_data = df.groupby('month').agg(
            total_amount=('amount', 'sum'),
            income=('amount', lambda x: sum(i for i in x if i > 0)),
            expenses=('amount', lambda x: sum(i for i in x if i < 0)),
            transactions=('amount', 'count')
        ).reset_index()
        monthly_data['expenses'] = monthly_data['expenses'].abs()
        
        if not monthly_data.empty:
            # Sort by month to ensure chronological order
            monthly_data = monthly_data.sort_values('month')
            monthly_data['month_name'] = monthly_data['month'].apply(lambda x: pd.to_datetime(x + '-01').strftime('%b %Y'))
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly_data['month_name'],
                y=monthly_data['income'],
                name='Income',
                marker_color='green',
                text=[f"${x:,.2f}" for x in monthly_data['income']],
                textposition='auto'
            ))
            fig.add_trace(go.Bar(
                x=monthly_data['month_name'],
                y=monthly_data['expenses'],
                name='Expenses',
                marker_color='red',
                text=[f"${x:,.2f}" for x in monthly_data['expenses']],
                textposition='auto'
            ))
            fig.add_trace(go.Scatter(
                x=monthly_data['month_name'],
                y=monthly_data['total_amount'],
                name='Net',
                mode='lines+markers',
                line=dict(color='blue', width=3),
                marker=dict(size=8),
                text=[f"${x:,.2f}" for x in monthly_data['total_amount']],
                hoverinfo='text+name'
            ))
            
            fig.update_layout(
                barmode='group',
                title='Monthly Income vs Expenses',
                xaxis_title='Month',
                yaxis_title='Amount ($)',
                legend_title='Type',
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough monthly data to display trends.")

    # INCOME ANALYSIS TAB
    with tab2:
        st.header("Income Analysis")
        
        # Income metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Income", f"${metrics['total_income']:,.2f}")
        col2.metric("Monthly Average", f"${metrics['avg_monthly_income']:,.2f}")
        
        # Find income trend direction
        income_trend_direction = "↗️" if metrics['income_trend'] > 0 else "↘️" if metrics['income_trend'] < 0 else "→"
        income_trend_percentage = abs(metrics['income_trend'] / metrics['avg_monthly_income'] * 100) if metrics['avg_monthly_income'] > 0 else 0
        
        col3.metric("Income Trend", f"{income_trend_direction} {income_trend_percentage:.1f}% per month")
        col4.metric("Income Volatility", f"{metrics['income_volatility'] * 100:.1f}%", 
                help="Higher values indicate greater monthly fluctuations")
        
        # Income sources analysis
        st.subheader("Income Sources")
        income_df = df[df['amount'] > 0]
        
        if not income_df.empty:
            # Get income by description
            income_by_description = income_df.groupby('description').agg(
                amount=('amount', 'sum'),
                transactions=('amount', 'count')
            ).reset_index().sort_values('amount', ascending=False).head(10)
            
            if not income_by_description.empty:
                # Format the table display
                display_data = income_by_description.copy()
                display_data['amount'] = display_data['amount'].apply(lambda x: f"${x:,.2f}")
                display_data.columns = ['Source', 'Amount', 'Transactions']
                
                st.subheader("Top Income Sources")
                st.dataframe(
                    display_data,
                    hide_index=True,
                    column_config={
                        "Source": st.column_config.TextColumn("Source"),
                        "Amount": st.column_config.TextColumn("Amount"),
                        "Transactions": st.column_config.NumberColumn("Transactions", format="%d")
                    }
                )
            
            # Monthly income trends
            st.subheader("Monthly Income Trends")
            income_by_month = income_df.groupby('month').agg(
                amount=('amount', 'sum'),
                transactions=('amount', 'count')
            ).reset_index()
            
            if not income_by_month.empty and len(income_by_month) > 1:
                # Sort by month to ensure chronological order
                income_by_month = income_by_month.sort_values('month')
                income_by_month['month_name'] = income_by_month['month'].apply(lambda x: pd.to_datetime(x + '-01').strftime('%b %Y'))
                
                fig = px.line(
                    income_by_month, 
                    x='month_name', 
                    y='amount',
                    markers=True,
                    title='Monthly Income Trend',
                    labels={'amount': 'Income ($)', 'month_name': 'Month'},
                    color_discrete_sequence=['green']
                )
                
                # Add trend line (simple linear regression)
                x = np.arange(len(income_by_month))
                y = income_by_month['amount'].values
                
                if len(x) > 1:
                    model = LinearRegression()
                    model.fit(x.reshape(-1, 1), y)
                    trend_y = model.predict(x.reshape(-1, 1))
                    
                    fig.add_trace(go.Scatter(
                        x=income_by_month['month_name'],
                        y=trend_y,
                        mode='lines',
                        name='Trend',
                        line=dict(color='darkgreen', width=2, dash='dash')
                    ))
                
                # Add data labels
                fig.update_traces(
                    texttemplate='$%{y:,.2f}',
                    textposition='top center'
                )
                
                # Layout adjustments
                fig.update_layout(
                    xaxis_title='Month',
                    yaxis_title='Income ($)',
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)

    # EXPENSE ANALYSIS TAB
    with tab3:
        st.header("Expense Analysis")
        
        # Expense metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Expenses", f"${metrics['total_expenses']:,.2f}")
        col2.metric("Monthly Average", f"${metrics['avg_monthly_expenses']:,.2f}")
        
        # Find expense trend direction
        expense_trend_direction = "↗️" if metrics['expense_trend'] > 0 else "↘️" if metrics['expense_trend'] < 0 else "→"
        expense_trend_percentage = abs(metrics['expense_trend'] / metrics['avg_monthly_expenses'] * 100) if metrics['avg_monthly_expenses'] > 0 else 0
        
        col3.metric("Expense Trend", f"{expense_trend_direction} {expense_trend_percentage:.1f}% per month")
        col4.metric("Expense Volatility", f"{metrics['expense_volatility'] * 100:.1f}%", 
                help="Higher values indicate greater monthly fluctuations")
        
        # Expense category analysis
        st.subheader("Expense Categories")
        expense_df = df[df['amount'] < 0].copy()
        expense_df['amount'] = expense_df['amount'].abs()  # Convert to positive for easier analysis
        
        if not expense_df.empty:
            # Get expenses by category
            expense_by_category = expense_df.groupby('category').agg(
                amount=('amount', 'sum'),
                transactions=('amount', 'count')
            ).reset_index().sort_values('amount', ascending=False)
            
            # Top expense categories visualization
            if not expense_by_category.empty:
                # Create pie chart for top categories
                top_categories = expense_by_category.head(8).copy()
                
                if len(expense_by_category) > 8:
                    other_sum = expense_by_category.iloc[8:]['amount'].sum()
                    other_count = expense_by_category.iloc[8:]['transactions'].sum()
                    
                    other_row = pd.DataFrame({
                        'category': ['Other'],
                        'amount': [other_sum],
                        'transactions': [other_count]
                    })
                    
                    top_categories = pd.concat([top_categories, other_row])
                
                # Create the pie chart
                fig = px.pie(
                    top_categories,
                    values='amount',
                    names='category',
                    title='Expense Categories',
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    hover_data=['transactions']
                )
                
                # Customize hover template
                fig.update_traces(
                    hovertemplate='<b>%{label}</b><br>Amount: $%{value:.2f}<br>Transactions: %{customdata[0]}<br>Percentage: %{percent:.1%}'
                )
                
                # Improve layout
                fig.update_layout(
                    height=450
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show expense breakdown table
                st.subheader("Expense Breakdown")
                
                # Format for display
                display_data = expense_by_category.copy()
                display_data['amount'] = display_data['amount'].apply(lambda x: f"${x:,.2f}")
                display_data['percentage'] = expense_by_category['amount'] / expense_by_category['amount'].sum() * 100
                display_data['percentage'] = display_data['percentage'].apply(lambda x: f"{x:.1f}%")
                display_data.columns = ['Category', 'Amount', 'Transactions', 'Percentage']
                
                st.dataframe(
                    display_data,
                    hide_index=True,
                    column_config={
                        "Category": st.column_config.TextColumn("Category"),
                        "Amount": st.column_config.TextColumn("Amount"),
                        "Transactions": st.column_config.NumberColumn("Transactions", format="%d"),
                        "Percentage": st.column_config.TextColumn("Percentage")
                    }
                )
            
            # Monthly expense trends
            st.subheader("Monthly Expense Trends")
            
            monthly_expenses = expense_df.groupby(['month', 'category']).agg(
                amount=('amount', 'sum')
            ).reset_index()
            
            if len(monthly_expenses) > 0:
                monthly_expenses = monthly_expenses.sort_values('month')
                monthly_expenses['month_name'] = monthly_expenses['month'].apply(lambda x: pd.to_datetime(x + '-01').strftime('%b %Y'))
                
                # Get top 5 categories for stacked area chart
                top_5_categories = expense_by_category.head(5)['category'].tolist()
                filtered_data = monthly_expenses[monthly_expenses['category'].isin(top_5_categories)]
                
                # Group all other categories as "Other"
                other_data = monthly_expenses[~monthly_expenses['category'].isin(top_5_categories)]
                if not other_data.empty:
                    other_grouped = other_data.groupby('month').agg(amount=('amount', 'sum')).reset_index()
                    other_grouped['category'] = 'Other'
                    other_grouped['month_name'] = other_grouped['month'].apply(lambda x: pd.to_datetime(x + '-01').strftime('%b %Y'))
                    
                    # Combine top 5 and "Other"
                    filtered_data = pd.concat([filtered_data, other_grouped])
                
                # Create stacked area chart
                fig = px.area(
                    filtered_data,
                    x='month_name',
                    y='amount',
                    color='category',
                    title='Expense Categories Over Time',
                    labels={'amount': 'Expenses ($)', 'month_name': 'Month', 'category': 'Category'}
                )
                
                fig.update_layout(
                    xaxis_title='Month',
                    yaxis_title='Expenses ($)',
                    height=400,
                    legend=dict(
                        title='Category',
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    ),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            # Expense patterns by day of month
            st.subheader("Expense Patterns by Day of Month")
            expense_by_day = expense_df.groupby('day').agg(
                amount=('amount', lambda x: abs(sum(x))),
                transactions=('amount', 'count')
            ).reset_index()
            
            if not expense_by_day.empty:
                # Create the chart
                fig = px.bar(
                    expense_by_day.sort_values('day'),
                    x='day',
                    y='amount',
                    title='Expenses by Day of Month',
                    color='amount',
                    color_continuous_scale='Reds',
                    text=[f"${x:,.0f}" for x in expense_by_day['amount']]
                )
                
                fig.update_layout(
                    xaxis_title='Day of Month',
                    yaxis_title='Expenses ($)',
                    height=350,
                    xaxis=dict(tickmode='linear', dtick=5)  # Show every 5th day
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No expense data available for analysis.")

    # FINANCIAL FORECASTING TAB
    with tab4:
        st.header("Financial Forecasting")
        
        # Validate that we have sufficient data
        if df.empty or len(df['month'].unique()) < 2:
            st.warning("Not enough data for forecasting. You need at least 2 months of transaction data.")
            st.info("Upload more bank statements with transactions across multiple months to enable forecasting.")
        else:
            # Forecasting period selector
            forecast_periods = st.slider("Forecast Months Ahead", min_value=1, max_value=12, value=6)
            
            # Generate unique hash for this data to verify it's being used
            data_hash = hash(str(df.head(10)))
            st.markdown(f"<span style='display:none'>Data Hash: {data_hash}</span>", unsafe_allow_html=True)
            
            # Regenerate forecasts with selected period
            historical_data, forecast_data = forecast_financials(df, forecast_periods)
            
            # Calculate volatility from historical data
            income_volatility = 0
            expense_volatility = 0
            
            if len(historical_data) >= 3:
                income_volatility = historical_data['income'].std() / historical_data['income'].mean() * 100 if historical_data['income'].mean() > 0 else 0
                expense_volatility = historical_data['expenses'].std() / historical_data['expenses'].mean() * 100 if historical_data['expenses'].mean() > 0 else 0
            
            # Display volatility metrics
            st.subheader("Forecast Accuracy Factors")
            
            col1, col2 = st.columns(2)
            
            # Volatility indicators
            vol_status = {
                'low': '🟢 Low',
                'medium': '🟡 Medium',
                'high': '🔴 High'
            }
            
            income_vol_status = vol_status['low'] if income_volatility < 10 else (vol_status['medium'] if income_volatility < 20 else vol_status['high'])
            expense_vol_status = vol_status['low'] if expense_volatility < 10 else (vol_status['medium'] if expense_volatility < 20 else vol_status['high'])
            
            col1.metric(
                "Income Volatility", 
                f"{income_volatility:.1f}%",
                income_vol_status
            )
            
            col2.metric(
                "Expense Volatility", 
                f"{expense_volatility:.1f}%",
                expense_vol_status
            )
            
            with st.expander("About Volatility in Forecasting"):
                st.markdown("""
                **Volatility** measures how much your income and expenses fluctuate from month to month. Higher volatility means:
                
                1. **Less predictable finances** - Large swings make forecasting more difficult
                2. **More variability in forecasts** - Our model incorporates this variability to provide more realistic projections
                3. **Need for larger emergency funds** - To handle unexpected variations
                
                *The forecasts shown include these volatility adjustments to give you more realistic scenarios.*
                """)
                
            with st.expander("About Our Advanced Forecasting Model"):
                st.markdown("""
                ### How Our Financial Forecasting Works
                
                Unlike simple trend projections that show the same growth pattern every month, our model incorporates:
                
                1. **Historical Volatility** - Using the actual month-to-month variation in your past financial data
                2. **Seasonal Patterns** - Recognizing common financial patterns throughout the year:
                   - Income boosts during tax refund season (March-April)
                   - Higher expenses during holiday shopping (November-December)
                   - Post-holiday spending reduction (January-February)
                   - Vacation spending increases in summer months
                   
                3. **Realistic Monthly Variations** - Instead of smooth lines, our forecast shows the ups and downs you're likely to experience
                
                This approach provides a more accurate picture of your future finances with realistic monthly variations rather than just showing average trends.
                """)
            
            # KPI forecast cards
            if not historical_data.empty:
                last_historical_month = historical_data.iloc[-1]
                # Get the month string, handling different DataFrame structures
                if isinstance(historical_data.index[-1], str):
                    last_historical_month_str = historical_data.index[-1]
                elif 'month' in last_historical_month:
                    last_historical_month_str = last_historical_month['month']
                else:
                    last_historical_month_str = 'N/A'
            else:
                last_historical_month = pd.Series({'income': 0, 'expenses': 0, 'net': 0})
                last_historical_month_str = 'N/A'
                
            if not forecast_data.empty:
                last_forecast_month = forecast_data.iloc[-1]
                # Get the month string, handling different DataFrame structures
                if isinstance(forecast_data.index[-1], str):
                    last_forecast_month_str = forecast_data.index[-1]
                elif 'month' in last_forecast_month:
                    last_forecast_month_str = last_forecast_month['month']
                else:
                    last_forecast_month_str = 'N/A'
            else:
                last_forecast_month = pd.Series({'income': 0, 'expenses': 0, 'net': 0})
                last_forecast_month_str = 'N/A'
            
            # Format month names
            if last_historical_month_str != 'N/A':
                try:
                    last_historical_month_name = pd.to_datetime(last_historical_month_str + '-01').strftime('%b %Y')
                except:
                    last_historical_month_name = last_historical_month_str
            else:
                last_historical_month_name = 'N/A'
                
            if last_forecast_month_str != 'N/A':
                try:
                    last_forecast_month_name = pd.to_datetime(last_forecast_month_str + '-01').strftime('%b %Y')
                except:
                    last_forecast_month_name = last_forecast_month_str
            else:
                last_forecast_month_name = 'N/A'
            
            # Calculate projected changes
            income_change = ((last_forecast_month['income'] / last_historical_month['income']) - 1) * 100 if last_historical_month['income'] > 0 else 0
            expense_change = ((last_forecast_month['expenses'] / last_historical_month['expenses']) - 1) * 100 if last_historical_month['expenses'] > 0 else 0
            net_change = ((last_forecast_month['net'] / last_historical_month['net']) - 1) * 100 if last_historical_month['net'] > 0 else 0
            
            # Display forecast metrics
            st.subheader(f"Projected Financial Changes ({last_historical_month_name} → {last_forecast_month_name})")
            
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Projected Income", 
                f"${last_forecast_month['income']:,.2f}", 
                f"{income_change:+.1f}%" if income_change != 0 else "No change"
            )
            
            col2.metric(
                "Projected Expenses", 
                f"${last_forecast_month['expenses']:,.2f}", 
                f"{expense_change:+.1f}%" if expense_change != 0 else "No change"
            )
            
            col3.metric(
                "Projected Net Profit", 
                f"${last_forecast_month['net']:,.2f}", 
                f"{net_change:+.1f}%" if net_change != 0 else "No change"
            )
            
            # Add volatility-adjusted projections
            st.subheader("Volatility-Adjusted Scenarios")
            st.write("How volatility affects your financial future")
            
            # Create scenario calculations if we have forecast data
            if not forecast_data.empty:
                # Calculate best and worst case scenarios for last forecast month
                best_income = last_forecast_month['income'] * (1 + (income_volatility / 100))
                worst_income = last_forecast_month['income'] * (1 - (income_volatility / 100))
                
                best_expenses = last_forecast_month['expenses'] * (1 - (expense_volatility / 100))
                worst_expenses = last_forecast_month['expenses'] * (1 + (expense_volatility / 100))
                
                # Calculate net profits for each scenario
                expected_net = last_forecast_month['income'] - last_forecast_month['expenses']
                best_net = best_income - best_expenses
                worst_net = worst_income - worst_expenses
                
                # Create tabular comparison
                scenario_data = {
                    "Scenario": ["Worst Case", "Expected", "Best Case"],
                    "Income": [f"${worst_income:,.2f}", f"${last_forecast_month['income']:,.2f}", f"${best_income:,.2f}"],
                    "Expenses": [f"${worst_expenses:,.2f}", f"${last_forecast_month['expenses']:,.2f}", f"${best_expenses:,.2f}"],
                    "Net Profit": [f"${worst_net:,.2f}", f"${expected_net:,.2f}", f"${best_net:,.2f}"]
                }
                
                # Display as a styled table with proper formatting
                scenario_df = pd.DataFrame(scenario_data)
                
                # Create three columns for the scenarios
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 😓 Worst Case")
                    st.metric("Income", f"${worst_income:,.2f}", f"{(worst_income/last_forecast_month['income'] - 1)*100:.1f}%")
                    st.metric("Expenses", f"${worst_expenses:,.2f}", f"{(worst_expenses/last_forecast_month['expenses'] - 1)*100:.1f}%")
                    st.metric("Net Profit", f"${worst_net:,.2f}", f"{(worst_net/expected_net - 1)*100:.1f}%" if expected_net > 0 else "N/A")
                    
                with col2:
                    st.markdown("#### 😐 Expected")
                    st.metric("Income", f"${last_forecast_month['income']:,.2f}")
                    st.metric("Expenses", f"${last_forecast_month['expenses']:,.2f}")
                    st.metric("Net Profit", f"${expected_net:,.2f}")
                    
                with col3:
                    st.markdown("#### 😀 Best Case")
                    st.metric("Income", f"${best_income:,.2f}", f"+{(best_income/last_forecast_month['income'] - 1)*100:.1f}%")
                    st.metric("Expenses", f"${best_expenses:,.2f}", f"-{(1 - best_expenses/last_forecast_month['expenses'])*100:.1f}%")
                    st.metric("Net Profit", f"${best_net:,.2f}", f"+{(best_net/expected_net - 1)*100:.1f}%" if expected_net > 0 else "N/A")
                
                # Recommendation based on volatility
                st.markdown("### Planning Recommendations")
                
                if income_volatility > 15 or expense_volatility > 15:
                    st.warning("""
                    **High Volatility Detected**: Consider building a larger emergency fund to cover unexpected fluctuations in your finances.
                    
                    **Recommended Emergency Fund**: ${:,.2f} (3-6 months of highest possible expenses)
                    """.format(worst_expenses * 6))
                else:
                    st.success("""
                    **Stable Financial Pattern**: Your income and expenses show relatively predictable patterns.
                    
                    **Recommended Emergency Fund**: ${:,.2f} (3 months of expected expenses)
                    """.format(last_forecast_month['expenses'] * 3))
                
                # Display savings potential
                savings_expected = expected_net * forecast_periods
                savings_best = best_net * forecast_periods
                
                st.info(f"""
                **Potential Savings** over the next {forecast_periods} months:
                - Expected: **${savings_expected:,.2f}**
                - Best Case: **${savings_best:,.2f}**
                """)
                
            # Combined historical and forecast chart
            st.subheader("Income and Expense Forecast with Best/Worst Case Scenarios")
            
            if not historical_data.empty and not forecast_data.empty:
                # Prepare historical data
                historical_df = historical_data.reset_index()
                historical_df['month_dt'] = pd.to_datetime(historical_df['month'] + '-01')
                historical_df['month_name'] = historical_df['month_dt'].dt.strftime('%b %Y')
                
                # Prepare forecast data
                forecast_df = forecast_data.reset_index()
                forecast_df['month_dt'] = pd.to_datetime(forecast_df['month'] + '-01')
                forecast_df['month_name'] = forecast_df['month_dt'].dt.strftime('%b %Y')
                
                # Calculate best and worst case for each month based on volatility
                forecast_df['best_income'] = forecast_df['income'] * (1 + (income_volatility/100))
                forecast_df['worst_income'] = forecast_df['income'] * (1 - (income_volatility/100))
                forecast_df['best_expenses'] = forecast_df['expenses'] * (1 - (expense_volatility/100))
                forecast_df['worst_expenses'] = forecast_df['expenses'] * (1 + (expense_volatility/100))
                forecast_df['best_net'] = forecast_df['best_income'] - forecast_df['best_expenses']
                forecast_df['worst_net'] = forecast_df['worst_income'] - forecast_df['worst_expenses']
                
                # Create figure with subplots - one for income and one for expenses
                fig = make_subplots(rows=2, cols=1, 
                                   shared_xaxes=True, 
                                   vertical_spacing=0.1,
                                   subplot_titles=("Income Forecasts (Expected, Best & Worst Case)", 
                                                  "Expense Forecasts (Expected, Best & Worst Case)"))
                
                # Income Plot (Top)
                # Historical income
                fig.add_trace(go.Scatter(
                    x=historical_df['month_name'],
                    y=historical_df['income'],
                    name='Historical Income',
                    mode='lines+markers',
                    line=dict(color='green', width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>%{x}</b><br>Income: $%{y:,.2f}<br>'
                ), row=1, col=1)
                
                # Expected forecast income
                fig.add_trace(go.Scatter(
                    x=forecast_df['month_name'],
                    y=forecast_df['income'],
                    name='Expected Income',
                    mode='lines+markers',
                    line=dict(color='green', width=3, dash='dash'),
                    marker=dict(size=8, symbol='circle-open'),
                    hovertemplate='<b>%{x}</b><br>Expected Income: $%{y:,.2f}<br>'
                ), row=1, col=1)
                
                # Best case income
                fig.add_trace(go.Scatter(
                    x=forecast_df['month_name'],
                    y=forecast_df['best_income'],
                    name='Best Case Income',
                    mode='lines+markers',
                    line=dict(color='darkgreen', width=2, dash='dot'),
                    marker=dict(size=6, symbol='triangle-up'),
                    hovertemplate='<b>%{x}</b><br>Best Case: $%{y:,.2f}<br>'
                ), row=1, col=1)
                
                # Worst case income
                fig.add_trace(go.Scatter(
                    x=forecast_df['month_name'],
                    y=forecast_df['worst_income'],
                    name='Worst Case Income',
                    mode='lines+markers',
                    line=dict(color='lightgreen', width=2, dash='dot'),
                    marker=dict(size=6, symbol='triangle-down'),
                    hovertemplate='<b>%{x}</b><br>Worst Case: $%{y:,.2f}<br>'
                ), row=1, col=1)
                
                # Expense Plot (Bottom)
                # Historical expenses
                fig.add_trace(go.Scatter(
                    x=historical_df['month_name'],
                    y=historical_df['expenses'],
                    name='Historical Expenses',
                    mode='lines+markers',
                    line=dict(color='red', width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>%{x}</b><br>Expenses: $%{y:,.2f}<br>'
                ), row=2, col=1)
                
                # Expected forecast expenses
                fig.add_trace(go.Scatter(
                    x=forecast_df['month_name'],
                    y=forecast_df['expenses'],
                    name='Expected Expenses',
                    mode='lines+markers',
                    line=dict(color='red', width=3, dash='dash'),
                    marker=dict(size=8, symbol='circle-open'),
                    hovertemplate='<b>%{x}</b><br>Expected Expenses: $%{y:,.2f}<br>'
                ), row=2, col=1)
                
                # Best case expenses (lower is better)
                fig.add_trace(go.Scatter(
                    x=forecast_df['month_name'],
                    y=forecast_df['best_expenses'],
                    name='Best Case Expenses',
                    mode='lines+markers',
                    line=dict(color='lightcoral', width=2, dash='dot'),
                    marker=dict(size=6, symbol='triangle-down'),
                    hovertemplate='<b>%{x}</b><br>Best Case: $%{y:,.2f}<br>'
                ), row=2, col=1)
                
                # Worst case expenses (higher is worse)
                fig.add_trace(go.Scatter(
                    x=forecast_df['month_name'],
                    y=forecast_df['worst_expenses'],
                    name='Worst Case Expenses',
                    mode='lines+markers',
                    line=dict(color='darkred', width=2, dash='dot'),
                    marker=dict(size=6, symbol='triangle-up'),
                    hovertemplate='<b>%{x}</b><br>Worst Case: $%{y:,.2f}<br>'
                ), row=2, col=1)
                
                # Add vertical line to mark forecast start
                if not historical_df.empty and not forecast_df.empty:
                    forecast_start_date = forecast_df['month_name'].iloc[0]
                    for i in range(1, 3):
                        fig.add_vline(
                            x=forecast_start_date,
                            line_width=2,
                            line_dash="dash",
                            line_color="gray",
                            opacity=0.7,
                            row=i, col=1
                        )
                
                # Update layout
                fig.update_layout(
                    title='Monthly Income and Expense Forecasts with Best/Worst Case Scenarios',
                    height=700,
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    ),
                    hovermode='x unified'
                )
                
                # Update x and y axis labels
                fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
                fig.update_yaxes(title_text="Amount ($)", row=2, col=1)
                fig.update_xaxes(title_text="Month", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation of the scenarios
                st.write("""
                **Chart Legend:**
                - **Solid Lines**: Historical data
                - **Dashed Lines**: Expected forecast
                - **Dotted Lines (Up Triangles)**: Maximum value with volatility
                - **Dotted Lines (Down Triangles)**: Minimum value with volatility
                
                The forecast shows different scenarios based on your historical data patterns:
                - **Expected**: The most likely outcome based on trends and patterns
                - **Best Case**: For income (higher is better), expenses (lower is better)
                - **Worst Case**: For income (lower is worse), expenses (higher is worse)
                """)
                
                # Add Net Income Forecast Chart (best, expected, worst)
                st.subheader("Net Profit Forecast Scenarios")
                
                fig = go.Figure()
                
                # Historical net
                fig.add_trace(go.Scatter(
                    x=historical_df['month_name'],
                    y=historical_df['net'],
                    name='Historical Net',
                    mode='lines+markers',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>%{x}</b><br>Net: $%{y:,.2f}<br>'
                ))
                
                # Expected forecast net
                fig.add_trace(go.Scatter(
                    x=forecast_df['month_name'],
                    y=forecast_df['net'],
                    name='Expected Net',
                    mode='lines+markers',
                    line=dict(color='blue', width=3, dash='dash'),
                    marker=dict(size=8, symbol='circle-open'),
                    hovertemplate='<b>%{x}</b><br>Expected Net: $%{y:,.2f}<br>'
                ))
                
                # Best case net
                fig.add_trace(go.Scatter(
                    x=forecast_df['month_name'],
                    y=forecast_df['best_net'],
                    name='Best Case Net',
                    mode='lines+markers',
                    line=dict(color='darkblue', width=2, dash='dot'),
                    marker=dict(size=6, symbol='triangle-up'),
                    hovertemplate='<b>%{x}</b><br>Best Case: $%{y:,.2f}<br>'
                ))
                
                # Worst case net
                fig.add_trace(go.Scatter(
                    x=forecast_df['month_name'],
                    y=forecast_df['worst_net'],
                    name='Worst Case Net',
                    mode='lines+markers',
                    line=dict(color='lightblue', width=2, dash='dot'),
                    marker=dict(size=6, symbol='triangle-down'),
                    hovertemplate='<b>%{x}</b><br>Worst Case: $%{y:,.2f}<br>'
                ))
                
                # Add vertical line to mark forecast start
                if not historical_df.empty and not forecast_df.empty:
                    forecast_start_date = forecast_df['month_name'].iloc[0]
                    fig.add_vline(
                        x=forecast_start_date,
                        line_width=2,
                        line_dash="dash",
                        line_color="gray",
                        opacity=0.7
                    )
                
                # Update layout
                fig.update_layout(
                    title='Net Profit Forecast (Best, Expected, and Worst Case)',
                    xaxis_title='Month',
                    yaxis_title='Net Profit ($)',
                    height=400,
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    ),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a table with specific month-by-month forecast values including scenarios
                st.subheader("Detailed Monthly Forecast Values (All Scenarios)")
                
                # Round values for better display
                forecast_table = forecast_df.copy()
                display_columns = [
                    'month_name',
                    'income', 'best_income', 'worst_income',
                    'expenses', 'best_expenses', 'worst_expenses',
                    'net', 'best_net', 'worst_net'
                ]
                forecast_table = forecast_table[display_columns]
                
                # Format table with currency
                formatted_table = forecast_table.copy()
                for col in formatted_table.columns:
                    if col != 'month_name':
                        formatted_table[col] = formatted_table[col].map('${:,.2f}'.format)
                
                # Rename columns for display
                formatted_table.columns = [
                    'Month',
                    'Expected Income', 'Best Income', 'Worst Income',
                    'Expected Expenses', 'Best Expenses', 'Worst Expenses',
                    'Expected Net Profit', 'Best Net Profit', 'Worst Net Profit'
                ]
                
                # Display the table with all scenarios
                st.dataframe(
                    formatted_table,
                    hide_index=True,
                    use_container_width=True
                )
                
                # Add detailed forecast table with monthly variations
                st.subheader("Monthly Percent Changes")
                st.write("Month-by-month percentage changes between forecasted months")
                
                # Calculate month-over-month percentage changes for forecast data
                if len(forecast_df) > 1:
                    forecast_mom = forecast_df.copy()
                    
                    # Calculate month-over-month percentage changes
                    forecast_mom['income_pct_change'] = forecast_mom['income'].pct_change() * 100
                    forecast_mom['expenses_pct_change'] = forecast_mom['expenses'].pct_change() * 100
                    
                    # First month has no change - set to 0
                    forecast_mom.iloc[0, forecast_mom.columns.get_loc('income_pct_change')] = 0
                    forecast_mom.iloc[0, forecast_mom.columns.get_loc('expenses_pct_change')] = 0
                    
                    # Create a figure for month-over-month percentage changes
                    fig = go.Figure()
                    
                    # Add income percentage change
                    fig.add_trace(go.Bar(
                        x=forecast_mom['month_name'].iloc[1:],  # Skip first month which has no change
                        y=forecast_mom['income_pct_change'].iloc[1:],
                        name='Income % Change',
                        marker_color='green',
                        text=[f"{x:+.1f}%" for x in forecast_mom['income_pct_change'].iloc[1:]],
                        textposition='auto'
                    ))
                    
                    # Add expenses percentage change
                    fig.add_trace(go.Bar(
                        x=forecast_mom['month_name'].iloc[1:],
                        y=forecast_mom['expenses_pct_change'].iloc[1:],
                        name='Expenses % Change',
                        marker_color='red',
                        text=[f"{x:+.1f}%" for x in forecast_mom['expenses_pct_change'].iloc[1:]],
                        textposition='auto'
                    ))
                    
                    # Add zero line
                    fig.add_shape(
                        type='line',
                        x0=-0.5,
                        x1=len(forecast_mom)-0.5,
                        y0=0,
                        y1=0,
                        line=dict(color='gray', width=2, dash='dash')
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title='Monthly Variations (% Change from Previous Month)',
                        xaxis_title='Month',
                        yaxis_title='Percent Change (%)',
                        barmode='group',
                        height=400,
                        legend=dict(
                            orientation='h',
                            yanchor='bottom', 
                            y=1.02,
                            xanchor='right',
                            x=1
                        ),
                        hovermode='x unified'
                    )
                    
                    # Add annotations explaining major variations
                    for i in range(1, len(forecast_mom)):
                        # Add annotation for significant changes (more than 10%)
                        income_change = forecast_mom['income_pct_change'].iloc[i]
                        expense_change = forecast_mom['expenses_pct_change'].iloc[i]
                        
                        # Extract month number directly from the date in month_name
                        month_dt = forecast_mom['month_dt'].iloc[i]
                        month_num = month_dt.month  # This is already a datetime
                        
                        # Explain significant income changes
                        if abs(income_change) > 10:
                            explanation = ""
                            if month_num in [3, 4] and income_change > 0:
                                explanation = "Tax refund season"
                            elif month_num in [12, 1] and income_change > 0:
                                explanation = "Holiday bonuses"
                            elif month_num in [6, 7] and income_change < 0:
                                explanation = "Summer income dip"
                            
                            if explanation:
                                fig.add_annotation(
                                    x=forecast_mom['month_name'].iloc[i],
                                    y=income_change,
                                    text=explanation,
                                    showarrow=True,
                                    arrowhead=2,
                                    yshift=10,
                                    bgcolor="rgba(255, 255, 255, 0.8)"
                                )
                        
                        # Explain significant expense changes
                        if abs(expense_change) > 10:
                            explanation = ""
                            if month_num in [11, 12] and expense_change > 0:
                                explanation = "Holiday shopping"
                            elif month_num in [1, 2] and expense_change < 0:
                                explanation = "Post-holiday reduction"
                            elif month_num in [7, 8] and expense_change > 0:
                                explanation = "Summer vacation"
                            
                            if explanation:
                                fig.add_annotation(
                                    x=forecast_mom['month_name'].iloc[i],
                                    y=expense_change,
                                    text=explanation,
                                    showarrow=True,
                                    arrowhead=2,
                                    yshift=-10,
                                    bgcolor="rgba(255, 255, 255, 0.8)"
                                )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add helpful explanation of monthly variations
                    st.info("""
                    **Understanding Monthly Variations:**
                    - **Positive values** indicate an increase compared to the previous month
                    - **Negative values** show a decrease from the previous month
                    - These variations factor in both your historical volatility patterns and typical seasonal financial behaviors
                    - Recognizing these patterns can help you better prepare for upcoming financial changes
                    """)
                
            else:
                st.info("Not enough historical data to generate a forecast.")
                
        # Add back button and data summary in sidebar
        if st.sidebar.button("Back to Results"):
            st.sidebar.markdown('<meta http-equiv="refresh" content="0;URL=/results/%s"/>' % session_id, unsafe_allow_html=True)

        # Data summary in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Data Summary")
        st.sidebar.markdown(f"Total Transactions: **{len(df)}**")
        st.sidebar.markdown(f"Date Range: **{df['date'].min().strftime('%b %d, %Y')}** to **{df['date'].max().strftime('%b %d, %Y')}**")
        st.sidebar.markdown(f"Total Income: **${metrics['total_income']:,.2f}**")
        st.sidebar.markdown(f"Total Expenses: **${metrics['total_expenses']:,.2f}**")
        st.sidebar.markdown(f"Net Profit: **${metrics['net_profit']:,.2f}**")
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Financial Health Indicators")
        st.sidebar.markdown(f"Savings Rate: **{metrics['savings_rate']:.1%}**")
        st.sidebar.markdown(f"Income/Expense Ratio: **{metrics['income_expense_ratio']:.2f}**")
        st.sidebar.markdown(f"Profit Margin: **{metrics['profit_margin']:.1%}**")

        # Add monthly averages
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Monthly Averages")
        st.sidebar.markdown(f"Avg. Monthly Income: **${metrics['avg_monthly_income']:,.2f}**")
        st.sidebar.markdown(f"Avg. Monthly Expenses: **${metrics['avg_monthly_expenses']:,.2f}**")
        st.sidebar.markdown(f"Avg. Monthly Savings: **${metrics['avg_monthly_savings']:,.2f}**")

        # Add trends if there's enough data
        if 'income_trend' in metrics and 'expense_trend' in metrics:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Financial Trends")
            
            income_trend = metrics['income_trend']
            expense_trend = metrics['expense_trend']
            
            income_trend_icon = "📈" if income_trend > 0 else "📉"
            expense_trend_icon = "📈" if expense_trend > 0 else "📉"
            
            st.sidebar.markdown(f"Income Trend: **{income_trend_icon} {'+'if income_trend > 0 else ''}{income_trend:.2f}/month**")
            st.sidebar.markdown(f"Expense Trend: **{expense_trend_icon} {'+'if expense_trend > 0 else ''}{expense_trend:.2f}/month**")

        # Add explanation of metrics
        with st.sidebar.expander("What do these metrics mean?"):
            st.markdown("""
            - **Savings Rate**: Percentage of income saved (higher is better)
            - **Income/Expense Ratio**: How many times income covers expenses (>1 is good)
            - **Profit Margin**: Percentage of income that becomes profit (higher is better)
            - **Income Trend**: Direction and rate of income change per month
            - **Expense Trend**: Direction and rate of expense change per month
            - **Income Volatility**: Variation in monthly income (lower is more stable)
            - **Expense Volatility**: Variation in monthly expenses (lower is more stable)
            """)

# Main function
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Financial KPI Dashboard")
    parser.add_argument("--session_id", type=str, help="Session ID for loading data")
    parser.add_argument("--cache_timestamp", type=str, help="Cache invalidation timestamp")
    args = parser.parse_args()
    
    # Use the parsed session_id or a default if not provided
    session_id = args.session_id if args.session_id else "test"
    
    # Create a container in the sidebar to show simplified session info
    with st.sidebar:
        # No need to display session ID again as it's already shown in Data Source section
        
        # Instead of showing paths, just show data status
        st.write("### Data Status")
        
        # Check if data files exist
        csv_folder = os.environ.get('CSV_FOLDER', '')
        output_folder = os.environ.get('OUTPUT_FOLDER', '')
        data_found = False
        
        if csv_folder:
            session_csv_dir = os.path.join(csv_folder, session_id)
            all_tx_path = os.path.join(session_csv_dir, 'all_transactions.csv')
            
            if os.path.exists(all_tx_path):
                try:
                    all_df = pd.read_csv(all_tx_path)
                    st.success(f"✅ {len(all_df)} transactions loaded")
                    data_found = True
                except Exception:
                    pass
        
        if output_folder and not data_found:
            output_session_dir = os.path.join(output_folder, session_id)
            output_tx_path = os.path.join(output_session_dir, 'all_transactions.csv')
            
            if os.path.exists(output_tx_path):
                try:
                    output_df = pd.read_csv(output_tx_path)
                    st.success(f"✅ {len(output_df)} transactions loaded")
                    data_found = True
                except Exception:
                    pass
                    
        if not data_found:
            st.warning("⚠️ No transaction data found")
    
    try:
        # Force clear cache again here to be extra sure
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Load and prepare data
        df = load_transaction_data(session_id)
        
        if not df.empty:
            st.sidebar.success(f"Successfully loaded {len(df)} transactions")
            df = prepare_data(df)
            
            # Calculate metrics
            metrics = calculate_financial_ratios(df)
            
            # Display dashboard
            display_dashboard(df, metrics)
        else:
            st.error("No transaction data found. Please upload bank statements first.")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.sidebar.error(f"Error: {str(e)}")
        
        # Show traceback for debugging
        import traceback
        with st.sidebar.expander("Detailed Error Information"):
            st.code(traceback.format_exc())