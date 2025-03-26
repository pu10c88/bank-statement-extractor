import os
import tempfile
import sys
from pathlib import Path

# App information
APP_NAME = "Bank Statement Extractor"
APP_VERSION = "2.0.0"
APP_AUTHOR = "SIGMA BI - Development Team"

# Add the Extractor Files directory to the system path - Fix the absolute path
extractor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Extractor Files'))
sys.path.insert(0, extractor_path)
print(f"Added to path: {extractor_path}")

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session, jsonify
import shutil
import threading
import time
import uuid
import json
import logging
import traceback
from datetime import datetime
import numpy as np

# Force matplotlib to use non-interactive backend (for any image generation needs)
import matplotlib
matplotlib.use('Agg')  # This must be set before importing pyplot
import matplotlib.pyplot as plt
import io
import base64
from werkzeug.utils import secure_filename

# Import the extraction functions from the existing scripts
from chase_statements_load import load_pdf as chase_load_pdf
from chase_statements_load import merge_csv_files
from bofa_statements_load import load_pdf as bofa_load_pdf

app = Flask(__name__)
app.secret_key = "bank_extractor_secret_key"
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Set APP_ROOT for file path references
app.config['APP_ROOT'] = os.path.dirname(os.path.abspath(__file__))

# Ensure that the upload directory exists (use absolute path)
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'uploads'))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"Upload folder: {UPLOAD_FOLDER}")

# Ensure that the CSV output directory exists (use absolute path from project root)
CSV_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../CSV Files'))
os.makedirs(CSV_FOLDER, exist_ok=True)
print(f"CSV folder: {CSV_FOLDER}")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CSV_FOLDER'] = CSV_FOLDER
app.config['OUTPUT_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Store processing status for each user session
processing_status = {}
processing_logs = {}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join(app.config['OUTPUT_FOLDER'], 'app.log')
)
logger = logging.getLogger(__name__)

# In-memory storage for job status
processing_jobs = {}
progress_data = {}

# Define functions that were previously imported from bank_extraction
def get_bank_types():
    """Return a list of supported bank types"""
    return ["chase", "bofa"]

def process_bank_statement(file_path, bank_type, output_dir):
    """
    Process a bank statement and extract transactions
    
    Args:
        file_path (Path): Path to the PDF statement file
        bank_type (str): Type of bank (chase, bofa)
        output_dir (Path): Output directory for CSV files
        
    Returns:
        list: List of transaction dictionaries
    """
    # Ensure output directory exists
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate bank type
    if bank_type.lower() not in get_bank_types():
        raise ValueError(f"Unsupported bank type: {bank_type}")
    
    # Load PDF based on bank type
    try:
        if bank_type.lower() == "chase":
            df = chase_load_pdf(file_path)
        elif bank_type.lower() == "bofa":
            df = bofa_load_pdf(file_path)
        else:
            raise ValueError(f"Unsupported bank type: {bank_type}")
        
        # Check if any transactions were found
        if df.empty:
            return []
        
        # Create output CSV filename
        csv_file = output_dir / file_path.with_suffix('.csv').name
        
        # Add transaction type if not present
        if 'type' not in df.columns:
            df['type'] = df.apply(lambda row: 'credit' if row['amount'] >= 0 else 'debit', axis=1)
        
        # Standardize date format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # Save to CSV
        df.to_csv(csv_file, index=False)
        
        # Convert to list of dictionaries for return
        transactions = df.to_dict('records')
        
        return transactions
    
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")
        raise

@app.route('/')
def index():
    session_id = get_session_id()
    
    # Check if we need to display financial summary
    show_summary = request.args.get('show_summary', 'false') == 'true'
    transactions = request.args.get('transactions', '0')
    income = request.args.get('income', '0.00')
    expenses = request.args.get('expenses', '0.00')
    profit = request.args.get('profit', '0.00')
    
    template_vars = {
        'session_id': session_id,
        'show_summary': show_summary,
        'transactions': transactions,
        'income': income,
        'expenses': expenses,
        'profit': profit,
        'app_name': APP_NAME,
        'app_version': APP_VERSION,
        'app_author': APP_AUTHOR
    }
    return render_template('index.html', **template_vars)

@app.route('/upload', methods=['POST'])
def upload_file():
    session_id = get_session_id()
    
    # Check if the post request has the file part
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    files = request.files.getlist('files[]')
    bank_type = request.form.get('bank_type', 'chase')
    
    # If no files selected
    if not files or files[0].filename == '':
        flash('No files selected')
        return redirect(request.url)
    
    # Create a session directory
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    # Clear any existing files
    for existing_file in os.listdir(session_dir):
        os.remove(os.path.join(session_dir, existing_file))
    
    # Save all uploaded files
    filenames = []
    for file in files:
        if file and file.filename.endswith('.pdf'):
            filename = file.filename
            file_path = os.path.join(session_dir, filename)
            file.save(file_path)
            filenames.append(filename)
    
    if filenames:
        # Start processing in a separate thread
        processing_status[session_id] = {
            'status': 'processing',
            'progress': 0,
            'total_files': len(filenames)
        }
        processing_logs[session_id] = []
        
        thread = threading.Thread(target=process_pdfs, args=(session_dir, bank_type, session_id))
        thread.daemon = True
        thread.start()
        
        flash(f'Uploaded {len(filenames)} files. Processing started.')
    else:
        flash('No valid PDF files uploaded')
    
    return redirect(url_for('status', session_id=session_id))

@app.route('/status/<session_id>')
def status(session_id):
    if session_id in processing_status:
        status_data = processing_status[session_id]
        logs = processing_logs.get(session_id, [])
        return render_template('status.html', 
                              status=status_data,
                              logs=logs,
                              session_id=session_id,
                              app_name=APP_NAME,
                              app_version=APP_VERSION,
                              app_author=APP_AUTHOR)
    else:
        flash('No processing session found')
        return redirect(url_for('index'))

@app.route('/status_data/<session_id>')
def status_data(session_id):
    if session_id in processing_status:
        status_data = processing_status[session_id]
        logs = processing_logs.get(session_id, [])
        return json.dumps({
            'status': status_data,
            'logs': logs
        })
    else:
        return json.dumps({'error': 'Session not found'})

@app.route('/results/<session_id>')
def results(session_id):
    session_csv_dir = os.path.join(app.config['CSV_FOLDER'], session_id)
    
    if not os.path.exists(session_csv_dir):
        flash('No results found for this session')
        return redirect(url_for('index'))
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(session_csv_dir) if f.endswith('.csv')]
    
    # Check if all_transactions.csv exists
    has_merged = 'all_transactions.csv' in csv_files
    csv_files = [f for f in csv_files if f != 'all_transactions.csv'] 
    
    # Variables for the template
    transactions = []
    income_total = "0.00"
    expense_total = "0.00"
    net_profit = "0.00"
    
    # Get transactions and summary information if merged file exists
    summary = None
    if has_merged:
        try:
            all_transactions_path = os.path.join(session_csv_dir, 'all_transactions.csv')
            df = pd.read_csv(all_transactions_path)
            
            # Add transaction type if not present
            if 'type' not in df.columns:
                df['type'] = df.apply(lambda row: 'credit' if float(row['amount']) >= 0 else 'debit', axis=1)
                # Save the updated dataframe with the type column
                df.to_csv(all_transactions_path, index=False)
            
            # Load transactions for the template
            transactions = df.to_dict('records')
            
            # Basic summary
            total_transactions = len(df)
            
            if 'amount' in df.columns:
                deposits = df[df['amount'] > 0]['amount'].sum()
                withdrawals = abs(df[df['amount'] < 0]['amount'].sum())
                net_change = deposits - withdrawals
                
                # Update template variables
                income_total = format_currency(deposits).replace('$', '')
                expense_total = format_currency(withdrawals).replace('$', '')
                net_profit = format_currency(net_change).replace('$', '')
                
                # Monthly breakdown
                df['date'] = pd.to_datetime(df['date'])
                df['month'] = df['date'].dt.strftime('%Y-%m')
                
                # Simplified monthly aggregation to avoid MultiIndex issues
                monthly_data = []
                for month_name in sorted(df['month'].unique(), reverse=True):
                    month_df = df[df['month'] == month_name]
                    count = len(month_df)
                    total = month_df['amount'].sum()
                    
                    # Calculate monthly income and expenses
                    month_income = month_df[month_df['amount'] > 0]['amount'].sum()
                    month_expenses = abs(month_df[month_df['amount'] < 0]['amount'].sum())
                    
                    monthly_data.append({
                        'month': month_name,
                        'count': count,
                        'total': round(total, 2),
                        'total_formatted': format_currency(total),
                        'income': round(month_income, 2),
                        'income_formatted': format_currency(month_income),
                        'expenses': round(month_expenses, 2),
                        'expenses_formatted': format_currency(month_expenses)
                    })
                
                summary = {
                    'total_transactions': total_transactions,
                    'deposits': round(deposits, 2),
                    'deposits_formatted': format_currency(deposits),
                    'withdrawals': round(withdrawals, 2),
                    'withdrawals_formatted': format_currency(withdrawals),
                    'net_change': round(net_change, 2),
                    'net_change_formatted': format_currency(net_change),
                    'monthly': monthly_data
                }
        except Exception as e:
            flash(f'Error generating summary: {str(e)}')
            app.logger.error(f"Error reading transactions: {str(e)}")
            print(f"Error reading transactions: {str(e)}")
    
    # If no merged file exists, try to load from individual CSVs
    elif csv_files:
        try:
            all_transactions = []
            
            for csv_file in csv_files:
                csv_path = os.path.join(session_csv_dir, csv_file)
                df = pd.read_csv(csv_path)
                
                # Add transaction type if not present
                if 'type' not in df.columns:
                    df['type'] = df.apply(lambda row: 'credit' if row['amount'] >= 0 else 'debit', axis=1)
                
                all_transactions.extend(df.to_dict('records'))
            
            if all_transactions:
                transactions = all_transactions
                
                # Calculate totals
                income = sum(float(t['amount']) for t in transactions if float(t['amount']) >= 0)
                expenses = sum(abs(float(t['amount'])) for t in transactions if float(t['amount']) < 0)
                profit = income - expenses
                
                income_total = format_currency(income).replace('$', '')
                expense_total = format_currency(expenses).replace('$', '')
                net_profit = format_currency(profit).replace('$', '')
        except Exception as e:
            flash(f'Error loading individual CSV files: {str(e)}')
            print(f"Error reading individual CSVs: {str(e)}")
    
    # Prepare template variables
    template_vars = {
        'session_id': session_id,
        'csv_files': csv_files,
        'has_merged': has_merged,
        'summary': summary,
        'transactions': transactions,
        'income_total': income_total,
        'expense_total': expense_total,
        'net_profit': net_profit,
        'csv_file': 'all_transactions.csv' if has_merged else None,
        'app_name': APP_NAME,
        'app_version': APP_VERSION,
        'app_author': APP_AUTHOR
    }
    return render_template('results.html', **template_vars)

@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    session_csv_dir = os.path.join(app.config['CSV_FOLDER'], session_id)
    file_path = os.path.join(session_csv_dir, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        flash('File not found')
        return redirect(url_for('results', session_id=session_id))

@app.route('/merge/<session_id>')
def merge_files(session_id):
    # Get the directory of CSV files for the session
    session_csv_dir = os.path.join(app.config['CSV_FOLDER'], session_id)
    
    if not os.path.exists(session_csv_dir):
        flash('No CSV files found for this session')
        return redirect(url_for('index'))
    
    # Get all CSV files in the session directory
    csv_files = [f for f in os.listdir(session_csv_dir) if f.endswith('.csv') and f != 'all_transactions.csv']
    
    if not csv_files:
        flash('No CSV files found for merging')
        return redirect(url_for('results', session_id=session_id))
    
    try:
        # Merge all CSV files into one
        merged_data = []
        for csv_file in csv_files:
            file_path = os.path.join(session_csv_dir, csv_file)
            try:
                df = pd.read_csv(file_path)
                
                # Ensure 'type' column exists with proper values
                if 'type' not in df.columns:
                    df['type'] = df.apply(lambda row: 'credit' if float(row['amount']) >= 0 else 'debit', axis=1)
                else:
                    # Make sure existing type values are correct
                    df['type'] = df.apply(lambda row: 'credit' if float(row['amount']) >= 0 else 'debit', axis=1)
                
                merged_data.append(df)
                app.logger.info(f"Added {len(df)} rows from {csv_file}")
            except Exception as e:
                app.logger.error(f"Error reading {csv_file}: {str(e)}")
        
        if not merged_data or len(merged_data) == 0:
            flash('No valid data found in CSV files')
            return redirect(url_for('results', session_id=session_id))
        
        # Concatenate all dataframes
        merged_df = pd.concat(merged_data, ignore_index=True)
        app.logger.info(f"Merged {len(merged_df)} total transactions")
        
        # Sort transactions by date (newest first)
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        merged_df = merged_df.sort_values('date', ascending=False)
        merged_df['date'] = merged_df['date'].dt.strftime('%Y-%m-%d')
        
        # Save the merged data to a new CSV file
        output_file = os.path.join(session_csv_dir, 'all_transactions.csv')
        merged_df.to_csv(output_file, index=False)
        app.logger.info(f"Saved merged transactions to {output_file}")
        
        flash('Files merged successfully! You can now download the combined CSV file.')
    except Exception as e:
        app.logger.error(f"Error merging files: {str(e)}")
        flash(f'Error merging files: {str(e)}')
    
    return redirect(url_for('results', session_id=session_id))

@app.route('/download_csv/<session_id>')
def download_csv(session_id):
    """Download the merged all_transactions.csv file"""
    session_csv_dir = os.path.join(app.config['CSV_FOLDER'], session_id)
    file_path = os.path.join(session_csv_dir, 'all_transactions.csv')
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name='all_transactions.csv')
    else:
        flash('Merged CSV file not found. Please merge your files first.')
        return redirect(url_for('results', session_id=session_id))

@app.route('/download_month/<session_id>/<month>')
def download_month(session_id, month):
    """Download transactions for a specific month"""
    session_csv_dir = os.path.join(app.config['CSV_FOLDER'], session_id)
    all_trans_path = os.path.join(session_csv_dir, 'all_transactions.csv')
    
    if not os.path.exists(all_trans_path):
        # Try to merge files first if they haven't been merged yet
        return redirect(url_for('merge_files', session_id=session_id))
    
    try:
        # Read the merged file
        df = pd.read_csv(all_trans_path)
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Extract year-month from date
        df['month'] = df['date'].dt.strftime('%Y-%m')
        
        # Filter by selected month
        month_df = df[df['month'] == month].copy()
        
        # Format dates back to string
        month_df['date'] = month_df['date'].dt.strftime('%Y-%m-%d')
        
        # Remove the temporary month column
        month_df = month_df.drop('month', axis=1)
        
        if month_df.empty:
            flash(f'No transactions found for {month}')
            return redirect(url_for('results', session_id=session_id))
        
        # Create a temporary file
        temp_file = os.path.join(session_csv_dir, f'month_{month}.csv')
        month_df.to_csv(temp_file, index=False)
        
        # Send the file for download
        filename = f'transactions_{month}.csv'
        return send_file(temp_file, as_attachment=True, download_name=filename)
        
    except Exception as e:
        flash(f'Error generating monthly report: {str(e)}')
        return redirect(url_for('results', session_id=session_id))

# Add a debugging route to check file locations
@app.route('/debug_paths/<session_id>')
def debug_paths(session_id):
    """Debug route to check file paths and transaction data"""
    # Check various file locations
    file_paths = {
        'output_dir': app.config['OUTPUT_FOLDER'],
        'csv_dir': app.config['CSV_FOLDER'],
        'session_output_dir': os.path.join(app.config['OUTPUT_FOLDER'], session_id),
        'session_csv_dir': os.path.join(app.config['CSV_FOLDER'], session_id),
    }
    
    # Check specific file existence
    file_existence = {
        'all_transactions_in_output': os.path.exists(os.path.join(app.config['OUTPUT_FOLDER'], session_id, 'all_transactions.csv')),
        'all_transactions_in_csv': os.path.exists(os.path.join(app.config['CSV_FOLDER'], session_id, 'all_transactions.csv')),
        'all_transactions_in_main_csv': os.path.exists(os.path.join(app.config['CSV_FOLDER'], 'all_transactions.csv')),
    }
    
    # Try to list files in relevant directories
    files_in_dirs = {}
    
    for name, path in file_paths.items():
        if os.path.exists(path):
            files_in_dirs[name] = os.listdir(path)
        else:
            files_in_dirs[name] = f"Directory doesn't exist: {path}"
    
    # Try to load data if a file exists
    data_summary = {}
    for name, exists in file_existence.items():
        if exists:
            path = None
            if name == 'all_transactions_in_output':
                path = os.path.join(app.config['OUTPUT_FOLDER'], session_id, 'all_transactions.csv')
            elif name == 'all_transactions_in_csv':
                path = os.path.join(app.config['CSV_FOLDER'], session_id, 'all_transactions.csv')
            elif name == 'all_transactions_in_main_csv':
                path = os.path.join(app.config['CSV_FOLDER'], 'all_transactions.csv')
            
            if path:
                try:
                    df = pd.read_csv(path)
                    data_summary[name] = {
                        'rows': len(df),
                        'columns': df.columns.tolist(),
                        'has_date': 'date' in df.columns,
                        'has_amount': 'amount' in df.columns,
                    }
                except Exception as e:
                    data_summary[name] = f"Error reading file: {str(e)}"
    
    # Render a debug template
    return jsonify({
        'file_paths': file_paths,
        'file_existence': file_existence,
        'files_in_dirs': files_in_dirs,
        'data_summary': data_summary,
    })

def load_transactions_data(session_id):
    """Load transaction data from various possible locations"""
    try:
        # Check multiple possible locations for transaction data
        possible_paths = [
            os.path.join(app.config['CSV_FOLDER'], session_id, 'all_transactions.csv'),
            os.path.join(app.config['OUTPUT_FOLDER'], session_id, 'all_transactions.csv'),
            # Fallback to demo data if needed (for testing)
            os.path.join(app.config['APP_ROOT'], 'static', 'demo_data', 'all_transactions.csv')
        ]
        
        transactions_file = None
        for path in possible_paths:
            if os.path.exists(path):
                transactions_file = path
                logger.info(f"Found transaction data at: {path}")
                break
        
        if not transactions_file:
            logger.error(f"No transaction data found. Searched paths: {possible_paths}")
            return pd.DataFrame()  # Return empty DataFrame
        
        # Load the data
        df = pd.read_csv(transactions_file)
        logger.info(f"Loaded {len(df)} transactions from {transactions_file}")
        
        # Ensure amount is numeric
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            
        # Ensure dates are in datetime format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        # Fill NaN values
        df = df.fillna({
            'description': 'Unknown',
            'amount': 0.0
        })
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading transaction data: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()  # Return empty DataFrame in case of error

def get_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def log_message(session_id, message):
    if session_id in processing_logs:
        processing_logs[session_id].append(message)

def format_currency(amount):
    """Format amount as currency with thousands separator and 2 decimal places"""
    return f"${amount:,.2f}"

def process_pdfs(directory, bank_type, session_id):
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]
    total_files = len(pdf_files)
    
    if total_files == 0:
        processing_status[session_id]['status'] = 'completed'
        processing_status[session_id]['progress'] = 100
        log_message(session_id, "No PDF files found in the uploaded files.")
        return
    
    # Create session-specific output directory
    session_csv_dir = os.path.join(app.config['CSV_FOLDER'], session_id)
    os.makedirs(session_csv_dir, exist_ok=True)
    
    log_message(session_id, f"Processing {total_files} PDF files...")
    log_message(session_id, f"Output CSV files will be saved to: {session_csv_dir}")
    
    processed_count = 0
    error_count = 0
    total_transactions = 0
    total_deposits = 0
    total_withdrawals = 0
    
    for i, pdf_file in enumerate(pdf_files):
        try:
            filename = os.path.basename(pdf_file)
            log_message(session_id, f"Loading PDF: {filename}...")
            
            try:
                # Load PDF based on selected bank
                if bank_type == 'chase':
                    df = chase_load_pdf(pdf_file)
                else:
                    df = bofa_load_pdf(pdf_file)
                
                if df.empty:
                    log_message(session_id, f"⚠️ No transactions found in {filename}")
                    continue
            except Exception as bank_error:
                # If the selected bank fails, try the other bank format
                log_message(session_id, f"⚠️ Error processing as {bank_type}: {str(bank_error)}")
                log_message(session_id, f"Attempting to process with alternative bank format...")
                
                try:
                    if bank_type == 'chase':
                        df = bofa_load_pdf(pdf_file)
                    else:
                        df = chase_load_pdf(pdf_file)
                    
                    if df.empty:
                        log_message(session_id, f"⚠️ No transactions found with alternative format in {filename}")
                        continue
                    
                    log_message(session_id, f"✓ Successfully extracted using alternative bank format")
                except Exception as alt_error:
                    raise Exception(f"Failed with both bank formats. Original error: {str(bank_error)}, Alternative error: {str(alt_error)}")
            
            # Add transaction type if not present
            if 'type' not in df.columns:
                df['type'] = df.apply(lambda row: 'credit' if float(row['amount']) >= 0 else 'debit', axis=1)
            
            log_message(session_id, f"✓ Successfully extracted {len(df)} transactions")
            
            # Calculate and display statistics
            if 'amount' in df.columns:
                deposits_df = df[df['amount'] > 0]
                statement_deposits = deposits_df['amount'].sum()
                total_deposits += statement_deposits
                log_message(session_id, f"✓ Statement deposits: {format_currency(statement_deposits)} ({len(deposits_df)} transactions)")
                
                withdrawals_df = df[df['amount'] < 0]
                statement_withdrawals = abs(withdrawals_df['amount'].sum())
                total_withdrawals += statement_withdrawals
                log_message(session_id, f"✓ Statement withdrawals: {format_currency(statement_withdrawals)} ({len(withdrawals_df)} transactions)")
                log_message(session_id, f"✓ Statement net change: {format_currency(statement_deposits - statement_withdrawals)}")
            else:
                log_message(session_id, f"⚠️ Warning: No 'amount' column found in extracted data")
            
            # Update running totals
            total_transactions += len(df)
            
            # Create output CSV filename
            csv_file = os.path.join(session_csv_dir, os.path.splitext(filename)[0] + '.csv')
            
            # Standardize date format
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            
            # Save extracted transactions to CSV file
            df.to_csv(csv_file, index=False)
            log_message(session_id, f"✓ Saved to CSV: {os.path.basename(csv_file)}")
            log_message(session_id, "-" * 50)
            
            # Increment counter for successfully processed files
            processed_count += 1
            
        except Exception as e:
            # If any error occurs during processing, catch and log it
            error_msg = f"❌ Error processing {os.path.basename(pdf_file)}: {str(e)}"
            log_message(session_id, error_msg)
            log_message(session_id, "-" * 50)
            error_count += 1
        
        # Update progress
        progress = int(((i + 1) / total_files) * 100)
        processing_status[session_id]['progress'] = progress
    
    # Print summary
    log_message(session_id, "\nProcess Completed!")
    log_message(session_id, "-" * 50)
    log_message(session_id, f"Successfully Processed: {processed_count} files")
    log_message(session_id, f"Total Transactions Processed: {total_transactions}")
    
    # Display financial summary
    net_profit = total_deposits - total_withdrawals
    log_message(session_id, "-" * 50)
    log_message(session_id, f"FINANCIAL SUMMARY:")
    log_message(session_id, f"Total Income/Deposits: {format_currency(total_deposits)}")
    log_message(session_id, f"Total Expenses/Withdrawals: {format_currency(total_withdrawals)}")
    log_message(session_id, f"Net Profit: {format_currency(net_profit)}")
    log_message(session_id, "-" * 50)
    
    log_message(session_id, f"Errors Encountered: {error_count} files")
    
    # Update processing status
    processing_status[session_id]['status'] = 'completed'
    processing_status[session_id]['processed_count'] = processed_count
    processing_status[session_id]['error_count'] = error_count
    processing_status[session_id]['total_transactions'] = total_transactions
    processing_status[session_id]['income'] = round(total_deposits, 2)
    processing_status[session_id]['expenses'] = round(total_withdrawals, 2)
    processing_status[session_id]['profit'] = round(net_profit, 2)
    processing_status[session_id]['income_formatted'] = format_currency(total_deposits)
    processing_status[session_id]['expenses_formatted'] = format_currency(total_withdrawals)
    processing_status[session_id]['profit_formatted'] = format_currency(net_profit)

def create_transaction_summary(transactions):
    """Create a summary of the transactions"""
    if not transactions:
        return None
    
    total_transactions = len(transactions)
    
    # Calculate the income and expenses
    income = sum(float(t['amount']) for t in transactions if t['type'] == 'credit')
    expenses = sum(float(t['amount']) for t in transactions if t['type'] == 'debit')
    net_profit = income - expenses
    
    # Format currency values with commas for thousands
    income_formatted = f"{income:,.2f}"
    expenses_formatted = f"{expenses:,.2f}"
    net_profit_formatted = f"{net_profit:,.2f}"
    
    return {
        'total_transactions': total_transactions,
        'income': income_formatted,
        'expenses': expenses_formatted,
        'net_profit': net_profit_formatted
    }

def get_template_vars():
    """Get common template variables"""
    return {
        'app_name': APP_NAME,
        'app_version': APP_VERSION,
        'app_author': APP_AUTHOR
    }

def process_files(job_id, files, bank_type, output_dir):
    """Process files in the background"""
    job = processing_jobs[job_id]
    all_transactions = []
    processed_count = 0
    
    for i, file_info in enumerate(files):
        try:
            # Update file status
            file_info['status'] = 'processing'
            job['files'] = files  # Update files in the job
            
            # Process the file
            logger.info(f"Processing file: {file_info['name']}")
            
            file_transactions = process_bank_statement(
                Path(file_info['path']), 
                bank_type, 
                Path(output_dir)
            )
            
            # Add to all transactions
            if file_transactions:
                all_transactions.extend(file_transactions)
            
            # Update status
            file_info['status'] = 'completed'
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing file {file_info['name']}: {str(e)}")
            logger.error(traceback.format_exc())
            file_info['status'] = 'error'
        
        # Update progress
        job['progress'] = int((i + 1) / len(files) * 100)
        job['processed_count'] = processed_count
        job['files'] = files  # Update files in the job
    
    # Create combined CSV with all transactions
    if all_transactions:
        try:
            import csv
            csv_path = os.path.join(output_dir, 'all_transactions.csv')
            
            # Sort transactions by date
            all_transactions.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'), reverse=True)
            
            if all_transactions and len(all_transactions) > 0:
                fieldnames = all_transactions[0].keys()
                
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for tx in all_transactions:
                        writer.writerow(tx)
        
        except Exception as e:
            logger.error(f"Error creating combined CSV: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Mark job as completed
    job['status'] = 'completed'
    job['completed'] = True
    job['total_transactions'] = len(all_transactions)
    job['end_time'] = datetime.now().isoformat()
    logger.info(f"Job {job_id} completed. Processed {processed_count} files with {len(all_transactions)} transactions.")

@app.route('/simple_streamlit/<session_id>')
def simple_streamlit(session_id):
    """Launch a Streamlit KPI dashboard with real data"""
    try:
        # Use a fixed port for Streamlit to make it more reliable
        streamlit_port = 8501
        logger.info(f"Using port for Streamlit: {streamlit_port}")
        
        # Create a Streamlit script that loads and analyzes real transaction data
        streamlit_script = '''
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import calendar
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Basic page configuration
st.set_page_config(page_title="Financial KPI Dashboard", page_icon="💰", layout="wide")

# Session ID for data loading
SESSION_ID = "SESSION_ID_PLACEHOLDER"

# Functions to load and process data
def load_transaction_data(session_id):
    # Load transaction data from CSV file
    # Check multiple possible locations for transaction data
    possible_paths = [
        os.path.join("/Users/paulocampos/Desktop/Work/Coding/Bank_Extraction_Code_V2/CSV Files", session_id, 'all_transactions.csv'),
        os.path.join("/Users/paulocampos/Desktop/Work/Coding/Bank_Extraction_Code_V2/GUI/web/output", session_id, 'all_transactions.csv'),
        # Fallback to demo data if needed
        os.path.join("/Users/paulocampos/Desktop/Work/Coding/Bank_Extraction_Code_V2/GUI/web/static/demo_data", 'all_transactions.csv')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            st.sidebar.success(f"Loaded data from: {os.path.basename(os.path.dirname(path))}")
            return pd.read_csv(path)
    
    # If no file found, create dummy data (shouldn't happen with demo data available)
    st.sidebar.error("No transaction data found. Using demo data.")
    return pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10, freq='M'),
        'description': ['Salary', 'Rent', 'Groceries', 'Utilities', 'Bonus', 'Restaurant', 'Gas', 'Insurance', 'Shopping', 'Entertainment'],
        'amount': [5000, -1500, -400, -200, 1000, -150, -80, -120, -350, -200],
        'type': ['credit', 'debit', 'debit', 'debit', 'credit', 'debit', 'debit', 'debit', 'debit', 'debit']
    })

def format_currency(amount):
    # Format amount as currency with thousands separator and 2 decimal places
    return f"${amount:,.2f}"

def prepare_data(df):
    # Clean and prepare the data for analysis
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
    
    # Add expense category based on description (simplified)
    def categorize(desc):
        desc = desc.lower()
        if any(word in desc for word in ['salary', 'deposit', 'income', 'bonus']):
            return 'Income'
        elif any(word in desc for word in ['grocery', 'food', 'market']):
            return 'Groceries'
        elif any(word in desc for word in ['restaurant', 'dining', 'cafe', 'coffee']):
            return 'Dining'
        elif any(word in desc for word in ['rent', 'mortgage']):
            return 'Housing'
        elif any(word in desc for word in ['gas', 'fuel', 'transit', 'uber', 'lyft']):
            return 'Transportation'
        elif any(word in desc for word in ['utility', 'electric', 'water', 'internet', 'phone']):
            return 'Utilities'
        elif any(word in desc for word in ['amazon', 'walmart', 'target', 'shop']):
            return 'Shopping'
        elif any(word in desc for word in ['doctor', 'medical', 'pharmacy']):
            return 'Healthcare'
        else:
            return 'Other'
    
    df['category'] = df['description'].astype(str).apply(categorize)
    
    return df

# Load and process data
df = load_transaction_data(SESSION_ID)
df = prepare_data(df)

# Calculate KPIs
income_df = df[df['amount'] > 0]
expense_df = df[df['amount'] < 0]

total_income = income_df['amount'].sum() if not income_df.empty else 0
total_expenses = abs(expense_df['amount'].sum()) if not expense_df.empty else 0
net_profit = total_income - total_expenses
income_expense_ratio = total_income / total_expenses if total_expenses > 0 else 0

# Group by month for trend analysis
monthly_data = df.groupby('month').agg(
    total_amount=('amount', 'sum'),
    income=('amount', lambda x: sum(i for i in x if i > 0)),
    expenses=('amount', lambda x: sum(i for i in x if i < 0)),
    transactions=('amount', 'count')
).reset_index()
monthly_data['expenses'] = monthly_data['expenses'].abs()

# Group by category for donut chart
category_data = df[df['amount'] < 0].groupby('category').agg(
    amount=('amount', lambda x: abs(sum(x))),
    count=('amount', 'count')
).reset_index()
category_data = category_data.sort_values('amount', ascending=False)

# Find highest income and expense months
if not monthly_data.empty:
    highest_income_month = monthly_data.loc[monthly_data['income'].idxmax()]
    highest_expense_month = monthly_data.loc[monthly_data['expenses'].idxmax()]
else:
    highest_income_month = pd.Series({'month': 'N/A', 'income': 0})
    highest_expense_month = pd.Series({'month': 'N/A', 'expenses': 0})

# Average monthly calculations
avg_monthly_income = total_income / len(monthly_data) if len(monthly_data) > 0 else 0
avg_monthly_expenses = total_expenses / len(monthly_data) if len(monthly_data) > 0 else 0

# DASHBOARD LAYOUT
st.title('Financial KPI Dashboard')

# Top KPI Cards
st.markdown("### Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Income", f"${total_income:,.2f}")
col2.metric("Total Expenses", f"${total_expenses:,.2f}")
col3.metric("Net Profit", f"${net_profit:,.2f}")
col4.metric("Income/Expense Ratio", f"{income_expense_ratio:.2f}")

# Second row of KPIs
st.markdown("### Financial Insights")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Monthly Income", f"${avg_monthly_income:,.2f}")
col2.metric("Avg Monthly Expenses", f"${avg_monthly_expenses:,.2f}")
col3.metric(f"Best Income Month", f"${highest_income_month['income']:,.2f} ({highest_income_month['month']})")
col4.metric(f"Highest Expense Month", f"${highest_expense_month['expenses']:,.2f} ({highest_expense_month['month']})")

# Income vs Expenses Trends
st.markdown("### Monthly Income vs Expenses")
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
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough monthly data to display trends.")

# Expense Breakdown
st.markdown("### Expense Breakdown by Category")
col1, col2 = st.columns([3, 2])

with col1:
    if not category_data.empty:
        # Create pie chart
        fig = px.pie(
            category_data, 
            values='amount', 
            names='category',
            title='Expense Distribution by Category',
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Plasma_r
        )
        
        # Update trace to add percentage and amount
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.2f}<br>Percentage: %{percent}'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No expense categories to display.")

with col2:
    if not category_data.empty:
        # Format the table display
        display_data = category_data.copy()
        display_data['amount'] = display_data['amount'].apply(lambda x: f"${x:,.2f}")
        display_data.columns = ['Category', 'Amount', 'Transactions']
        
        st.markdown("### Top Expense Categories")
        st.dataframe(
            display_data,
            hide_index=True,
            column_config={
                "Category": st.column_config.TextColumn("Category"),
                "Amount": st.column_config.TextColumn("Amount"),
                "Transactions": st.column_config.NumberColumn("Transactions")
            },
            height=400
        )
    else:
        st.info("No expense categories to display.")

# Transaction volume by month
st.markdown("### Transaction Volume by Month")
if not monthly_data.empty:
    fig = px.bar(
        monthly_data, 
        x='month_name', 
        y='transactions',
        text_auto=True,
        title='Number of Transactions per Month',
        color='transactions', 
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Transaction Count',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough data to display transaction volume.")

# Add a back button
if st.sidebar.button("Back to Results"):
    st.sidebar.markdown('<meta http-equiv="refresh" content="0;URL=/results/%s"/>' % SESSION_ID, unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown(f"Total Transactions: **{len(df)}**")
st.sidebar.markdown(f"Data Range: **{df['date'].min().strftime('%b %d, %Y')}** to **{df['date'].max().strftime('%b %d, %Y')}**")
'''
        
        # Replace SESSION_ID with the actual session_id
        streamlit_script = streamlit_script.replace('SESSION_ID_PLACEHOLDER', session_id)
        
        # Save the Streamlit script to a file
        temp_script_path = os.path.join(app.config['APP_ROOT'], f'temp_simple_streamlit_{session_id}.py')
        with open(temp_script_path, 'w') as f:
            f.write(streamlit_script)
        
        logger.info(f"Created KPI dashboard Streamlit script at {temp_script_path}")
        
        # Instead of embedding in an iframe, provide a link to the user with instructions
        flash("KPI dashboard is starting. Click the link below to access it.", "info")
        
        # Start Streamlit in a separate process without waiting for it
        import subprocess
        subprocess.Popen(
            [
                'streamlit', 'run', 
                temp_script_path, 
                '--server.port', str(streamlit_port),
                '--server.headless', 'true',
                '--browser.serverAddress', 'localhost',
                '--server.enableCORS', 'true',
                '--server.enableXsrfProtection', 'false'
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Return a page with a direct link to the Streamlit dashboard
        return render_template(
            'streamlit_redirect.html',
            session_id=session_id,
            streamlit_url=f"http://localhost:{streamlit_port}"
        )
        
    except Exception as e:
        logger.error(f"Error launching KPI dashboard: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f"Error launching dashboard: {str(e)}", "danger")
        return redirect(url_for('results', session_id=session_id))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4446) 