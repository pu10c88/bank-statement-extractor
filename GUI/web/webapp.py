import os
import tempfile
import sys
from pathlib import Path
from functools import wraps
import multiprocessing
import subprocess
import atexit  # Add atexit for cleanup when app exits

# App information
APP_NAME = "Bank Statement Extractor"
APP_VERSION = "2.1.0"  # Updated version number
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
from datetime import datetime, timedelta
import numpy as np
from flask_wtf.csrf import CSRFProtect  # Import CSRF protection
from werkzeug.exceptions import HTTPException

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

# Import CSV processing module
from csv_load import process_csv_file, is_csv_file, is_excel_file

# Initialize Flask app with static folder configuration
static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
if not os.path.exists(static_folder):
    os.makedirs(os.path.join(static_folder, 'images'), exist_ok=True)

app = Flask(__name__, static_folder=static_folder, static_url_path='/static')
app.secret_key = os.environ.get('SECRET_KEY', "bank_extractor_secret_key")

# Set session timeout to 30 minutes
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Ensure logo file exists
logo_path = os.path.join(static_folder, 'images', 'icone_1.png')
if not os.path.exists(logo_path):
    # Try to copy from Logos directory
    logos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Logos'))
    source_logo = os.path.join(logos_dir, 'icone_1.png')
    if os.path.exists(source_logo):
        shutil.copy(source_logo, logo_path)
        print(f"Copied logo from {source_logo} to {logo_path}")
    else:
        print(f"Warning: Logo file not found at {source_logo}")

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

# Configure enhanced logging
LOG_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs'))
os.makedirs(LOG_FOLDER, exist_ok=True)

# Create a custom formatter for the logs
class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.ERROR:
            return f"⚠️ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {record.getMessage()}"
        elif record.levelno == logging.WARNING:
            return f"⚡ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - WARNING - {record.getMessage()}"
        else:
            return f"ℹ️ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - {record.getMessage()}"

# Set up file handler
file_handler = logging.FileHandler(os.path.join(LOG_FOLDER, 'app.log'))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(CustomFormatter())

# Set up console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(CustomFormatter())

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# In-memory storage for job status
processing_jobs = {}
progress_data = {}
active_streamlit_processes = {}  # Store active Streamlit processes

# Store active sessions and their files
active_sessions = set()

# Add this function to clean up all files when the application is closed
def cleanup_all_files():
    """Clean up all files from uploads and CSV directories"""
    logger.info(f"Application shutting down. Cleaning up all files and sessions...")
    try:
        # Clean up all sessions first
        for session_id in list(active_sessions):
            try:
                cleanup_session_files(session_id)
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {str(e)}")
        
        # Then clean up main directories
        
        # Clean up uploads directory
        uploads_dir = app.config['UPLOAD_FOLDER']
        if os.path.exists(uploads_dir):
            try:
                # First remove all files in subdirectories
                for root, dirs, files in os.walk(uploads_dir, topdown=False):
                    for file in files:
                        try:
                            os.remove(os.path.join(root, file))
                            logger.info(f"Removed file: {os.path.join(root, file)}")
                        except Exception as e:
                            logger.error(f"Error deleting file {os.path.join(root, file)}: {str(e)}")
                
                # Then remove all subdirectories
                for root, dirs, files in os.walk(uploads_dir, topdown=False):
                    for dir in dirs:
                        try:
                            dir_path = os.path.join(root, dir)
                            os.rmdir(dir_path)
                            logger.info(f"Removed directory: {dir_path}")
                        except Exception as e:
                            logger.error(f"Error removing directory {os.path.join(root, dir)}: {str(e)}")
            except Exception as e:
                logger.error(f"Error cleaning uploads directory: {str(e)}")
        
        # Clean up CSV directory
        csv_dir = app.config['CSV_FOLDER']
        if os.path.exists(csv_dir):
            try:
                # First remove all files in subdirectories
                for root, dirs, files in os.walk(csv_dir, topdown=False):
                    for file in files:
                        try:
                            os.remove(os.path.join(root, file))
                            logger.info(f"Removed file: {os.path.join(root, file)}")
                        except Exception as e:
                            logger.error(f"Error deleting file {os.path.join(root, file)}: {str(e)}")
                
                # Then remove all subdirectories
                for root, dirs, files in os.walk(csv_dir, topdown=False):
                    for dir in dirs:
                        try:
                            dir_path = os.path.join(root, dir)
                            os.rmdir(dir_path)
                            logger.info(f"Removed directory: {dir_path}")
                        except Exception as e:
                            logger.error(f"Error removing directory {os.path.join(root, dir)}: {str(e)}")
            except Exception as e:
                logger.error(f"Error cleaning CSV directory: {str(e)}")
                
        # Clean up output directory
        output_dir = app.config['OUTPUT_FOLDER']
        if os.path.exists(output_dir):
            try:
                # First remove all files in subdirectories
                for root, dirs, files in os.walk(output_dir, topdown=False):
                    for file in files:
                        try:
                            os.remove(os.path.join(root, file))
                            logger.info(f"Removed file: {os.path.join(root, file)}")
                        except Exception as e:
                            logger.error(f"Error deleting file {os.path.join(root, file)}: {str(e)}")
                
                # Then remove all subdirectories
                for root, dirs, files in os.walk(output_dir, topdown=False):
                    for dir in dirs:
                        try:
                            dir_path = os.path.join(root, dir)
                            os.rmdir(dir_path)
                            logger.info(f"Removed directory: {dir_path}")
                        except Exception as e:
                            logger.error(f"Error removing directory {os.path.join(root, dir)}: {str(e)}")
            except Exception as e:
                logger.error(f"Error cleaning output directory: {str(e)}")
                
        # Kill all remaining Streamlit processes
        try:
            import signal
            for session_id, pids in active_streamlit_processes.items():
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        logger.info(f"Terminated Streamlit process {pid}")
                    except Exception as e:
                        logger.warning(f"Could not terminate process {pid}: {str(e)}")
            # Clear the streamlit processes dictionary
            active_streamlit_processes.clear()
        except Exception as e:
            logger.error(f"Error terminating Streamlit processes: {str(e)}")
        
        # Clear all global session tracking data
        active_sessions.clear()
        processing_status.clear()
        processing_logs.clear()
        processing_jobs.clear()
        progress_data.clear()
        
        logger.info("All files and sessions cleaned up successfully")
    except Exception as e:
        logger.error(f"Error in cleanup_all_files: {str(e)}")

# Register the cleanup function to run when the app exits
atexit.register(cleanup_all_files)

# Replace the old cleanup_all_sessions function with our new comprehensive one
def cleanup_all_sessions():
    """Redirect to the more comprehensive cleanup_all_files function"""
    cleanup_all_files()

# Custom error handler for all exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(traceback.format_exc())
    
    # Handle HTTP exceptions differently
    if isinstance(e, HTTPException):
        return render_template('error.html', 
                               error_code=e.code,
                               error_name=e.name,
                               error_description=e.description,
                               app_name=APP_NAME,
                               app_version=APP_VERSION,
                               app_author=APP_AUTHOR), e.code
    
    # Handle all other exceptions
    return render_template('error.html', 
                           error_code=500,
                           error_name="Internal Server Error",
                           error_description=str(e),
                           app_name=APP_NAME,
                           app_version=APP_VERSION,
                           app_author=APP_AUTHOR), 500

# A decorator to ensure user session is active
def require_session(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'session_id' not in session:
            flash("Your session has expired. Please start again.")
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

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
    
    # Add current session to active sessions
    active_sessions.add(session_id)
    
    # Clear existing files when loading the initial page
    try:
        cleanup_session_files(session_id)
    except Exception as e:
        logger.error(f"Error clearing files for session {session_id}: {str(e)}")
    
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
    pdf_filenames = []
    csv_excel_filenames = []
    
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(session_dir, filename)
            file.save(file_path)
            
            if filename.lower().endswith('.pdf'):
                pdf_filenames.append(filename)
            elif is_csv_file(filename) or is_excel_file(filename):
                csv_excel_filenames.append(filename)
    
    # Initialize processing
    processing_status[session_id] = {
        'status': 'processing',
        'progress': 0,
        'total_files': len(pdf_filenames) + len(csv_excel_filenames)
    }
    processing_logs[session_id] = []
    
    # Process files based on their type
    if pdf_filenames:
        processing_logs[session_id].append(f"Found {len(pdf_filenames)} PDF files to process.")
        thread = threading.Thread(target=process_pdfs, args=(session_dir, bank_type, session_id))
        thread.daemon = True
        thread.start()
    
    if csv_excel_filenames:
        processing_logs[session_id].append(f"Found {len(csv_excel_filenames)} CSV/Excel files to process.")
        thread = threading.Thread(target=process_csv_files, args=(session_id, app.config, processing_logs[session_id]))
        thread.daemon = True
        thread.start()
    
    if pdf_filenames or csv_excel_filenames:
        flash(f'Uploaded {len(pdf_filenames)} PDF files and {len(csv_excel_filenames)} CSV/Excel files. Processing started.')
    else:
        flash('No valid files uploaded')
    
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
    """Show processing results"""
    session_csv_dir = os.path.join(app.config['CSV_FOLDER'], session_id)
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
    
    # Ensure directories exist
    os.makedirs(session_csv_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for all_transactions.csv file
    all_transactions_file = os.path.join(session_csv_dir, 'all_transactions.csv')
    transactions = []
    
    if os.path.exists(all_transactions_file):
        try:
            df = pd.read_csv(all_transactions_file)
            transactions = df.to_dict('records')
        except Exception as e:
            flash(f"Error loading transactions: {str(e)}")
            logger.error(f"Error loading transactions: {str(e)}")
    
    # Count individual transaction files (CSV files that aren't all_transactions.csv)
    uploaded_file_count = 0
    try:
        for file in os.listdir(session_csv_dir):
            if file.endswith('.csv') and file != 'all_transactions.csv':
                uploaded_file_count += 1
    except Exception as e:
        logger.error(f"Error counting CSV files: {str(e)}")
    
    # Generate summary data
    summary = create_transaction_summary(transactions)
    
    # Calculate totals - simplified approach using the sign of the amount
    total_income = sum(transaction['amount'] for transaction in transactions if transaction['amount'] > 0)
    total_expenses = sum(abs(transaction['amount']) for transaction in transactions if transaction['amount'] < 0)
    net_profit = total_income - total_expenses
    
    # Format currency values
    income_total = format_currency(total_income)
    expense_total = format_currency(total_expenses)
    net_profit_formatted = format_currency(net_profit)
    
    # Add standard template variables
    template_vars = get_template_vars()
    
    return render_template(
        'results.html',
        transactions=transactions,
        session_id=session_id,
        summary=summary,
        income_total=income_total,
        expense_total=expense_total,
        net_profit=net_profit_formatted,
        uploaded_file_count=uploaded_file_count,
        **template_vars
    )

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
    session_dir = os.path.join(app.config['CSV_FOLDER'], session_id)
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if the directory exists
    if not os.path.exists(session_dir):
        flash(f"No CSV files found for session {session_id}")
        return redirect(url_for('results', session_id=session_id))
    
    try:
        # Force cleanup of previous merged file
        all_transactions_path = os.path.join(session_dir, 'all_transactions.csv')
        output_transactions_path = os.path.join(output_dir, 'all_transactions.csv')
        
        # Delete any existing merged file to ensure complete reprocessing
        if os.path.exists(all_transactions_path):
            try:
                os.remove(all_transactions_path)
                logger.info(f"Removed existing merged file: {all_transactions_path}")
            except Exception as e:
                logger.warning(f"Could not remove existing file {all_transactions_path}: {str(e)}")
                
        if os.path.exists(output_transactions_path):
            try:
                os.remove(output_transactions_path)
                logger.info(f"Removed existing merged file: {output_transactions_path}")
            except Exception as e:
                logger.warning(f"Could not remove existing file {output_transactions_path}: {str(e)}")
        
        # Find all CSV files in the session directory
        csv_files = []
        for file in os.listdir(session_dir):
            if file.endswith('.csv') and file != 'all_transactions.csv':
                csv_files.append(os.path.join(session_dir, file))
        
        if not csv_files:
            flash("No CSV files found to merge")
            return redirect(url_for('results', session_id=session_id))
        
        # Use pandas to load and merge all CSV files
        all_transactions = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                if not df.empty:
                    all_transactions.append(df)
            except Exception as e:
                logger.error(f"Error loading {os.path.basename(file)}: {str(e)}")
        
        if not all_transactions:
            flash("No valid transactions found in any of the processed files")
            return redirect(url_for('results', session_id=session_id))
        
        # Concatenate all dataframes
        combined_df = pd.concat(all_transactions, ignore_index=True)
        
        # Ensure date column is in datetime format
        if 'date' in combined_df.columns:
            combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
            
            # Sort by date
            combined_df = combined_df.sort_values('date')
            
            # Convert back to string format for storage
            combined_df['date'] = combined_df['date'].dt.strftime('%Y-%m-%d')
        
        # Add transaction type if not present
        if 'type' not in combined_df.columns:
            combined_df['type'] = combined_df.apply(lambda row: 'credit' if row['amount'] >= 0 else 'debit', axis=1)
        
        # Calculate transaction count
        transaction_count = len(combined_df)
        flash(f"Successfully merged {len(csv_files)} CSV files with {transaction_count} transactions")
    except Exception as e:
        logger.error(f"Error merging CSV files: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f"Error merging CSV files: {str(e)}")
    
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
        # Check for actual user transaction data in the session output directories
        possible_paths = [
            os.path.join(app.config['CSV_FOLDER'], session_id, 'all_transactions.csv'),
            os.path.join(app.config['OUTPUT_FOLDER'], session_id, 'all_transactions.csv'),
        ]
        
        transactions_file = None
        for path in possible_paths:
            if os.path.exists(path):
                transactions_file = path
                logger.info(f"Found user transaction data at: {path}")
                break
        
        if not transactions_file:
            logger.error(f"No transaction data found for session {session_id}. Searched paths: {possible_paths}")
            flash("No transaction data found. Please upload bank statements first.", "warning")
            return pd.DataFrame()  # Return empty DataFrame
        
        # Load the data
        df = pd.read_csv(transactions_file)
        logger.info(f"Loaded {len(df)} user transactions from {transactions_file}")
        
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
    """Process all PDFs in the directory"""
    try:
        # Get list of PDF files in the directory
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]
        
        if not files:
            processing_logs[session_id].append("No PDF files found to process.")
            return
        
        # Update status
        pdf_count = len(files)
        processing_status[session_id]['pdf_total'] = pdf_count
        processing_status[session_id]['pdf_processed'] = 0
        processing_logs[session_id].append(f"Processing {pdf_count} PDF files...")
        
        # Create session-specific CSV output directory
        session_csv_dir = os.path.join(app.config['CSV_FOLDER'], session_id)
        os.makedirs(session_csv_dir, exist_ok=True)
        
        # Create session-specific output directory
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each file
        for i, file_path in enumerate(files):
            try:
                # Update progress
                pdf_progress = ((i + 1) / pdf_count) * 100
                processing_status[session_id]['pdf_progress'] = pdf_progress
                processing_status[session_id]['pdf_processed'] = i + 1
                
                # Update overall progress (estimate)
                if 'csv_progress' in processing_status[session_id]:
                    total_progress = (pdf_progress + processing_status[session_id].get('csv_progress', 0)) / 2
                else:
                    total_progress = pdf_progress
                processing_status[session_id]['progress'] = total_progress
                
                processing_logs[session_id].append(f"Processing {os.path.basename(file_path)}...")
                
                # Convert to Path object for better path handling
                file_path_obj = Path(file_path)
                output_csv_path = Path(session_csv_dir) / file_path_obj.with_suffix('.csv').name
                
                # Process based on bank type
                if bank_type.lower() == 'chase':
                    processing_logs[session_id].append(f"Using Chase bank statement processor...")
                    # Process Chase statement
                    df = chase_load_pdf(file_path_obj)
                elif bank_type.lower() == 'bofa':
                    processing_logs[session_id].append(f"Using Bank of America statement processor...")
                    # Process Bank of America statement
                    df = bofa_load_pdf(file_path_obj)
                else:
                    processing_logs[session_id].append(f"Unsupported bank type: {bank_type}")
                    continue
                
                # Save transactions to CSV
                if df is not None and not df.empty:
                    df.to_csv(output_csv_path, index=False)
                    processing_logs[session_id].append(f"Extracted {len(df)} transactions from {os.path.basename(file_path)}")
                else:
                    processing_logs[session_id].append(f"No transactions found in {os.path.basename(file_path)}")
                
                # Add brief pause to prevent UI freezing
                time.sleep(0.1)
            
            except Exception as e:
                error_message = f"Error processing {os.path.basename(file_path)}: {str(e)}"
                processing_logs[session_id].append(error_message)
                logger.error(error_message)
                logger.error(traceback.format_exc())
        
        # Check if we need to wait for CSV processing to complete
        if 'csv_processing' in processing_status[session_id] and processing_status[session_id]['csv_processing']:
            processing_logs[session_id].append("PDF processing complete. Waiting for CSV processing to finish...")
        else:
            # Automatically merge all CSV files after processing
            merge_csv_files_for_session(session_id)
        
    except Exception as e:
        error_message = f"Error during processing: {str(e)}"
        processing_logs[session_id].append(error_message)
        logger.error(error_message)
        logger.error(traceback.format_exc())
        processing_status[session_id]['status'] = 'error'

def merge_csv_files_for_session(session_id):
    """Merge all CSV files for a session into one file."""
    try:
        processing_logs[session_id].append("Automatically merging all CSV files...")
        # Find all CSV files in the session directory
        session_csv_dir = os.path.join(app.config['CSV_FOLDER'], session_id)
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        
        if not os.path.exists(session_csv_dir):
            processing_logs[session_id].append("CSV directory does not exist. Nothing to merge.")
            processing_status[session_id]['status'] = 'completed'
            return
            
        csv_files = []
        for file in os.listdir(session_csv_dir):
            if file.endswith('.csv') and file != 'all_transactions.csv':
                csv_files.append(os.path.join(session_csv_dir, file))
        
        if csv_files:
            # Merge all CSV files into one
            all_transactions = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if not df.empty:
                        all_transactions.append(df)
                except Exception as e:
                    processing_logs[session_id].append(f"Error loading {os.path.basename(csv_file)}: {str(e)}")
            
            if all_transactions:
                # Concatenate all dataframes
                combined_df = pd.concat(all_transactions, ignore_index=True)
                
                # Ensure date column is in datetime format
                if 'date' in combined_df.columns:
                    combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
                    
                    # Sort by date
                    combined_df = combined_df.sort_values('date')
                    
                    # Convert back to string format for storage
                    combined_df['date'] = combined_df['date'].dt.strftime('%Y-%m-%d')
                
                # Add transaction type if not present, based on amount sign
                if 'type' not in combined_df.columns:
                    combined_df['type'] = combined_df.apply(lambda row: 'credit' if row['amount'] >= 0 else 'debit', axis=1)
                
                # Save to CSV in both locations
                merged_file = os.path.join(session_csv_dir, 'all_transactions.csv')
                combined_df.to_csv(merged_file, index=False)
                
                output_merged_file = os.path.join(output_dir, 'all_transactions.csv')
                combined_df.to_csv(output_merged_file, index=False)
                
                transaction_count = len(combined_df)
                processing_logs[session_id].append(f"Successfully merged {len(csv_files)} CSV files with {transaction_count} transactions")
            else:
                processing_logs[session_id].append("No valid transactions found in any of the processed files")
        else:
            processing_logs[session_id].append("No CSV files found to merge")
    except Exception as e:
        error_message = f"Error merging CSV files: {str(e)}"
        processing_logs[session_id].append(error_message)
        logger.error(error_message)
        logger.error(traceback.format_exc())
    
    # Update final status
    processing_status[session_id]['status'] = 'completed'

def create_transaction_summary(transactions):
    """Generate summary data from transactions list for the template"""
    if not transactions:
        return {
            'total_transactions': 0,
            'deposits': 0,
            'deposits_formatted': format_currency(0),
            'withdrawals': 0,
            'withdrawals_formatted': format_currency(0),
            'net_change': 0,
            'net_change_formatted': format_currency(0),
            'monthly': []
        }
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(transactions)
    
    # Basic summary
    total_transactions = len(df)
    
    if 'amount' in df.columns:
        # Fix data types if needed
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Calculate metrics
        deposits = df[df['amount'] > 0]['amount'].sum()
        withdrawals = abs(df[df['amount'] < 0]['amount'].sum())
        net_change = deposits - withdrawals
        
        # Monthly breakdown
        try:
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
        except Exception as e:
            logger.error(f"Error creating monthly breakdown: {str(e)}")
            monthly_data = []
    else:
        deposits = 0
        withdrawals = 0
        net_change = 0
        monthly_data = []
    
    return {
        'total_transactions': total_transactions,
        'deposits': round(deposits, 2),
        'deposits_formatted': format_currency(deposits),
        'withdrawals': round(withdrawals, 2),
        'withdrawals_formatted': format_currency(withdrawals),
        'net_change': round(net_change, 2),
        'net_change_formatted': format_currency(net_change),
        'monthly': monthly_data
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
        # Find an available port between 8501-8510 for Streamlit
        streamlit_port = 8501
        max_port = 8510
        
        # Check if we already have a running process for this session
        if session_id in active_streamlit_processes:
            logger.info(f"Reusing existing Streamlit process for session {session_id}")
            # Just redirect to the existing dashboard
            template_data = {
                'streamlit_url': f"http://localhost:{streamlit_port}?embed=true",
                'session_id': session_id,
            }
            return render_template('streamlit_dashboard.html', **template_data)
        
        # Check if port 8501 is already in use by another process
        import socket
        while streamlit_port <= max_port:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', streamlit_port))
            sock.close()
            if result != 0:  # Port is available
                break
            streamlit_port += 1
            
        if streamlit_port > max_port:
            logger.error(f"No available ports found for Streamlit in range 8501-{max_port}")
            flash("Unable to start Streamlit dashboard - all ports are busy. Please try again later.")
            return redirect(url_for('results', session_id=session_id))
            
        logger.info(f"Using port for Streamlit: {streamlit_port}")
        
        # Always use the standalone Streamlit dashboard
        streamlit_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'streamlit', 'financial_dashboard.py')
        
        if not os.path.exists(streamlit_path):
            logger.error(f"Streamlit script not found at: {streamlit_path}")
            flash("Streamlit dashboard file not found.")
            return redirect(url_for('results', session_id=session_id))
        
        # Clean up Streamlit cache to ensure new data is loaded
        streamlit_cache_dir = os.path.join(os.path.expanduser("~"), ".streamlit/cache")
        if os.path.exists(streamlit_cache_dir):
            try:
                logger.info(f"Cleaning Streamlit cache directory: {streamlit_cache_dir}")
                for cache_file in os.listdir(streamlit_cache_dir):
                    file_path = os.path.join(streamlit_cache_dir, cache_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        logger.warning(f"Error cleaning cache file {file_path}: {str(e)}")
            except Exception as e:
                logger.warning(f"Error accessing Streamlit cache directory: {str(e)}")
        
        # Kill any existing Streamlit processes before starting a new one
        try:
            import signal
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline'] if proc.info['cmdline'] else []
                    cmdline_str = ' '.join(cmdline)
                    if 'streamlit' in cmdline_str and f'--server.port {streamlit_port}' in cmdline_str:
                        logger.info(f"Terminating existing Streamlit process on port {streamlit_port}: {proc.info['pid']}")
                        os.kill(proc.info['pid'], signal.SIGTERM)
                except Exception as proc_err:
                    logger.warning(f"Error checking process: {str(proc_err)}")
        except Exception as e:
            logger.warning(f"Error cleaning up Streamlit processes: {str(e)}")
        
        # Create a process to run the Streamlit app with the session ID
        process = multiprocessing.Process(
            target=run_streamlit_process,
            args=(streamlit_path, session_id, streamlit_port)
        )
        process.daemon = True
        process.start()
        
        logger.info(f"Started Streamlit process with PID {process.pid} for session {session_id}")
        
        # Add to active processes for later cleanup
        if session_id not in active_streamlit_processes:
            active_streamlit_processes[session_id] = []
        active_streamlit_processes[session_id].append(process.pid)
        
        # Return the streamlit webpage
        template_data = {
            'streamlit_url': f"http://localhost:{streamlit_port}?embed=true",
            'session_id': session_id,
        }
        return render_template('streamlit_dashboard.html', **template_data)

    except Exception as e:
        error_message = f"Failed to start Streamlit dashboard: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        flash("An error occurred starting the Streamlit dashboard.")
        return redirect(url_for('results', session_id=session_id))

def run_streamlit_process(script_path, session_id, port):
    """Run a Streamlit process with the given script path and session ID"""
    try:
        # Make sure shutil is imported
        import shutil
        
        # Create a merged CSV file in the session directory if it doesn't exist yet
        session_csv_dir = os.path.join(app.config['CSV_FOLDER'], session_id)
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        
        # Ensure both directories exist
        os.makedirs(session_csv_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if all_transactions.csv exists in either directory
        session_all_tx = os.path.join(session_csv_dir, 'all_transactions.csv')
        output_all_tx = os.path.join(output_dir, 'all_transactions.csv')
        
        # If CSV exists in one place but not the other, copy it to ensure consistency
        if os.path.exists(session_all_tx) and not os.path.exists(output_all_tx):
            logger.info(f"Copying transactions from {session_all_tx} to {output_all_tx}")
            shutil.copy2(session_all_tx, output_all_tx)
        elif os.path.exists(output_all_tx) and not os.path.exists(session_all_tx):
            logger.info(f"Copying transactions from {output_all_tx} to {session_all_tx}")
            shutil.copy2(output_all_tx, session_all_tx)
        
        # If all_transactions.csv doesn't exist in either directory, create a sample one
        if not os.path.exists(session_all_tx) and not os.path.exists(output_all_tx):
            logger.info(f"No transaction data found for session {session_id}, creating sample data")
            
            # Copy sample transactions if they exist
            sample_path = os.path.join(os.path.dirname(script_path), 'sample_transactions.csv')
            if os.path.exists(sample_path):
                shutil.copy(sample_path, output_all_tx)
                shutil.copy(sample_path, session_all_tx)
                logger.info(f"Copied sample transactions to both output and CSV directories")
        
        # Set environment variables to help Streamlit locate data
        os.environ['STREAMLIT_SESSION_ID'] = session_id
        os.environ['CSV_FOLDER'] = app.config['CSV_FOLDER']
        os.environ['OUTPUT_FOLDER'] = app.config['OUTPUT_FOLDER']
        
        # Add timestamp to force cache invalidation - update with each process start
        cache_timestamp = str(time.time())
        os.environ['STREAMLIT_CACHE_TIMESTAMP'] = cache_timestamp
        logger.info(f"Set cache timestamp: {cache_timestamp}")
        
        # Clean up Streamlit cache directory before starting new process
        try:
            streamlit_cache_dir = os.path.expanduser("~/.streamlit/cache")
            if os.path.exists(streamlit_cache_dir):
                logger.info(f"Removing Streamlit cache directory: {streamlit_cache_dir}")
                shutil.rmtree(streamlit_cache_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Error cleaning Streamlit cache: {str(e)}")
        
        # Run the Streamlit command with enhanced arguments
        cmd = [
            "streamlit", "run", script_path,
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.serverAddress", "localhost",
            "--browser.gatherUsageStats", "false",
            "--server.enableCORS", "true",
            "--server.enableXsrfProtection", "false",
            "--server.maxUploadSize", "100",
            "--client.showErrorDetails", "true",
            "--client.toolbarMode", "minimal",
            "--", "--session_id", session_id, "--cache_timestamp", cache_timestamp
        ]
        
        logger.info(f"Running Streamlit command: {' '.join(cmd)}")
        subprocess.run(cmd)
    except Exception as e:
        logger.error(f"Error in Streamlit process: {str(e)}")
        logger.error(traceback.format_exc())

@app.route('/help')
def help_page():
    """Show help and documentation page"""
    return render_template('help.html', 
                           app_name=APP_NAME,
                           app_version=APP_VERSION,
                           app_author=APP_AUTHOR)

@app.route('/export_chart/<session_id>/<chart_type>', methods=['GET'])
@require_session
def export_chart(session_id, chart_type):
    """Generate and download a chart as PNG"""
    try:
        # Load transaction data
        df = load_transactions_data(session_id)
        
        if df.empty:
            flash("No transaction data available for charts")
            return redirect(url_for('results', session_id=session_id))
            
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        # Create a BytesIO object to store the image
        img_bytes = io.BytesIO()
        
        # Set figure parameters based on chart type
        plt.figure(figsize=(8, 5), dpi=100)
        
        if chart_type == 'monthly_income':
            # Filter for income transactions (positive amounts)
            income_df = df[df['amount'] > 0].copy()
            
            # Add month column for grouping
            income_df['month'] = income_df['date'].dt.strftime('%Y-%m')
            
            # Group by month
            monthly_income = income_df.groupby('month')['amount'].sum().reset_index()
            
            # Sort by month
            monthly_income = monthly_income.sort_values('month')
            
            # Plot
            ax = plt.subplot(111)
            bars = ax.bar(monthly_income['month'], monthly_income['amount'], color='green')
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'${height:,.2f}',
                        ha='center', va='bottom', rotation=45)
            
            plt.title('Monthly Income')
            plt.xlabel('Month')
            plt.ylabel('Income ($)')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
        elif chart_type == 'monthly_expenses':
            # Filter for expense transactions (negative amounts)
            expense_df = df[df['amount'] < 0].copy()
            
            # Convert amounts to positive for easier visualization
            expense_df['amount'] = expense_df['amount'].abs()
            
            # Add month column for grouping
            expense_df['month'] = expense_df['date'].dt.strftime('%Y-%m')
            
            # Group by month
            monthly_expenses = expense_df.groupby('month')['amount'].sum().reset_index()
            
            # Sort by month
            monthly_expenses = monthly_expenses.sort_values('month')
            
            # Plot
            ax = plt.subplot(111)
            bars = ax.bar(monthly_expenses['month'], monthly_expenses['amount'], color='red')
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'${height:,.2f}',
                        ha='center', va='bottom', rotation=45)
            
            plt.title('Monthly Expenses')
            plt.xlabel('Month')
            plt.ylabel('Expenses ($)')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
        elif chart_type == 'income_vs_expenses':
            # Add month column
            df['month'] = df['date'].dt.strftime('%Y-%m')
            
            # Calculate monthly income and expenses
            monthly_income = df[df['amount'] > 0].groupby('month')['amount'].sum()
            monthly_expenses = df[df['amount'] < 0].groupby('month')['amount'].sum().abs()
            
            # Combine into a dataframe
            monthly_data = pd.DataFrame({
                'Income': monthly_income,
                'Expenses': monthly_expenses
            }).fillna(0)
            
            # Sort by month
            monthly_data = monthly_data.sort_index()
            
            # Plot
            ax = plt.subplot(111)
            
            # Plot bars
            x = np.arange(len(monthly_data.index))
            width = 0.35
            
            income_bars = ax.bar(x - width/2, monthly_data['Income'], width, label='Income', color='green')
            expense_bars = ax.bar(x + width/2, monthly_data['Expenses'], width, label='Expenses', color='red')
            
            # Add data labels
            for bars in [income_bars, expense_bars]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'${height:,.0f}',
                            ha='center', va='bottom', size=8)
            
            # Set chart properties
            ax.set_title('Monthly Income vs Expenses')
            ax.set_xlabel('Month')
            ax.set_ylabel('Amount ($)')
            ax.set_xticks(x)
            ax.set_xticklabels(monthly_data.index, rotation=45)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
        elif chart_type == 'expense_categories':
            # Check if description column exists
            if 'description' not in df.columns:
                plt.text(0.5, 0.5, 'No description data available for category analysis', 
                       ha='center', va='center', transform=plt.gca().transAxes)
            else:
                # Filter for expenses
                expense_df = df[df['amount'] < 0].copy()
                expense_df['amount'] = expense_df['amount'].abs()
                
                # Generate basic categories based on description
                expense_df['category'] = expense_df['description'].apply(categorize_transaction)
                
                # Group by category
                category_expenses = expense_df.groupby('category')['amount'].sum().sort_values(ascending=False)
                
                # Use top categories and group the rest as "Other"
                top_categories = category_expenses.head(5)
                if len(category_expenses) > 5:
                    other_sum = category_expenses[5:].sum()
                    top_categories['Other'] = other_sum
                
                # Plot pie chart
                plt.figure(figsize=(8, 8))  # Square figure for pie chart
                ax = plt.subplot(111)
                
                # Create pie chart with percentage and value labels
                total = top_categories.sum()
                
                def autopct_format(pct):
                    value = (pct / 100) * total
                    return f'${value:,.2f}\n({pct:.1f}%)'
                
                wedges, texts, autotexts = ax.pie(
                    top_categories, 
                    labels=top_categories.index,
                    autopct=autopct_format,
                    startangle=90,
                    shadow=False
                )
                
                # Styling
                for autotext in autotexts:
                    autotext.set_size(9)
                    autotext.set_weight('bold')
                
                plt.title('Expense Categories')
                plt.axis('equal')  # Equal aspect ratio ensures pie is circular
                
        elif chart_type == 'savings_trend':
            # Add month column
            df['month'] = df['date'].dt.strftime('%Y-%m')
            
            # Calculate net savings by month
            monthly_net = df.groupby('month')['amount'].sum().reset_index()
            
            # Sort by month
            monthly_net = monthly_net.sort_values('month')
            
            # Calculate cumulative savings
            monthly_net['cumulative'] = monthly_net['amount'].cumsum()
            
            # Plot
            ax = plt.subplot(111)
            ax.plot(monthly_net['month'], monthly_net['cumulative'], marker='o', linewidth=2, color='blue')
            
            # Add data points
            for i, v in enumerate(monthly_net['cumulative']):
                ax.text(i, v, f'${v:,.2f}', ha='left', va='bottom')
            
            # Add zero line
            ax.axhline(y=0, color='red', linestyle='-', alpha=0.3)
            
            # Fill between curve and zero line
            ax.fill_between(
                monthly_net['month'], 
                monthly_net['cumulative'], 
                0, 
                where=(monthly_net['cumulative'] >= 0), 
                color='green', 
                alpha=0.3,
                label='Savings'
            )
            ax.fill_between(
                monthly_net['month'], 
                monthly_net['cumulative'], 
                0, 
                where=(monthly_net['cumulative'] < 0), 
                color='red', 
                alpha=0.3,
                label='Deficit'
            )
            
            plt.title('Cumulative Savings Trend')
            plt.xlabel('Month')
            plt.ylabel('Cumulative Amount ($)')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
        
        else:
            # Default chart if type not recognized
            plt.text(0.5, 0.5, f'Chart type "{chart_type}" not recognized', 
                   ha='center', va='center', transform=plt.gca().transAxes)
        
        # Save to BytesIO object
        plt.savefig(img_bytes, format='png', dpi=100)
        img_bytes.seek(0)
        plt.close()
        
        # Create appropriate filename
        filename = f"{chart_type}_{datetime.now().strftime('%Y%m%d')}.png"
        
        # Return the PNG file as an attachment
        return send_file(
            img_bytes,
            mimetype='image/png',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Error generating chart for export: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f"Error generating chart: {str(e)}")
        return redirect(url_for('results', session_id=session_id))


def categorize_transaction(description):
    """Basic categorization of transactions based on description"""
    description = description.lower()
    
    # Income categories
    if any(word in description for word in ['salary', 'payroll', 'direct deposit']):
        return 'Income:Salary'
    elif any(word in description for word in ['interest', 'dividend']):
        return 'Income:Investments'
    
    # Housing categories
    if any(word in description for word in ['rent', 'mortgage', 'hoa', 'housing']):
        return 'Housing'
    
    # Utilities
    if any(word in description for word in ['electric', 'water', 'gas', 'utility', 'utilities', 'internet', 'cable', 'phone']):
        return 'Utilities'
    
    # Food categories
    if any(word in description for word in ['restaurant', 'dining', 'food', 'cafe', 'coffee', 'doordash', 'grubhub', 'uber eats']):
        return 'Food & Dining'
    elif any(word in description for word in ['grocery', 'supermarket', 'market', 'walmart', 'target', 'costco']):
        return 'Groceries'
    
    # Transportation categories
    if any(word in description for word in ['uber', 'lyft', 'taxi', 'transit', 'train', 'subway', 'metro', 'bus']):
        return 'Transportation'
    elif any(word in description for word in ['gas', 'fuel', 'exxon', 'shell', 'chevron']):
        return 'Auto:Fuel'
    elif any(word in description for word in ['auto', 'car', 'vehicle', 'insurance']):
        return 'Auto:Other'
        
    # Shopping categories
    if any(word in description for word in ['amazon', 'ebay', 'etsy', 'shop', 'store']):
        return 'Shopping'
    
    # Entertainment categories
    if any(word in description for word in ['movie', 'theatre', 'theater', 'netflix', 'hulu', 'spotify', 'entertainment']):
        return 'Entertainment'
    
    # Health categories
    if any(word in description for word in ['doctor', 'health', 'medical', 'pharmacy', 'fitness', 'gym']):
        return 'Health & Fitness'
    
    # Travel categories
    if any(word in description for word in ['hotel', 'airbnb', 'airline', 'air', 'flight', 'travel']):
        return 'Travel'
    
    # Subscription
    if any(word in description for word in ['subscription', 'membership']):
        return 'Subscriptions'
        
    # Default for anything not matching above categories
    return 'Uncategorized'

@app.route('/kpi_dashboard/<session_id>')
@require_session
def kpi_dashboard(session_id):
    """Show KPI dashboard for processed data"""
    try:
        # Load transaction data
        df = load_transactions_data(session_id)
        
        if df.empty:
            flash("No transaction data available for KPI analysis")
            return redirect(url_for('results', session_id=session_id))
        
        # Calculate KPIs
        kpi_data = calculate_kpis(df)
        
        # Generate chart images
        charts = generate_kpi_charts(df)
        
        # Generate forecast if we have enough data
        forecast = None
        if kpi_data.get('num_months', 0) >= 3:
            forecast = generate_forecast(df)
        
        return render_template('kpi_dashboard.html',
                              app_name=APP_NAME,
                              app_version=APP_VERSION,
                              app_author=APP_AUTHOR,
                              session_id=session_id,
                              kpi_data=kpi_data,
                              charts=charts,
                              forecast=forecast)
                              
    except Exception as e:
        logger.error(f"Error generating KPI dashboard: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f"Error generating KPI dashboard: {str(e)}")
        return redirect(url_for('results', session_id=session_id))

def calculate_kpis(df):
    """Calculate key performance indicators from transaction data"""
    kpi_data = {}
    
    try:
        # Make sure data is properly formatted
        if 'date' not in df.columns or 'amount' not in df.columns:
            return kpi_data
            
        # Ensure date is in datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Add month information for aggregations
        df['month'] = df['date'].dt.strftime('%Y-%m')
        
        # Calculate total income and expenses
        total_income = df[df['amount'] > 0]['amount'].sum()
        total_expenses = abs(df[df['amount'] < 0]['amount'].sum())
        
        # Calculate income to expense ratio
        if total_expenses > 0:
            income_expense_ratio = total_income / total_expenses
        else:
            income_expense_ratio = float('inf')  # Avoid division by zero
            
        # Calculate average monthly income and expenses
        # Group by month and calculate sums
        monthly_data = df.groupby('month').agg({
            'amount': lambda x: (x[x > 0].sum(), abs(x[x < 0].sum()))
        })
        
        # Extract monthly income and expenses
        monthly_income = [x[0] for x in monthly_data['amount']]
        monthly_expenses = [x[1] for x in monthly_data['amount']]
        
        # Calculate averages
        num_months = len(monthly_income)
        avg_monthly_income = sum(monthly_income) / num_months if num_months > 0 else 0
        avg_monthly_expenses = sum(monthly_expenses) / num_months if num_months > 0 else 0
        
        # Find month with highest income
        if num_months > 0:
            monthly_income_dict = {month: x[0] for month, x in zip(monthly_data.index, monthly_data['amount'])}
            highest_income_month = max(monthly_income_dict.items(), key=lambda x: x[1])
        else:
            highest_income_month = ('N/A', 0)
            
        # Store KPI data
        kpi_data = {
            'total_income': total_income,
            'total_expenses': total_expenses,
            'income_expense_ratio': income_expense_ratio,
            'num_months': num_months,
            'avg_monthly_income': avg_monthly_income,
            'avg_monthly_expenses': avg_monthly_expenses,
            'highest_income_month': highest_income_month,
            'net_profit': total_income - total_expenses
        }
        
    except Exception as e:
        logger.error(f"Error calculating KPIs: {str(e)}")
        logger.error(traceback.format_exc())
        
    return kpi_data


def generate_forecast(df, months_ahead=3):
    """Generate financial forecast based on historical data"""
    forecast = []
    
    try:
        # Ensure date is in datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Add month information for aggregations
        df['month'] = df['date'].dt.strftime('%Y-%m')
        
        # Group by month and calculate income and expenses
        monthly_data = pd.DataFrame({
            'income': df[df['amount'] > 0].groupby('month')['amount'].sum(),
            'expenses': abs(df[df['amount'] < 0].groupby('month')['amount'].sum())
        }).fillna(0)
        
        # If we don't have enough data, return empty forecast
        if len(monthly_data) < 3:
            return forecast
            
        # Sort by month for time series analysis
        monthly_data = monthly_data.sort_index()
        
        # Calculate basic statistics
        avg_income = monthly_data['income'].mean()
        avg_expenses = monthly_data['expenses'].mean()
        income_growth = calculate_growth_rate(monthly_data['income'])
        expense_growth = calculate_growth_rate(monthly_data['expenses'])
        
        # Get the last month in data
        last_month = pd.to_datetime(monthly_data.index[-1] + '-01')
        
        # Generate forecast
        for i in range(1, months_ahead + 1):
            next_month = last_month + pd.DateOffset(months=i)
            month_name = next_month.strftime('%Y-%m')
            
            # Project income and expenses with growth factors
            projected_income = avg_income * (1 + income_growth * i)
            projected_expenses = avg_expenses * (1 + expense_growth * i)
            projected_savings = projected_income - projected_expenses
            
            forecast.append({
                'month': month_name,
                'income': projected_income,
                'expenses': projected_expenses,
                'savings': projected_savings
            })
            
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        logger.error(traceback.format_exc())
        
    return forecast


def calculate_growth_rate(series):
    """Calculate average month-over-month growth rate"""
    if len(series) < 2 or series.iloc[0] == 0:
        return 0
        
    try:
        # Calculate percentage changes
        pct_changes = series.pct_change().dropna()
        
        # Remove outliers (changes > 100%)
        filtered_changes = pct_changes[pct_changes.abs() <= 1]
        
        # If all changes were outliers, use a small default growth
        if len(filtered_changes) == 0:
            return 0.01  # Default to 1% growth
            
        # Return average growth rate
        return filtered_changes.mean()
        
    except Exception as e:
        logger.error(f"Error calculating growth rate: {str(e)}")
        return 0

@app.route('/clear_streamlit_cache/<session_id>')
@require_session
def clear_streamlit_cache(session_id):
    """Clear Streamlit cache and restart the dashboard process"""
    try:
        logger.info(f"Clearing Streamlit cache for session {session_id}")
        
        # Kill any existing Streamlit processes for this session
        if session_id in active_streamlit_processes:
            try:
                import signal
                import psutil
                for pid in active_streamlit_processes[session_id]:
                    try:
                        logger.info(f"Terminating Streamlit process {pid} for session {session_id}")
                        os.kill(pid, signal.SIGTERM)
                    except Exception as e:
                        logger.warning(f"Error terminating process {pid}: {str(e)}")
                
                # Clear the list of active processes for this session
                active_streamlit_processes[session_id] = []
            except Exception as e:
                logger.warning(f"Error cleaning up processes: {str(e)}")
        
        # Clear the Streamlit cache directory
        try:
            import shutil
            streamlit_cache_dir = os.path.expanduser("~/.streamlit/cache")
            if os.path.exists(streamlit_cache_dir):
                logger.info(f"Removing Streamlit cache directory: {streamlit_cache_dir}")
                shutil.rmtree(streamlit_cache_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Error cleaning Streamlit cache: {str(e)}")
        
        # Flash success message and redirect to Streamlit dashboard
        flash("Successfully cleared Streamlit cache. Dashboard will reload with fresh data.")
        return redirect(url_for('simple_streamlit', session_id=session_id))
    
    except Exception as e:
        logger.error(f"Error clearing Streamlit cache: {str(e)}")
        logger.error(traceback.format_exc())
        flash("Error clearing Streamlit cache. Please try again.")
        return redirect(url_for('results', session_id=session_id))

@app.route('/reset_and_upload')
def reset_and_upload():
    """Clear existing data and redirect to upload page for new files"""
    session_id = get_session_id()
    
    try:
        cleanup_session_files(session_id)
        flash("All previous files cleared. You can now upload new files.")
    except Exception as e:
        flash(f"Error clearing files: {str(e)}")
        logger.error(f"Error clearing files for session {session_id}: {str(e)}")
    
    return redirect(url_for('index'))

# Function to clean up files for a specific session
def cleanup_session_files(session_id):
    """Clean up all files associated with a session"""
    try:
        # Clear uploads directory (including PDF files)
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if os.path.exists(upload_dir):
            for file in os.listdir(upload_dir):
                file_path = os.path.join(upload_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed file: {file_path}")
            logger.info(f"Cleared upload directory for session {session_id}")
            
            # Try to remove the directory itself after files are deleted
            try:
                os.rmdir(upload_dir)
                logger.info(f"Removed upload directory for session {session_id}")
            except Exception as e:
                logger.warning(f"Could not remove upload directory: {str(e)}")
        
        # Clear CSV directory
        csv_dir = os.path.join(app.config['CSV_FOLDER'], session_id)
        if os.path.exists(csv_dir):
            for file in os.listdir(csv_dir):
                file_path = os.path.join(csv_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed file: {file_path}")
            logger.info(f"Cleared CSV directory for session {session_id}")
            
            # Try to remove the directory itself after files are deleted
            try:
                os.rmdir(csv_dir)
                logger.info(f"Removed CSV directory for session {session_id}")
            except Exception as e:
                logger.warning(f"Could not remove CSV directory: {str(e)}")
        
        # Clear output directory
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed file: {file_path}")
            logger.info(f"Cleared output directory for session {session_id}")
            
            # Try to remove the directory itself after files are deleted
            try:
                os.rmdir(output_dir)
                logger.info(f"Removed output directory for session {session_id}")
            except Exception as e:
                logger.warning(f"Could not remove output directory: {str(e)}")
        
        # Reset processing status
        if session_id in processing_status:
            del processing_status[session_id]
        if session_id in processing_logs:
            del processing_logs[session_id]
            
        # If this session had Streamlit processes, kill them
        if session_id in active_streamlit_processes:
            try:
                import signal
                for pid in active_streamlit_processes[session_id]:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        logger.info(f"Terminated Streamlit process {pid}")
                    except Exception as e:
                        logger.warning(f"Could not terminate process {pid}: {str(e)}")
                del active_streamlit_processes[session_id]
            except Exception as e:
                logger.warning(f"Error cleaning up Streamlit processes: {str(e)}")
                
        # Remove from active sessions
        if session_id in active_sessions:
            active_sessions.remove(session_id)
            logger.info(f"Removed session {session_id} from active sessions")
    except Exception as e:
        logger.error(f"Error cleaning up files for session {session_id}: {str(e)}")

@app.route('/import_csv/<session_id>')
def import_csv_to_dashboard(session_id):
    """Directly import CSV data to the dashboard without saving external files."""
    session_csv_dir = os.path.join(app.config['CSV_FOLDER'], session_id)
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
    
    # Create dashboard data directory if needed
    dashboard_dir = os.path.join(output_dir, 'dashboard_data')
    os.makedirs(dashboard_dir, exist_ok=True)
    
    # Check for all_transactions.csv file
    all_transactions_file = os.path.join(session_csv_dir, 'all_transactions.csv')
    
    if not os.path.exists(all_transactions_file):
        flash("No transaction data found. Please upload and process files first.")
        return redirect(url_for('results', session_id=session_id))
    
    try:
        # Load transactions without saving additional files
        df = pd.read_csv(all_transactions_file)
        transactions = df.to_dict('records')
        
        # Start Streamlit process with --no-save option to avoid creating additional files
        return start_streamlit_dashboard(session_id, no_save=True)
    except Exception as e:
        flash(f"Error importing CSV data: {str(e)}")
        logger.error(f"Error importing CSV data: {str(e)}")
        logger.error(traceback.format_exc())
        return redirect(url_for('results', session_id=session_id))

def process_csv_files(session_id, app_config, logs):
    """Process CSV and Excel files for the session."""
    try:
        # Get session directory
        session_dir = os.path.join(app_config['UPLOAD_FOLDER'], session_id)
        csv_output_dir = os.path.join(app_config['CSV_FOLDER'], session_id)
        
        # Ensure output directory exists
        os.makedirs(csv_output_dir, exist_ok=True)
        
        # Find all CSV and Excel files
        files = []
        for filename in os.listdir(session_dir):
            file_path = os.path.join(session_dir, filename)
            if os.path.isfile(file_path) and (is_csv_file(filename) or is_excel_file(filename)):
                files.append(file_path)
        
        total_files = len(files)
        logs.append(f"Found {total_files} CSV/Excel files to process.")
        
        # Process each file
        for i, file_path in enumerate(files):
            try:
                filename = os.path.basename(file_path)
                logs.append(f"Processing {filename}...")
                
                # If it's an Excel file, convert to CSV
                if is_excel_file(filename):
                    logs.append(f"Converting Excel file to CSV: {filename}")
                    from csv_load import convert_xlsx_to_csv
                    csv_file_path = convert_xlsx_to_csv(file_path)
                    if csv_file_path:
                        file_path = csv_file_path
                
                # Process the CSV file
                from csv_load import process_csv_file
                transactions = process_csv_file(file_path)
                
                # Save processed data to the CSV directory
                output_filename = os.path.splitext(filename)[0] + '.csv'
                output_path = os.path.join(csv_output_dir, output_filename)
                
                # Convert transactions to DataFrame and save
                if transactions:
                    df = pd.DataFrame(transactions)
                    df.to_csv(output_path, index=False)
                    logs.append(f"Saved {len(transactions)} transactions to {output_filename}")
                else:
                    logs.append(f"No transactions found in {filename}")
                
                # Update progress
                progress = int(((i + 1) / total_files) * 100)
                if session_id in processing_status:
                    processing_status[session_id]['progress'] = progress
            
            except Exception as e:
                error_msg = f"Error processing {os.path.basename(file_path)}: {str(e)}"
                logs.append(error_msg)
                logger.error(error_msg)
                traceback.print_exc()
        
        # Update processing status
        if session_id in processing_status:
            processing_status[session_id]['status'] = 'completed'
            processing_status[session_id]['progress'] = 100
        
        logs.append("CSV/Excel processing completed.")
        
        # Merge all CSV files
        if total_files > 0:
            logs.append("Merging all CSV files...")
            merge_csv_files_for_session(session_id)
            logs.append("Merge completed.")
    
    except Exception as e:
        error_msg = f"Error in CSV/Excel processing: {str(e)}"
        logs.append(error_msg)
        logger.error(error_msg)
        traceback.print_exc()
        
        # Update status to error
        if session_id in processing_status:
            processing_status[session_id]['status'] = 'error'
            processing_status[session_id]['message'] = str(e)

def merge_csv_files_for_session(session_id):
    """Merge all CSV files for a session into one file."""
    try:
        processing_logs[session_id].append("Automatically merging all CSV files...")
        # Find all CSV files in the session directory
        session_csv_dir = os.path.join(app.config['CSV_FOLDER'], session_id)
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        
        if not os.path.exists(session_csv_dir):
            processing_logs[session_id].append("CSV directory does not exist. Nothing to merge.")
            processing_status[session_id]['status'] = 'completed'
            return
            
        csv_files = []
        for file in os.listdir(session_csv_dir):
            if file.endswith('.csv') and file != 'all_transactions.csv':
                csv_files.append(os.path.join(session_csv_dir, file))
        
        if csv_files:
            # Merge all CSV files into one
            all_transactions = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if not df.empty:
                        all_transactions.append(df)
                except Exception as e:
                    processing_logs[session_id].append(f"Error loading {os.path.basename(csv_file)}: {str(e)}")
            
            if all_transactions:
                # Concatenate all dataframes
                combined_df = pd.concat(all_transactions, ignore_index=True)
                
                # Ensure date column is in datetime format
                if 'date' in combined_df.columns:
                    combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
                    
                    # Sort by date
                    combined_df = combined_df.sort_values('date')
                    
                    # Convert back to string format for storage
                    combined_df['date'] = combined_df['date'].dt.strftime('%Y-%m-%d')
                
                # Add transaction type if not present, based on amount sign
                if 'type' not in combined_df.columns:
                    combined_df['type'] = combined_df.apply(lambda row: 'credit' if row['amount'] >= 0 else 'debit', axis=1)
                
                # Save to CSV in both locations
                merged_file = os.path.join(session_csv_dir, 'all_transactions.csv')
                combined_df.to_csv(merged_file, index=False)
                
                output_merged_file = os.path.join(output_dir, 'all_transactions.csv')
                combined_df.to_csv(output_merged_file, index=False)
                
                transaction_count = len(combined_df)
                processing_logs[session_id].append(f"Successfully merged {len(csv_files)} CSV files with {transaction_count} transactions")
            else:
                processing_logs[session_id].append("No valid transactions found in any of the processed files")
        else:
            processing_logs[session_id].append("No CSV files found to merge")
    except Exception as e:
        error_message = f"Error merging CSV files: {str(e)}"
        processing_logs[session_id].append(error_message)
        logger.error(error_message)
        logger.error(traceback.format_exc())
    
    # Update final status
    processing_status[session_id]['status'] = 'completed'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4446) 