import pandas as pd
import pdfplumber as pdf
from pathlib import Path
import re
import glob
import os
import datetime

def extract_transactions(text):
    # Split text into lines and remove empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    transactions = []
    current_date = None
    in_deposits_section = False
    in_withdrawals_section = False
    
    for i, line in enumerate(lines):
        # Check for section headers
        if "DEPOSITS AND ADDITIONS" in line:
            in_deposits_section = True
            in_withdrawals_section = False
            continue
        elif "ELECTRONIC WITHDRAWALS" in line or "CHECKS PAID" in line or "ATM & DEBIT CARD WITHDRAWALS" in line:
            in_deposits_section = False
            in_withdrawals_section = True
            continue
        elif "CHECKING SUMMARY" in line or "DAILY ENDING BALANCE" in line or "SAVINGS SUMMARY" in line:
            # Skip summary sections
            in_deposits_section = False
            in_withdrawals_section = False
            continue
        elif "Beginning Balance" in line or "Ending Balance" in line or "Total Deposits" in line or "Total Electronic Withdrawals" in line:
            # Skip summary lines
            continue
        
        # Look for transactions which typically start with a date (MM/DD format)
        date_match = re.match(r'^(\d{2}/\d{2})\s', line)
        
        if date_match and (in_deposits_section or in_withdrawals_section):
            current_date = date_match.group(1)
            
            # Look for amount patterns at the end of the line (dollar amounts)
            amount_match = re.search(r'\$?([\d,]+\.\d{2})\s*$', line)
            
            if amount_match:
                amount_str = amount_match.group(1).replace(',', '')
                amount = float(amount_str)
                
                # Make withdrawals negative based on the section
                if in_withdrawals_section:
                    amount = -amount  # Withdrawals are negative
                
                # Extract description (everything between date and amount)
                description = line[date_match.end():amount_match.start()].strip()
                
                # Add transaction with the MM/DD date format
                transactions.append({
                    'date': current_date,
                    'description': description,
                    'amount': amount,
                    'type': 'deposit' if in_deposits_section else 'withdrawal'
                })
    
    return transactions

def load_pdf(path):
    """
    Load a Chase statement PDF and extract transactions
    Returns a pandas DataFrame with columns: date, description, amount, type
    """
    all_transactions = []
    statement_year = None
    
    # Open the PDF file using pdfplumber
    with pdf.open(path) as pdf_file:
        # Iterate through each page in the PDF
        for page in pdf_file.pages:
            # Extract the text content from the current page
            text = page.extract_text()
            
            # Try to extract statement period/date for year information
            if statement_year is None:
                # Look for statement period with year
                date_period_match = re.search(r'(?:Statement Period|From|Statement Date)[:\s]+\w+\s+\d{1,2},?\s+(\d{4})', text, re.IGNORECASE)
                if date_period_match:
                    statement_year = date_period_match.group(1)
                else:
                    # Try another pattern common in Chase statements
                    date_period_match = re.search(r'\w+\s+\d{1,2},\s+(\d{4})\s+(?:through|to)\s+\w+\s+\d{1,2},\s+\d{4}', text)
                    if date_period_match:
                        statement_year = date_period_match.group(1)
                    else:
                        # Try another format that might appear in Chase statements
                        date_period_match = re.search(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+(\d{4})', text)
                        if date_period_match:
                            statement_year = date_period_match.group(1)
            
            # Parse the text to find transactions using extract_transactions()
            transactions = extract_transactions(text)
            # Add the found transactions to our running list
            all_transactions.extend(transactions)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_transactions)
    
    if not df.empty:
        # Use current year if we couldn't extract it from the statement
        if statement_year is None:
            statement_year = str(datetime.datetime.now().year)
        
        # Create a new column with the full date (MM/DD/YY)
        df['full_date'] = df['date'] + '/' + statement_year[-2:]
        
        # Convert date strings to datetime
        df['date'] = pd.to_datetime(df['full_date'], format='%m/%d/%y', errors='coerce')
        
        # Drop the temporary column
        df.drop('full_date', axis=1, inplace=True)
    
    return df

def process_multiple_statements(directory):
    """
    Process multiple PDF statements in a directory and save them to CSV
    """
    # Input directory for PDF files
    directory = Path(directory)
    print(f"\nLooking for PDF files in: {directory}")
    
    # Create output directory for CSV files
    csv_dir = Path('./CSV Files')
    csv_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
    
    print(f"CSV files will be saved to: {csv_dir}\n")
    
    # Initialize variables for tracking totals
    processed_count = 0
    error_count = 0
    total_transactions = 0
    total_deposits = 0
    total_withdrawals = 0
    cost_transactions = 0
    price_transaction = 0.50 * (1+0.30)
    min_price_book = 300
    
    for pdf_file in directory.glob('*.pdf'):
        try:
            # Attempt to process each PDF file
            print(f"Loading PDF: {pdf_file.name}...")
            
            # Load the PDF and extract all transactions into a DataFrame
            df = load_pdf(pdf_file)
            
            if df.empty:
                print(f"⚠️ No transactions found in {pdf_file.name}")
                continue
                
            print(f"✓ Successfully extracted {len(df)} transactions")
            
            # Calculate total deposits (income) for this statement
            deposits_df = df[df['amount'] > 0]
            statement_deposits = deposits_df['amount'].sum()
            total_deposits += statement_deposits
            print(f"✓ Statement deposits: ${statement_deposits:.2f} ({len(deposits_df)} transactions)")
            
            # Calculate total withdrawals (expenses) for this statement
            withdrawals_df = df[df['amount'] < 0]
            statement_withdrawals = abs(withdrawals_df['amount'].sum())
            total_withdrawals += statement_withdrawals
            print(f"✓ Statement withdrawals: ${statement_withdrawals:.2f} ({len(withdrawals_df)} transactions)")
            print(f"✓ Statement net change: ${statement_deposits - statement_withdrawals:.2f}")
            
            # Update running totals
            total_transactions += len(df)
            
            # Create output CSV filename in the csv_files directory
            # Uses same name as PDF but with .csv extension
            csv_file = csv_dir / pdf_file.with_suffix('.csv').name
            
            # Save extracted transactions to CSV file
            df.to_csv(csv_file, index=False)
            print(f"✓ Saved to CSV: {csv_file.name}")
            print("-" * 50)
            
            # Increment counter for successfully processed files
            processed_count += 1
            
        except Exception as e:
            # If any error occurs during processing, catch and log it
            print(f"❌ Error processing {pdf_file.name}: {str(e)}")
            print("-" * 50)
            error_count += 1
    
    # Print summary
    if total_transactions <= 300:
        cost_transactions = min_price_book * 12
    else:
        cost_transactions = float((total_transactions * price_transaction) + min_price_book * 12)
    
    print("\nProcess Completed!")
    print("-" * 50)
    print(f"Successfully Processed: {processed_count} files")
    print(f"Total Transactions Processed: {total_transactions}")
    print(f"Total Deposits (Income): ${total_deposits:.2f}")
    print(f"Total Withdrawals (Expenses): ${total_withdrawals:.2f}")
    print(f"Total Net Change: ${total_deposits - total_withdrawals:.2f}")
    print("-" * 50)
    print(f"Total Cost: ${cost_transactions:.2f}")
    print(f"Total Monthly Cost: ${cost_transactions/12:.2f}")
    print("-" * 50)
    print(f"Errors Encountered: {error_count} files")
    if processed_count > 0:
        print(f"CSV files are saved in: {csv_dir}")

def merge_csv_files(output_directory):
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(output_directory, '*.csv'))
    
    # Create an empty list to store all dataframes
    all_dfs = []
    
    # Read each CSV file and append to the list
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_dfs.append(df)
    
    # Concatenate all dataframes
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        # Convert dates using a more flexible approach
        merged_df['date'] = pd.to_datetime(merged_df['date'])  # Let pandas auto-detect format
        merged_df = merged_df.sort_values('date')
        
        # Convert date back to a consistent format (YYYY-MM-DD)
        merged_df['date'] = merged_df['date'].dt.strftime('%Y-%m-%d')
        
        # Save the merged dataframe to a new CSV
        merged_output_path = os.path.join(output_directory, 'all_transactions.csv')
        merged_df.to_csv(merged_output_path, index=False)
        print(f"Created merged file: {merged_output_path}")

def main():
    process_multiple_statements('./Chase_Statements')
    
    # After processing all PDFs and creating individual CSVs
    merge_csv_files('./CSV Files')

if __name__ == "__main__":
    main() 