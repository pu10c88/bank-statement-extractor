import pandas as pd
import pdfplumber as pdf
from pathlib import Path
import re
import glob
import os

def extract_transactions(text):
    # Split text into lines and remove empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    transactions = []
    for line in lines:
        # Look for date patterns (MM/DD/YY) followed by transaction details
        date_match = re.search(r'\d{2}/\d{2}/\d{2}', line)
        if date_match:
            # Look for amount patterns at the end of the line
            amount_match = re.search(r'-?\d+,?\d*\.\d{2}\s*$', line)
            if amount_match:
                date = date_match.group()
                amount = float(amount_match.group().replace(',', ''))
                # Extract description (everything between date and amount)
                description = line[date_match.end():amount_match.start()].strip()
                transactions.append({
                    'date': date,
                    'description': description,
                    'amount': amount
                })
    
    return transactions

def load_pdf(path):
    """
    Load a Bank of America statement PDF and extract transactions
    Returns a pandas DataFrame with columns: date, description, amount
    """
    all_transactions = []
    
    # Open the PDF file using pdfplumber
    with pdf.open(path) as pdf_file:
        # Iterate through each page in the PDF
        for page in pdf_file.pages:
            # Extract the text content from the current page
            text = page.extract_text()
            # Parse the text to find transactions using extract_transactions()
            transactions = extract_transactions(text)
            # Add the found transactions to our running list
            all_transactions.extend(transactions)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_transactions)
    
    # Convert date strings to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    return df

def process_multiple_statements(directory):
    """
    Process multiple PDF statements in a directory and save them to CSV
    """
    # Input directory for PDF files
    directory = Path('./Bofa_Statements')
    print(f"\nLooking for PDF files in: {directory}")
    
    # Create output directory for CSV files
    csv_dir = Path('./CSV Files')
    csv_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
    print(f"CSV files will be saved to: {csv_dir}\n")
    
    # Initialize variables for tracking totals
    processed_count = 0
    error_count = 0
    total_transactions = 0
    total_revenue = 0  # New variable for tracking revenue
    cost_transactions = 0
    price_transaction = 0.50 * (1+0.30)
    min_price_book = 300
    
    for pdf_file in directory.glob('*.pdf'):
        try:
            # Attempt to process each PDF file
            print(f"Loading PDF: {pdf_file.name}...")
            
            # Load the PDF and extract all transactions into a DataFrame
            df = load_pdf(pdf_file)
            print(f"✓ Successfully extracted {len(df)} transactions")
            # Calculate total revenue for this statement 
            statement_revenue = df[df['amount'] > 0]['amount'].sum()
            total_revenue += statement_revenue
            print(f"✓ Statement revenue: ${statement_revenue:.2f}")
            
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
    print(f"Total Revenue: ${total_revenue:.2f}")
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
    process_multiple_statements('./Bofa_Statements')
    
    # After processing all PDFs and creating individual CSVs
    merge_csv_files('./CSV Files')

if __name__ == "__main__":
    main()

            








