import csv
import os
import pandas as pd
from datetime import datetime
import glob
import re
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

def is_csv_file(filename):
    """Check if a file is a CSV file based on extension."""
    return filename.lower().endswith('.csv')

def is_excel_file(filename):
    """Check if a file is an Excel file based on extension."""
    return filename.lower().endswith(('.xlsx', '.xls', '.xlsm'))

def detect_date_format(date_str):
    """Detect and convert date format to YYYY-MM-DD."""
    # Try different date formats
    date_formats = [
        '%Y-%m-%d',        # 2024-01-31
        '%m/%d/%Y',        # 01/31/2024
        '%m/%d/%y',        # 01/31/24
        '%d/%m/%Y',        # 31/01/2024
        '%B %d, %Y',       # January 31, 2024
        '%b %d, %Y'        # Jan 31, 2024
    ]
    
    for fmt in date_formats:
        try:
            date_obj = datetime.strptime(date_str.strip(), fmt)
            return date_obj.strftime('%Y-%m-%d')
        except (ValueError, AttributeError):
            continue
    
    return None

def detect_transaction_type(row, headers):
    """Detect if a transaction is income (credit) or expense (debit)."""
    # Check if there's a type column
    if 'type' in headers:
        if 'credit' in str(row.get('type', '')).lower():
            return 'credit'
        elif 'debit' in str(row.get('type', '')).lower():
            return 'debit'
    
    # Check amount for sign
    amount = row.get('amount', 0)
    if isinstance(amount, str):
        amount = amount.replace(',', '').replace('$', '').replace('"', '').strip()
        # Handle parentheses which often indicate negative numbers
        if amount.startswith('(') and amount.endswith(')'):
            amount = '-' + amount[1:-1]
        try:
            amount = float(amount)
        except ValueError:
            amount = 0
    
    # Check for negative sign in description
    description = str(row.get('description', ''))
    if '-' in description and description.startswith('-'):
        return 'debit'
    
    # Some files might use negative numbers for debits and positive for credits
    if amount < 0:
        return 'debit'
    
    # Look for keywords in description
    if any(keyword in description.lower() for keyword in ['payment to', 'purchase', 'withdrawal', 'paid']):
        return 'debit'
    if any(keyword in description.lower() for keyword in ['payment from', 'deposit', 'credit', 'received']):
        return 'credit'
    
    # Default: positive amounts are credits, negative are debits
    return 'credit' if amount >= 0 else 'debit'

def convert_xlsx_to_csv(file_path):
    """Convert Excel file to CSV."""
    try:
        # Read Excel file
        df = pd.read_excel(file_path)
        
        # Create CSV file path
        csv_file_path = file_path.rsplit('.', 1)[0] + '.csv'
        
        # Save as CSV
        df.to_csv(csv_file_path, index=False)
        
        print(f"  Converted {file_path} to {csv_file_path}")
        return csv_file_path
    except Exception as e:
        print(f"  Error converting {file_path} to CSV: {e}")
        return None

def categorize_transaction(description):
    """Categorize a transaction based on its description."""
    description = description.lower()
    
    categories = {
        'Food & Grocery': ['grocery', 'food', 'restaurant', 'supermarket', 'market', 'aldi', 'publix', 
                          'walmart', 'target', 'whole foods', 'crema', 'bagel', 'farm', 'sushi', 
                          'poke', 'domino', 'cuban', 'tres', 'meat', 'farmers', 'seafood', 'fresh'],
        
        'Transportation': ['gas', 'fuel', 'uber', 'lyft', 'car', 'auto', 'vehicle', 'transport', 'subway', 
                          'train', 'parking', 'west palmetto', 'sunpass', 'jomar petroleum', 'shell', 'orion'],
        
        'Housing': ['rent', 'mortgage', 'apartment', 'housing', 'home', 'landlord', 'landloard', 'house rental'],
        
        'Utilities': ['electric', 'water', 'gas', 'internet', 'phone', 'cable', 'utility', 'utilities', 'bill', 
                    'entergy', 'comcast', 't-mobile', 't mobile', 'tmobile', 'boost mobile'],
        
        'Entertainment': ['movie', 'cinema', 'theatre', 'theater', 'concert', 'netflix', 'spotify', 'amazon prime', 
                         'subscription', 'itunes', 'cmx', 'screen', 'stream', 'subscription'],
        
        'Shopping': ['amazon', 'retail', 'clothing', 'shop', 'store', 'merchandise', 'purchase', 'online', 
                    'apple', 'bestbuy', 'best buy', 'walm', 'walmart'],
        
        'Health': ['doctor', 'hospital', 'medical', 'health', 'pharmacy', 'medicine', 'clinic', 'dental', 
                  'dentist', 'therapy', 'wellness', 'fitness', 'gym', 'farmacia'],
        
        'Personal Care': ['salon', 'spa', 'beauty', 'hair', 'haircut', 'massage', 'massagem', 'braids', 'blind', 'usa'],
        
        'Education': ['school', 'college', 'university', 'tuition', 'book', 'course'],
        
        'Business': ['mma masters', 'zuffa', 'fernandes', 'ufc', 'u gym', 'masters', 'borges', 'atn cable', 'business'],
        
        'Insurance': ['insurance', 'coverage', 'policy', 'protection', 'bmag'],
        
        'Taxes': ['tax', 'irs', 'government', 'treasury'],
        
        'Investments': ['invest', 'stock', 'etf', 'fund', 'bond', 'portfolio', 'market', 'trade', 'dividend', 'capital'],
        
        'Fees': ['fee', 'charge', 'service fee', 'monthly fee', 'atm fee', 'withdraw fee', 'adp fee'],
        
        'Transfer': ['transfer', 'send', 'receive', 'moved', 'zelle', 'online transfer', 'ach', 'wire', 'western']
    }
    
    # Check if description matches any category
    for category, keywords in categories.items():
        if any(keyword in description for keyword in keywords):
            return category
    
    return 'Other'  # Default category

def normalize_amount(amount_str):
    """Convert amount string to float."""
    if isinstance(amount_str, (int, float)):
        return float(amount_str)
    
    if not amount_str:
        return 0.0
    
    # Remove currency symbols, commas, and quotes
    amount_str = str(amount_str).replace('$', '').replace(',', '').replace('"', '').strip()
    
    # Handle parentheses (common for negative values)
    if amount_str.startswith('(') and amount_str.endswith(')'):
        amount_str = '-' + amount_str[1:-1]
    
    try:
        return float(amount_str)
    except ValueError:
        return 0.0

def process_csv_file(file_path):
    """Process a CSV file and extract transaction data."""
    transactions = []
    
    try:
        # Try different delimiters
        delimiters = [',', ';', '\t', '|']
        
        for delimiter in delimiters:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    # Read the first few lines to detect the structure
                    sample = ''.join([next(file) for _ in range(10)])
                    file.seek(0)
                    
                    # If this delimiter doesn't appear enough in the sample, try the next one
                    if sample.count(delimiter) < 5:
                        continue
                    
                    # Read the CSV with the current delimiter
                    reader = csv.reader(file, delimiter=delimiter)
                    
                    # Skip rows until we find the header row or use first row as header
                    headers = []
                    header_row_found = False
                    
                    for row in reader:
                        if not row or all(not cell.strip() for cell in row):
                            continue  # Skip empty rows
                            
                        # Look for potential header row
                        if any(keyword.lower() in ' '.join(row).lower() for keyword in 
                               ['date', 'description', 'amount', 'transaction', 'deposit', 'withdrawal']):
                            headers = [h.lower() for h in row]
                            header_row_found = True
                            break
                    
                    # If no header was found, use generic headers
                    if not headers:
                        file.seek(0)
                        headers = ['date', 'description', 'amount', 'type']
                    
                    # Map common header variations to our standard header names
                    header_mapping = {
                        'transaction date': 'date',
                        'trans date': 'date',
                        'post date': 'date',
                        'posting date': 'date',
                        'transaction description': 'description',
                        'trans description': 'description',
                        'memo': 'description',
                        'details': 'description',
                        'transaction amount': 'amount',
                        'debit': 'amount',
                        'withdrawal': 'amount',
                        'credit': 'amount',
                        'deposit': 'amount',
                        'category': 'type'
                    }
                    
                    normalized_headers = []
                    for h in headers:
                        h_lower = h.lower().strip()
                        if h_lower in header_mapping:
                            normalized_headers.append(header_mapping[h_lower])
                        else:
                            normalized_headers.append(h_lower)
                    
                    # Process rows
                    file.seek(0)
                    if header_row_found:
                        # Skip to the row after the header
                        for _ in range(reader.line_num):
                            next(file)
                    
                    dict_reader = csv.DictReader(file, fieldnames=normalized_headers, delimiter=delimiter)
                    
                    for row in dict_reader:
                        # Skip header row if it was included in the data
                        if row.get('date', '').lower() == 'date':
                            continue
                            
                        # Skip empty rows
                        if not any(row.values()):
                            continue
                        
                        # Extract date and convert to standard format
                        date_str = None
                        for key in ['date', 'description']:
                            if key in row and row[key]:
                                date_match = re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}', str(row[key]))
                                if date_match:
                                    date_str = date_match.group(0)
                                    break
                        
                        if not date_str:
                            # Try to find a date in description
                            for key in row:
                                if row[key] and isinstance(row[key], str):
                                    date_match = re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}', row[key])
                                    if date_match:
                                        date_str = date_match.group(0)
                                        break
                        
                        if not date_str:
                            continue  # Skip rows without a valid date
                        
                        date = detect_date_format(date_str)
                        if not date:
                            continue
                        
                        # Extract amount
                        amount = 0.0
                        if 'amount' in row and row['amount']:
                            amount = normalize_amount(row['amount'])
                        
                        # Get description
                        description = row.get('description', '')
                        
                        # Determine transaction type
                        transaction_type = detect_transaction_type(row, normalized_headers)
                        
                        # Categorize transaction
                        category = categorize_transaction(str(description))
                        
                        # Create transaction record with signed amount based on type
                        transaction = {
                            'date': date,
                            'description': description,
                            'amount': -abs(amount) if transaction_type == 'debit' else abs(amount),  # Use negative value for debits, positive for credits
                            'type': transaction_type,
                            'category': category
                        }
                        
                        transactions.append(transaction)
                    
                    # If we found some transactions, don't try other delimiters
                    if transactions:
                        break
                    
            except Exception as e:
                print(f"  Error with delimiter '{delimiter}': {e}")
                continue
        
    except Exception as e:
        print(f"  Error processing file: {e}")
    
    return transactions

def calculate_monthly_summary(transactions):
    """Calculate monthly income, expenses, and profit."""
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(transactions)
    
    if df.empty:
        return []
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract year and month
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Group by year and month
    monthly_summary = []
    for (year, month), group in df.groupby(['year', 'month']):
        income = group[group['type'] == 'credit']['amount'].sum()
        expenses = group[group['type'] == 'debit']['amount'].sum()
        profit = income - expenses
        
        # Calculate expenses by category
        category_expenses = {}
        for category, category_group in group[group['type'] == 'debit'].groupby('category'):
            category_expenses[category] = category_group['amount'].sum()
        
        monthly_summary.append({
            'year': year,
            'month': month,
            'income': income,
            'expenses': expenses,
            'profit': profit,
            'category_expenses': category_expenses
        })
    
    # Sort by year and month
    monthly_summary.sort(key=lambda x: (x['year'], x['month']))
    
    return monthly_summary

def get_month_name(month_num):
    """Convert month number to name."""
    month_names = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    return month_names[month_num - 1]

def create_charts(monthly_summary, output_dir):
    """Create charts for monthly summary data."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert to DataFrame for easier plotting
    data = []
    for item in monthly_summary:
        row = {
            'year': item['year'],
            'month': item['month'],
            'month_name': get_month_name(item['month']),
            'income': item['income'],
            'expenses': item['expenses'],
            'profit': item['profit']
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df.sort_values('date', inplace=True)
    
    # Create time-axis labels
    x_labels = [f"{row['month_name'][:3]} {row['year']}" for _, row in df.iterrows()]
    
    # Create income vs expenses chart
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['income'], 'g-', label='Income', linewidth=2)
    plt.plot(df.index, df['expenses'], 'r-', label='Expenses', linewidth=2)
    plt.fill_between(df.index, df['income'], alpha=0.3, color='green')
    plt.fill_between(df.index, df['expenses'], alpha=0.3, color='red')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Monthly Income vs Expenses', fontsize=16)
    plt.ylabel('Amount ($)', fontsize=12)
    plt.xticks(df.index, x_labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'income_vs_expenses.png'), dpi=300)
    
    # Create profit chart
    plt.figure(figsize=(12, 6))
    plt.bar(df.index, df['profit'], color=['green' if x >= 0 else 'red' for x in df['profit']])
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.title('Monthly Profit', fontsize=16)
    plt.ylabel('Amount ($)', fontsize=12)
    plt.xticks(df.index, x_labels, rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on each bar
    for i, value in enumerate(df['profit']):
        color = 'black' if value >= 0 else 'white'
        plt.text(i, value + (0.05 * value if value >= 0 else -0.05 * abs(value)), 
                 f'${value:,.0f}', ha='center', va='bottom' if value >= 0 else 'top', 
                 fontweight='bold', color=color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_profit.png'), dpi=300)
    
    # Create expense categories chart for last 3 months
    last_three_months = monthly_summary[-3:] if len(monthly_summary) >= 3 else monthly_summary
    
    for i, month_data in enumerate(last_three_months):
        if not month_data['category_expenses']:
            continue
            
        # Create pie chart for expense categories
        plt.figure(figsize=(10, 8))
        
        # Get categories and amounts
        categories = list(month_data['category_expenses'].keys())
        amounts = list(month_data['category_expenses'].values())
        
        # Sort by amount (descending)
        sorted_indices = sorted(range(len(amounts)), key=lambda i: amounts[i], reverse=True)
        categories = [categories[i] for i in sorted_indices]
        amounts = [amounts[i] for i in sorted_indices]
        
        # Take top 7 categories and group the rest as 'Other'
        if len(categories) > 7:
            other_amount = sum(amounts[7:])
            categories = categories[:7] + ['Other']
            amounts = amounts[:7] + [other_amount]
        
        # Create explode array (pull out the largest slice)
        explode = [0.1 if i == 0 else 0 for i in range(len(categories))]
        
        plt.pie(amounts, labels=categories, autopct='%1.1f%%', startangle=90, explode=explode,
               shadow=True, textprops={'fontsize': 10})
        
        month_year = f"{get_month_name(month_data['month'])} {month_data['year']}"
        plt.title(f'Expense Categories - {month_year}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'expense_categories_{month_data["year"]}_{month_data["month"]:02d}.png'), dpi=300)
    
    print(f"Charts saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Process CSV files with transaction data and generate monthly summaries.')
    parser.add_argument('files', nargs='*', help='Individual files to process')
    parser.add_argument('--dir', default='Samples', help='Directory containing CSV files')
    parser.add_argument('--output', default='monthly_summary', help='Base name for output files (without extension)')
    parser.add_argument('--charts', action='store_true', help='Generate charts')
    parser.add_argument('--charts-dir', default='charts', help='Directory to save charts')
    parser.add_argument('--start-date', help='Filter transactions starting from this date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Filter transactions up to this date (YYYY-MM-DD)')
    parser.add_argument('--csv-only', action='store_true', help='Generate only CSV output (no Excel)')
    parser.add_argument('--no-save', action='store_true', help='Process data without saving files (for dashboard use)')
    
    args = parser.parse_args()
    
    # Find files to process
    input_files = []
    
    # First, handle explicitly specified files
    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        # If it's an Excel file, convert it to CSV
        if is_excel_file(file_path):
            print(f"Found Excel file: {file_path}")
            csv_file = convert_xlsx_to_csv(file_path)
            if csv_file:
                input_files.append(csv_file)
        else:
            input_files.append(file_path)
    
    # If no individual files were specified, search the directory
    if not input_files:
        # First find and convert any Excel files in the directory
        xlsx_files = glob.glob(os.path.join(args.dir, '*.xlsx')) + glob.glob(os.path.join(args.dir, '*.xls'))
        for xlsx_file in xlsx_files:
            print(f"Found Excel file: {xlsx_file}")
            csv_file = convert_xlsx_to_csv(xlsx_file)
            if csv_file:
                input_files.append(csv_file)
        
        # Then add any existing CSV files
        csv_files = glob.glob(os.path.join(args.dir, '*.csv'))
        input_files.extend(csv_files)
    
    # Remove duplicates (in case we just converted an Excel file to a CSV that already existed)
    input_files = list(set(input_files))
    
    if not input_files:
        print(f"No CSV or Excel files found to process")
        return
    
    all_transactions = []
    
    for file_path in input_files:
        print(f"Processing {file_path}...")
        transactions = process_csv_file(file_path)
        all_transactions.extend(transactions)
        print(f"  Found {len(transactions)} transactions.")
    
    if not all_transactions:
        print("No transactions found in any files.")
        return
    
    # Filter transactions by date if requested
    if args.start_date or args.end_date:
        filtered_transactions = []
        for transaction in all_transactions:
            date = transaction['date']
            if args.start_date and date < args.start_date:
                continue
            if args.end_date and date > args.end_date:
                continue
            filtered_transactions.append(transaction)
        
        all_transactions = filtered_transactions
        print(f"Filtered to {len(all_transactions)} transactions between {args.start_date or 'earliest'} and {args.end_date or 'latest'}")
    
    monthly_summary = calculate_monthly_summary(all_transactions)
    
    # Skip saving files if no-save option is set
    if args.no_save:
        print("\nProcessed data without saving files (no-save mode)")
        return monthly_summary, all_transactions
    
    # Print summary
    print("\nMonthly Summary:")
    print("=" * 60)
    print(f"{'Year-Month':<15} {'Income':>14} {'Expenses':>14} {'Profit':>14}")
    print("-" * 60)
    
    for item in monthly_summary:
        month_name = get_month_name(item['month'])
        month_year = f"{month_name} {item['year']}"
        print(f"{month_year:<15} ${item['income']:>13,.2f} ${item['expenses']:>13,.2f} ${item['profit']:>13,.2f}")
    
    # Save to CSV and Excel
    summary_df = pd.DataFrame(monthly_summary)
    
    # Add month name column
    summary_df['month_name'] = summary_df['month'].apply(get_month_name)
    
    # Format columns for better readability in CSV
    summary_df['income_formatted'] = summary_df['income'].apply(lambda x: f"${x:,.2f}")
    summary_df['expenses_formatted'] = summary_df['expenses'].apply(lambda x: f"${x:,.2f}")
    summary_df['profit_formatted'] = summary_df['profit'].apply(lambda x: f"${x:,.2f}")
    
    # Save to CSV
    csv_output_file = f"{args.output}.csv"
    # Remove category_expenses from CSV output as it's a dictionary
    csv_df = summary_df.copy()
    if 'category_expenses' in csv_df.columns:
        csv_df = csv_df.drop('category_expenses', axis=1)
    csv_df.to_csv(csv_output_file, index=False)
    print(f"\nSummary saved to {csv_output_file}")
    
    # Save detailed transactions to CSV
    transactions_df = pd.DataFrame(all_transactions)
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    transactions_df = transactions_df.sort_values('date')
    transactions_output_file = f"transactions_detail.csv"
    transactions_df.to_csv(transactions_output_file, index=False)
    print(f"Detailed transactions saved to {transactions_output_file}")
    
    # Save to Excel with better formatting if not csv-only mode
    if not args.csv_only:
        try:
            excel_output_file = f"{args.output}.xlsx"
            
            # Create Excel file with nice formatting
            writer = pd.ExcelWriter(excel_output_file, engine='xlsxwriter')
            
            # Convert to simpler DataFrame for Excel
            excel_df = pd.DataFrame({
                'Year': summary_df['year'],
                'Month': summary_df['month_name'],
                'Income': summary_df['income'],
                'Expenses': summary_df['expenses'],
                'Profit': summary_df['profit']
            })
            
            # Add category expenses as additional sheets
            # Create expense by category summary
            category_totals = defaultdict(float)
            
            for item in monthly_summary:
                for category, amount in item.get('category_expenses', {}).items():
                    category_totals[category] += amount
            
            # Sort categories by total amount
            sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
            
            # Create category summary data
            category_data = {
                'Category': [cat for cat, _ in sorted_categories],
                'Total': [amount for _, amount in sorted_categories]
            }
            
            category_df = pd.DataFrame(category_data)
            
            # Write to Excel
            excel_df.to_excel(writer, sheet_name='Monthly Summary', index=False)
            category_df.to_excel(writer, sheet_name='Expense Categories', index=False)
            
            # Also add transactions to the Excel file
            transactions_df.to_excel(writer, sheet_name='All Transactions', index=False)
            
            # Get the xlsxwriter workbook and worksheet objects
            workbook = writer.book
            summary_worksheet = writer.sheets['Monthly Summary']
            categories_worksheet = writer.sheets['Expense Categories']
            
            # Add formats
            money_format = workbook.add_format({'num_format': '$#,##0.00', 'align': 'right'})
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'align': 'center',
                'border': 1
            })
            
            # Apply formats to summary worksheet
            summary_worksheet.set_column('A:A', 8)
            summary_worksheet.set_column('B:B', 12)
            summary_worksheet.set_column('C:E', 15, money_format)
            
            # Format header row
            for col_num, value in enumerate(excel_df.columns.values):
                summary_worksheet.write(0, col_num, value, header_format)
            
            # Add conditional formatting for profit column
            summary_worksheet.conditional_format(1, 4, len(excel_df) + 1, 4, {
                'type': '3_color_scale',
                'min_color': '#FF6666',  # Red for negative
                'mid_color': '#FFFFFF',  # White for zero
                'max_color': '#66FF66'   # Green for positive
            })
            
            # Apply formats to categories worksheet
            categories_worksheet.set_column('A:A', 20)
            categories_worksheet.set_column('B:B', 15, money_format)
            
            # Format header row
            for col_num, value in enumerate(category_df.columns.values):
                categories_worksheet.write(0, col_num, value, header_format)
            
            # Close the Excel writer
            writer.close()
            
            print(f"Summary also saved to {excel_output_file}")
        except Exception as e:
            print(f"Could not create Excel file: {e}")
            print("Install xlsxwriter package to enable Excel output: pip install xlsxwriter")
    else:
        print("Excel output skipped (CSV only mode)")
    
    # Generate charts if requested
    if args.charts:
        try:
            create_charts(monthly_summary, args.charts_dir)
        except ImportError:
            print("Matplotlib is required for chart generation. Install with: pip install matplotlib")
    
    return monthly_summary, all_transactions

if __name__ == "__main__":
    main()
