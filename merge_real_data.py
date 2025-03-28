import pandas as pd
import os
import glob
from datetime import datetime

# Session ID
session_id = "2864665d-7f84-404f-9c95-0d31ef06f913"

# Directories
csv_dir = f"CSV Files/{session_id}"
output_dir = f"GUI/web/output/{session_id}"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all CSV files from the session directory
csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

print(f"Found {len(csv_files)} CSV files in {csv_dir}")

# Create an empty list to store all dataframes
all_dfs = []

# Read each CSV file and append to the list
for csv_file in csv_files:
    try:
        print(f"Processing {os.path.basename(csv_file)}...")
        df = pd.read_csv(csv_file)
        
        # Basic validation
        if 'date' not in df.columns or 'amount' not in df.columns:
            print(f"Skipping {os.path.basename(csv_file)} - missing required columns")
            continue
            
        # Add transaction type if not present
        if 'type' not in df.columns:
            df['type'] = df.apply(lambda row: 'credit' if float(row['amount']) >= 0 else 'debit', axis=1)
            
        all_dfs.append(df)
        print(f"Added {len(df)} transactions from {os.path.basename(csv_file)}")
    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")

# Merge all dataframes
if all_dfs:
    print("\nMerging all transactions...")
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Ensure date is in datetime format for sorting
    merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')
    
    # Sort by date (most recent first)
    merged_df = merged_df.sort_values('date', ascending=False)
    
    # Convert date back to a consistent string format
    merged_df['date'] = merged_df['date'].dt.strftime('%Y-%m-%d')
    
    # Save the merged dataframe to output directory
    output_file = os.path.join(output_dir, "all_transactions.csv")
    merged_df.to_csv(output_file, index=False)
    
    print(f"Created merged file with {len(merged_df)} transactions at {output_file}")
else:
    print("No valid transaction data found to merge") 