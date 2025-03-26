# CSV Files Directory

This directory is used to store the output CSV files containing extracted transaction data from bank statements.

## Structure

- Each processing session creates a subdirectory with a unique session ID
- `all_transactions.csv` is created when multiple statement files are merged
- Individual statement CSV files are named after their source PDF files

## Notes

- This directory is excluded from Git to avoid committing user data
- The application will create necessary subdirectories during runtime
