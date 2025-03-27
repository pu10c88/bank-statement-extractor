# CSV Files Directory

This directory stores extracted transaction data in CSV format.

Each processing session creates its own subdirectory identified by a session ID.
The application stores all the transaction data exported from bank statements in CSV format.

## Structure

- Session-specific subdirectories (e.g., `UUID-format-session-id/`)
- Each session directory contains:
  - Individual statement CSV files (e.g., `eStmt_2024-05-31.csv`)
  - Monthly summary files (e.g., `January.csv`)
  - Combined transaction data (`all_transactions.csv`)

## Note

The actual transaction data files are not tracked in git for privacy and security reasons.
Only this README file is included in the repository. 