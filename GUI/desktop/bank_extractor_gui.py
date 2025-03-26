import sys
import os
from pathlib import Path

# App information
APP_NAME = "Bank Statement Extractor"
APP_VERSION = "1.0.0"
APP_AUTHOR = "SIGMA BI - Development Team"

# Add the Extractor Files directory to the system path - Fix the absolute path
extractor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Extractor Files'))
sys.path.insert(0, extractor_path)
print(f"Added to path: {extractor_path}")

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QVBoxLayout, QHBoxLayout, QFileDialog, QWidget, 
                            QComboBox, QTextEdit, QTabWidget, QGroupBox,
                            QProgressBar, QMessageBox, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import pandas as pd

# Import the extraction functions from the existing scripts
from chase_statements_load import load_pdf as chase_load_pdf
from chase_statements_load import merge_csv_files
from bofa_statements_load import load_pdf as bofa_load_pdf

class BankExtractorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize variables
        self.input_directory = None
        # Use absolute path for default output directory
        default_csv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../CSV Files'))
        self.output_directory = Path(default_csv_dir)
        self.output_directory.mkdir(exist_ok=True)
        print(f"Default output directory: {self.output_directory}")
        
        # Set up the main window
        self.setWindowTitle(f'{APP_NAME} v{APP_VERSION}')
        self.setGeometry(100, 100, 800, 600)
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Add app info header with logo
        header_layout = QHBoxLayout()
        
        # Load and add logo
        logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Logos/icone_1.png'))
        if os.path.exists(logo_path):
            logo_label = QLabel()
            logo_pixmap = QPixmap(logo_path)
            # Scale logo to a reasonable size
            logo_pixmap = logo_pixmap.scaledToHeight(60, Qt.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
            logo_label.setFixedSize(60, 60)
            header_layout.addWidget(logo_label)
        
        # Add app title and version info
        title_info_layout = QVBoxLayout()
        app_title = QLabel(f"<h2>{APP_NAME}</h2>")
        app_title.setTextFormat(Qt.RichText)
        title_info_layout.addWidget(app_title)
        
        app_version = QLabel(f"<p>Version {APP_VERSION}<br>by {APP_AUTHOR}</p>")
        app_version.setTextFormat(Qt.RichText)
        title_info_layout.addWidget(app_version)
        
        header_layout.addLayout(title_info_layout)
        header_layout.setStretch(1, 1)  # Make the title section expandable
        
        main_layout.addLayout(header_layout)
        
        # Add a horizontal line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Create tabs
        extraction_tab = QWidget()
        reports_tab = QWidget()
        
        tabs.addTab(extraction_tab, "Extract Statements")
        tabs.addTab(reports_tab, "View Reports")
        
        # Set up extraction tab
        extraction_layout = QVBoxLayout(extraction_tab)
        
        # Input directory selection
        input_group = QGroupBox("Input/Output Files")
        input_layout = QVBoxLayout()
        
        input_dir_layout = QHBoxLayout()
        self.input_dir_label = QLabel("No input directory selected")
        input_dir_button = QPushButton("Select Input Directory")
        input_dir_button.clicked.connect(self.select_input_directory)
        input_dir_layout.addWidget(self.input_dir_label)
        input_dir_layout.addWidget(input_dir_button)
        
        # Output directory selection
        output_dir_layout = QHBoxLayout()
        self.output_dir_label = QLabel(str(self.output_directory))
        output_dir_button = QPushButton("Select Output Directory")
        output_dir_button.clicked.connect(self.select_output_directory)
        output_dir_layout.addWidget(self.output_dir_label)
        output_dir_layout.addWidget(output_dir_button)
        
        input_layout.addLayout(input_dir_layout)
        input_layout.addLayout(output_dir_layout)
        input_group.setLayout(input_layout)
        extraction_layout.addWidget(input_group)
        
        # Bank selection
        bank_group = QGroupBox("Bank Selection")
        bank_layout = QHBoxLayout()
        bank_layout.addWidget(QLabel("Select Bank:"))
        self.bank_selector = QComboBox()
        self.bank_selector.addItems(["Chase", "Bank of America"])
        bank_layout.addWidget(self.bank_selector)
        bank_group.setLayout(bank_layout)
        extraction_layout.addWidget(bank_group)
        
        # Process controls
        process_group = QGroupBox("Process")
        process_layout = QVBoxLayout()
        
        self.process_button = QPushButton("Process Statements")
        self.process_button.clicked.connect(self.process_statements)
        process_layout.addWidget(self.process_button)
        
        self.merge_button = QPushButton("Merge All CSV Files")
        self.merge_button.clicked.connect(self.merge_all_csvs)
        process_layout.addWidget(self.merge_button)
        
        self.progress_bar = QProgressBar()
        process_layout.addWidget(self.progress_bar)
        
        process_group.setLayout(process_layout)
        extraction_layout.addWidget(process_group)
        
        # Log window
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        extraction_layout.addWidget(log_group)
        
        # Reports tab
        reports_layout = QVBoxLayout(reports_tab)
        
        # Load report button
        load_report_button = QPushButton("Load All Transactions Report")
        load_report_button.clicked.connect(self.load_report)
        reports_layout.addWidget(load_report_button)
        
        # Report display
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        reports_layout.addWidget(self.report_text)
        
        # Add tabs to main layout
        main_layout.addWidget(tabs)
        
        # Set the central widget
        self.setCentralWidget(main_widget)
    
    def select_input_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory with PDF Statements")
        if directory:
            self.input_directory = Path(directory)
            self.input_dir_label.setText(str(self.input_directory))
            self.log_message(f"Selected input directory: {self.input_directory}")
    
    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory for CSV Files")
        if directory:
            self.output_directory = Path(directory)
            self.output_directory.mkdir(exist_ok=True)
            self.output_dir_label.setText(str(self.output_directory))
            self.log_message(f"Selected output directory: {self.output_directory}")
    
    def log_message(self, message):
        self.log_text.append(message)
    
    def format_currency(self, amount):
        """Format amount as currency with thousands separator and 2 decimal places"""
        return f"${amount:,.2f}"
    
    def process_statements(self):
        if not self.input_directory:
            QMessageBox.warning(self, "Warning", "Please select an input directory first.")
            return
        
        self.log_message(f"\nProcessing PDF files from: {self.input_directory}")
        self.log_message(f"Output CSV files will be saved to: {self.output_directory}")
        
        bank = self.bank_selector.currentText()
        processed_count = 0
        error_count = 0
        total_transactions = 0
        total_deposits = 0
        total_withdrawals = 0
        
        # Get list of PDF files
        pdf_files = list(self.input_directory.glob('*.pdf'))
        
        if not pdf_files:
            self.log_message("No PDF files found in the selected directory.")
            return
        
        self.progress_bar.setMaximum(len(pdf_files))
        self.progress_bar.setValue(0)
        
        for i, pdf_file in enumerate(pdf_files):
            try:
                self.log_message(f"Loading PDF: {pdf_file.name}...")
                
                try:
                    # Load PDF based on selected bank
                    if bank == "Chase":
                        df = chase_load_pdf(pdf_file)
                    else:
                        df = bofa_load_pdf(pdf_file)
                    
                    if df.empty:
                        self.log_message(f"⚠️ No transactions found in {pdf_file.name}")
                        continue
                except Exception as bank_error:
                    # If the selected bank fails, try the other bank format
                    self.log_message(f"⚠️ Error processing as {bank}: {str(bank_error)}")
                    self.log_message(f"Attempting to process with alternative bank format...")
                    
                    try:
                        if bank == "Chase":
                            df = bofa_load_pdf(pdf_file)
                        else:
                            df = chase_load_pdf(pdf_file)
                        
                        if df.empty:
                            self.log_message(f"⚠️ No transactions found with alternative format in {pdf_file.name}")
                            continue
                        
                        self.log_message(f"✓ Successfully extracted using alternative bank format")
                    except Exception as alt_error:
                        raise Exception(f"Failed with both bank formats. Original error: {str(bank_error)}, Alternative error: {str(alt_error)}")
                
                self.log_message(f"✓ Successfully extracted {len(df)} transactions")
                
                # Calculate and display statistics
                if 'amount' in df.columns:
                    deposits_df = df[df['amount'] > 0]
                    statement_deposits = deposits_df['amount'].sum()
                    total_deposits += statement_deposits
                    self.log_message(f"✓ Statement deposits: {self.format_currency(statement_deposits)} ({len(deposits_df)} transactions)")
                    
                    withdrawals_df = df[df['amount'] < 0]
                    statement_withdrawals = abs(withdrawals_df['amount'].sum())
                    total_withdrawals += statement_withdrawals
                    self.log_message(f"✓ Statement withdrawals: {self.format_currency(statement_withdrawals)} ({len(withdrawals_df)} transactions)")
                    self.log_message(f"✓ Statement net change: {self.format_currency(statement_deposits - statement_withdrawals)}")
                else:
                    self.log_message(f"⚠️ Warning: No 'amount' column found in extracted data")
                
                # Update running totals
                total_transactions += len(df)
                
                # Create output CSV filename
                csv_file = self.output_directory / pdf_file.with_suffix('.csv').name
                
                # Save extracted transactions to CSV file
                df.to_csv(csv_file, index=False)
                self.log_message(f"✓ Saved to CSV: {csv_file.name}")
                self.log_message("-" * 50)
                
                # Increment counter for successfully processed files
                processed_count += 1
                
            except Exception as e:
                # If any error occurs during processing, catch and log it
                error_msg = f"❌ Error processing {pdf_file.name}: {str(e)}"
                self.log_message(error_msg)
                self.log_message("-" * 50)
                error_count += 1
                
                # Show error in dialog for critical errors
                if "Failed with both bank formats" in str(e):
                    QMessageBox.critical(self, "Processing Error", 
                                         f"Failed to process {pdf_file.name} with any known format.\n\n{str(e)}")
            
            # Update progress bar
            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()  # Ensure UI updates
        
        # Print summary
        self.log_message("\nProcess Completed!")
        self.log_message("-" * 50)
        self.log_message(f"Successfully Processed: {processed_count} files")
        self.log_message(f"Total Transactions Processed: {total_transactions}")
        self.log_message("-" * 50)
        
        # Display financial summary
        net_profit = total_deposits - total_withdrawals
        self.log_message(f"FINANCIAL SUMMARY:")
        self.log_message(f"Total Income/Deposits: {self.format_currency(total_deposits)}")
        self.log_message(f"Total Expenses/Withdrawals: {self.format_currency(total_withdrawals)}")
        self.log_message(f"Net Profit: {self.format_currency(net_profit)}")
        self.log_message("-" * 50)
        
        self.log_message(f"Errors Encountered: {error_count} files")
        if processed_count > 0:
            self.log_message(f"CSV files are saved in: {self.output_directory}")
        
        # Show summary in a dialog if there were any errors
        if error_count > 0:
            QMessageBox.warning(self, "Processing Complete with Errors", 
                               f"Processed {processed_count} files with {error_count} errors.\n\nSee the log for details.")
        elif processed_count > 0:
            QMessageBox.information(self, "Processing Complete", 
                                   f"Successfully processed {processed_count} files with {total_transactions} transactions.\n\n"
                                   f"Total Income: {self.format_currency(total_deposits)}\n"
                                   f"Total Expenses: {self.format_currency(total_withdrawals)}\n"
                                   f"Net Profit: {self.format_currency(net_profit)}")
    
    def merge_all_csvs(self):
        try:
            self.log_message("\nMerging all CSV files...")
            merge_csv_files(str(self.output_directory))
            self.log_message("✓ Merge completed successfully")
            
            # After merging, show a comprehensive summary
            all_transactions_path = self.output_directory / 'all_transactions.csv'
            if all_transactions_path.exists():
                self.show_summary_after_merge(all_transactions_path)
        except Exception as e:
            self.log_message(f"❌ Error merging CSV files: {str(e)}")
    
    def show_summary_after_merge(self, file_path):
        try:
            df = pd.read_csv(file_path)
            total_transactions = len(df)
            
            if 'amount' in df.columns:
                deposits = df[df['amount'] > 0]['amount'].sum()
                withdrawals = abs(df[df['amount'] < 0]['amount'].sum())
                net_profit = deposits - withdrawals
                
                self.log_message("\nCOMPREHENSIVE FINANCIAL SUMMARY:")
                self.log_message(f"Total Transactions: {total_transactions}")
                self.log_message(f"Total Income/Deposits: {self.format_currency(deposits)}")
                self.log_message(f"Total Expenses/Withdrawals: {self.format_currency(withdrawals)}")
                self.log_message(f"Net Profit: {self.format_currency(net_profit)}")
                
                # Show in dialog for better visibility
                QMessageBox.information(self, "Comprehensive Financial Summary", 
                                       f"Total Transactions: {total_transactions}\n\n"
                                       f"Total Income: {self.format_currency(deposits)}\n"
                                       f"Total Expenses: {self.format_currency(withdrawals)}\n"
                                       f"Net Profit: {self.format_currency(net_profit)}")
        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.warning(self, "Summary Error", error_msg)
    
    def load_report(self):
        all_transactions_path = self.output_directory / 'all_transactions.csv'
        
        if not all_transactions_path.exists():
            QMessageBox.warning(self, "Warning", "No merged transactions file found. Please process and merge statements first.")
            return
        
        try:
            df = pd.read_csv(all_transactions_path)
            
            # Basic report
            self.report_text.clear()
            self.report_text.append("═══════════ TRANSACTION REPORT ═══════════\n")
            self.report_text.append(f"Total Transactions: {len(df)}")
            
            # Check if 'amount' column exists
            if 'amount' in df.columns:
                # Calculate totals
                deposits = df[df['amount'] > 0]['amount'].sum()
                withdrawals = abs(df[df['amount'] < 0]['amount'].sum())
                net_profit = deposits - withdrawals
                
                self.report_text.append(f"\n════════ FINANCIAL SUMMARY ════════")
                self.report_text.append(f"Total Income/Deposits: {self.format_currency(deposits)}")
                self.report_text.append(f"Total Expenses/Withdrawals: {self.format_currency(withdrawals)}")
                self.report_text.append(f"Net Profit: {self.format_currency(net_profit)}")
                
                # Display transactions by month
                self.report_text.append(f"\n════════ MONTHLY BREAKDOWN ════════")
                df['date'] = pd.to_datetime(df['date'])
                df['month'] = df['date'].dt.strftime('%Y-%m')
                
                monthly = df.groupby('month').agg({
                    'amount': ['count', 'sum']
                })
                
                for month, data in monthly.iterrows():
                    count = data[('amount', 'count')]
                    total = data[('amount', 'sum')]
                    monthly_income = df[(df['month'] == month) & (df['amount'] > 0)]['amount'].sum()
                    monthly_expenses = abs(df[(df['month'] == month) & (df['amount'] < 0)]['amount'].sum())
                    
                    self.report_text.append(f"\n{month}:")
                    self.report_text.append(f"  Transactions: {count}")
                    self.report_text.append(f"  Income: {self.format_currency(monthly_income)}")
                    self.report_text.append(f"  Expenses: {self.format_currency(monthly_expenses)}")
                    self.report_text.append(f"  Net: {self.format_currency(total)}")
            else:
                self.report_text.append("\n⚠️ Warning: No 'amount' column found in the data.")
        
        except Exception as e:
            error_msg = f"Error loading report: {str(e)}"
            self.report_text.setText(error_msg)
            QMessageBox.critical(self, "Report Error", error_msg)


def main():
    app = QApplication(sys.argv)
    window = BankExtractorGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 