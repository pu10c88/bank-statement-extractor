# Bank Statement Extractor - Installation Guide

## Overview

Bank Statement Extractor is a desktop application that allows you to extract transaction data from bank statements (PDF files) and analyze your financial information. This guide will help you install and set up the application on your computer.

## System Requirements

- Windows 10/11, macOS 10.14+, or Linux
- At least 4GB RAM
- 500MB free disk space

## Installation Options

There are two ways to install Bank Statement Extractor:

### Option 1: Pre-built Installers (Recommended)

1. **Download the installer** appropriate for your operating system:
   - For Windows: `Bank_Statement_Extractor_Windows_Setup.exe`
   - For macOS: `Bank_Statement_Extractor_macOS.dmg`
   - For Linux: `Bank_Statement_Extractor_Linux.AppImage`

2. **Run the installer**:
   - Windows: Double-click the .exe file and follow the installation wizard
   - macOS: Open the .dmg file, drag the application to your Applications folder
   - Linux: Make the AppImage executable (`chmod +x Bank_Statement_Extractor_Linux.AppImage`) and run it

3. **Launch the application** from your applications menu/desktop shortcut

### Option 2: Python Package Installation

If you're comfortable with Python, you can install the application as a Python package:

1. **Ensure you have Python 3.7 or newer installed**
   ```
   python --version
   ```

2. **Clone the repository or download the source code**
   ```
   git clone [repository-url]
   cd Bank_Extraction_Code
   ```

3. **Install the package**
   ```
   pip install -e .
   ```

4. **Run the application**
   ```
   bank_extractor
   ```

## First-time Setup

1. When you first launch the application, you'll need to:
   - Select an input directory containing your PDF bank statements
   - Choose an output directory for extracted CSV files
   - Select your bank type (currently supports Chase and Bank of America)

2. Click "Process Statements" to extract transaction data from your PDFs

3. Use "Merge All CSV Files" to combine multiple statements into one report

## Troubleshooting

### Common Issues:

1. **Application won't start**:
   - Windows: Make sure Microsoft Visual C++ Redistributable is installed
   - macOS: Right-click the app and select "Open" to bypass Gatekeeper
   - Linux: Make sure the AppImage has execute permissions

2. **PDF extraction fails**:
   - Ensure your PDF files are not password protected
   - Check that the statements are from supported banks
   - Verify the PDF format matches expected layouts

3. **Dependencies issues**:
   - If using the Python package method, ensure all required packages are installed:
     ```
     pip install -r requirements.txt
     ```

### Contact Support:

If you encounter any issues, please contact us at:
- Email: support@sigmaBI.com
- Support website: https://sigmaBI.com/support

## Updates

- Windows/macOS: The application will automatically check for updates
- Python package: Update with `pip install -e .` or `git pull` in the project directory

---

© 2024 SIGMA BI - Development Team 