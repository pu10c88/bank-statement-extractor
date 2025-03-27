# Bank Statement Extractor v3.1.0 - Installation Guide

## Overview

Bank Statement Extractor is a professional desktop application that allows you to extract transaction data from bank statements (PDF, CSV, and Excel files) and analyze your financial information. With the new v3.1.0 update, it features a modernized dark UI optimized for productivity and eye comfort. This guide will help you install and set up the application on your computer.

## What's New in v3.1.0

- **Modern Dark UI**: Completely redesigned interface with professional dark mode and Sigma brand colors
- **Enhanced User Experience**: Improved workflow with intuitive navigation and responsive design
- **Better CSV Support**: Advanced CSV handling capabilities for import and export
- **Improved Data Tables**: Enhanced transaction tables with better filtering and sorting
- **Monthly Data Analysis**: New tools for analyzing transactions by month

## System Requirements

- Windows 10/11, macOS 10.14+, or Linux
- At least 4GB RAM
- 500MB free disk space
- Modern web browser (Chrome, Firefox, Safari, Edge recommended)

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

1. **Ensure you have Python 3.9 or newer installed**
   ```
   python --version
   ```

2. **Clone the repository or download the source code**
   ```
   git clone [repository-url]
   cd Bank_Extraction_Code
   ```

3. **Install the required packages**
   ```
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```
   cd GUI/web
   python webapp.py
   ```

5. **Access the application** by opening a web browser and navigating to:
   ```
   http://localhost:4446
   ```

## Using the New Interface

1. When you first launch the application, you'll see the new dark-themed interface.

2. **Upload files**:
   - Drag and drop files directly into the upload area
   - Or click "Select Files" to browse your computer
   - Select your bank type (for PDF files) from the dropdown menu

3. **View Results**:
   - Financial summary cards display key metrics
   - Monthly breakdown section shows period analysis
   - Transaction table provides detailed view with search and filter options
   - Use the KPI Dashboard for visual data analysis

4. **Export Options**:
   - Download complete transaction set as CSV
   - Export specific monthly data
   - Access the data directly from the KPI dashboard

## Troubleshooting

### Common Issues:

1. **Application won't start**:
   - Windows: Make sure Microsoft Visual C++ Redistributable is installed
   - macOS: Right-click the app and select "Open" to bypass Gatekeeper
   - Linux: Make sure the AppImage has execute permissions
   - Web version: Ensure port 4446 is available and not blocked by firewall

2. **UI display issues**:
   - Try using a different modern browser
   - Ensure your browser is up to date
   - Disable any browser extensions that might interfere with the application

3. **Data extraction fails**:
   - Ensure your files are not password protected
   - Check that the statements are from supported banks
   - Verify the PDF/CSV format matches expected layouts

4. **Dependencies issues**:
   - If using the Python package method, ensure all required packages are installed:
     ```
     pip install -r requirements.txt
     ```

### Contact Support:

If you encounter any issues, please contact us at:
- Email: paulo.loureiro@sigmabusinessint.com
- GitHub: Create an issue on the repository

## Updates

- Windows/macOS: The application will automatically check for updates
- Python package: Update with `git pull` in the project directory and reinstall requirements if needed

---

© 2024 SIGMA BI - Development Team 