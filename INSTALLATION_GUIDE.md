# Bank Statement Extractor v3.1.1 - Web Application Installation Guide

## Overview

Bank Statement Extractor is a professional web application that allows you to extract transaction data from bank statements (PDF, CSV, and Excel files) and analyze your financial information. With the v3.1.1 update, it features a modernized dark UI optimized for productivity and eye comfort. This guide will help you install and set up the web application on your server.

## What's New in v3.1.1

- **Modern Dark UI**: Completely redesigned interface with professional dark mode and Sigma brand colors
- **Enhanced User Experience**: Improved workflow with intuitive navigation and responsive design
- **Better CSV Support**: Advanced CSV handling capabilities for import and export
- **Improved Data Tables**: Enhanced transaction tables with better filtering and sorting
- **Monthly Data Analysis**: New tools for analyzing transactions by month

## System Requirements

- Python 3.9+
- Modern web browser (Chrome, Firefox, Safari, Edge recommended)
- 4GB RAM (recommended)
- 500MB free disk space

## Installation

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

## Using the Web Interface

1. When you first access the application, you'll see the dark-themed interface.

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

1. **Web server won't start**:
   - Ensure port 4446 is available and not blocked by firewall
   - Check that you have the necessary permissions to run the server
   - Verify all dependencies are installed correctly

2. **UI display issues**:
   - Try using a different modern browser
   - Ensure your browser is up to date
   - Disable any browser extensions that might interfere with the application

3. **Data extraction fails**:
   - Ensure your files are not password protected
   - Check that the statements are from supported banks
   - Verify the PDF/CSV format matches expected layouts

4. **Dependencies issues**:
   - Ensure all required packages are installed:
     ```
     pip install -r requirements.txt
     ```

### Contact Support:

If you encounter any issues, please contact us at:
- Email: paulo.loureiro@sigmabusinessint.com
- GitHub: Create an issue on the repository

## Updates

- Update with `git pull` in the project directory and reinstall requirements if needed:
  ```
  git pull
  pip install -r requirements.txt
  ```

---

© 2024 SIGMA BI - Development Team 