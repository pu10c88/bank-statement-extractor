# Bank Statement Extractor v3.1.0

A professional tool for extracting transaction data from bank statements and generating financial insights with a modern tech-focused UI.

![SIGMA BI Logo](Logos/icone_1.png)

## Features

- **PDF Statement Extraction**: Extract transaction data from Chase and Bank of America (BoFA) PDF statements
- **Support for Multiple File Types**: Process PDF, CSV and Excel files with transaction data
- **Modern Dark UI**: Tech-focused interface with Sigma colors for enhanced usability and reduced eye strain
- **Financial Analysis**: View comprehensive summaries, charts, and KPIs based on your transaction data
- **Data Visualization**: Interactive KPI dashboard with income/expense breakdown and trends
- **Advanced CSV Handling**: Generate, download, and process CSV files with automatic formatting
- **Monthly Data Breakdown**: View and export transaction data by month for better period analysis
- **Responsive Design**: Optimized for both desktop and mobile use

## New in Version 3.1.0

- Completely redesigned dark mode UI with Sigma brand colors and tech-focused aesthetics
- Enhanced data tables with improved filtering, sorting, and pagination
- Modernized dashboard with financial summary cards and visual indicators
- Improved mobile responsiveness with adaptive layouts
- Better visual hierarchy and information organization for enhanced UX
- Technical improvements for DataTables integration and performance
- More intuitive file upload interface with drag-and-drop support
- Enhanced CSV export with better formatting and monthly breakdown options

## UI Features

The redesigned user interface includes:

- **Dark Theme**: Professional dark theme optimized for reduced eye strain
- **Dynamic Cards**: Financial metrics displayed in easy-to-read cards with color coding
- **Interactive Elements**: Improved buttons, dropdowns, and input fields with responsive feedback
- **Table Enhancements**: Sortable, filterable transaction tables with modern styling
- **Progress Indicators**: Visual feedback during file processing
- **Consistent Branding**: Sigma color palette applied throughout the application
- **Improved Typography**: Enhanced readability with optimized fonts and spacing

## CSV Functionality

The application offers comprehensive CSV capabilities:

- **CSV Generation**: Automatically format and generate CSV files from extracted transaction data
- **Monthly CSV Export**: Download transaction data filtered by specific months
- **CSV Import**: Process existing CSV files with transaction data
- **Format Detection**: Intelligent detection of CSV structure and column mapping
- **Data Cleaning**: Automatic cleanup of imported CSV data for consistency
- **Export Options**: Choose between different CSV formats and structures

## Screenshots

![Main Interface](screenshots/main_interface.png)
![Results Dashboard](screenshots/results_dashboard.png)

## Installation

### Prerequisites

- Python 3.9+
- pip (Python package manager)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/YOUR-USERNAME/Bank_Extraction_Code.git
   cd Bank_Extraction_Code
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the web application:
   ```
   cd GUI/web
   python webapp.py
   ```

2. Open your browser and go to: `http://localhost:4446`

3. Upload your bank statements and select the bank type:
   - Drag and drop files into the upload area
   - Or click "Select Files" to browse your computer
   - Choose your bank type from the dropdown (for PDF statements)

4. View results, including:
   - Financial summary with income, expenses, and profit metrics
   - Monthly breakdown of transactions
   - Detailed transaction table with searching and filtering
   - KPI Dashboard with visual charts and trends

5. Export your data:
   - Download complete transaction set as CSV
   - Export specific monthly data
   - Access formatted data for further analysis

## Project Structure

- **Extractor Files/**: Core extraction scripts for different bank types
- **CSV Files/**: Output directory for extracted transaction data
- **GUI/web/**: Web application files
  - **static/**: Static assets (CSS, JS, images)
  - **templates/**: HTML templates with modern UI design
  - **uploads/**: Temporary storage for uploaded files
  - **webapp.py**: Main Flask application
- **Logos/**: Application branding assets
- **build_installer.bat/sh**: Scripts for building standalone installers

## Building Installers

For Windows:
```
./build_installer.bat
```

For macOS/Linux:
```
chmod +x build_installer.sh
./build_installer.sh
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Developed by Paulo Loureiro Campos for Sigma Business Intelligence

## Support

For support, please contact: paulo.loureiro@sigmabusinessint.com 