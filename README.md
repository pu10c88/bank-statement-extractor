# Bank Statement Extractor v2.0.0

A tool for extracting transaction data from bank statements and generating financial insights.

## Features

- **PDF Statement Extraction**: Extract transaction data from Chase and Bank of America (BoFA) PDF statements
- **Web Interface**: User-friendly web application for uploading and processing bank statements
- **Financial Analysis**: View summaries, charts, and KPIs based on your transaction data
- **Data Visualization**: Interactive KPI dashboard with income/expense breakdown and trends
- **Export to CSV**: Download transaction data in CSV format for further analysis

## New in Version 2.0.0

- Added interactive KPI dashboard with Streamlit integration
- Enhanced data visualizations with Plotly charts
- Improved transaction categorization
- Simplified user interface with consolidated dashboard
- Added support for CORS and better error handling
- Included demo data for testing and development

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

3. Upload your bank statements and select the bank type

4. View the results and KPI dashboard

## Project Structure

- **Extractor Files/**: Core extraction scripts for different bank types
- **CSV Files/**: Output directory for extracted transaction data
- **GUI/web/**: Web application files
  - **static/**: Static assets (CSS, JS, images)
  - **templates/**: HTML templates
  - **uploads/**: Temporary storage for uploaded files
  - **webapp.py**: Main Flask application

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Developed by SIGMA BI - Development Team 