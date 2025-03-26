# Bank Statement Extractor

Extract and analyze transaction data from bank statements (PDF files).

## Features

- Extract transactions from Chase and Bank of America statements
- Merge multiple statements into a single CSV file
- View financial summaries and reports
- Filter and analyze transaction data

## Building Installation Packages

### Prerequisites

- Python 3.7 or newer
- PyInstaller (`pip install pyinstaller`)
- For Windows installer: NSIS (Nullsoft Scriptable Install System)
- For macOS DMG: create-dmg (`brew install create-dmg`)
- For Linux AppImage: appimagetool

### Building Installers

1. Ensure all requirements are installed:
   ```
   pip install -r requirements.txt
   ```

2. Run the appropriate build script for your platform:

   **Windows:**
   ```
   build_installer.bat
   ```

   **macOS/Linux:**
   ```
   ./build_installer.sh
   ```

3. The installers will be created in the `installers` directory:
   - Windows: `Bank_Statement_Extractor_Windows_Setup.exe` or `Bank_Statement_Extractor_Windows.zip`
   - macOS: `Bank_Statement_Extractor_macOS.dmg` or `Bank_Statement_Extractor_macOS.zip`
   - Linux: `Bank_Statement_Extractor_Linux.AppImage` or `Bank_Statement_Extractor_Linux.tar.gz`

### Sending to Clients

1. Verify the installer works on a clean system
2. Send the appropriate installer file to your client along with the `INSTALLATION_GUIDE.md` file
3. Provide your contact information for support

## Development Setup

1. Clone the repository:
   ```
   git clone [repository-url]
   cd Bank_Extraction_Code
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the desktop application:
   ```
   python GUI/desktop/bank_extractor_gui.py
   ```

5. Run the web application:
   ```
   cd GUI/web
   python webapp.py
   ```

## Project Structure

- `Extractor Files/` - Core extraction logic
- `GUI/desktop/` - Desktop application (PyQt5)
- `GUI/web/` - Web application (Flask)
- `Logos/` - Application icons and branding

## License

This project is licensed under the terms of the [LICENSE.txt](LICENSE.txt) file.

## Support

For support, contact the SIGMA BI Development Team. 