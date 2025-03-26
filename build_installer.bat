@echo off
echo Building Bank Statement Extractor installer for Windows...

REM Install required packages if needed
pip install -r requirements.txt
pip install pyinstaller

REM Clear previous build directories
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"

REM Build the application using PyInstaller
pyinstaller bank_extractor.spec

REM Create output directory for installers
if not exist "installers" mkdir installers

REM Use NSIS if available, otherwise just copy the dist folder
where nsis >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Packaging with NSIS...
    REM This assumes you have an NSIS script file named installer.nsi
    makensis installer.nsi
) else (
    echo NSIS not found, creating ZIP package instead...
    powershell Compress-Archive -Path "dist\Bank Statement Extractor" -DestinationPath "installers\Bank_Statement_Extractor_Windows.zip" -Force
)

echo Build complete! Check the installers directory for the output files.
pause 