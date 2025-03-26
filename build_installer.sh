#!/bin/bash

echo "Building Bank Statement Extractor installer..."

# Determine OS
OS="$(uname)"

# Install required packages
pip install -r requirements.txt
pip install pyinstaller

# Clear previous build directories
rm -rf dist build

# Build the application using PyInstaller
pyinstaller bank_extractor.spec

# Create output directory for installers
mkdir -p installers

if [[ "$OS" == "Darwin" ]]; then
    echo "Building for macOS..."
    
    # Create a DMG if create-dmg is available
    if command -v create-dmg &> /dev/null; then
        echo "Creating DMG package..."
        create-dmg \
            --volname "Bank Statement Extractor" \
            --volicon "Logos/icone_1.png" \
            --window-pos 200 120 \
            --window-size 800 400 \
            --icon-size 100 \
            --icon "Bank Statement Extractor.app" 200 190 \
            --hide-extension "Bank Statement Extractor.app" \
            --app-drop-link 600 185 \
            "installers/Bank_Statement_Extractor_macOS.dmg" \
            "dist/Bank Statement Extractor.app"
    else
        echo "create-dmg not found, creating ZIP package instead..."
        (cd dist && zip -r "../installers/Bank_Statement_Extractor_macOS.zip" "Bank Statement Extractor.app")
    fi
    
    echo "macOS package created in installers directory"
    
elif [[ "$OS" == "Linux" ]]; then
    echo "Building for Linux..."
    
    # Check if appimagetool is available
    if command -v appimagetool &> /dev/null; then
        echo "Creating AppImage..."
        
        # Create AppDir structure
        mkdir -p AppDir/usr/bin
        mkdir -p AppDir/usr/share/applications
        mkdir -p AppDir/usr/share/icons/hicolor/256x256/apps
        
        # Copy files
        cp -r "dist/Bank Statement Extractor"/* AppDir/usr/bin/
        cp "Logos/icone_1.png" AppDir/usr/share/icons/hicolor/256x256/apps/bank-statement-extractor.png
        
        # Create desktop file
        cat > AppDir/usr/share/applications/bank-statement-extractor.desktop << EOF
[Desktop Entry]
Name=Bank Statement Extractor
Exec=bank-statement-extractor
Icon=bank-statement-extractor
Type=Application
Categories=Office;Finance;
EOF
        
        # Create AppRun
        cat > AppDir/AppRun << EOF
#!/bin/bash
HERE="\$(dirname "\$(readlink -f "\${0}")")"
EXEC="\${HERE}/usr/bin/Bank Statement Extractor"
exec "\${EXEC}" "\$@"
EOF
        
        chmod +x AppDir/AppRun
        
        # Build AppImage
        appimagetool AppDir "installers/Bank_Statement_Extractor_Linux.AppImage"
        
        # Clean up
        rm -rf AppDir
    else
        echo "appimagetool not found, creating tarball instead..."
        (cd dist && tar -czf "../installers/Bank_Statement_Extractor_Linux.tar.gz" "Bank Statement Extractor")
    fi
    
    echo "Linux package created in installers directory"
else
    echo "Unsupported OS: $OS"
    exit 1
fi

echo "Build complete! Check the installers directory for the output files." 