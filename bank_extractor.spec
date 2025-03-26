# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

# Define paths
block_cipher = None
base_path = os.path.abspath(os.path.dirname('__file__'))
gui_path = os.path.join(base_path, 'GUI', 'desktop')
extractor_path = os.path.join(base_path, 'Extractor Files')
logo_path = os.path.join(base_path, 'Logos')

# Add data files
data_files = [
    (os.path.join(logo_path, 'icone_1.png'), 'Logos'),
]

# Find all Python files in Extractor Files directory
extractor_files = []
for root, dirs, files in os.walk(extractor_path):
    for file in files:
        if file.endswith('.py'):
            source_path = os.path.join(root, file)
            dest_path = os.path.join('Extractor Files', os.path.relpath(source_path, extractor_path))
            extractor_files.append((source_path, os.path.dirname(dest_path)))

data_files.extend(extractor_files)

a = Analysis(
    [os.path.join(gui_path, 'bank_extractor_gui.py')],
    pathex=[base_path, extractor_path],
    binaries=[],
    datas=data_files,
    hiddenimports=['pandas', 'PyQt5', 'PyPDF2', 'tabula', 'numpy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Bank Statement Extractor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(logo_path, 'icone_1.png'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Bank Statement Extractor',
)

# For macOS
app = BUNDLE(
    coll,
    name='Bank Statement Extractor.app',
    icon=os.path.join(logo_path, 'icone_1.png'),
    bundle_identifier='com.sigmaBI.bankstatementextractor',
    info_plist={
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': 'True',
    },
) 