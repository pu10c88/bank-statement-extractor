from setuptools import setup, find_packages

setup(
    name="BankStatementExtractor",
    version="1.0.0",
    description="Extract transactions from bank statements",
    author="SIGMA BI - Development Team",
    packages=find_packages(),
    install_requires=[
        "PyQt5>=5.15.0",
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "PyPDF2>=2.0.0",
        "tabula-py>=2.0.0",
        "JPype1>=1.2.0",  # Required by tabula
        "openpyxl>=3.0.0",  # For Excel export
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "bank_extractor=GUI.desktop.bank_extractor_gui:main",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 