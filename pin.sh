#!/bin/bash
set -e

# Upgrade pip and setuptools
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools

# Check if pip-tools is installed
if ! pip show pip-tools > /dev/null 2>&1; then
    echo "pip-tools is not installed. Installing it now..."
    pip install pip-tools
fi

# Define input and output files
INPUT_FILE="requirements.in"
OUTPUT_FILE="requirements.txt"

# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Input file '$INPUT_FILE' not found. Please create it before running this script."
    exit 1
fi

# Create a temporary virtual environment
TEMP_ENV_DIR=$(mktemp -d)
echo "Creating temporary virtual environment in $TEMP_ENV_DIR..."
python -m venv "$TEMP_ENV_DIR"

# Activate the virtual environment
source "$TEMP_ENV_DIR/bin/activate"

# Upgrade pip and setuptools in the temporary environment
pip install --upgrade pip setuptools

# Install pip-tools in the temporary environment
pip install pip-tools

# Run pip-compile in the temporary environment
pip-compile --output-file="$OUTPUT_FILE" "$INPUT_FILE"

# Deactivate and remove the temporary virtual environment
deactivate
rm -rf "$TEMP_ENV_DIR"

echo "Temporary virtual environment removed."
echo "Environment pinned successfully for Python. Output written to '$OUTPUT_FILE'."