#!/bin/bash
# filepath: /Users/Omid.Solari/workspace/projects/mayo/create_env.sh

set -e  # Exit on any error

# Extract the package name from setup.py
PACKAGE_NAME=$(grep 'name=' setup.py | sed -E 's/.*name="([^"]+)".*/\1/')
if [ -z "$PACKAGE_NAME" ]; then
    PACKAGE_NAME="interview"
    echo "Could not extract the package name from setup.py. Falling back to default: $PACKAGE_NAME"
fi

PACKAGE_NAME="interview"

# Extract the Python version from requirements.in
PYTHON_VERSION=$(grep "^# Python version:" requirements.in | awk -F': ' '{print $2}' | head -n 1)
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"  # Default to 3.8 if not specified

# Check if the Conda environment already exists
if conda env list | grep -q "^$PACKAGE_NAME\s"; then
    echo "Conda environment '$PACKAGE_NAME' already exists."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$PACKAGE_NAME"

    # Check if the Python version matches
    CURRENT_PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [ "$CURRENT_PYTHON_VERSION" != "$PYTHON_VERSION" ]; then
        echo "Python version in the environment is $CURRENT_PYTHON_VERSION, but $PYTHON_VERSION is required."
        echo "Please recreate the environment with the correct Python version."
        exit 1
    fi
else
    echo "Creating a new Conda environment named '$PACKAGE_NAME' with Python $PYTHON_VERSION..."
    conda create -y -n "$PACKAGE_NAME" python="$PYTHON_VERSION"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$PACKAGE_NAME"
fi

# Upgrade pip and setuptools
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools

# Install the package in editable mode
echo "Installing the package in editable mode..."
pip install -e .

echo "Environment setup complete. To activate the environment, run: conda activate $PACKAGE_NAME"