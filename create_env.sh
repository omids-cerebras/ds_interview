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

# Check if the Conda environment already exists
if conda env list | grep -q "^$PACKAGE_NAME\s"; then
    echo "Conda environment '$PACKAGE_NAME' already exists."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$PACKAGE_NAME"
else
    echo "Creating a new Conda environment named '$PACKAGE_NAME'..."
    conda create -y -n "$PACKAGE_NAME" python
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