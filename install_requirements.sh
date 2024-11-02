#!/bin/bash

# Define the virtual environment name (replace 'venv' with your desired name)
VENV_DIR="venv"

# Check if a virtual environment is currently activated
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating current virtual environment: $VIRTUAL_ENV"
    deactivate
else
    echo "No active virtual environment found."
fi

# Remove the existing virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment: $VENV_DIR"
    rm -rf "$VENV_DIR"
else
    echo "No existing virtual environment found."
fi

# Create a new virtual environment
echo "Creating a new virtual environment: $VENV_DIR"
python3 -m venv "$VENV_DIR" || { echo "Failed to create virtual environment"; exit 1; }

# Activate the new virtual environment
echo "Activating the new virtual environment: $VENV_DIR"
source "$VENV_DIR/bin/activate" || { echo "Failed to activate virtual environment"; exit 1; }

# Upgrade pip and install required packages
echo "Upgrading pip..."
pip install --upgrade pip || { echo "Failed to upgrade pip"; exit 1; }

# Install requirements
echo "Installing main packages..."
pip install -r requirements.txt || { echo "Failed to install main packages"; exit 1; }

echo "Installing torch-related packages..."
pip install -r torch_requirements.txt || { echo "Failed to install torch-related packages"; exit 1; }

echo "All packages installed successfully!"

source venv/bin/activate