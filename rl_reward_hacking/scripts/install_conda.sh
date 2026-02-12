#!/bin/bash

# Script to install Miniconda
# This script downloads and installs Miniconda3 to the user's home directory

set -e  # Exit on error

CONDA_INSTALL_DIR=${CONDA_INSTALL_DIR:-$HOME/miniconda3}
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
INSTALLER_PATH="/tmp/miniconda_installer.sh"

echo "Installing Miniconda..."

# Check if conda is already installed
if command -v conda &> /dev/null; then
    echo "Conda is already installed at $(which conda)"
    conda --version
    exit 0
fi

# Check if miniconda directory already exists
if [ -d "$CONDA_INSTALL_DIR" ]; then
    echo "Miniconda directory already exists at $CONDA_INSTALL_DIR"
    echo "To reinstall, please remove this directory first: rm -rf $CONDA_INSTALL_DIR"
    exit 1
fi

echo "1. Downloading Miniconda installer..."
wget -q "$MINICONDA_URL" -O "$INSTALLER_PATH" || {
    echo "Error: Failed to download Miniconda installer"
    exit 1
}

echo "2. Installing Miniconda to $CONDA_INSTALL_DIR..."
chmod +x "$INSTALLER_PATH"
bash "$INSTALLER_PATH" -b -p "$CONDA_INSTALL_DIR" || {
    echo "Error: Failed to install Miniconda"
    rm -f "$INSTALLER_PATH"
    exit 1
}

echo "3. Initializing conda..."
"$CONDA_INSTALL_DIR/bin/conda" init bash || {
    echo "Warning: Failed to initialize conda in bash"
}

# Clean up installer
rm -f "$INSTALLER_PATH"

echo "4. Verifying installation..."
"$CONDA_INSTALL_DIR/bin/conda" --version

echo ""
echo "Miniconda installation completed successfully!"
echo "Location: $CONDA_INSTALL_DIR"
echo ""
echo "To use conda in this session, run:"
echo "  source $CONDA_INSTALL_DIR/etc/profile.d/conda.sh"
echo ""
echo "Or restart your terminal to have conda available automatically."
echo ""
echo "To create the verl environment as specified in README.md, run:"
echo "  $CONDA_INSTALL_DIR/bin/conda create -n verl python==3.10"
echo "  $CONDA_INSTALL_DIR/bin/conda activate verl"

