#!/bin/bash

echo "Starting setup..."

# Check if npm is installed
if ! command -v npm &> /dev/null
then
    echo "npm could not be found. Please install Node.js."
    exit
fi

# We can optionally install globally if not present
echo "Checking for Truffle and Ganache..."
truffle version || npm install -g truffle ganache

echo "Setting up Python virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! To activate the environment, run: source venv/bin/activate"
