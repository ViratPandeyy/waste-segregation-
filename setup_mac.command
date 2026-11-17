#!/bin/bash
cd "$(dirname "$0")"
echo "Creating Python virtual environment..."
python3.11 -m venv .venv || { echo "Python 3.11 not found. Install Python 3.11 from python.org."; read -p "Press Return to close..."; exit 1; }
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete. Run ./run_mac.command to start the app."
read -p "Press Return to close..."
