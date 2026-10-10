#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

echo "==> [Render Build] Upgrading pip..."
python -m pip install --upgrade pip

echo "==> [Render Build] Installing Python dependencies..."
pip install -r requirements.txt

echo "==> [Render Build] Downloading and verifying AI model weights..."
python download_models.py

echo "==> [Render Build] Build completed successfully."
