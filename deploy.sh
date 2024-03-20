#!/bin/bash

# Activate the virtual environment
source /home/ubuntu/ruu/venv/bin/activate

# Navigate to the application directory
cd /home/ubuntu/ruu

# Pull the latest changes from the repository
git pull origin main

# Install any new dependencies
pip install -r requirements.txt

# Restart the Gunicorn service for your Flask app
sudo systemctl restart ruuapi.service

echo "Deployment completed successfully."