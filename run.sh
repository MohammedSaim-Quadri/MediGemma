#!/bin/bash
echo "🏥 Starting Medi-Gemma CDSS..."

# Load Environment Variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Run the App
streamlit run app.py
