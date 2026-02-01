#!/bin/bash
echo "🏥 Starting Medi-Gemma CDSS..."

# Load Environment Variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)
# Run the App
streamlit run src/interface/app_main.py
