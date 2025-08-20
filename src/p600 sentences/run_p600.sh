#!/bin/bash

# P600 Sentence Processing Runner Script
# This script sets up the environment and runs the P600 sentence processing

echo "Setting up P600 sentence processing environment..."

# Check if we're in the right directory
if [ ! -f "p600.py" ]; then
    echo "Error: Please run this script from the 'p600 sentences' directory"
    exit 1
fi

# Check if API key is set (OpenAI is the default)
api_key_env="OPENAI_API_KEY"
api_key=$OPENAI_API_KEY

if [ -z "$api_key" ]; then
    echo "Error: $api_key_env environment variable is not set"
    echo "Please set it first:"
    echo "export $api_key_env=your_api_key_here"
    echo ""
    echo "Note: The script automatically detects OpenAI models and uses OPENAI_API_KEY."
    echo "For other model providers, you may need to set NEURONPEDIA_API_KEY instead."
    exit 1
fi

echo "✓ $api_key_env environment variable is set"
echo "✓ Running P600 sentence processing..."

# Run the processing
python p600.py

echo "✓ Processing complete!"
echo "Check the 'results' directory for output files."
