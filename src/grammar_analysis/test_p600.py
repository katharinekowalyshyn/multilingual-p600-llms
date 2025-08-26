#!/usr/bin/env python3
"""
Test script for P600 sentence processing setup.
This script tests the configuration and file paths without running the full LLM processing.
"""

import os
import sys
import pandas as pd

# Ensure src is in the Python path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

def test_configuration():
    """Test that the configuration and file paths are set up correctly."""
    print("Testing P600 sentence processing configuration...")
    
    # Test config path
    config_path = os.path.join(SCRIPT_DIR, "..", "conf", "config.yaml")
    if os.path.exists(config_path):
        print(f"✓ Config file found at: {config_path}")
    else:
        print(f"✗ Config file not found at: {config_path}")
        return False
    
    # Test input files
    input_files = [
        "gardenpath_sample.csv",
        "control_gardenpath_sample.csv", 
        "p600_sample.csv"
    ]
    
    for filename in input_files:
        file_path = os.path.join(SCRIPT_DIR, filename)
        if os.path.exists(file_path):
            print(f"✓ Input file found: {filename}")
            
            # Test CSV reading
            try:
                df = pd.read_csv(file_path)
                print(f"  - Contains {len(df)} sentences")
                if 'sentence' in df.columns:
                    print(f"  - Has 'sentence' column ✓")
                else:
                    print(f"  - Missing 'sentence' column ✗")
            except Exception as e:
                print(f"  - Error reading CSV: {e}")
        else:
            print(f"✗ Input file not found: {filename}")
            return False
    
    # Test output directory creation
    output_dir = os.path.join(SCRIPT_DIR, "results")
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Output directory ready: {output_dir}")
    except Exception as e:
        print(f"✗ Error creating output directory: {e}")
        return False
    
    # Test environment variable
    api_key_env = "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_env)
    if api_key:
        print(f"✓ {api_key_env} environment variable is set")
    else:
        print(f"⚠ {api_key_env} environment variable is not set")
        print("  You'll need to set this before running the full script:")
        print(f"  export {api_key_env}=your_api_key_here")
        print("  Note: The script will automatically detect if you're using OpenAI models")
        print("  and use the appropriate API key.")
    
    print("\nConfiguration test completed!")
    return True

if __name__ == "__main__":
    test_configuration()
