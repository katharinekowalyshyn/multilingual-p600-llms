"""
P600 Sentence SAE Analysis using Neuronpedia API

This script analyzes how SAE features change when processing control sentences vs. P600 sentences
using Neuronpedia's API for model inference and SAE feature extraction.

Key features:
- Uses Neuronpedia API for remote model inference (no local model loading)
- Extracts SAE features from model activations
- Compares feature changes between control and P600 sentences
- Creates visualizations of feature changes over time
"""

import hydra
from omegaconf import DictConfig
import pandas as pd
import torch
from typing import Literal, Dict, List, Tuple
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import requests
import json
import os
from transformers import AutoTokenizer
import time

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class NeuronpediaAPI:
    """Interface to Neuronpedia API for model inference and SAE features"""
    
    def __init__(self, api_key: str, base_url: str = "https://www.neuronpedia.org"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_available_models(self):
        """Get list of available models and their SAE sources"""
        # Try different possible API endpoints
        possible_endpoints = [
            f"{self.base_url}/api/models",
            f"{self.base_url}/api/sources",
            f"{self.base_url}/api/releases",
            f"{self.base_url}/api"
        ]
        
        for endpoint in possible_endpoints:
            try:
                print(f"Checking endpoint: {endpoint}")
                response = requests.get(endpoint, headers=self.headers, timeout=30)
                
                print(f"Response status: {response.status_code}")
                print(f"Response content type: {response.headers.get('content-type', 'unknown')}")
                
                if response.status_code == 200:
                    # Check if response is HTML (which suggests wrong endpoint)
                    if response.text.strip().startswith('<!DOCTYPE') or response.text.strip().startswith('<html'):
                        print(f"⚠️ {endpoint} returns HTML, not JSON")
                        continue
                    
                    try:
                        data = response.json()
                        print(f"✓ Success with endpoint: {endpoint}")
                        return data
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON decode error at {endpoint}: {e}")
                        continue
                else:
                    print(f"❌ {endpoint} failed: {response.status_code}")
                    continue
                    
            except Exception as e:
                print(f"Error with endpoint {endpoint}: {e}")
                continue
        
        print("❌ No working API endpoints found")
        return None
    
    def get_available_sources(self, model_name: str):
        """Get available sources for a specific model"""
        # Try to get available sources from the available-resources page
        try:
            url = f"{self.base_url}/available-resources"
            print(f"Checking available resources at: {url}")
            
            response = requests.get(url, headers=self.headers, timeout=30)
            if response.status_code == 200:
                # This will return HTML, but we can extract source information
                # For now, let's use some common source IDs based on the user's example
                if model_name == "gpt2-small":
                    return ["9-res-jb", "1-res-jb", "2-res-jb"]  # Common source IDs
                else:
                    return ["default"]
            else:
                print(f"Failed to get available resources: {response.status_code}")
                return ["default"]
                
        except Exception as e:
            print(f"Error getting available sources: {e}")
            return ["default"]
    
    def get_model_activations(self, model_name: str, prompt: str, layer: int, 
                             submodule: str = "resid") -> Dict:
        """Get model activations for a specific layer and submodule"""
        
        # Use the correct endpoint based on the user's example
        endpoint = f"{self.base_url}/api/activation/new"
        
        # Get available sources for this model
        available_sources = self.get_available_sources(model_name)
        print(f"Available sources for {model_name}: {available_sources}")
        
        # Try each available source until one works
        for source in available_sources:
            payload = {
                "feature": {
                    "modelId": model_name,
                    "source": source,
                    "index": str(layer)
                },
                "customText": prompt
            }
            
            try:
                print(f"Trying with source: {source}")
                print(f"Payload: {payload}")
                
                response = requests.post(endpoint, headers=self.headers, json=payload, timeout=30)
                
                print(f"Response status: {response.status_code}")
                print(f"Response content type: {response.headers.get('content-type', 'unknown')}")
                
                if response.status_code == 200:
                    print("✓ API call successful")
                    
                    # Check if response has content
                    if not response.text.strip():
                        print("⚠️ Response is empty")
                        continue
                    
                    try:
                        return response.json()
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON decode error: {e}")
                        print(f"Raw response: {response.text[:500]}...")
                        continue
                        
                elif response.status_code == 401:
                    print("❌ Authentication failed - check your API key")
                    return None
                elif response.status_code == 500:
                    error_msg = response.text[:200] if response.text else "Unknown error"
                    print(f"⚠️ API error with source {source}: {error_msg}")
                    # Try to parse JSON error if possible
                    try:
                        error_json = response.json()
                        if 'message' in error_json:
                            print(f"Error details: {error_json['message']}")
                    except:
                        pass
                    continue  # Try next source
                else:
                    print(f"❌ API error {response.status_code}: {response.text[:200]}...")
                    continue
                    
            except Exception as e:
                print(f"❌ API call failed with source {source}: {e}")
                continue
        
        print("❌ All sources failed")
        return None
    
    def get_sae_features(self, model_name: str, prompt: str, layer: int,
                        submodule: str = "resid") -> np.ndarray:
        """Extract SAE features from model activations"""
        
        activations = self.get_model_activations(model_name, prompt, layer, submodule)
        if activations and 'values' in activations:
            # The 'values' field contains the SAE feature activations
            values = activations['values']
            print(f"✓ Extracted SAE features from 'values' field")
            print(f"Feature shape: {len(values)} values")
            return np.array(values)
        elif activations and 'sae_features' in activations:
            return np.array(activations['sae_features'])
        elif activations and 'features' in activations:
            return np.array(activations['features'])
        elif activations and 'activations' in activations:
            # If we get raw activations, we might need to process them differently
            return np.array(activations['activations'])
        else:
            if activations:
                print(f"Unexpected response structure: {list(activations.keys())}")
                print(f"Available fields: {activations}")
            return None

    def test_api_connection(self):
        """Test basic API connectivity and authentication"""
        print("🔍 Testing Neuronpedia API connectivity...")
        
        # Try different authentication approaches
        test_headers = [
            {"Authorization": f"Bearer {self.api_key}"},
            {"X-API-Key": self.api_key},
            {"api-key": self.api_key},
            {"token": self.api_key}
        ]
        
        # Try different base URLs
        possible_base_urls = [
            "https://www.neuronpedia.org",
            "https://api.neuronpedia.org",
            "https://neuronpedia.org"
        ]
        
        for base_url in possible_base_urls:
            for headers in test_headers:
                try:
                    # Try a simple health check endpoint
                    endpoint = f"{base_url}/api/health"
                    print(f"Testing: {endpoint} with headers: {list(headers.keys())}")
                    
                    response = requests.get(endpoint, headers=headers, timeout=10)
                    print(f"Status: {response.status_code}")
                    
                    if response.status_code == 200:
                        print(f"✓ Success with {base_url} and headers {list(headers.keys())}")
                        return base_url, headers
                    elif response.status_code == 401:
                        print(f"⚠️ Authentication failed with {base_url}")
                    elif response.status_code == 404:
                        print(f"⚠️ Endpoint not found: {endpoint}")
                    else:
                        print(f"⚠️ Unexpected status: {response.status_code}")
                        
                except Exception as e:
                    print(f"Error testing {base_url}: {e}")
                    continue
        
        print("❌ No working API configuration found")
        return None, None

def analyze_feature_changes(control_features, p600_features):
    """Analyze how SAE features change between control and P600 sentences"""
    
    print("Analyzing feature changes...")
    
    # Find the maximum feature length across all sentences
    max_length = 0
    for submodule_name in control_features:
        for features in control_features[submodule_name]:
            if features is not None:
                max_length = max(max_length, len(features))
        for features in p600_features[submodule_name]:
            if features is not None:
                max_length = max(max_length, len(features))
    
    print(f"Maximum feature length: {max_length}")
    
    # Pad all features to the same length
    def pad_features(features_list, target_length):
        """Pad features to target length with zeros"""
        padded_features = []
        for features in features_list:
            if features is not None:
                # Pad with zeros to reach target length
                if len(features) < target_length:
                    padded = np.pad(features, (0, target_length - len(features)), 
                                  mode='constant', constant_values=0)
                else:
                    # Truncate if longer (shouldn't happen, but just in case)
                    padded = features[:target_length]
                padded_features.append(padded)
            else:
                # If no features, create zero array
                padded_features.append(np.zeros(target_length))
        return padded_features
    
    # Pad both datasets
    padded_control = {}
    padded_p600 = {}
    
    for submodule_name in control_features:
        print(f"Padding {submodule_name} features...")
        padded_control[submodule_name] = pad_features(control_features[submodule_name], max_length)
        padded_p600[submodule_name] = pad_features(p600_features[submodule_name], max_length)
    
    # Now analyze the padded features
    feature_changes = {}
    
    for submodule_name in padded_control:
        print(f"Analyzing {submodule_name}...")
        
        # Stack features (now all same shape)
        control_stack = np.stack(padded_control[submodule_name], axis=0)
        p600_stack = np.stack(padded_p600[submodule_name], axis=0)
        
        # Calculate statistics
        control_mean = np.mean(control_stack, axis=0)
        p600_mean = np.mean(p600_stack, axis=0)
        
        # Calculate difference
        mean_diff = p600_mean - control_mean
        
        # Calculate variance
        control_var = np.var(control_stack, axis=0)
        p600_var = np.var(p600_stack, axis=0)
        
        feature_changes[submodule_name] = {
            'control_mean': control_mean,
            'p600_mean': p600_mean,
            'mean_difference': mean_diff,
            'control_variance': control_var,
            'p600_variance': p600_var,
            'control_stack': control_stack,
            'p600_stack': p600_stack
        }
        
        print(f"  {submodule_name}: Control mean {control_mean.shape}, P600 mean {p600_mean.shape}")
    
    return feature_changes

def create_visualizations(feature_changes, output_dir):
    """Create visualizations of feature changes over time"""
    
    print("Creating visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Mean feature differences across submodules
    plt.figure(figsize=(15, 10))
    
    submodules = list(feature_changes.keys())
    mean_diffs = [feature_changes[sub]['mean_difference'].mean() for sub in submodules]
    
    plt.subplot(2, 2, 1)
    plt.bar(range(len(submodules)), mean_diffs)
    plt.xlabel('Submodule')
    plt.ylabel('Mean Feature Difference (P600 - Control)')
    plt.title('Average Feature Changes Across Submodules')
    plt.xticks(range(len(submodules)), submodules, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Feature-by-feature comparison for a specific submodule
    plt.subplot(2, 2, 2)
    # Use the first submodule as example
    example_sub = submodules[0]
    control_mean = feature_changes[example_sub]['control_mean']
    p600_mean = feature_changes[example_sub]['p600_mean']
    
    plt.plot(range(len(control_mean)), control_mean, 'b-', label='Control', alpha=0.7)
    plt.plot(range(len(p600_mean)), p600_mean, 'r-', label='P600', alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Activation')
    plt.title(f'Feature Activations: {example_sub}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Variance comparison
    plt.subplot(2, 2, 3)
    control_vars = [feature_changes[sub]['control_variance'].mean() for sub in submodules]
    p600_vars = [feature_changes[sub]['p600_variance'].mean() for sub in submodules]
    
    x = np.arange(len(submodules))
    width = 0.35
    
    plt.bar(x - width/2, control_vars, width, label='Control', alpha=0.7)
    plt.bar(x + width/2, p600_vars, width, label='P600', alpha=0.7)
    plt.xlabel('Submodule')
    plt.ylabel('Mean Feature Variance')
    plt.title('Feature Variance Comparison')
    plt.xticks(x, submodules, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Heatmap of feature differences
    plt.subplot(2, 2, 4)
    # Create a matrix of feature differences across submodules
    diff_matrix = np.array([feature_changes[sub]['mean_difference'] for sub in submodules])
    
    plt.imshow(diff_matrix, cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='Feature Difference (P600 - Control)')
    plt.xlabel('Feature Index')
    plt.ylabel('Submodule')
    plt.title('Feature Differences Heatmap')
    plt.yticks(range(len(submodules)), submodules)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualizations saved to {output_dir}")
    
    # Save detailed results to CSV
    results_data = []
    for submodule in submodules:
        control_mean = feature_changes[submodule]['control_mean']
        p600_mean = feature_changes[submodule]['p600_mean']
        mean_diff = feature_changes[submodule]['mean_difference']
        
        for i in range(len(control_mean)):
            results_data.append({
                'submodule': submodule,
                'feature_index': i,
                'control_mean': control_mean[i],
                'p600_mean': p600_mean[i],
                'difference': mean_diff[i],
                'abs_difference': abs(mean_diff[i])
            })
    
    results_df = pd.DataFrame(results_data)
    results_csv_path = os.path.join(output_dir, 'feature_analysis_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"✓ Detailed results saved to {results_csv_path}")
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print(f"Total features analyzed: {len(results_df)}")
    print(f"Submodules: {len(submodules)}")
    print(f"Features per submodule: {len(results_df) // len(submodules)}")
    
    # Find most changed features
    top_changes = results_df.nlargest(10, 'abs_difference')
    print(f"\n🔍 Top 10 Most Changed Features:")
    for _, row in top_changes.iterrows():
        print(f"  {row['submodule']} feature {row['feature_index']}: {row['difference']:.4f}")
    
    return results_df

def main(cfg: DictConfig):
    """Main function to analyze P600 sentences using SAEs"""
    
    print("Starting P600 SAE analysis with Neuronpedia API...")
    
    # Check for API key
    api_key = os.environ.get("NEURONPEDIA_API_KEY")
    if not api_key:
        print("❌ NEURONPEDIA_API_KEY environment variable not set")
        print("Please set it: export NEURONPEDIA_API_KEY=your_key_here")
        return
    
    # Initialize Neuronpedia API
    api = NeuronpediaAPI(api_key=api_key)
    print("✓ Neuronpedia API initialized")
    
    # Model configuration
    model_name = "gpt2-small"  # Model available on Neuronpedia
    print(f"Using model: {model_name}")
    
    # Define which submodules to analyze (reduced to keep API calls under 100)
    # Total: 2 datasets × 8 submodules = 16 operations × ~25 sentences = ~400 API calls
    all_submodule_names = [
        'embed',           # Input embeddings
        'attn_0',          # First attention layer
        'mlp_0',           # First MLP layer
        'resid_0',         # First residual connection
        'attn_6',          # Middle attention layer
        'mlp_6',           # Middle MLP layer
        'resid_6',         # Middle residual connection
        'resid_11'         # Final layer (last residual)
    ]
    
    print(f"Analyzing {len(all_submodule_names)} submodules (GPT-2 model)...")
    
    # Load sentence data
    print("Loading sentence data...")
    control_df = pd.read_csv("../grammar_analysis/control_gardenpath_sample.csv")
    p600_df = pd.read_csv("../grammar_analysis/p600_sample.csv")
    
    print(f"Loaded {len(control_df)} control sentences")
    print(f"Loaded {len(p600_df)} P600 sentences")
    
    # Extract SAE features using Neuronpedia API
    print("Extracting SAE features...")
    
    # Initialize progress tracking
    total_operations = len(all_submodule_names) * 2  # 2 for control and P600
    completed_operations = 0
    start_time = time.time()
    
    print(f"Total operations: {total_operations} (2 datasets × {len(all_submodule_names)} submodules)")
    print("=" * 60)
    
    # Extract features for control sentences
    print(f"\n📊 Processing CONTROL sentences ({len(control_df)} sentences)...")
    control_features = {}
    
    for i, submodule_name in enumerate(all_submodule_names):
        print(f"\n🔍 Submodule {i+1}/{len(all_submodule_names)}: {submodule_name}")
        features_list = []
        
        for prompt in tqdm(control_df['sentence'], desc=f"  {submodule_name}", unit="sent"):
            if submodule_name == 'embed':
                layer = 0
            else:
                # Parse submodule name to get layer number
                parts = submodule_name.split('_')
                if len(parts) == 2:
                    layer = int(parts[1])
                else:
                    layer = 0
            
            features = api.get_sae_features(model_name, prompt, layer, submodule_name)
            if features is not None:
                features_list.append(features)
            else:
                print(f"    ⚠️ Failed to get features for: {prompt[:50]}...")
        
        if features_list:
            control_features[submodule_name] = features_list
            print(f"    ✅ {submodule_name}: {len(features_list)} features extracted")
        else:
            print(f"    ❌ {submodule_name}: No features extracted")
        
        # Update overall progress
        completed_operations += 1
        elapsed_time = time.time() - start_time
        avg_time_per_op = elapsed_time / completed_operations
        remaining_ops = total_operations - completed_operations
        estimated_remaining = remaining_ops * avg_time_per_op
        
        print(f"    📈 Progress: {completed_operations}/{total_operations} ({completed_operations/total_operations*100:.1f}%)")
        print(f"    ⏱️  Elapsed: {elapsed_time/60:.1f} min, Est. remaining: {estimated_remaining/60:.1f} min")
    
    # Extract features for P600 sentences
    print(f"\n📊 Processing P600 sentences ({len(p600_df)} sentences)...")
    p600_features = {}
    
    for i, submodule_name in enumerate(all_submodule_names):
        print(f"\n🔍 Submodule {i+1}/{len(all_submodule_names)}: {submodule_name}")
        features_list = []
        
        for prompt in tqdm(p600_df['sentence'], desc=f"  {submodule_name}", unit="sent"):
            if submodule_name == 'embed':
                layer = 0
            else:
                # Parse submodule name to get layer number
                parts = submodule_name.split('_')
                if len(parts) == 2:
                    layer = int(parts[1])
                else:
                    layer = 0
            
            features = api.get_sae_features(model_name, prompt, layer, submodule_name)
            if features is not None:
                features_list.append(features)
            else:
                print(f"    ⚠️ Failed to get features for: {prompt[:50]}...")
        
        if features_list:
            p600_features[submodule_name] = features_list
            print(f"    ✅ {submodule_name}: {len(features_list)} features extracted")
        else:
            print(f"    ❌ {submodule_name}: No features extracted")
        
        # Update overall progress
        completed_operations += 1
        elapsed_time = time.time() - start_time
        avg_time_per_op = elapsed_time / completed_operations
        remaining_ops = total_operations - completed_operations
        estimated_remaining = remaining_ops * avg_time_per_op
        
        print(f"    📈 Progress: {completed_operations}/{total_operations} ({completed_operations/total_operations*100:.1f}%)")
        print(f"    ⏱️  Elapsed: {elapsed_time/60:.1f} min, Est. remaining: {estimated_remaining/60:.1f} min")
    
    print("\n" + "=" * 60)
    total_time = time.time() - start_time
    print(f"🎉 Feature extraction complete in {total_time/60:.1f} minutes!")
    print(f"📊 Control features: {len(control_features)} submodules")
    print(f"📊 P600 features: {len(p600_features)} submodules")
    
    # Analyze feature changes
    print("Analyzing feature changes...")
    feature_changes = analyze_feature_changes(control_features, p600_features)
    
    # Create visualizations
    print("Creating visualizations...")
    output_dir = "src/incremental_analysis/results"
    create_visualizations(feature_changes, output_dir)
    
    # Save results
    print("Saving results...")
    results = {
        'control_features': {k: [f for f in v] for k, v in control_features.items()},
        'p600_features': {k: [f for f in v] for k, v in p600_features.items()},
        'feature_changes': {k: v for k, v in feature_changes.items()}
    }
    
    # Save as numpy arrays
    np.savez(f'{output_dir}/sae_analysis_results.npz', **results)
    
    print(f"Analysis complete! Results saved to {output_dir}/")
    print("Generated visualizations:")
    print(f"  - {output_dir}/feature_analysis.png")
    print(f"  - {output_dir}/feature_analysis_results.csv")

if __name__ == "__main__":
    main(None)  # For now, run without Hydra config
