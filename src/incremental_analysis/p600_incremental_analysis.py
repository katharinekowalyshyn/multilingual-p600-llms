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
from src.utils import get_neuronpedia_api_key, neuronpedia_fetch_sae_features
import time

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def fetch_sae_features(api_key: str, prompt: str, model_id: str, source: str, index: int,
                      base_url: str = "https://www.neuronpedia.org") -> np.ndarray:
    """Get SAE features from Neuronpedia API.

    Parameters
    - api_key: Bearer token for Neuronpedia
    - prompt: text to evaluate
    - model_id: e.g., "gpt2-small"
    - source: e.g., "9-res-jb"
    - index: layer/source index as expected by Neuronpedia
    - base_url: Neuronpedia host
    """
    
    endpoint = f"{base_url}/api/activation/new"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "feature": {"modelId": model_id, "source": source, "index": str(index)},
        "customText": prompt,
    }
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        # Prefer 'values' if present
        if isinstance(data, dict):
            for key in ("values", "sae_features", "features", "activations"):
                if key in data:
                    arr = np.array(data[key])
                    return arr
    except Exception:
        return None
    return None

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
    try:
        api_key = get_neuronpedia_api_key()
    except RuntimeError as e:
        print(f"❌ {e}")
        return
    
    print("✓ Neuronpedia API configured")
    
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
            
            # Simple mapping: choose a likely source id for GPT-2; users can adjust
            source = "9-res-jb" if layer > 0 else "1-res-jb"
            features = neuronpedia_fetch_sae_features(api_key, prompt, model_name, source, layer)
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
            
            source = "9-res-jb" if layer > 0 else "1-res-jb"
            features = neuronpedia_fetch_sae_features(api_key, prompt, model_name, source, layer)
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
