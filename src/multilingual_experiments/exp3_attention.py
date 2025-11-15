"""
Experiment 3: Attention Visualization

Maps which attention heads are sensitive to disambiguation across languages.
Analyzes attention patterns to identify reanalysis mechanisms.

Key measures:
- Are the same heads active across languages or language-specific?
- Resource-level patterns in attention allocation
- Attention shifts at disambiguation points
"""

import os
import sys
import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.multilingual_experiments.model_loader import load_model_from_config
from src.multilingual_experiments.chunking import get_chunking_strategy
from src.utils import save_dataframe_to_csv

sns.set_style("whitegrid")


def extract_attention_weights(model, tokenizer, sentence: str, 
                              chunking_strategy, layers_to_analyze: Optional[List[int]] = None) -> Dict:
    """
    Extract attention weights for a sentence incrementally.
    
    Args:
        model: Loaded model
        tokenizer: Model tokenizer
        sentence: Sentence to analyze
        chunking_strategy: Chunking strategy instance
        layers_to_analyze: Specific layers to extract (None for all)
        
    Returns:
        Dictionary with attention weights at each chunk
    """
    chunks = chunking_strategy.chunk(sentence, tokenizer)
    attention_data = []
    
    for chunk_idx, (chunk_text, start_token, end_token) in enumerate(chunks):
        # Tokenize chunk
        tokens = tokenizer.encode(chunk_text, return_tensors="pt")
        if hasattr(model, 'to'):
            tokens = tokens.to(next(model.parameters()).device)
        
        # Extract attention if available
        if hasattr(model, 'run_with_cache'):
            _, cache = model.run_with_cache(tokens)
            
            # Get attention weights from cache
            attention_weights = {}
            for key in cache.keys():
                if 'attn' in key.lower() or 'attention' in key.lower():
                    # Extract layer number
                    layer_num = None
                    for part in key.split('.'):
                        if part.isdigit():
                            layer_num = int(part)
                            break
                    
                    if layers_to_analyze is None or layer_num in layers_to_analyze:
                        attn = cache[key]
                        # Average across heads if multi-head
                        if attn.dim() > 3:
                            attn = attn.mean(dim=1)  # Average across heads
                        attention_weights[key] = attn.cpu().numpy()
            
            attention_data.append({
                "chunk_index": chunk_idx,
                "chunk_text": chunk_text,
                "attention_weights": attention_weights,
                "num_layers": len(attention_weights)
            })
        else:
            # Model doesn't support attention extraction
            attention_data.append({
                "chunk_index": chunk_idx,
                "chunk_text": chunk_text,
                "attention_weights": {},
                "num_layers": 0
            })
    
    return attention_data


def analyze_attention_shifts(attention_data: List[Dict]) -> pd.DataFrame:
    """
    Analyze attention shifts between chunks to detect reanalysis.
    
    Args:
        attention_data: List of attention data dictionaries
        
    Returns:
        DataFrame with attention shift analysis
    """
    results = []
    
    for i in range(1, len(attention_data)):
        prev_attn = attention_data[i-1]['attention_weights']
        curr_attn = attention_data[i]['attention_weights']
        
        # Compare attention patterns
        for key in prev_attn.keys():
            if key in curr_attn:
                prev = prev_attn[key]
                curr = curr_attn[key]
                
                # Calculate attention shift (e.g., KL divergence or cosine distance)
                if prev.shape == curr.shape:
                    # Flatten for comparison
                    prev_flat = prev.flatten()
                    curr_flat = curr.flatten()
                    
                    # Cosine similarity
                    dot_product = np.dot(prev_flat, curr_flat)
                    norm_prev = np.linalg.norm(prev_flat)
                    norm_curr = np.linalg.norm(curr_flat)
                    cosine_sim = dot_product / (norm_prev * norm_curr + 1e-10)
                    
                    # L2 distance
                    l2_distance = np.linalg.norm(prev_flat - curr_flat)
                    
                    results.append({
                        "chunk_index": i,
                        "layer_key": key,
                        "cosine_similarity": float(cosine_sim),
                        "l2_distance": float(l2_distance),
                        "attention_shift_detected": l2_distance > 0.1  # Threshold
                    })
    
    return pd.DataFrame(results)


def visualize_attention_patterns(attention_data: List[Dict], output_path: Path):
    """
    Create visualization of attention patterns.
    
    Args:
        attention_data: List of attention data dictionaries
        output_path: Path to save visualization
    """
    # Create heatmap of attention across chunks and layers
    # This is a simplified version - could be expanded
    
    fig, axes = plt.subplots(len(attention_data), 1, figsize=(12, 4 * len(attention_data)))
    if len(attention_data) == 1:
        axes = [axes]
    
    for idx, attn_data in enumerate(attention_data):
        if attn_data['num_layers'] > 0:
            # Aggregate attention across layers
            all_attn = []
            for key, attn in attn_data['attention_weights'].items():
                if attn.size > 0:
                    all_attn.append(attn.mean())
            
            if all_attn:
                axes[idx].bar(range(len(all_attn)), all_attn)
                axes[idx].set_title(f"Chunk {idx}: {attn_data['chunk_text'][:50]}...")
                axes[idx].set_xlabel("Layer")
                axes[idx].set_ylabel("Mean Attention")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for Experiment 3."""
    if not cfg.multilingual_p600.enabled:
        print("Multilingual P600 extension is disabled.")
        return
    
    exp_cfg = cfg.multilingual_p600.experiments.experiment_3_attention
    if not exp_cfg.enabled:
        print("Experiment 3 is disabled in config.")
        return
    
    # Load dataset
    dataset_path = Path(hydra.utils.to_absolute_path(
        cfg.multilingual_p600.dataset.output_dir
    )) / "multilingual_gardenpath_dataset.csv"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Run dataset generation first.")
    
    df = pd.read_csv(dataset_path)
    gardenpath_df = df[df['sentence_type'] == 'gardenpath']
    
    print(f"Loaded {len(gardenpath_df)} garden-path sentences")
    
    # Get enabled models
    enabled_models = [
        (name, model_cfg) 
        for name, model_cfg in cfg.multilingual_p600.models.items()
        if model_cfg.get('enabled', False)
    ]
    
    if not enabled_models:
        print("No models enabled in config.")
        return
    
    # Determine layers to analyze
    layers_to_analyze = None
    if exp_cfg.layers_to_analyze != "all":
        layers_to_analyze = exp_cfg.layers_to_analyze
    
    # Process each model
    all_results = []
    
    for model_name, model_cfg in enabled_models:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Load model
            model, tokenizer, loader = load_model_from_config(model_cfg)
            
            # Get chunking strategy
            chunk_strategy = get_chunking_strategy("syntactic", language='en')
            
            # Process sentences
            for idx, row in tqdm(gardenpath_df.iterrows(), 
                               total=len(gardenpath_df), 
                               desc=f"Processing {model_name}"):
                try:
                    attention_data = extract_attention_weights(
                        model=model,
                        tokenizer=tokenizer,
                        sentence=row['sentence'],
                        chunking_strategy=chunk_strategy,
                        layers_to_analyze=layers_to_analyze
                    )
                    
                    # Analyze attention shifts
                    shift_df = analyze_attention_shifts(attention_data)
                    shift_df['model'] = model_name
                    shift_df['sentence_id'] = idx
                    shift_df['language'] = row['language']
                    shift_df['ambiguity_type'] = row['ambiguity_type']
                    
                    all_results.append(shift_df)
                    
                    # Create visualization for first sentence of each language
                    if idx == gardenpath_df[gardenpath_df['language'] == row['language']].index[0]:
                        output_dir = Path(hydra.utils.to_absolute_path(exp_cfg.output_dir))
                        output_dir.mkdir(parents=True, exist_ok=True)
                        viz_path = output_dir / f"attention_{model_name}_{row['language']}_{idx}.png"
                        visualize_attention_patterns(attention_data, viz_path)
                    
                except Exception as e:
                    print(f"Error processing sentence {idx}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save results
        output_dir = Path(hydra.utils.to_absolute_path(exp_cfg.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "attention_analysis_results.csv"
        
        save_dataframe_to_csv(combined_df, str(output_path))
        
        print(f"\n{'='*60}")
        print("Experiment 3 complete!")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()

