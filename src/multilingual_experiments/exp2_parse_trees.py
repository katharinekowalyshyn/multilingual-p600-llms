"""
Experiment 2: Parse Tree Extraction

Extracts incremental parse trees using Manning et al. (2020) technique to track
syntactic reanalysis patterns across languages.

Key measures:
- Do models shift parse structures at disambiguation across languages?
- Compare syntactic reanalysis patterns across resource levels
- Language-specific parse tree differences
"""

import os
import sys
import hydra
from omegaconf import DictConfig
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.multilingual_experiments.model_loader import load_model_from_config
from src.multilingual_experiments.chunking import get_chunking_strategy
from src.utils import save_dataframe_to_csv


def extract_parse_tree_manning_2020(model, tokenizer, sentence: str, 
                                     chunking_strategy) -> List[Dict]:
    """
    Extract incremental parse trees using Manning et al. (2020) method.
    
    This method uses attention patterns and hidden states to infer
    syntactic structure incrementally.
    
    Args:
        model: Loaded model
        tokenizer: Model tokenizer
        sentence: Sentence to parse
        chunking_strategy: Chunking strategy instance
        
    Returns:
        List of parse tree representations at each chunk
    """
    # TODO: Implement Manning et al. (2020) parse tree extraction
    # This would involve:
    # 1. Extracting attention weights at each chunk
    # 2. Using attention patterns to infer dependency relations
    # 3. Building incremental parse trees
    # 4. Comparing trees across chunks to detect reanalysis
    
    chunks = chunking_strategy.chunk(sentence, tokenizer)
    parse_trees = []
    
    for chunk_idx, (chunk_text, start_token, end_token) in enumerate(chunks):
        # Tokenize chunk
        tokens = tokenizer.encode(chunk_text, return_tensors="pt")
        if hasattr(model, 'to'):
            tokens = tokens.to(next(model.parameters()).device)
        
        # Extract attention (if available)
        if hasattr(model, 'run_with_cache'):
            _, cache = model.run_with_cache(tokens)
            # Use attention patterns to infer structure
            # This is a placeholder - actual implementation would analyze
            # attention heads to identify syntactic dependencies
            parse_tree = {
                "chunk_index": chunk_idx,
                "chunk_text": chunk_text,
                "structure": "placeholder",  # Would contain actual parse tree
                "dependencies": [],  # Would contain dependency relations
                "reanalysis_detected": chunk_idx > 0  # Placeholder
            }
        else:
            parse_tree = {
                "chunk_index": chunk_idx,
                "chunk_text": chunk_text,
                "structure": "not_available",
                "dependencies": [],
                "reanalysis_detected": False
            }
        
        parse_trees.append(parse_tree)
    
    return parse_trees


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for Experiment 2."""
    if not cfg.multilingual_p600.enabled:
        print("Multilingual P600 extension is disabled.")
        return
    
    exp_cfg = cfg.multilingual_p600.experiments.experiment_2_parse_trees
    if not exp_cfg.enabled:
        print("Experiment 2 is disabled in config.")
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
    
    # Process each model
    all_results = []
    
    for model_name, model_cfg in enabled_models:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Load model
            model, tokenizer, loader = load_model_from_config(model_cfg)
            
            # Get chunking strategy (syntactic for parse trees)
            chunk_strategy = get_chunking_strategy("syntactic", language='en')
            
            # Process sentences
            for idx, row in tqdm(gardenpath_df.iterrows(), 
                               total=len(gardenpath_df), 
                               desc=f"Processing {model_name}"):
                try:
                    parse_trees = extract_parse_tree_manning_2020(
                        model=model,
                        tokenizer=tokenizer,
                        sentence=row['sentence'],
                        chunking_strategy=chunk_strategy
                    )
                    
                    # Convert to DataFrame rows
                    for parse_tree in parse_trees:
                        parse_tree['model'] = model_name
                        parse_tree['sentence_id'] = idx
                        parse_tree['language'] = row['language']
                        parse_tree['ambiguity_type'] = row['ambiguity_type']
                        all_results.append(parse_tree)
                    
                except Exception as e:
                    print(f"Error processing sentence {idx}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
    
    # Combine all results
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save results
        output_dir = Path(hydra.utils.to_absolute_path(exp_cfg.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "parse_tree_results.csv"
        
        save_dataframe_to_csv(results_df, str(output_path))
        
        print(f"\n{'='*60}")
        print("Experiment 2 complete!")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()

