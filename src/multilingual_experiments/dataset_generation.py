"""
Multilingual Garden-Path Sentence Generation

Generates garden-path sentences across multiple languages.
Also generates matched unambiguous control sentences.

All generation uses Ollama API with language-specific prompts.
"""

import os
import sys
import hydra
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from pathlib import Path
import json
import re

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.multilingual_experiments.model_loader import ModelLoader
from src.utils import save_dataframe_to_csv


def load_prompt_template(language: str, prompt_type: str = "gardenpath") -> str:
    """
    Load prompt template for a specific language and prompt type.
    
    Args:
        language: Language name (e.g., "Chinese", "Spanish")
        prompt_type: Either "gardenpath" or "control"
        
    Returns:
        Prompt template string
    """
    prompt_file_name = language.lower()
    prompt_file = f"{prompt_type}_prompt_{prompt_file_name}.txt"
    prompt_path = Path(SRC_DIR) / "prompts" / prompt_file
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    return template


def generate_gardenpath_sentence(language: str, loader: ModelLoader) -> Dict:
    """
    Generate a garden-path sentence for a specific language using Ollama.
    
    Args:
        language: Target language
        loader: ModelLoader instance (Ollama)
        
    Returns:
        Dictionary with sentence information
    """
    # Load language-specific prompt template
    prompt_template = load_prompt_template(language, "gardenpath")
    
    # Fill in template placeholders (only language if needed)
    prompt = prompt_template.format(language=language) if "{language}" in prompt_template else prompt_template

    try:
        response = loader.generate(prompt, options={"temperature": 0.7, "num_predict": 100})
        sentence = response.strip()
        
        # Clean up the response
        # Remove quotes if present
        if sentence.startswith('"') and sentence.endswith('"'):
            sentence = sentence[1:-1]
        if sentence.startswith("'") and sentence.endswith("'"):
            sentence = sentence[1:-1]
        
        # Remove any explanatory text (look for common patterns)
        # If response contains "Sentence:" or similar, extract just the sentence
        if "Sentence:" in sentence or "sentence:" in sentence:
            parts = re.split(r'[Ss]entence:\s*', sentence)
            if len(parts) > 1:
                sentence = parts[-1].strip()
        
        # Remove markdown formatting
        sentence = sentence.replace('```', '').strip()
        
        # Take first sentence if multiple sentences
        sentence = sentence.split('.')[0] + '.' if '.' in sentence else sentence
        sentence = sentence.split('!')[0] + '!' if '!' in sentence else sentence
        sentence = sentence.split('?')[0] + '?' if '?' in sentence else sentence
        
        return {
            "language": language,
            "sentence": sentence,
            "sentence_type": "gardenpath"
        }
    except Exception as e:
        print(f"Error generating sentence for {language}: {e}")
        return None


def generate_control_sentence(language: str, gardenpath_sentence: str, loader: ModelLoader) -> Dict:
    """
    Generate an unambiguous control sentence matched to a garden-path sentence using Ollama.
    
    Args:
        language: Target language
        gardenpath_sentence: The garden-path sentence to match
        loader: ModelLoader instance (Ollama)
        
    Returns:
        Dictionary with control sentence information
    """
    # Load language-specific prompt template
    prompt_template = load_prompt_template(language, "control")
    
    # Fill in template placeholders
    prompt = prompt_template.format(
        language=language,
        gardenpath_sentence=gardenpath_sentence
    )

    try:
        response = loader.generate(prompt, options={"temperature": 0.5, "num_predict": 100})
        sentence = response.strip()
        
        # Clean up the response
        if sentence.startswith('"') and sentence.endswith('"'):
            sentence = sentence[1:-1]
        if sentence.startswith("'") and sentence.endswith("'"):
            sentence = sentence[1:-1]
        
        if "Control:" in sentence or "control:" in sentence:
            parts = re.split(r'[Cc]ontrol:\s*', sentence)
            if len(parts) > 1:
                sentence = parts[-1].strip()
        
        sentence = sentence.replace('```', '').strip()
        sentence = sentence.split('.')[0] + '.' if '.' in sentence else sentence
        
        return {
            "language": language,
            "sentence": sentence,
            "sentence_type": "control",
            "matched_gardenpath": gardenpath_sentence
        }
    except Exception as e:
        print(f"Error generating control for {language}: {e}")
        return None


def generate_multilingual_dataset(cfg: DictConfig) -> pd.DataFrame:
    """
    Generate garden-path and control sentences for all configured languages using Ollama.
    
    Args:
        cfg: Configuration object with multilingual_p600 settings
        
    Returns:
        DataFrame with all generated sentences
    """
    mp600_cfg = cfg.multilingual_p600
    
    # Collect all languages
    all_languages = []
    all_languages.extend(mp600_cfg.languages.high_resource)
    all_languages.extend(mp600_cfg.languages.mid_resource)
    all_languages.extend(mp600_cfg.languages.low_resource)
    
    # Get configuration
    num_gardenpath = mp600_cfg.dataset.num_gardenpath_sentences_per_language
    num_control = mp600_cfg.dataset.num_control_sentences_per_language
    
    # Get a model for generation (use first enabled model)
    enabled_models = [
        (name, model_cfg) 
        for name, model_cfg in mp600_cfg.models.items()
        if model_cfg.get('enabled', False)
    ]
    
    if not enabled_models:
        raise ValueError("No models enabled in config. Enable at least one model for generation.")
    
    # Use first enabled model for generation
    model_name, model_cfg = enabled_models[0]
    ollama_base_url = mp600_cfg.get('ollama_base_url', 'http://localhost:11434')
    
    loader = ModelLoader(
        model_name=model_cfg['model_name'],
        ollama_base_url=ollama_base_url
    )
    loader.load()
    
    print(f"Using model: {model_name} ({model_cfg['model_name']}) for generation")
    print(f"Generating multilingual garden-path dataset...")
    print(f"Languages: {len(all_languages)}")
    print(f"Garden-path sentences per language: {num_gardenpath}")
    print(f"Control sentences per language: {num_control}")
    print("=" * 60)
    
    all_sentences = []
    
    for language in tqdm(all_languages, desc="Languages"):
        print(f"\nProcessing {language}...")
        
        # Generate garden-path sentences
        gardenpath_sentences = []
        print(f"  Generating {num_gardenpath} garden-path sentences...")
        for _ in range(num_gardenpath):
            sentence_data = generate_gardenpath_sentence(
                language, loader
            )
            if sentence_data:
                gardenpath_sentences.append(sentence_data)
                all_sentences.append(sentence_data)
        
        # Generate control sentences (matched to garden-path sentences)
        print(f"  Generating {num_control} control sentences...")
        for gardenpath_data in gardenpath_sentences[:num_control]:
            control_data = generate_control_sentence(
                language,
                gardenpath_data["sentence"],
                loader
            )
            if control_data:
                all_sentences.append(control_data)
    
    df = pd.DataFrame(all_sentences)
    return df


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for multilingual dataset generation."""
    if not cfg.multilingual_p600.enabled:
        print("Multilingual P600 extension is disabled in config.")
        return
    
    # Generate dataset
    try:
        df = generate_multilingual_dataset(cfg)
        
        # Save to file
        output_dir = Path(hydra.utils.to_absolute_path(cfg.multilingual_p600.dataset.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "multilingual_gardenpath_dataset.csv"
        
        save_dataframe_to_csv(df, str(output_path))
        
        print(f"\n{'='*60}")
        print("Dataset generation complete!")
        print(f"Total sentences: {len(df)}")
        print(f"Saved to: {output_path}")
        print(f"\nBreakdown by language:")
        for lang in df['language'].unique():
            lang_df = df[df['language'] == lang]
            gardenpath_count = len(lang_df[lang_df['sentence_type'] == 'gardenpath'])
            control_count = len(lang_df[lang_df['sentence_type'] == 'control'])
            print(f"  {lang}: {len(lang_df)} sentences ({gardenpath_count} gardenpath, {control_count} control)")
        print(f"{'='*60}")
            
    except Exception as e:
        print(f"Error during dataset generation: {e}")
        raise


if __name__ == "__main__":
    main()

