"""
P600 Sentence Generation Script

This script generates P600 sentences in multiple languages using Large Language Models (LLMs) via DSPy.
P600 sentences contain grammatical errors that trigger P600 ERP responses in brain studies.

Key Features:
- Generates P600 sentences in multiple languages (English, Español, Deutsch)
- Creates sentences with specific grammatical violations (subject-verb agreement, tense errors, etc.)
- Uses DSPy to ensure unique sentence generation and avoid repetition
- Integrates with Hydra configuration system for easy customization
- Progress tracking with tqdm for long generation runs

P600 Sentences:
These are sentences containing grammatical errors that native speakers immediately detect,
triggering P600 ERP responses in brain studies. Examples include:
- Subject-verb agreement: "The dogs barks loudly"
- Tense inconsistencies: "Yesterday he walks to the store"
- Number agreement: "The group of boys are playing"
- Article-noun mismatches: "Der Mann sah den Frau"

Configuration:
- Set languages in config.yaml: languages list
- Configure number of sentences: p600_generation.num_sentences_per_language
- Set LLM model: llm.model
- Define prompt file: p600_generation.prompt_file
- Set output file: p600_generation.output_file

Usage:
1. Set environment variable: export OPENAI_API_KEY=your_key
2. Configure settings in src/conf/config.yaml
3. Run: python src/dataset_generation/generate_p600_sentences.py

Output:
- CSV file with columns: language, sentence, error_type
- Each sentence contains a grammatical error appropriate for P600 studies
- Sentences are unique within each language and across runs

Created: 2025-08-21
"""

import hydra
from omegaconf import DictConfig
import dspy
import pandas as pd
import os
import sys
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm
import re
from src.utils import save_dataframe_to_csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

def generate_p600_sentences(language, llm, cfg, n):
    """
    Generate P600 sentences for a specific language.
    
    Args:
        language (str): Language to generate sentences in
        llm: The configured LLM
        cfg: Configuration object
        n (int): Number of sentences to generate
    
    Returns:
        list: List of dictionaries with sentence and error_type information
    """
    if not hasattr(cfg.p600_generation, 'prompt_file'):
        raise ValueError("No P600 generation prompt file specified in config.")
    
    prompt_file = os.path.join(SRC_DIR, cfg.p600_generation.prompt_file)
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        raise ValueError(f"P600 generation prompt file not found: {prompt_file}")

    # Use dspy Predict module to enforce unique P600 sentence generation
    class P600Signature(dspy.Signature):
        """Generate a unique P600 sentence with grammatical error."""
        instruction: str = dspy.InputField()
        previous_sentences: str = dspy.InputField(desc="A list of sentences to avoid repeating.")
        sentence: str = dspy.OutputField(desc="A single, unique P600 sentence with a grammatical error.")
        error_type: str = dspy.OutputField(desc="Brief description of the grammatical error (e.g., 'subject-verb agreement', 'tense error')")

    p600_predict = dspy.Predict(P600Signature)
    unique_sentences = set()
    max_attempts = n * 5  # avoid infinite loops if LLM repeats
    attempts = 0
    
    sentences_data = []
    
    with tqdm(total=n, desc=f"{language} P600 sentences", unit="sent") as pbar:
        while len(unique_sentences) < n and attempts < max_attempts:
            avoid = "\n".join(unique_sentences) if unique_sentences else ""
            prompt = prompt_template.format(language=language)
            
            try:
                result = p600_predict(instruction=prompt, previous_sentences=avoid)
                sentence = result.sentence.strip()
                error_type = result.error_type.strip()
                
                if sentence and sentence not in unique_sentences:
                    unique_sentences.add(sentence)
                    sentences_data.append({
                        "language": language,
                        "sentence": sentence,
                        "error_type": error_type
                    })
                    pbar.update(1)
                attempts += 1
                
            except Exception as e:
                print(f"Error generating sentence for {language}: {e}")
                attempts += 1
                continue
    
    if len(unique_sentences) < n:
        print(f"Warning: Only generated {len(unique_sentences)} unique P600 sentences for {language} after {attempts} attempts.")
    
    return sentences_data

def generate_multilingual_p600_dataset(cfg):
    """
    Generate P600 sentences for all configured languages.
    
    Args:
        cfg: Configuration object
    
    Returns:
        list: List of dictionaries with language, sentence, and error_type
    """
    all_sentences = []
    
    for language in cfg.p600_generation.languages:
        print(f"\nGenerating P600 sentences for {language}...")
        sentences = generate_p600_sentences(
            language, 
            None,  # LLM is configured globally
            cfg, 
            cfg.p600_generation.num_sentences_per_language
        )
        all_sentences.extend(sentences)
    
    return all_sentences

# Dynamically determine the config path relative to this script
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "conf")

@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to generate P600 sentences for multiple languages.
    """
    # Check if P600 generation is enabled
    if not cfg.p600_generation.enabled:
        print("P600 generation is disabled in config. Set 'enabled: true' to run.")
        return
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your bash profile or session.")
    
    lm = dspy.LM(f"openai/{cfg.llm.model}", api_key=api_key)
    dspy.configure(lm=lm)
    
    print("Starting P600 sentence generation...")
    print(f"Using LLM: {cfg.llm.model}")
    print(f"Languages: {', '.join(cfg.p600_generation.languages)}")
    print(f"Sentences per language: {cfg.p600_generation.num_sentences_per_language}")
    
    try:
        # Generate P600 sentences for all languages
        all_sentences = generate_multilingual_p600_dataset(cfg)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(all_sentences)
        output_path = os.path.join(SCRIPT_DIR, cfg.p600_generation.output_file)
        save_dataframe_to_csv(df, output_path)
        
        print(f"\n{'='*50}")
        print("P600 sentence generation completed successfully!")
        print(f"Total sentences generated: {len(all_sentences)}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*50}")
        
        # Print summary by language
        for language in cfg.p600_generation.languages:
            lang_sentences = df[df['language'] == language]
            print(f"{language}: {len(lang_sentences)} sentences")
        
        # Print summary by error type
        print(f"\nError types generated:")
        error_counts = df['error_type'].value_counts()
        for error_type, count in error_counts.items():
            print(f"  {error_type}: {count}")
            
    except Exception as e:
        print(f"Error during P600 generation: {e}")
        raise

if __name__ == "__main__":
    main() 