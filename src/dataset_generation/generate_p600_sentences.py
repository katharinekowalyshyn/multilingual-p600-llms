"""
P600 Sentence Generation Script

This script generates P600 sentences in multiple languages using Large Language Models (LLMs) via DSPy.
P600 sentences contain grammatical errors that trigger P600 ERP responses in brain studies.

Key Features:
- Generates P600 sentences in multiple languages (English, Español, Deutsch)
- Creates sentences with specific grammatical violations (subject-verb agreement, tense errors, etc.)
- Uses DSPy to ensure unique sentence generation and avoid repetition
- Ensures equal distribution of different error types for each language
- Integrates with Hydra configuration system for easy customization
- Progress tracking with tqdm for long generation runs

Equal Distribution Feature:
This script ensures that each language gets an equal number of sentences for each P600 error type:
- Morphological errors (e.g., gender agreement, tense consistency)
- Semantic reversal anomalies (e.g., agent-patient role reversals)
- Article-noun mismatches (e.g., case agreement errors)
- Word order violations (e.g., structural grammar errors)

For example, if you request 10 sentences per language, you'll get:
- 2-3 sentences of each error type per language
- Any remainder sentences are distributed evenly across the first few error types
- This ensures balanced datasets for linguistic research and ERP studies

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
- Equal distribution of error types for each language

Created: 2025-08-21
Updated: 2025-08-26
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

P600_ERROR_TYPES = [
    "morphological_error",
    "semantic_reversal_anomaly", 
    "article_noun_mismatch",
    "word_order_violation"
]

def generate_p600_sentences_for_error_type(language, error_type, llm, cfg, n_per_type):
    """
    Generate P600 sentences for a specific language and error type.
    
    Args:
        language (str): Language to generate sentences in
        error_type (str): Specific type of grammatical error to generate
        llm: The configured LLM
        cfg: Configuration object
        n_per_type (int): Number of sentences to generate for this error type
    
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

    error_type_prompt = f"""Generate {n_per_type} unique P600 sentences in {language} that specifically contain a {error_type.replace('_', ' ')}.

Focus on creating sentences with this specific grammatical error:
- {error_type.replace('_', ' ').title()}

The sentences should:
- Be grammatically incorrect in a way that native speakers would immediately detect
- Sound natural enough that the error isn't obvious until the reader processes the full sentence
- Be appropriate for linguistic research and ERP studies
- Be written in {language}
- Each sentence must be unique and different from the others

Generate exactly {n_per_type} sentences, one per line."""

    class P600ErrorTypeSignature(dspy.Signature):
        """Generate unique P600 sentences with a specific grammatical error type."""
        instruction: str = dspy.InputField()
        sentences: str = dspy.OutputField(desc=f"Exactly {n_per_type} unique P600 sentences with {error_type.replace('_', ' ')} errors, one per line")

    p600_predict = dspy.Predict(P600ErrorTypeSignature)
    unique_sentences = set()
    max_attempts = n_per_type * 3  # Allow some retries
    attempts = 0
    
    sentences_data = []
    
    with tqdm(total=n_per_type, desc=f"{language} {error_type}", unit="sent") as pbar:
        while len(unique_sentences) < n_per_type and attempts < max_attempts:
            try:
                result = p600_predict(instruction=error_type_prompt)
                sentences_text = result.sentences.strip()
                sentences = [s.strip() for s in sentences_text.split('\n') if s.strip()]
                
                for sentence in sentences:
                    if sentence and sentence not in unique_sentences and len(unique_sentences) < n_per_type:
                        unique_sentences.add(sentence)
                        sentences_data.append({
                            "language": language,
                            "sentence": sentence,
                            "error_type": error_type.replace('_', ' ').title()
                        })
                        pbar.update(1)
                
                attempts += 1
                
            except Exception as e:
                print(f"Error generating {error_type} sentences for {language}: {e}")
                attempts += 1
                continue
    
    if len(unique_sentences) < n_per_type:
        print(f"Warning: Only generated {len(unique_sentences)} unique {error_type} sentences for {language} after {attempts} attempts.")
    
    return sentences_data

def generate_p600_sentences(language, llm, cfg, n):
    """
    Generate P600 sentences for a specific language with equal distribution of error types.
    
    Args:
        language (str): Language to generate sentences in
        llm: The configured LLM
        cfg: Configuration object
        n (int): Total number of sentences to generate
    
    Returns:
        list: List of dictionaries with sentence and error_type information
    """
    n_per_type = n // len(P600_ERROR_TYPES)
    remainder = n % len(P600_ERROR_TYPES)
    
    if n_per_type == 0:
        print(f"Warning: {n} sentences is too few for {len(P600_ERROR_TYPES)} error types. Minimum should be {len(P600_ERROR_TYPES)}.")
        n_per_type = 1
        remainder = 0
    
    all_sentences = []
    
    for i, error_type in enumerate(P600_ERROR_TYPES):
        current_n = n_per_type + (1 if i < remainder else 0)
        if current_n > 0:
            print(f"  Generating {current_n} {error_type.replace('_', ' ')} sentences...")
            try:
                sentences = generate_p600_sentences_for_error_type(
                    language, 
                    error_type, 
                    llm, 
                    cfg, 
                    current_n
                )
                all_sentences.extend(sentences)
            except Exception as e:
                print(f"Error generating {error_type} sentences for {language}: {e}")
                continue
    
    return all_sentences

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

CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "conf")

@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
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
    print(f"Error types: {', '.join(P600_ERROR_TYPES)}")
    print(f"Sentences per error type per language: {cfg.p600_generation.num_sentences_per_language // len(P600_ERROR_TYPES)}")
    
    try:
        all_sentences = generate_multilingual_p600_dataset(cfg)
        df = pd.DataFrame(all_sentences)
        output_path = os.path.join(SCRIPT_DIR, cfg.p600_generation.output_file)
        save_dataframe_to_csv(df, output_path)
        
        print(f"\n{'='*50}")
        print("P600 sentence generation completed successfully!")
        print(f"Total sentences generated: {len(all_sentences)}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*50}")
        
        for language in cfg.p600_generation.languages:
            lang_sentences = df[df['language'] == language]
            print(f"{language}: {len(lang_sentences)} sentences")
        
        print(f"\nError types generated:")
        error_counts = df['error_type'].value_counts()
        for error_type, count in error_counts.items():
            print(f"  {error_type}: {count}")
            
        print(f"\nDistribution by language and error type:")
        for language in cfg.p600_generation.languages:
            print(f"\n{language}:")
            lang_df = df[df['language'] == language]
            expected_per_type = cfg.p600_generation.num_sentences_per_language // len(P600_ERROR_TYPES)
            for error_type in P600_ERROR_TYPES:
                error_type_count = len(lang_df[lang_df['error_type'] == error_type.replace('_', ' ').title()])
                expected = expected_per_type + (1 if P600_ERROR_TYPES.index(error_type) < cfg.p600_generation.num_sentences_per_language % len(P600_ERROR_TYPES) else 0)
                status = "✓" if error_type_count == expected else f"✗ (expected {expected})"
                print(f"  {error_type.replace('_', ' ').title()}: {error_type_count} {status}")
            
    except Exception as e:
        print(f"Error during P600 generation: {e}")
        raise

if __name__ == "__main__":
    main() 