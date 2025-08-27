"""
Multilingual Garden Path Sentence Generator

This script generates garden path sentences in multiple languages using Large Language Models (LLMs) via DSPy.
Each sentence is entirely in one language with no code-switching, designed to cause temporary parsing difficulties.

Key Features:
- Generates garden path sentences in multiple languages (English, Español, Deutsch)
- Each sentence is entirely in one language (no mixing)
- Uses DSPy to ensure unique sentence generation and avoid repetition
- Integrates with Hydra configuration system for easy customization

Garden Path Sentences:
These are sentences that initially lead readers to an incorrect interpretation
due to ambiguous structure, requiring re-parsing to understand correctly.
Examples: "The horse raced past the barn fell" (initially seems like "The horse raced past the barn" but "fell" requires re-parsing).

Configuration:
- Set languages in config.yaml: gardenpath_generation.languages
- Configure number of sentences: gardenpath_generation.num_sentences
- Set LLM model: llm.model
- Define prompt files: prompt_files for each language

Usage:
1. Set environment variable: export OPENAI_API_KEY=your_key
2. Configure settings in src/conf/config.yaml
3. Run: python src/dataset_generation/generate_multilingual_gardenpaths.py

Output:
- CSV file with columns: language, sentence
- Each sentence is entirely in one language with garden path ambiguity
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

def generate_garden_path_sentences(language, llm, cfg, n):
    if language not in cfg.prompt_files:
        raise ValueError(f"No prompt file specified for language '{language}'. Please add it to the config under 'prompt_files'.")
    
    prompt_file = os.path.join(SRC_DIR, cfg.prompt_files[language])
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        raise ValueError(f"Prompt file not found for {language}: {prompt_file}")

    class GardenPathSignature(dspy.Signature):
        """Generate a unique garden path sentence in a single language."""
        instruction: str = dspy.InputField()
        previous_sentences: str = dspy.InputField(desc="A list of sentences to avoid repeating.")
        sentence: str = dspy.OutputField(desc="A single, unique garden path sentence entirely in the specified language.")

    garden_path_predict = dspy.Predict(GardenPathSignature)
    unique_sentences = set()
    max_attempts = n * 5  # avoid infinite loops if LLM repeats
    attempts = 0
    
    sentences_data = []
    
    with tqdm(total=n, desc=f"{language} garden path sentences", unit="sent") as pbar:
        while len(unique_sentences) < n and attempts < max_attempts:
            avoid = "\n".join(unique_sentences) if unique_sentences else ""
            prompt = prompt_template.format(language=language)
            
            try:
                result = garden_path_predict(instruction=prompt, previous_sentences=avoid)
                sentence = result.sentence.strip()
                
                if sentence and sentence not in unique_sentences:
                    unique_sentences.add(sentence)
                    sentences_data.append({
                        "language": language,
                        "sentence": sentence
                    })
                    pbar.update(1)
                attempts += 1
                
            except Exception as e:
                print(f"Error generating sentence for {language}: {e}")
                attempts += 1
                continue
    
    if len(unique_sentences) < n:
        print(f"Warning: Only generated {len(unique_sentences)} unique garden path sentences for {language} after {attempts} attempts.")
    
    return sentences_data

def generate_multilingual_gardenpath_dataset(cfg):
    all_sentences = []
    
    for language in cfg.gardenpath_generation.languages:
        print(f"\nGenerating garden path sentences for {language}...")
        sentences = generate_garden_path_sentences(
            language, 
            None,  
            cfg, 
            cfg.gardenpath_generation.num_sentences
        )
        all_sentences.extend(sentences)
    
    return all_sentences

CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "conf")

@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    if not cfg.gardenpath_generation.enabled:
        print("Garden path generation is disabled in config. Set 'enabled: true' to run.")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your bash profile or session.")
    
    lm = dspy.LM(f"openai/{cfg.llm.model}", api_key=api_key)
    dspy.configure(lm=lm)
    
    print("Starting multilingual garden path sentence generation...")
    print(f"Using LLM: {cfg.llm.model}")
    print(f"Languages: {', '.join(cfg.gardenpath_generation.languages)}")
    print(f"Sentences per language: {cfg.gardenpath_generation.num_sentences}")
    
    try:
        all_sentences = generate_multilingual_gardenpath_dataset(cfg)
        df = pd.DataFrame(all_sentences)
        output_path = os.path.join(SCRIPT_DIR, cfg.gardenpath_generation.output_file)
        save_dataframe_to_csv(df, output_path)
        
        print(f"\n{'='*50}")
        print("Multilingual garden path generation completed successfully!")
        print(f"Total sentences generated: {len(all_sentences)}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*50}")
        
        for language in cfg.gardenpath_generation.languages:
            lang_sentences = df[df['language'] == language]
            print(f"{language}: {len(lang_sentences)} sentences")
            
    except Exception as e:
        print(f"Error during garden path generation: {e}")
        raise

if __name__ == "__main__":
    main() 