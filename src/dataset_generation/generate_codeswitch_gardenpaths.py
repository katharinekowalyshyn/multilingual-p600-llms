"""
NOTE: This script is not used in the final dataset generation yet. It could be used later on and I've tested it with the code-switching prompt before.

Code-Switched Garden Path Sentence Generator

This script generates code-switched garden path sentences that combine two languages
within the same sentence. It creates unique, linguistically challenging sentences
that mix languages while maintaining garden path ambiguity.

Key Features:
- Generates code-switched garden path sentences between language pairs
- Creates sentences that mix two languages within the same sentence
- Uses DSPy to ensure unique sentence generation and avoid repetition
- Integrates with Hydra configuration system for easy customization

Code-Switched Garden Path Sentences:
These are sentences that combine two languages while maintaining garden path ambiguity.
Examples: "The horse raced past la granja fell" (English/Spanish mix with garden path structure).

Configuration:
- Set bilingual language pairs in config.yaml: codeswitch_generation.bilingual_languages
- Configure number of sentences: codeswitch_generation.num_sentences
- Set LLM model: llm.model
- Define code-switch prompt: prompt_files.code_switch_prompt

Usage:
1. Set environment variable: export OPENAI_API_KEY=your_key
2. Configure settings in src/conf/config.yaml
3. Run: python src/dataset_generation/generate_codeswitch_gardenpaths.py

Output:
- CSV file with columns: language_pair, sentence
- Each sentence combines two languages with garden path ambiguity
- Sentences are unique and appropriate for code-switching research

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

def generate_code_switched_gardenpath_sentences(language1, language2, total_sentences, code_switch_prompt, llm, cfg):
    """
    Generate code-switched garden path sentences between two languages.
    
    Args:
        language1 (str): First language
        language2 (str): Second language
        total_sentences (int): Total number of sentences to generate
        code_switch_prompt (str): The prompt template for code-switching
        llm: The configured LLM
        cfg: Configuration object
    
    Returns:
        list: List of dictionaries with 'language_pair' and 'sentence' keys
    """
    data = []
    
    print(f"Generating {total_sentences} code-switched garden path sentences ({language1}/{language2})...")

    prompt = code_switch_prompt.format(language1=language1, language2=language2)

    class CodeSwitchGardenPathSignature(dspy.Signature):
        """Generate a code-switched garden path sentence."""
        instruction: str = dspy.InputField()
        previous_sentences: str = dspy.InputField(desc="A list of sentences to avoid repeating.")
        sentence: str = dspy.OutputField(desc="A single, unique code-switched garden path sentence that combines both languages.")

    code_switch_predict = dspy.Predict(CodeSwitchGardenPathSignature)
    unique_sentences = set()
    max_attempts = total_sentences * 5
    attempts = 0
    
    with tqdm(total=total_sentences, desc=f"Code-switched garden path sentences", unit="sent") as pbar:
        while len(unique_sentences) < total_sentences and attempts < max_attempts:
            avoid = "\n".join(unique_sentences) if unique_sentences else ""
            result = code_switch_predict(instruction=prompt, previous_sentences=avoid)
            sentence = result.sentence.strip()
            if sentence and sentence not in unique_sentences:
                unique_sentences.add(sentence)
                pbar.update(1)
            attempts += 1
    
    if len(unique_sentences) < total_sentences:
        print(f"Warning: Only generated {len(unique_sentences)} unique code-switched sentences after {attempts} attempts.")
    
    for sentence in unique_sentences:
        data.append({
            "language_pair": f"{language1}/{language2}",
            "sentence": sentence
        })
    
    return data

CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "conf")

@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    if not cfg.codeswitch_generation.enabled:
        print("Code-switching generation is disabled in config. Set 'enabled: true' to run.")
        return
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your bash profile or session.")
    
    lm = dspy.LM(f"openai/{cfg.llm.model}", api_key=api_key)
    dspy.configure(lm=lm)
    
    print("Starting code-switched garden path sentence generation...")
    print(f"Using LLM: {cfg.llm.model}")
    print(f"Language pairs: {cfg.codeswitch_generation.bilingual_languages}")
    print(f"Sentences per pair: {cfg.codeswitch_generation.num_sentences}")
    
    try:
        all_sentences = []
        for language_pair in cfg.codeswitch_generation.bilingual_languages:
            language1, language2 = language_pair
            print(f"\n{'='*50}")
            print(f"Processing language pair: {language1}/{language2}")
            print(f"{'='*50}")
            
            sentences = generate_code_switched_gardenpath_sentences(
                language1, 
                language2, 
                cfg.codeswitch_generation.num_sentences, 
                cfg.prompt_files.code_switch_prompt, 
                lm, 
                cfg
            )
            all_sentences.extend(sentences)

        df = pd.DataFrame(all_sentences)
        output_path = os.path.join(SCRIPT_DIR, cfg.codeswitch_generation.output_file)
        save_dataframe_to_csv(df, output_path)
        
        print(f"\n{'='*50}")
        print("Code-switched garden path generation completed successfully!")
        print(f"Total sentences generated: {len(all_sentences)}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*50}")
        
        for language_pair in cfg.codeswitch_generation.bilingual_languages:
            lang1, lang2 = language_pair
            pair_sentences = df[df['language_pair'] == f"{lang1}/{lang2}"]
            print(f"{lang1}/{lang2}: {len(pair_sentences)} sentences")
            
    except Exception as e:
        print(f"Error during code-switched generation: {e}")
        raise

if __name__ == "__main__":
    main() 