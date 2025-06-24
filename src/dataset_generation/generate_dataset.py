import hydra
from omegaconf import DictConfig
import dspy
import pandas as pd
import os
import sys
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm
import re

# Ensure src is in the Python path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from utils import save_dataframe_to_csv

# The OpenAI API key is read from the environment variable OPENAI_API_KEY.
# Set it in your bash profile or session, e.g., export OPENAI_API_KEY=sk-...

def generate_garden_path_sentences(language, llm, cfg, n):
    if language not in cfg.prompt_files:
        raise ValueError(f"No prompt file specified for language '{language}'. Please add it to the config under 'prompt_files'.")
    prompt_file = os.path.join(SRC_DIR, cfg.prompt_files[language])
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Use dspy Predict module to enforce a single, unique garden path sentence output
    class GardenPathSignature(dspy.Signature):
        """Generate a unique garden path sentence."""
        instruction: str = dspy.InputField()
        previous_sentences: str = dspy.InputField(desc="A list of sentences to avoid repeating.")
        sentence: str = dspy.OutputField(desc="A single, unique garden path sentence.")

    garden_path_predict = dspy.Predict(GardenPathSignature)
    unique_sentences = set()
    max_attempts = n * 5  # avoid infinite loops if LLM repeats
    attempts = 0
    with tqdm(total=n, desc=f"{language} unique sentences", unit="sent") as pbar:
        while len(unique_sentences) < n and attempts < max_attempts:
            avoid = "\n".join(unique_sentences) if unique_sentences else ""
            prompt = prompt_template.format(language=language)
            result = garden_path_predict(instruction=prompt, previous_sentences=avoid)
            sentence = result.sentence.strip()
            if sentence and sentence not in unique_sentences:
                unique_sentences.add(sentence)
                pbar.update(1)
            attempts += 1
    if len(unique_sentences) < n:
        print(f"Warning: Only generated {len(unique_sentences)} unique sentences for {language} after {attempts} attempts.")
    return list(unique_sentences)

# Dynamically determine the config path relative to this script
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "conf")

@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Set up LLM
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your bash profile or session.")
    lm = dspy.LM(f"openai/{cfg.llm.model}", api_key=api_key)
    dspy.configure(lm=lm)
    
    data = []
    for language in cfg.languages:
        sentences = generate_garden_path_sentences(language, lm, cfg, cfg.num_sentences)
        for sentence in sentences:
            data.append({
                "language": language,
                "sentence": sentence
            })
    
    # Save to CSV in src/dataset_generation/dataset.csv
    output_path = os.path.join(SCRIPT_DIR, "multillingual_dataset.csv")
    save_dataframe_to_csv(pd.DataFrame(data), output_path)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    main() 