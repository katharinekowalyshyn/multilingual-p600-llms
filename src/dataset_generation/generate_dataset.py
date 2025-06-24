import hydra
from omegaconf import DictConfig
import dspy
import pandas as pd
import os
from hydra.core.hydra_config import HydraConfig
from src.utils import save_dataframe_to_csv

def generate_garden_path_sentence(language, llm, cfg):
    # Require a prompt file for each language; raise an error if missing
    if language not in cfg.prompt_files:
        raise ValueError(f"No prompt file specified for language '{language}'. Please add it to the config under 'prompt_files'.")
    prompt_file = cfg.prompt_files[language]
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    prompt = prompt_template.format(language=language)
    response = llm(prompt)
    return response

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Set up LLM
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running.")
    lm = dspy.LM(f"openai/{cfg.llm.model}", api_key=api_key)
    dspy.configure(lm=lm)
    
    data = []
    for language in cfg.languages:
        for _ in range(cfg.num_sentences):
            sentence = generate_garden_path_sentence(language, lm, cfg)
            data.append({
                "language": language,
                "sentence": sentence
            })
    
    # Save to CSV
    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dataset_generation", cfg.output_file)
    save_dataframe_to_csv(pd.DataFrame(data), output_path)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    main() 