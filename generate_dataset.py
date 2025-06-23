import hydra
from omegaconf import DictConfig
import dspy
import pandas as pd
import os
from hydra.core.hydra_config import HydraConfig
from utils import save_dataframe_to_csv

def generate_garden_path_sentence(language, llm):
    prompt = f"Generate a garden path sentence in {language}. A garden path sentence is a grammatically correct sentence that starts in such a way that a reader's most likely interpretation will be incorrect; the reader is led down the 'garden path.' Provide only the sentence."
    response = llm(prompt)
    return response

@hydra.main(config_path="conf", config_name="config", version_base=None)
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
            sentence = generate_garden_path_sentence(language, lm)
            data.append({
                "language": language,
                "sentence": sentence
            })
    
    # Save to CSV
    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, cfg.output_file)
    save_dataframe_to_csv(pd.DataFrame(data), output_path)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    main() 