"""
Multilingual Garden-Path Sentence Generation

Generates garden-path sentences across 14 languages with different ambiguity types:
- Attachment ambiguities
- Case marking ambiguities  
- Relative clause ambiguities

Also generates matched unambiguous control sentences.
"""

import os
import sys
import hydra
from omegaconf import DictConfig
import dspy
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple
from pathlib import Path

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.utils import save_dataframe_to_csv


AMBIGUITY_TYPES = {
    "attachment_ambiguity": {
        "description": "Sentences where a phrase can attach to multiple positions (e.g., 'I saw the man with binoculars')",
        "examples": {
            "English": "While the man hunted, the deer ran through the woods",
            "Spanish": "Mientras el hombre cazaba, el ciervo corrió por el bosque"
        }
    },
    "case_marking_ambiguity": {
        "description": "Sentences with ambiguous case marking (especially in languages with case systems)",
        "examples": {
            "English": "The teacher saw the student with the book",
            "German": "Der Lehrer sah den Schüler mit dem Buch"
        }
    },
    "relative_clause_ambiguity": {
        "description": "Sentences with ambiguous relative clause attachment",
        "examples": {
            "English": "The daughter of the colonel who was on the balcony",
            "French": "La fille du colonel qui était sur le balcon"
        }
    }
}


def generate_gardenpath_sentence(language: str, ambiguity_type: str, llm, cfg) -> Dict:
    """
    Generate a garden-path sentence for a specific language and ambiguity type.
    
    Args:
        language: Target language
        ambiguity_type: Type of ambiguity to create
        llm: Configured LLM
        cfg: Configuration object
        
    Returns:
        Dictionary with sentence information
    """
    ambiguity_info = AMBIGUITY_TYPES[ambiguity_type]
    
    prompt = f"""Generate a garden-path sentence in {language} that contains a {ambiguity_type.replace('_', ' ')}.

A garden-path sentence is one where the initial interpretation leads the reader down the wrong path,
requiring reanalysis when disambiguating information arrives.

Ambiguity type: {ambiguity_type.replace('_', ' ').title()}
Description: {ambiguity_info['description']}

The sentence should:
1. Be natural and grammatically correct in {language}
2. Create initial misinterpretation that requires reanalysis
3. Have a clear disambiguation point
4. Be appropriate for linguistic research
5. Be written entirely in {language}

Generate only the sentence, no explanations."""

    class GardenPathSignature(dspy.Signature):
        """Generate a garden-path sentence with specific ambiguity type."""
        instruction: str = dspy.InputField()
        sentence: str = dspy.OutputField(desc=f"One garden-path sentence in {language} with {ambiguity_type}")

    predict = dspy.Predict(GardenPathSignature)
    
    try:
        result = predict(instruction=prompt)
        sentence = result.sentence.strip()
        
        # Remove quotes if present
        if sentence.startswith('"') and sentence.endswith('"'):
            sentence = sentence[1:-1]
        if sentence.startswith("'") and sentence.endswith("'"):
            sentence = sentence[1:-1]
        
        return {
            "language": language,
            "sentence": sentence,
            "ambiguity_type": ambiguity_type,
            "sentence_type": "gardenpath"
        }
    except Exception as e:
        print(f"Error generating sentence for {language}, {ambiguity_type}: {e}")
        return None


def generate_control_sentence(language: str, gardenpath_sentence: str, ambiguity_type: str, llm) -> Dict:
    """
    Generate an unambiguous control sentence matched to a garden-path sentence.
    
    Args:
        language: Target language
        gardenpath_sentence: The garden-path sentence to match
        ambiguity_type: Type of ambiguity in the garden-path sentence
        llm: Configured LLM
        
    Returns:
        Dictionary with control sentence information
    """
    prompt = f"""Generate an unambiguous control sentence in {language} that matches this garden-path sentence:

Garden-path sentence: "{gardenpath_sentence}"

The control sentence should:
1. Have the same meaning and structure as the garden-path sentence
2. Be unambiguous (no garden-path effect)
3. Be grammatically correct in {language}
4. Use similar vocabulary and length
5. Resolve the {ambiguity_type.replace('_', ' ')} ambiguity clearly

Generate only the control sentence, no explanations."""

    class ControlSentenceSignature(dspy.Signature):
        """Generate an unambiguous control sentence."""
        instruction: str = dspy.InputField()
        sentence: str = dspy.OutputField(desc=f"One unambiguous control sentence in {language}")

    predict = dspy.Predict(ControlSentenceSignature)
    
    try:
        result = predict(instruction=prompt)
        sentence = result.sentence.strip()
        
        # Remove quotes if present
        if sentence.startswith('"') and sentence.endswith('"'):
            sentence = sentence[1:-1]
        if sentence.startswith("'") and sentence.endswith("'"):
            sentence = sentence[1:-1]
        
        return {
            "language": language,
            "sentence": sentence,
            "ambiguity_type": ambiguity_type,
            "sentence_type": "control",
            "matched_gardenpath": gardenpath_sentence
        }
    except Exception as e:
        print(f"Error generating control for {language}: {e}")
        return None


def generate_multilingual_dataset(cfg: DictConfig) -> pd.DataFrame:
    """
    Generate garden-path and control sentences for all configured languages.
    
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
    ambiguity_types = mp600_cfg.dataset.ambiguity_types
    
    # Calculate sentences per ambiguity type
    sentences_per_type = num_gardenpath // len(ambiguity_types)
    remainder = num_gardenpath % len(ambiguity_types)
    
    all_sentences = []
    
    print(f"Generating multilingual garden-path dataset...")
    print(f"Languages: {len(all_languages)}")
    print(f"Garden-path sentences per language: {num_gardenpath}")
    print(f"Control sentences per language: {num_control}")
    print(f"Ambiguity types: {', '.join(ambiguity_types)}")
    print("=" * 60)
    
    for language in tqdm(all_languages, desc="Languages"):
        print(f"\nProcessing {language}...")
        
        # Generate garden-path sentences
        gardenpath_sentences = []
        for i, ambiguity_type in enumerate(ambiguity_types):
            n = sentences_per_type + (1 if i < remainder else 0)
            
            for _ in range(n):
                sentence_data = generate_gardenpath_sentence(
                    language, ambiguity_type, None, cfg
                )
                if sentence_data:
                    gardenpath_sentences.append(sentence_data)
                    all_sentences.append(sentence_data)
        
        # Generate control sentences (matched to garden-path sentences)
        for gardenpath_data in gardenpath_sentences[:num_control]:
            control_data = generate_control_sentence(
                language,
                gardenpath_data["sentence"],
                gardenpath_data["ambiguity_type"],
                None
            )
            if control_data:
                all_sentences.append(control_data)
    
    df = pd.DataFrame(all_sentences)
    return df


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for multilingual dataset generation."""
    if not cfg.multilingual_p600.enabled:
        print("Multilingual P600 extension is disabled in config.")
        return
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    
    # Configure DSPy
    lm = dspy.LM(f"openai/{cfg.llm.model}", api_key=api_key)
    dspy.configure(lm=lm)
    
    # Generate dataset
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
        print(f"  {lang}: {len(lang_df)} sentences ({len(lang_df[lang_df['sentence_type']=='gardenpath'])} gardenpath, {len(lang_df[lang_df['sentence_type']=='control'])} control)")
    print(f"\nBreakdown by ambiguity type:")
    for amb_type in df['ambiguity_type'].unique():
        print(f"  {amb_type}: {len(df[df['ambiguity_type']==amb_type])} sentences")


if __name__ == "__main__":
    main()

