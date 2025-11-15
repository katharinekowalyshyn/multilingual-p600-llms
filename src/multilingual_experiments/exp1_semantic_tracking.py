"""
Experiment 1: Semantic Interpretation Tracking

Tracks how model interpretations change incrementally as sentences are processed
chunk-by-chunk. Uses question-answering probes to detect misinterpretations vs
correct interpretations at each chunk position.

Key measures:
- Does disambiguation happen at same chunk position across languages?
- Resource-level differences in lingering misinterpretations
- Language-specific features that aid/hinder reanalysis
"""

import os
import sys
import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json
from src.multilingual_experiments.model_loader import ModelLoader, load_model_from_config
from src.multilingual_experiments.chunking import get_chunking_strategy
from src.utils import save_dataframe_to_csv
import dspy

"""
# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
"""



def generate_qa_pairs(sentence: str, language: str, ambiguity_type: str, 
                 num_questions: int = 5, loader=None) -> List[Dict]:
    """
    Generate question-answer pairs to probe semantic interpretation using Ollama.
    
    Args:
        sentence: The sentence to generate questions about
        language: Language of the sentence
        ambiguity_type: Type of ambiguity in the sentence
        num_questions: Number of QA pairs to generate
        loader: ModelLoader instance (Ollama) for question generation
        
    Returns:
        List of dictionaries with 'question', 'correct_answer', 'misinterpretation_answer'
    """
    if loader is None:
        # Fallback: generate simple template questions
        return _generate_template_questions(sentence, language, ambiguity_type, num_questions)
    
    prompt = f"""Generate {num_questions} question-answer pairs to probe semantic interpretation of this garden-path sentence in {language}:

Sentence: "{sentence}"

Ambiguity type: {ambiguity_type.replace('_', ' ').title()}

For each question, provide:
1. A question that tests whether the reader has correctly interpreted the sentence
2. The correct answer (after disambiguation)
3. The misinterpretation answer (initial garden-path interpretation)

The questions should:
- Test understanding of who did what to whom
- Probe the ambiguous attachment/resolution
- Be answerable from the sentence content
- Be in {language}

Format your response as a JSON array with keys: question, correct_answer, misinterpretation_answer

Example format:
[
  {{"question": "...", "correct_answer": "...", "misinterpretation_answer": "..."}},
  ...
]"""

    try:
        response = loader.generate(prompt, options={"temperature": 0.3})
        
        # Try to extract JSON from response
        qa_text = response.strip()
        
        # Remove markdown code blocks if present
        if '```' in qa_text:
            parts = qa_text.split('```')
            for part in parts:
                if '[' in part and '{' in part:
                    qa_text = part
                    if qa_text.startswith('json'):
                        qa_text = qa_text[4:]
                    break
        
        # Try to parse JSON
        qa_pairs = json.loads(qa_text)
        if isinstance(qa_pairs, list) and len(qa_pairs) > 0:
            return qa_pairs[:num_questions]  # Limit to requested number
        else:
            raise ValueError("Invalid JSON format")
    except Exception as e:
        print(f"Error generating QA pairs with Ollama: {e}")
        return _generate_template_questions(sentence, language, ambiguity_type, num_questions)


def _generate_template_questions(sentence: str, language: str, ambiguity_type: str, 
                                   num_questions: int) -> List[Dict]:
    """Fallback template-based question generation."""
    # Simple template questions (can be improved)
    questions = []
    
    if ambiguity_type == "attachment_ambiguity":
        questions.append({
            "question": f"What is the main action in: {sentence}?",
            "correct_answer": "To be determined from sentence",
            "misinterpretation_answer": "Initial misinterpretation"
        })
    elif ambiguity_type == "relative_clause_ambiguity":
        questions.append({
            "question": f"Who or what does the relative clause modify in: {sentence}?",
            "correct_answer": "To be determined from sentence",
            "misinterpretation_answer": "Initial misinterpretation"
        })
    else:
        questions.append({
            "question": f"What is the meaning of: {sentence}?",
            "correct_answer": "To be determined from sentence",
            "misinterpretation_answer": "Initial misinterpretation"
        })
    
    # Repeat to reach num_questions
    while len(questions) < num_questions:
        questions.append(questions[0].copy())
    
    return questions[:num_questions]


def probe_interpretation(loader, chunk_text: str, question: str, 
                        language: str) -> Dict[str, float]:
    """
    Probe model's interpretation at a given chunk using a question via Ollama.
    
    Args:
        loader: ModelLoader instance (Ollama)
        chunk_text: Text chunk processed so far
        question: Question to probe interpretation
        language: Language of the text
        
    Returns:
        Dictionary with interpretation scores
    """
    # Create prompt: chunk + question
    prompt = f"{chunk_text}\n\nQuestion: {question}\nAnswer:"
    
    # Generate answer using Ollama
    try:
        response = loader.generate(prompt, options={"temperature": 0.1, "num_predict": 20})
        
        # For analysis, we'll use the response text
        # In a more sophisticated version, we could get logprobs if Ollama supports it
        return {
            "response": response.strip(),
            "response_length": len(response),
            "entropy": 0.0  # Would need logprobs from Ollama to calculate
        }
    except Exception as e:
        print(f"Error probing interpretation: {e}")
        return {
            "response": "",
            "response_length": 0,
            "entropy": 0.0
        }


def run_incremental_qa_analysis(sentence: str, language: str, ambiguity_type: str,
                                loader, chunking_strategy, 
                                num_qa_probes: int = 5) -> pd.DataFrame:
    """
    Run incremental QA analysis on a sentence.
    
    Args:
        sentence: Sentence to analyze
        language: Language of sentence
        ambiguity_type: Type of ambiguity
        loader: ModelLoader instance (Ollama)
        chunking_strategy: Chunking strategy instance (syntactic units)
        num_qa_probes: Number of QA pairs to generate
        
    Returns:
        DataFrame with incremental analysis results
    """
    # Generate QA pairs using Ollama
    qa_pairs = generate_qa_pairs(sentence, language, ambiguity_type, num_qa_probes, loader)
    
    # Chunk the sentence using syntactic units
    # Use loader's tokenizer if available, otherwise use a fallback
    tokenizer = loader.tokenizer if loader.tokenizer else None
    if tokenizer is None:
        # Fallback: create a simple tokenizer-like object
        class SimpleTokenizer:
            def encode(self, text, **kwargs):
                return loader.tokenize(text, add_bos=False)
            def decode(self, tokens, **kwargs):
                # Simple decode - just join tokens if they're strings
                if isinstance(tokens, list) and tokens and isinstance(tokens[0], str):
                    return " ".join(tokens)
                return str(tokens)
        tokenizer = SimpleTokenizer()
    
    chunks = chunking_strategy.chunk(sentence, tokenizer)
    
    results = []
    
    # Process incrementally
    accumulated_text = ""
    for chunk_idx, (chunk_text, start_token, end_token) in enumerate(chunks):
        accumulated_text += " " + chunk_text if accumulated_text else chunk_text
        
        # Probe with each question
        for qa_idx, qa_pair in enumerate(qa_pairs):
            probe_result = probe_interpretation(
                loader, accumulated_text, qa_pair['question'], language
            )
            
            # Check if interpretation matches correct or misinterpretation
            response_text = probe_result.get('response', '').lower()
            correct_answer = qa_pair.get('correct_answer', '').lower()
            misinterpretation_answer = qa_pair.get('misinterpretation_answer', '').lower()
            
            # Simple keyword matching (could be improved with semantic similarity)
            matches_correct = any(word in response_text for word in correct_answer.split() if len(word) > 3)
            matches_misinterpretation = any(word in response_text for word in misinterpretation_answer.split() if len(word) > 3)
            
            results.append({
                "chunk_index": chunk_idx,
                "chunk_text": chunk_text,
                "accumulated_text": accumulated_text,
                "start_token": start_token,
                "end_token": end_token,
                "question_index": qa_idx,
                "question": qa_pair.get('question', ''),
                "correct_answer": correct_answer,
                "misinterpretation_answer": misinterpretation_answer,
                "model_response": probe_result.get('response', ''),
                "matches_correct": matches_correct,
                "matches_misinterpretation": matches_misinterpretation,
                "entropy": probe_result.get('entropy', 0.0),
                "response_length": probe_result.get('response_length', 0)
            })
    
    return pd.DataFrame(results)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for Experiment 1."""
    if not cfg.multilingual_p600.enabled:
        print("Multilingual P600 extension is disabled.")
        return
    
    exp_cfg = cfg.multilingual_p600.experiments.experiment_1_semantic_tracking
    if not exp_cfg.enabled:
        print("Experiment 1 is disabled in config.")
        return
    
    # Load dataset
    # TODO: Generate or add garden-path sentences for all 14 languages
    # The dataset should contain columns: language, sentence, ambiguity_type, sentence_type
    # For now, we assume the dataset exists. To generate it, run:
    # python src/multilingual_experiments/dataset_generation.py
    dataset_path = Path(hydra.utils.to_absolute_path(
        cfg.multilingual_p600.dataset.output_dir
    )) / "multilingual_gardenpath_dataset.csv"
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Please generate garden-path sentences first by running dataset_generation.py, "
            "or manually create a CSV with columns: language, sentence, ambiguity_type, sentence_type"
        )
    
    df = pd.read_csv(dataset_path)
    gardenpath_df = df[df['sentence_type'] == 'gardenpath']
    
    print(f"Loaded {len(gardenpath_df)} garden-path sentences")
    
    # Setup Ollama for QA generation
    # We'll use the same model loader for QA generation
    llm = None  # Will be set per model
    
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
            # Load model (Ollama)
            model_name, tokenizer, loader = load_model_from_config({
                **model_cfg,
                'ollama_base_url': cfg.multilingual_p600.get('ollama_base_url', 'http://localhost:11434')
            })
            
            # Get chunking strategy (syntactic units)
            # TODO: Make language-specific chunking based on row['language']
            chunk_strategy = get_chunking_strategy(
                exp_cfg.chunk_strategy,  # Should be "syntactic"
                language='en'  # TODO: Use row['language'] for language-specific chunking
            )
            
            # Process sentences
            for idx, row in tqdm(gardenpath_df.iterrows(), 
                               total=len(gardenpath_df), 
                               desc=f"Processing {model_name}"):
                try:
                    results_df = run_incremental_qa_analysis(
                        sentence=row['sentence'],
                        language=row['language'],
                        ambiguity_type=row['ambiguity_type'],
                        loader=loader,
                        chunking_strategy=chunk_strategy,
                        num_qa_probes=exp_cfg.num_qa_probes
                    )
                    
                    # Add metadata
                    results_df['model'] = model_name
                    results_df['sentence_id'] = idx
                    results_df['language'] = row['language']
                    results_df['ambiguity_type'] = row['ambiguity_type']
                    
                    all_results.append(results_df)
                    
                except Exception as e:
                    print(f"Error processing sentence {idx}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save results
        output_dir = Path(hydra.utils.to_absolute_path(exp_cfg.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "semantic_tracking_results.csv"
        
        save_dataframe_to_csv(combined_df, str(output_path))
        
        print(f"\n{'='*60}")
        print("Experiment 1 complete!")
        print(f"Results saved to: {output_path}")
        print(f"Total records: {len(combined_df)}")
        print(f"{'='*60}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()

