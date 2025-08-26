"""
Created: 2025-08-21

P600 sentence grammaticality processing.

This Hydra-driven script classifies sentences in the configured P600 CSVs as
grammatical (1) or non-grammatical (0) using DSPy with the LLM specified in
`cfg.llm.model`.

Key behavior:
- Reads input CSVs from `cfg.p600_processing.input_files` (each must include a
  `sentence` column).
- Uses `cfg.llm.model`; API key is auto-detected based on provider. For OpenAI
  models (`gpt*` or `openai*`), reads `OPENAI_API_KEY` if present.
- Produces per-file outputs under `cfg.p600_processing.output_dir/<file_key>/`
  containing an added `grammaticality` column (1 or 0).
- Logs failures and tracks "false negatives" where outputs default to 0 due to
  invalid LLM responses or API errors; saves per-file failure logs and an
  overall failure summary.
- Mirrors the structure and Hydra integration patterns of
  `src/dataset_generation/generate_dataset.py`.

Run from the project (Hydra) context, e.g.:
    python src/p600 sentences/p600.py
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

def process_sentences_for_grammaticality(csv_file_path, llm, cfg):
    """
    Process sentences from a CSV file to determine grammaticality.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        llm: The configured LLM
        cfg: Configuration object
    
    Returns:
        tuple: (DataFrame with original data plus grammaticality column, dict with failure statistics)
    """
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Processing {len(df)} sentences from {os.path.basename(csv_file_path)}")
    except FileNotFoundError:
        print(f"Error: File {csv_file_path} not found")
        return None, {}
    
    # Initialize failure tracking
    failure_stats = {
        'invalid_outputs': 0,
        'api_errors': 0,
        'total_failures': 0,
        'failed_sentences': []
    }
    
    class GrammaticalitySignature(dspy.Signature):
        """Classify if a sentence is grammatically correct."""
        sentence: str = dspy.InputField(desc="The sentence to evaluate")
        instruction: str = dspy.InputField(desc="Instructions for classification")
        grammaticality: str = dspy.OutputField(desc="Must be exactly '1' for grammatical or '0' for non-grammatical")
    
    grammaticality_predict = dspy.Predict(GrammaticalitySignature)
    
    grammaticality_scores = []
    # If test_mode and test_n are set, optionally truncate the DataFrame
    if getattr(cfg.p600_processing, 'test_mode', False) and getattr(cfg.p600_processing, 'test_n', 0):
        df = df.head(int(cfg.p600_processing.test_n))

    with tqdm(total=len(df), desc="Processing sentences", unit="sent") as pbar:
        for idx, row in df.iterrows():
            sentence = row['sentence']
            # Load prompt: prefer file if configured
            prompt_text = None
            if hasattr(cfg.p600_processing, 'prompt_file') and cfg.p600_processing.prompt_file:
                try:
                    prompt_path = os.path.join(SRC_DIR, cfg.p600_processing.prompt_file)
                    with open(prompt_path, 'r', encoding='utf-8') as pf:
                        prompt_text = pf.read()
                except Exception as _:
                    print(f"Error: Failed to load prompt from {cfg.p600_processing.prompt_file}")
                    break
            if not prompt_text:
                print(f"Error: No prompt file found in {cfg.p600_processing.prompt_file}")
                break

            prompt = prompt_text.format(sentence=sentence)
            
            if getattr(cfg.p600_processing, 'skip_llm', False):
                # In skip mode, assign a deterministic placeholder (e.g., 0) without API usage
                grammaticality_scores.append(0)
            else:
                try:
                    result = grammaticality_predict(sentence=sentence, instruction=prompt)
                    score = result.grammaticality.strip()
                    
                    if score not in ['0', '1']:
                        print(f"Warning: Invalid output '{score}' for sentence '{sentence[:50]}...'. Defaulting to '0'")
                        score = '0'
                        failure_stats['invalid_outputs'] += 1
                        failure_stats['failed_sentences'].append({
                            'index': idx,
                            'sentence': sentence,
                            'error_type': 'invalid_output',
                            'llm_output': result.grammaticality.strip()
                        })
                    
                    grammaticality_scores.append(int(score))
                    
                except Exception as e:
                    print(f"Error processing sentence '{sentence[:50]}...': {e}. Defaulting to '0'")
                    grammaticality_scores.append(0)
                    failure_stats['api_errors'] += 1
                    failure_stats['failed_sentences'].append({
                        'index': idx,
                        'sentence': sentence,
                        'error_type': 'api_error',
                        'error_message': str(e)
                    })
            
            pbar.update(1)
    
    # Calculate total failures
    failure_stats['total_failures'] = failure_stats['invalid_outputs'] + failure_stats['api_errors']
    
    df['grammaticality'] = grammaticality_scores
    
    return df, failure_stats

def process_all_p600_files(cfg):
    """
    Process all P600 sentence files and save results.
    
    Args:
        cfg: Configuration object
    """
    # LLM setup (skipped when skip_llm=true)
    lm = None
    if not getattr(cfg.p600_processing, 'skip_llm', False):
        model = cfg.llm.model
        if model.startswith(('gpt', 'openai')):
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your bash profile or session.")
            if model.startswith('openai/'):
                model = model[8:]
            dspy_model = f"openai/{model}"
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No API key found. Set OPENAI_API_KEY environment variable.")
            dspy_model = f"openai/{model}"
        print(f"Using model: {model}")
        lm = dspy.LM(dspy_model, api_key=api_key)
        dspy.configure(lm=lm)
    
    # Base output directory (for overall summary)
    base_output_dir = os.path.join(SRC_DIR, cfg.p600_processing.output_dir)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Per-file specific output directories
    output_dirs = {}
    for file_type, file_path in cfg.p600_processing.input_files.items():
        # Use the input file's base name (without extension) for output file naming
        input_base = os.path.splitext(os.path.basename(file_path))[0]
        print("Input base: ", input_base)
        specific_output_dir = os.path.join(SRC_DIR, cfg.p600_processing.output_dir, file_type)
        os.makedirs(specific_output_dir, exist_ok=True)
        output_dirs[file_path] = (specific_output_dir, input_base)
    
    # Select files: all or only the test file
    if getattr(cfg.p600_processing, 'test_mode', False):
        key = cfg.p600_processing.test_file_key
        input_files = {key: cfg.p600_processing.input_files[key]}
    else:
        input_files = cfg.p600_processing.input_files
    
    # Track overall failure statistics
    overall_failure_stats = {
        'total_files_processed': 0,
        'total_sentences_processed': 0,
        'total_failures': 0,
        'total_invalid_outputs': 0,
        'total_api_errors': 0,
        'file_specific_stats': {}
    }
    
    for file_type, file_path in input_files.items():
        print(f"\n{'='*50}")
        print(f"Processing {file_type} sentences...")
        print(f"{'='*50}")
        
        # Construct full path to input file
        full_input_path = os.path.join(SRC_DIR, file_path)
        
        # Process the file
        result_df, failure_stats = process_sentences_for_grammaticality(full_input_path, lm, cfg)
        
        if result_df is not None:
            # Resolve specific output dir and filename
            specific_output_dir, input_base = output_dirs[file_path]
            output_filename = f"{input_base}_grammaticality_results.csv"
            output_path = os.path.join(specific_output_dir, output_filename)
            
            # Save results
            save_dataframe_to_csv(result_df, output_path)
            print(f"Results saved to: {output_path}")
            
            # Print summary statistics
            total_sentences = len(result_df)
            grammatical_count = result_df['grammaticality'].sum()
            non_grammatical_count = total_sentences - grammatical_count
            
            print(f"Summary for {file_type}:")
            print(f"  Total sentences: {total_sentences}")
            print(f"  Grammatical (1): {grammatical_count}")
            print(f"  Non-grammatical (0): {non_grammatical_count}")
            print(f"  Grammaticality rate: {grammatical_count/total_sentences*100:.1f}%")
            
            # Print failure statistics
            if failure_stats['total_failures'] > 0:
                print(f"  ⚠️  Failures detected:")
                print(f"    - Invalid outputs: {failure_stats['invalid_outputs']}")
                print(f"    - API errors: {failure_stats['api_errors']}")
                print(f"    - Total failures: {failure_stats['total_failures']}")
                print(f"    - Failure rate: {failure_stats['total_failures']/total_sentences*100:.1f}%")
            else:
                print(f"  ✓ No failures detected")
            
            # Update overall statistics
            overall_failure_stats['total_files_processed'] += 1
            overall_failure_stats['total_sentences_processed'] += total_sentences
            overall_failure_stats['total_failures'] += failure_stats['total_failures']
            overall_failure_stats['total_invalid_outputs'] += failure_stats['invalid_outputs']
            overall_failure_stats['total_api_errors'] += failure_stats['api_errors']
            overall_failure_stats['file_specific_stats'][file_type] = failure_stats
            
            # Save detailed failure log if there were failures
            if failure_stats['total_failures'] > 0:
                failure_log_filename = f"{input_base}_failure_log.csv"
                failure_log_path = os.path.join(specific_output_dir, failure_log_filename)
                
                failure_df = pd.DataFrame(failure_stats['failed_sentences'])
                save_dataframe_to_csv(failure_df, failure_log_path)
                print(f"  📋 Detailed failure log saved to: {failure_log_path}")
        else:
            print(f"Failed to process {file_type} file")
    
    # Print overall failure summary
    print(f"\n{'='*50}")
    print("DEBUG: About to print overall failure summary")
    print(f"Total failures found: {overall_failure_stats['total_failures']}")
    print(f"{'='*50}")
    
    if overall_failure_stats['total_failures'] > 0:
        print("OVERALL FAILURE SUMMARY")
        print(f"Total files processed: {overall_failure_stats['total_files_processed']}")
        print(f"Total sentences processed: {overall_failure_stats['total_sentences_processed']}")
        print(f"Total failures: {overall_failure_stats['total_failures']}")
        print(f"Total invalid outputs: {overall_failure_stats['total_invalid_outputs']}")
        print(f"Total API errors: {overall_failure_stats['total_api_errors']}")
        print(f"Overall failure rate: {overall_failure_stats['total_failures']/overall_failure_stats['total_sentences_processed']*100:.1f}%")
        
        # Save overall failure summary
        summary_filename = "overall_failure_summary.csv"
        summary_path = os.path.join(base_output_dir, summary_filename)
        
        summary_data = {
            'metric': ['total_files_processed', 'total_sentences_processed', 'total_failures', 
                      'total_invalid_outputs', 'total_api_errors', 'overall_failure_rate_percent'],
            'value': [
                overall_failure_stats['total_files_processed'],
                overall_failure_stats['total_sentences_processed'],
                overall_failure_stats['total_failures'],
                overall_failure_stats['total_invalid_outputs'],
                overall_failure_stats['total_api_errors'],
                round(overall_failure_stats['total_failures']/overall_failure_stats['total_sentences_processed']*100, 2)
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        save_dataframe_to_csv(summary_df, summary_path)
        print(f"📊 Overall failure summary saved to: {summary_path}")
    else:
        print("🎉 NO FAILURES DETECTED - ALL PROCESSING SUCCESSFUL!")

CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "conf")

@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to process P600 sentences for grammaticality classification.
    
    This script:
    1. Reads three CSV files containing different types of sentences
    2. Uses an LLM to classify each sentence as grammatical (1) or non-grammatical (0)
    3. Outputs three new CSV files with the grammaticality classifications
    4. Uses DSPY to enforce binary output format
    5. Tracks and reports all failures and false negatives
    """
    # Check if P600 processing is enabled
    if not cfg.p600_processing.enabled:
        print("P600 processing is disabled in config. Set 'enabled: true' to run.")
        return
    
    print("Starting P600 sentence grammaticality analysis...")
    print(f"Using LLM: {cfg.llm.model}")
    print(f"Output directory: {cfg.p600_processing.output_dir}")
    
    try:
        process_all_p600_files(cfg)
        print("\n" + "="*50)
        #print("P600 sentence processing completed successfully!")
        print("="*50)
    except Exception as e:
        print(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()
