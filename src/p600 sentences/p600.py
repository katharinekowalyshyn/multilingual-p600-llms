import hydra
from omegaconf import DictConfig
import dspy
import pandas as pd
import os
import sys
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm
import re
from utils import save_dataframe_to_csv

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
    with tqdm(total=len(df), desc="Processing sentences", unit="sent") as pbar:
        for idx, row in df.iterrows():
            sentence = row['sentence']
            prompt = cfg.p600_processing.prompt.format(sentence=sentence)
            
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
    # Set up LLM with automatic API key detection
    model = cfg.llm.model
    
    # Automatically detect API key based on model type
    if model.startswith(('gpt-', 'openai/')):
        # OpenAI models
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your bash profile or session.")
        # Remove 'openai/' prefix if present for DSPY
        if model.startswith('openai/'):
            model = model[8:]  # Remove 'openai/' prefix
        dspy_model = f"openai/{model}"
    else:
        # For other model providers, you can extend this logic
        # For now, we'll use a generic approach
        api_key = os.environ.get("NEURONPEDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No API key found. Set either NEURONPEDIA_API_KEY or OPENAI_API_KEY environment variable.")
        dspy_model = f"openai/{model}"  # Default to OpenAI format
    
    print(f"Using model: {model}")
    print(f"API key type: {'OpenAI' if model.startswith(('gpt-', 'openai/')) else 'Other'}")
    
    lm = dspy.LM(dspy_model, api_key=api_key)
    dspy.configure(lm=lm)
    
    # Create output directory
    output_dir = os.path.join(SRC_DIR, cfg.p600_processing.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each input file
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
            # Generate output filename
            output_filename = f"{file_type}_grammaticality_results.csv"
            output_path = os.path.join(output_dir, output_filename)
            
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
                failure_log_filename = f"{file_type}_failure_log.csv"
                failure_log_path = os.path.join(output_dir, failure_log_filename)
                
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
        summary_path = os.path.join(output_dir, summary_filename)
        
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
        print("P600 sentence processing completed successfully!")
        print("="*50)
    except Exception as e:
        print(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()
