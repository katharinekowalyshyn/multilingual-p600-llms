# multilingual-gardenpath-llms

This project is designed to generate and evaluate multilingual garden path sentences using large language models (LLMs). It is organized in three main parts:
1. **Dataset Generation**: Automatically generate datasets of garden path sentences in multiple languages using an LLM.
2. **P600 Sentence Processing**: Process and classify P600 sentences for grammaticality using LLMs.
3. **LLM Evaluation**: (Planned) Evaluate how well LLMs understand and process these sentences.

## Features
- Multilingual support (easily configurable languages)
- Modular, experiment-friendly codebase using Hydra for configuration
- LLM output management with [dspy](https://github.com/stanfordnlp/dspy)
- Utility functions for experiment pipelines
- Progress bar for dataset generation
- Ensures unique garden path sentences per language
- P600 sentence grammaticality classification with binary output (0/1)

## Setup

1. **Clone the repository**
2. **Create and activate a conda environment**
   ```bash
   conda create -y -n gardenpath python=3.12
   conda activate gardenpath
   ```
3. **Install dependencies**
   ```bash
   pip install hydra-core dspy-ai pandas tqdm
   ```
4. **Set your API keys**
   ```bash
   # For dataset generation and P600 processing (OpenAI models)
   export OPENAI_API_KEY=sk-...yourkey...
   
   # For other model providers (if not using OpenAI)
   export NEURONPEDIA_API_KEY=your_neuronpedia_key_here
   ```

## Configuration
Edit `src/conf/config.yaml` to set:
- `gardenpath_generation`: Configuration for multilingual garden path sentences
- `codeswitch_generation`: Configuration for code-switched garden path sentences
- `p600_generation`: Configuration for P600 sentence generation
- `p600_processing`: Configuration for P600 sentence processing
- `llm.model`: OpenAI model to use (e.g., `gpt-4o-mini`)
- `prompt_files`: Mapping of language to prompt file

## Usage

### Dataset Generation

The project supports three types of dataset generation:

#### 1. Multilingual Garden Path Sentences
Generate garden path sentences in multiple languages:
```bash
cd src/dataset_generation
python generate_multilingual_gardenpaths.py
```
Output: `multilingual_gardenpath_dataset.csv`

#### 2. Code-Switched Garden Path Sentences
Generate garden path sentences that mix two languages:
```bash
cd src/dataset_generation
python generate_codeswitch_gardenpaths.py
```
Output: `codeswitch_gardenpath_dataset.csv`

#### 3. P600 Sentences
Generate P600 sentences for grammaticality analysis:
```bash
cd src/dataset_generation
python generate_p600_sentences.py
```
Output: `p600_generated_sentences.csv`

**Configuration Options:**
- Set `enabled: true/false` for each generation type in `config.yaml`
- Adjust `num_sentences` or `num_sentences_per_language` as needed
- Modify `languages` list to include/exclude specific languages
- Customize output file names

### Grammatical Analysis (P600 Processing)

The P600 analysis system evaluates the grammaticality of three types of sentences:

#### 1. Test the Configuration
Before running the full analysis, test your setup:
```bash
cd src/grammar_analysis
python test_p600.py
```
This will process a small sample (3 sentences) to verify everything works correctly.

#### 2. Run the Full Analysis
Process all sentence files for grammaticality classification:
```bash
cd src/grammar_analysis
python p600_analysis.py
# or use the convenient shell script:
./run_p600.sh
```

#### 3. Input Files
The system processes these CSV files (located in `src/grammar_analysis/`):
- `gardenpath_sample.csv` - Garden path sentences
- `control_gardenpath_sample.csv` - Control garden path sentences  
- `p600_sample.csv` - P600 sample sentences

#### 4. Output
Results are saved in `src/grammar_analysis/results/` with:
- Original sentence data
- `grammaticality` column (1 for grammatical, 0 for non-grammatical)
- Summary statistics with counts and percentages

**Features:**
- Uses DSPY to enforce binary output format (0 or 1 only)
- Progress tracking for each file being processed
- Error handling and automatic fallback for invalid outputs
- Fully integrated with Hydra configuration system

## Project Structure
```
src/
├── conf/
│   └── config.yaml                    # Main configuration file
├── dataset_generation/
│   ├── generate_multilingual_gardenpaths.py    # Multilingual sentences
│   ├── generate_codeswitch_gardenpaths.py      # Code-switched sentences
│   ├── generate_p600_sentences.py              # P600 sentences
│   └── *.csv                          # Generated datasets
├── grammar_analysis/
│   ├── p600_analysis.py              # Main P600 processing script
│   ├── test_p600.py                  # Configuration test script
│   ├── run_p600.sh                   # Convenient runner script
│   ├── *.csv                         # Input sentence files
│   └── results/                      # Output directory
├── prompts/                           # Prompt templates
└── utils.py                          # Utility functions
```

## Utilities
- Common helper functions are in `src/utils.py` (e.g., for saving DataFrames).
- Configuration testing and validation scripts for each component.

## Extending the Project
- Add new utility functions to `src/utils.py` as needed.
- The P600 processing system can be extended to handle different sentence types or languages.
- The second part of the project (LLM evaluation) will be implemented in future scripts.

## Troubleshooting

### Dataset Generation
- Ensure `OPENAI_API_KEY` is set in your environment
- Check that prompt files exist in the specified locations
- Verify that the generation type is enabled in `config.yaml`

### P600 Processing
- Ensure `OPENAI_API_KEY` is set in your environment
- Verify all CSV files exist and have proper formatting
- Check the `results/` directory for output files
- Invalid LLM outputs automatically default to 0
- Use `test_p600.py` to debug configuration issues

## License
MIT

## Acknowledgments
- [DSPy](https://github.com/stanfordnlp/dspy) for LLM management
- [Hydra](https://hydra.cc/) for configuration management