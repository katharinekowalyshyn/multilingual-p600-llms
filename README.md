# multilingual-gardenpath-llms

This project is designed to generate and evaluate multilingual garden path sentences using large language models (LLMs). It is organized in three main parts:
1. **Dataset Generation**: Automatically generate a dataset of garden path sentences in multiple languages using an LLM.
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
- `languages`: List of languages to generate sentences in
- `num_sentences`: Number of unique sentences per language
- `llm.model`: OpenAI model to use (e.g., `gpt-4o`) - used by both dataset generation and P600 processing
- `prompt_files`: Mapping of language to prompt file (relative to `src/`)
- `p600_processing`: Configuration for P600 sentence processing (enabled/disabled, input files, output directory, prompt)

## Usage

### Dataset Generation
To generate a dataset, run from the project root or `src/` directory:
```bash
cd multilingual-gardenpath-llms
python src/dataset_generation/generate_dataset.py
```
The output CSV will be saved as `src/dataset_generation/multillingual_dataset.csv`, with columns for `language` and `sentence`.

- Each language will have the number of unique sentences specified in the config.
- A progress bar will show generation progress for each language.

### P600 Sentence Processing
To process P600 sentences for grammaticality classification:

1. **Test the configuration:**
   ```bash
   cd "src/p600 sentences"
   python test_p600.py
   ```

2. **Run the processing:**
   ```bash
   python p600.py
   # or use the shell script:
   ./run_p600.sh
   ```

The system processes three CSV files:
- `gardenpath_sample.csv` - Garden path sentences
- `control_gardenpath_sample.csv` - Control garden path sentences  
- `p600_sample.csv` - P600 sample sentences

Output files are saved in `src/p600 sentences/results/` with the original data plus a `grammaticality` column (1 for grammatical, 0 for non-grammatical).

**Features:**
- Uses DSPY to enforce binary output format (0 or 1 only)
- Progress tracking for each file being processed
- Error handling and automatic fallback for invalid outputs
- Summary statistics with counts and percentages
- Fully integrated with Hydra configuration system

## Project Structure
```
src/
├── conf/
│   └── config.yaml          # Main configuration file
├── dataset_generation/
│   ├── generate_dataset.py  # Dataset generation script
│   └── ...                  # Generated datasets
├── p600 sentences/
│   ├── p600.py             # P600 processing script
│   ├── test_p600.py        # Configuration test script
│   ├── run_p600.sh         # Convenient runner script
│   ├── *.csv               # Input sentence files
│   └── results/            # Output directory
├── prompts/                 # Prompt templates
└── utils.py                # Utility functions
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

### P600 Processing
- Ensure `NEURONPEDIA_API_KEY` is set in your environment
- Verify all CSV files exist and have proper formatting
- Check the `results/` directory for output files
- Invalid LLM outputs automatically default to 0

## License
MIT

## Acknowledgments
- [DSPy](https://github.com/stanfordnlp/dspy) for LLM management
- [Hydra](https://hydra.cc/) for configuration management