- **New in this update**: Added a Neuronpedia-powered SAE analysis of P600 sentences (`src/incremental_analysis/p600_analysis.py`).

# multilingual-gardenpath-llms

This project is designed to generate and evaluate multilingual garden path sentences using large language models (LLMs). It is organized in three main parts:
1. **Dataset Generation**: Automatically generate datasets of garden path sentences in multiple languages using an LLM.
2. **P600 Sentence Processing**: Process and classify P600 sentences for grammaticality using LLMs.
3. **SAE-based Internal Analysis (Neuronpedia)**: Analyze how Sparse Autoencoder (SAE) features change for control vs P600 sentences using remote SAEs via the Neuronpedia API.

## Features
- Multilingual support (easily configurable languages)
- Modular, experiment-friendly codebase using Hydra for configuration
- LLM output management with [dspy](https://github.com/stanfordnlp/dspy)
- Utility functions for experiment pipelines
- Progress bar for dataset generation
- Ensures unique garden path sentences per language
- P600 sentence grammaticality classification with binary output (0/1)
- Neuronpedia API integration for SAE feature analysis (no heavy local models)

## Quickstart

- Create environment
```bash
conda create -y -n gardenpath python=3.12
conda activate gardenpath
```

- Install core deps for generation/processing
```bash
pip install hydra-core dspy-ai pandas tqdm
```

- Install deps for SAE analysis (Neuronpedia API)
```bash
pip install requests numpy matplotlib seaborn tqdm
```

- Set API keys (as needed)
```bash
# For dataset generation / grammaticality via LLMs (if applicable)
export OPENAI_API_KEY=sk-...yourkey...

# For Neuronpedia SAE API
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

### SAE-Based P600 Analysis (Neuronpedia)

Analyze how SAE features change between control sentences and P600 sentences, using the Neuronpedia API (no local model loading).

- Inputs (CSV):
  - `src/grammar_analysis/control_gardenpath_sample.csv`
  - `src/grammar_analysis/p600_sample.csv`

- Script:
```bash
cd src/incremental_analysis
export NEURONPEDIA_API_KEY=your_neuronpedia_key_here
python p600_analysis.py
```

- What it does:
  - Calls Neuronpedia `POST /api/activation/new` to extract SAE features for `gpt2-small`
  - Compares per-feature means between Control and P600
  - Pads variable-length sequences (different token counts) to a common length
  - Produces plots and CSVs
  - Includes overall progress and ETA per submodule

- Outputs (default):
  - `src/incremental_analysis/results/feature_analysis.png`
  - `src/incremental_analysis/results/feature_analysis_results.csv`
  - Log output with progress, feature shapes, and timing

- API usage and limits:
  - Default configuration processes 8 submodules across 2 datasets × 25 sentences ≈ ~400 API calls (< 1000/hour limit)
  - To change submodules, edit `all_submodule_names` near the top of `src/incremental_analysis/p600_analysis.py` (e.g., add/remove `attn_k`, `mlp_k`, `resid_k` indices)

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
│   └── results/                      # Output directory (grammaticality)
├── incremental_analysis/
│   ├── p600_analysis.py              # Neuronpedia SAE analysis (control vs P600)
│   └── results/                      # feature_analysis.png / feature_analysis_results.csv
├── prompts/                           # Prompt templates
└── utils.py                          # Utility functions
```

## Utilities
- Common helper functions are in `src/utils.py` (e.g., for saving DataFrames).
- Configuration testing and validation scripts for each component.

## Extending the Project
- Add new utility functions to `src/utils.py` as needed.
- The P600 processing system can be extended to handle different sentence types or languages.
- The SAE analysis can be extended by adding/removing submodules or switching Neuronpedia models if available.

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

### SAE Analysis (Neuronpedia)
- Ensure `NEURONPEDIA_API_KEY` is set
- If you get `429`, reduce submodules or wait for limit reset
- Confirm you are on `https://www.neuronpedia.org` (not the old `api.` host)

## License
MIT

## Acknowledgments
- [DSPy](https://github.com/stanfordnlp/dspy) for LLM management
- [Hydra](https://hydra.cc/) for configuration management
- [Neuronpedia](https://www.neuronpedia.org) for remote SAE features

## Running P600 Incremental Analysis (Gemma + SAELens)

### Local
```bash
pip install -r requirements.txt
python src/incremental_analysis/gemma_p600_incremental_analysis.py \
  incremental_p600.model.gemma_checkpoint=google/gemma-2-2b-it \
  incremental_p600.model.sae_release_id=gemma-2-2b-res-jb-l20 \
  incremental_p600.batch_size=8
```

### Runpod
1. Ensure your repo is linked in Runpod and set to build from Dockerfile.
2. Build the image from the root (Runpod will detect Dockerfile).
3. Launch a GPU pod with at least 12GB VRAM (larger recommended for bigger Gemma checkpoints).
4. Optionally set environment variables to override Hydra config:
   - `GEMMA_CHECKPOINT`, `SAE_RELEASE_ID`, `BATCH_SIZE`, `CONTROL_CSV`, `P600_CSV`, `OUTPUT_DIR`, `SAVE_PER_FEATURE`
5. Start the container command:
```bash
bash runpod_start.sh
```
Outputs will be written to `incremental_p600.output_dir` (default `src/incremental_analysis/results`).