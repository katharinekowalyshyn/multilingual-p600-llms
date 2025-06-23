# multilingual-gardenpath-llms

This project is designed to generate and evaluate multilingual garden path sentences using large language models (LLMs). It is organized in two main parts:
1. **Dataset Generation**: Automatically generate a dataset of garden path sentences in multiple languages using an LLM.
2. **LLM Evaluation**: (Planned) Evaluate how well LLMs understand and process these sentences.

## Features
- Multilingual support (easily configurable languages)
- Modular, experiment-friendly codebase using Hydra for configuration
- LLM output management with [dspy](https://github.com/stanfordnlp/dspy)
- Utility functions for experiment pipelines

## Setup

1. **Clone the repository**
2. **Create and activate a conda environment**
   ```bash
   conda create -y -n gardenpath python=3.12
   conda activate gardenpath
   ```
3. **Install dependencies**
   ```bash
   pip install hydra-core dspy-ai pandas
   ```
4. **Set your OpenAI API key**
   ```bash
   export OPENAI_API_KEY=sk-...yourkey...
   ```

## Configuration
Edit `conf/config.yaml` to set:
- `languages`: List of languages to generate sentences in
- `num_sentences`: Number of sentences per language
- `llm.model`: OpenAI model to use (e.g., `gpt-3.5-turbo`)
- `output_file`: Name of the output CSV file

## Usage
To generate a dataset:
```bash
python generate_dataset.py
```
The output CSV will be saved in a Hydra-generated output directory, with columns for `language` and `sentence`.

## Utilities
- Common helper functions are in `utils.py` (e.g., for saving DataFrames).

## Extending the Project
- Add new utility functions to `utils.py` as needed.
- The second part of the project (LLM evaluation) will be implemented in future scripts.

## License
MIT

## Acknowledgments
- [DSPy](https://github.com/stanfordnlp/dspy) for LLM management
- [Hydra](https://hydra.cc/) for configuration management