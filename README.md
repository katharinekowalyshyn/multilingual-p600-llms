# Multilingual P600 Extension Experiments

This directory contains the implementation of four experiments for analyzing P600-like effects across 14 languages in large language models.

## Overview

The multilingual extension tests how garden-path disambiguation patterns occur across typologically diverse languages, examining:
- Cross-linguistic P600 manifestation
- Resource-level processing differences (high/mid/low-resource languages)
- Model architecture effects in multilingual contexts

## Setup

### Prerequisites

1. **Ollama**: All models are accessed via Ollama API. Make sure Ollama is installed and running:
   ```bash
   # Install Ollama from https://ollama.ai
   # Start Ollama service
   ollama serve
   ```

2. **Pull Required Models**:
   ```bash
   ollama pull llama3.1
   ollama pull gemma2:7b
   ollama pull gemma2:2b
   # Add other models as needed
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Edit `src/conf/config.yaml` to:
- Enable/disable models and experiments
- Configure language lists
- Set output directories
- Adjust experiment parameters

## Dataset

**IMPORTANT**: Before running experiments, you need garden-path sentences for all 14 languages.

The dataset should be a CSV file with columns:
- `language`: Language name
- `sentence`: Garden-path sentence text
- `ambiguity_type`: Type of ambiguity (attachment_ambiguity, case_marking_ambiguity, relative_clause_ambiguity)
- `sentence_type`: Either "gardenpath" or "control"

To generate the dataset:
```bash
python src/multilingual_experiments/dataset_generation.py
```

Or manually create `src/multilingual_experiments/datasets/multilingual_gardenpath_dataset.csv`

## Experiments

### Experiment 1: Semantic Interpretation Tracking

Tracks how model interpretations change incrementally as sentences are processed chunk-by-chunk using syntactic units.

**Key measures**:
- Does disambiguation happen at same chunk position across languages?
- Resource-level differences in lingering misinterpretations
- Language-specific features that aid/hinder reanalysis

**Run**:
```bash
python src/multilingual_experiments/exp1_semantic_tracking.py
```

**Output**: `src/multilingual_experiments/exp1_semantic_tracking/results/semantic_tracking_results.csv`

### Experiment 2: Parse Tree Extraction

Extracts incremental parse trees using Manning et al. (2020) technique.

**Status**: Skeleton implemented, needs full parse tree extraction logic.

### Experiment 3: Attention Visualization

Maps which attention heads are sensitive to disambiguation across languages.

**Status**: Skeleton implemented, needs Ollama integration for attention extraction.

### Experiment 4: Unambiguous Controls

Compares garden-path sentences with matched unambiguous controls.

**Status**: To be implemented.

## Languages

### High-Resource (8 languages)
- Chinese, Spanish, French, German, Portuguese, Italian, Japanese, Korean

### Mid-Resource (2 languages)
- Arabic, Indonesian

### Low-Resource (4 languages)
- Hindi, Bengali, Swahili, Yoruba

## Models

All models are accessed via Ollama:
- **LLaMA 3.1**: `llama3.1`
- **Gemma 7B**: `gemma2:7b`
- **Gemma 2B**: `gemma2:2b`
- **GPT-OSS-20b**: (when available)

## Chunking Strategy

Experiments use **syntactic chunking** by default, which breaks sentences into phrases/clauses based on:
- Punctuation marks
- Conjunctions
- Relative pronouns
- Language-specific boundary markers

This ensures chunks align with syntactic units rather than arbitrary token counts.

## Notes

- The old Neuronpedia-based analysis (`p600_incremental_analysis.py`) has been removed
- All model interactions now go through Ollama API
- Tokenizers are loaded from HuggingFace for chunking purposes, but inference is via Ollama
- Language-specific chunking rules can be added to `chunking.py` for better cross-linguistic support

