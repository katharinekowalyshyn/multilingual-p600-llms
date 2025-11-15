"""
Model Loading and Management for Multilingual P600 Experiments

Provides a unified interface for loading different model architectures
via Ollama API.

All models are accessed through Ollama API for consistency.
"""

import os
import requests
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
import warnings
import numpy as np
from transformers import AutoTokenizer

class ModelLoader:
    """
    Unified model loader for Ollama API.
    
    All models are accessed through Ollama API for consistency.
    Models should be pulled/available in Ollama before use.
    """
    
    def __init__(self, model_name: str, ollama_base_url: str = "http://localhost:11434", **kwargs):
        """
        Initialize Ollama model loader.
        
        Args:
            model_name: Ollama model name (e.g., "llama3.1", "gemma2:7b")
            ollama_base_url: Base URL for Ollama API
            **kwargs: Additional arguments (unused for now)
        """
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.kwargs = kwargs
        self.tokenizer = None
        
        # Try to load a tokenizer for the model (for chunking/tokenization)
        # We'll use a generic tokenizer or model-specific one if available
        self._load_tokenizer()
        
    def _load_tokenizer(self):
        """Load tokenizer for tokenization purposes."""

        tokenizer_name = None
        if "llama" in self.model_name.lower():
            tokenizer_name = "meta-llama/Llama-3.1-8B-Instruct"
        elif "gemma" in self.model_name.lower():
            if "7b" in self.model_name.lower() or "7" in self.model_name:
                tokenizer_name = "google/gemma:7b"
            else:
                tokenizer_name = "google/gemma:2b"
        
        if tokenizer_name:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                print(f"✓ Loaded tokenizer: {tokenizer_name}")
            except Exception as e:
                print(f"Warning: Could not load tokenizer {tokenizer_name}: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None
    
    def load(self):
        """Verify model is available in Ollama."""
        
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                if self.model_name in model_names:
                    print(f"✓ Model {self.model_name} available in Ollama")
                    return self.model_name, self.tokenizer
                else:
                    print(f"Warning: Model {self.model_name} not found in Ollama. Pull the model with: ollama pull {self.model_name}.")
                    return self.model_name, self.tokenizer
            else:
                raise ConnectionError(f"Ollama API returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.ollama_base_url}. "
                "Make sure Ollama is running."
            )
        except Exception as e:
            print(f"Warning: Could not verify Ollama model: {e}")
            return self.model_name, self.tokenizer
    
    def tokenize(self, text: str, add_bos: bool = True) -> List[int]:
        """Tokenize text using the model's tokenizer or Ollama API."""
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if add_bos and hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id:
                tokens = [self.tokenizer.bos_token_id] + tokens
            return tokens
        else:
            # Fallback: use Ollama API for tokenization
            try:
                response = requests.post(
                    f"{self.ollama_base_url}/api/tokenize",
                    json={"model": self.model_name, "prompt": text},
                    timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    return result.get("tokens", [])
                else:
                    # Simple fallback: split by spaces
                    return text.split()
            except Exception as e:
                print(f"Warning: Tokenization failed: {e}. Using simple split.")
                return text.split()
    
    def generate(self, prompt: str, options: Optional[Dict] = None) -> str:
        """
        Generate text using Ollama API.
        
        Args:
            prompt: Input prompt
            options: Optional generation parameters (temperature, top_p, etc.)
            
        Returns:
            Generated text
        """
        if options is None:
            options = {}
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            **options
        }
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=60
            )
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                raise RuntimeError(f"Ollama API returned status {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling Ollama API: {e}")
    
    def embeddings(self, prompt: str) -> np.ndarray:
        """
        Get embeddings for text using Ollama API.
        
        Args:
            prompt: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt
        }
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                return np.array(embedding)
            else:
                raise RuntimeError(f"Ollama API returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling Ollama embeddings API: {e}")


def load_model_from_config(model_config: Dict[str, Any]) -> tuple:
    """
    Load a model from configuration dictionary.
    
    Args:
        model_config: Dictionary with 'model_name' (Ollama model name), 'ollama_base_url', etc.
        
    Returns:
        Tuple of (model_name, tokenizer, ModelLoader instance)
    """
    loader = ModelLoader(
        model_name=model_config['model_name'],
        ollama_base_url=model_config.get('ollama_base_url', 'http://localhost:11434'),
        **model_config.get('kwargs', {})
    )
    model_name, tokenizer = loader.load()
    return model_name, tokenizer, loader

