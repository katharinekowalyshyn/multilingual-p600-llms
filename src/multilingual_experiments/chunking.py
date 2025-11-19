"""
Chunking Strategies for Incremental Analysis

Provides different chunking strategies for breaking sentences into
incremental chunks for analysis:
- Syntactic chunking (phrase/clause boundaries)
- Token count chunking (fixed number of tokens)
- Word boundary chunking (word-by-word)
"""

import re
from typing import List, Tuple, Optional

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ChunkingStrategy:
    
    def chunk(self, text: str, tokenizer) -> List[Tuple[str, int, int]]:
        """
        Chunk text into incremental segments.
        
        Args:
            text: Input text to chunk
            tokenizer: Tokenizer to use for tokenization
            
        Returns:
            List of tuples: (chunk_text, start_token_idx, end_token_idx)
        """
        raise NotImplementedError


class SyntacticChunking(ChunkingStrategy):
    """
    Chunk based on syntactic boundaries (phrases, clauses).
    
    Uses heuristics to identify phrase/clause boundaries:
    - Punctuation marks (., ;, :, ,)
    - Conjunctions (and, or, but, while, etc.)
    - Relative pronouns (who, which, that, etc.)
    """
    
    # Common conjunctions and relative pronouns across languages
    BOUNDARY_MARKERS = {
        'en': [r'\b(and|or|but|while|when|where|that|which|who|whom|whose)\b'],
        'es': [r'\b(y|o|pero|mientras|cuando|donde|que|quien|cual)\b'],
        'de': [r'\b(und|oder|aber|während|wenn|wo|dass|der|die|das|welche)\b'],
        'fr': [r'\b(et|ou|mais|pendant|quand|où|que|qui|lequel)\b'],
        'zh': [r'[，。；：、]'],  # Chinese punctuation
        'ja': [r'[、。，]'],  # Japanese punctuation
        'ko': [r'[，。；：]'],  # Korean punctuation
        'ar': [r'[،؛]'],  # Arabic punctuation
        'hi': [r'[।,]'],  # Hindi punctuation
        'bn': [r'[।,]'],  # Bengali punctuation
    }
    
    def __init__(self, language: str = 'en'):
        """
        Initialize syntactic chunker.
        
        Args:
            language: Language code for boundary markers
        """
        self.language = language.lower()[:2]
        if self.language not in self.BOUNDARY_MARKERS:
            self.language = 'en'  # Default to English
    
    def chunk(self, text: str, tokenizer) -> List[Tuple[str, int, int]]:
        """Chunk text at syntactic boundaries."""
        # Tokenize full text first
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_texts = tokenizer.convert_ids_to_tokens(tokens)
        
        # Find boundary positions
        boundaries = self._find_boundaries(text)
        
        chunks = []
        start_idx = 0
        
        for boundary_pos in boundaries:
            # Find token position corresponding to character position
            char_count = 0
            token_idx = 0
            for i, token_text in enumerate(token_texts):
                # Approximate character position (rough heuristic)
                char_count += len(token_text.replace('##', '').replace('▁', ''))
                if char_count >= boundary_pos:
                    token_idx = i + 1
                    break
            
            if token_idx > start_idx:
                chunk_tokens = tokens[start_idx:token_idx]
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append((chunk_text, start_idx, token_idx))
                start_idx = token_idx
        
        # Add final chunk
        if start_idx < len(tokens):
            chunk_tokens = tokens[start_idx:]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append((chunk_text, start_idx, len(tokens)))
        
        return chunks if chunks else [(text, 0, len(tokens))]
    
    def _find_boundaries(self, text: str) -> List[int]:
        """Find syntactic boundary positions in text."""
        boundaries = []
        
        # Get patterns for this language
        patterns = self.BOUNDARY_MARKERS.get(self.language, self.BOUNDARY_MARKERS['en'])
        
        # Find punctuation boundaries
        for match in re.finditer(r'[.,;:，。；：、]', text):
            boundaries.append(match.end())
        
        # Find conjunction/relative pronoun boundaries (before the marker)
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                boundaries.append(match.start())
        
        return sorted(set(boundaries))


class TokenCountChunking(ChunkingStrategy):
    """Chunk based on fixed token count."""
    
    def __init__(self, chunk_size: int = 3):
        """
        Initialize token count chunker.
        
        Args:
            chunk_size: Number of tokens per chunk
        """
        self.chunk_size = chunk_size
    
    def chunk(self, text: str, tokenizer) -> List[Tuple[str, int, int]]:
        """Chunk text into fixed-size token chunks."""
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            end_idx = min(i + self.chunk_size, len(tokens))
            chunk_tokens = tokens[i:end_idx]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append((chunk_text, i, end_idx))
        
        return chunks if chunks else [(text, 0, len(tokens))]


class WordBoundaryChunking(ChunkingStrategy):
    """Chunk word-by-word (one word per chunk)."""
    
    def chunk(self, text: str, tokenizer) -> List[Tuple[str, int, int]]:
        """Chunk text word-by-word."""
        # Split by whitespace
        words = text.split()
        
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_texts = tokenizer.convert_ids_to_tokens(tokens)
        
        chunks = []
        token_idx = 0
        
        for word in words:
            # Find tokens corresponding to this word
            word_tokens = []
            start_idx = token_idx
            
            # Try to match word tokens (heuristic)
            remaining_word = word.lower()
            while token_idx < len(token_texts) and remaining_word:
                token_text = token_texts[token_idx].replace('##', '').replace('▁', '').lower()
                if remaining_word.startswith(token_text):
                    word_tokens.append(tokens[token_idx])
                    remaining_word = remaining_word[len(token_text):]
                    token_idx += 1
                else:
                    break
            
            if word_tokens:
                chunk_text = tokenizer.decode(word_tokens, skip_special_tokens=True)
                chunks.append((chunk_text, start_idx, token_idx))
            else:
                # Fallback: use single token
                if token_idx < len(tokens):
                    chunk_text = tokenizer.decode([tokens[token_idx]], skip_special_tokens=True)
                    chunks.append((chunk_text, token_idx, token_idx + 1))
                    token_idx += 1
        
        return chunks if chunks else [(text, 0, len(tokens))]


def get_chunking_strategy(strategy_name: str, **kwargs) -> ChunkingStrategy:
    """
    Factory function to get chunking strategy.
    
    Args:
        strategy_name: Name of strategy ("syntactic", "token_count", "word_boundary")
        **kwargs: Strategy-specific arguments
        
    Returns:
        ChunkingStrategy instance
    """
    if strategy_name == "syntactic":
        language = kwargs.get('language', 'en')
        return SyntacticChunking(language=language)
    elif strategy_name == "token_count":
        chunk_size = kwargs.get('chunk_size', 3)
        return TokenCountChunking(chunk_size=chunk_size)
    elif strategy_name == "word_boundary":
        return WordBoundaryChunking()
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy_name}")

