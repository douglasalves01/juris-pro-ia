"""Módulos de ML para extração e pré-processamento de texto jurídico."""

from .preprocessor import TextPreprocessor
from .text_extractor import DocumentMetadata, TextExtractor

__all__ = ["DocumentMetadata", "TextExtractor", "TextPreprocessor"]
