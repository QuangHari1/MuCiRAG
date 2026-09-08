"""Pluggable embedding providers used by offline and online stages."""

from .providers import EmbeddingProvider, create_embedding_provider

__all__ = ["EmbeddingProvider", "create_embedding_provider"]
