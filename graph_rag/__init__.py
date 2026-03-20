"""
Graph RAG with Local SLM
========================
A Retrieval-Augmented Generation system using a knowledge graph
and a locally-running Small Language Model (via Ollama).
"""

from .graph_builder import GraphBuilder
from .retriever import GraphRetriever
from .slm import LocalSLM
from .pipeline import GraphRAGPipeline

__all__ = ["GraphBuilder", "GraphRetriever", "LocalSLM", "GraphRAGPipeline"]
__version__ = "0.1.0"
