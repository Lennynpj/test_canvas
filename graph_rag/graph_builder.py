"""
graph_builder.py
----------------
Builds a knowledge graph from raw text documents.

Architecture:
  - Nodes  : entities extracted from text (nouns, named entities)
  - Edges  : relationships between entities (co-occurrence + explicit triples)
  - Each node stores the source chunks so we can retrieve original text.

Dependencies: networkx, spacy (en_core_web_sm), sentence-transformers
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import networkx as nx

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


class GraphBuilder:
    """
    Builds and persists a knowledge graph from documents.

    Usage
    -----
    builder = GraphBuilder()
    builder.add_document("my_doc", "Alice works at SUEZ. Bob manages Alice.")
    builder.add_document("doc2",   "SUEZ is a water management company.")
    graph   = builder.build()
    builder.save("knowledge_graph.json")
    """

    # Small embedding model that runs fully offline after first download
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, use_embeddings: bool = True, chunk_size: int = 200):
        self.graph: nx.Graph = nx.Graph()
        self.chunks: list[dict[str, Any]] = []   # {id, doc_id, text, embedding}
        self.chunk_size = chunk_size
        self.use_embeddings = use_embeddings and _ST_AVAILABLE

        self._nlp = None
        self._embedder = None

        if self.use_embeddings:
            print(f"[GraphBuilder] Loading embedding model '{self.EMBEDDING_MODEL}' …")
            self._embedder = SentenceTransformer(self.EMBEDDING_MODEL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_document(self, doc_id: str, text: str) -> None:
        """Chunk a document, extract entities and add them to the graph."""
        chunks = self._chunk_text(text)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{idx}"
            embedding = (
                self._embedder.encode(chunk).tolist()
                if self.use_embeddings
                else None
            )
            self.chunks.append({"id": chunk_id, "doc_id": doc_id, "text": chunk, "embedding": embedding})
            entities = self._extract_entities(chunk)
            self._add_to_graph(chunk_id, chunk, entities)

    def add_documents_from_dir(self, directory: str | Path, glob: str = "*.txt") -> None:
        """Load all text files in a directory."""
        for path in Path(directory).glob(glob):
            text = path.read_text(encoding="utf-8", errors="ignore")
            self.add_document(path.stem, text)
            print(f"[GraphBuilder] Added '{path.name}' ({len(text)} chars)")

    def build(self) -> nx.Graph:
        """Return the constructed knowledge graph."""
        print(f"[GraphBuilder] Graph: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges, {len(self.chunks)} chunks")
        return self.graph

    def save(self, path: str | Path) -> None:
        """Persist graph + chunks to a JSON file."""
        data = {
            "graph": nx.node_link_data(self.graph),
            "chunks": [
                {**c, "embedding": c["embedding"]} for c in self.chunks
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[GraphBuilder] Saved to '{path}'")

    @classmethod
    def load(cls, path: str | Path) -> "GraphBuilder":
        """Restore a previously saved GraphBuilder (no re-indexing needed)."""
        instance = cls.__new__(cls)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        instance.graph = nx.node_link_graph(data["graph"])
        instance.chunks = data["chunks"]
        instance.use_embeddings = any(c["embedding"] is not None for c in instance.chunks)
        instance._nlp = None
        instance._embedder = None
        if instance.use_embeddings and _ST_AVAILABLE:
            instance._embedder = SentenceTransformer(cls.EMBEDDING_MODEL)
        print(f"[GraphBuilder] Loaded from '{path}': "
              f"{instance.graph.number_of_nodes()} nodes, {len(instance.chunks)} chunks")
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        overlap = self.chunk_size // 4
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunks.append(" ".join(words[start:end]))
            start += self.chunk_size - overlap
        return chunks or [text]

    def _extract_entities(self, text: str) -> list[str]:
        """
        Extract entities using spaCy if available, otherwise use simple regex
        heuristic (capitalised words / noun phrases).
        """
        if _SPACY_AVAILABLE:
            if self._nlp is None:
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    self._nlp = False   # mark unavailable so we don't retry
            if self._nlp:
                doc = self._nlp(text)
                entities = [ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 1]
                # also add noun chunks (deduplicated)
                noun_chunks = [nc.root.text.lower() for nc in doc.noun_chunks]
                return list(dict.fromkeys(entities + noun_chunks))

        # Fallback: capitalised words / sequences as proxy for named entities
        return list(dict.fromkeys(re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", text)))

    def _add_to_graph(self, chunk_id: str, text: str, entities: list[str]) -> None:
        """Add entities as nodes and co-occurrence edges to the graph."""
        # Add chunk node
        self.graph.add_node(chunk_id, type="chunk", text=text[:120])

        for entity in entities:
            if not self.graph.has_node(entity):
                self.graph.add_node(entity, type="entity", mention_count=0)
            self.graph.nodes[entity]["mention_count"] = (
                self.graph.nodes[entity].get("mention_count", 0) + 1
            )
            # chunk → entity edge
            self.graph.add_edge(chunk_id, entity, relation="contains")

        # entity ↔ entity co-occurrence edges (within same chunk)
        for i, a in enumerate(entities):
            for b in entities[i + 1:]:
                if a != b:
                    if self.graph.has_edge(a, b):
                        self.graph[a][b]["weight"] = self.graph[a][b].get("weight", 1) + 1
                    else:
                        self.graph.add_edge(a, b, relation="co-occurs", weight=1)
