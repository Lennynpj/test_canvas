"""
retriever.py
------------
Graph-aware retriever: given a user query, find the most relevant chunks
by walking the knowledge graph from the most semantically similar entities.

Strategy
--------
1. Encode the query with sentence-transformers (same model as GraphBuilder).
2. Cosine-similarity search over chunk embeddings   → top-K seed chunks.
3. From each seed chunk, expand along graph edges (BFS, depth=2) to gather
   related entity nodes and neighbouring chunks      → context window.
4. Return a ranked, deduplicated list of text chunks ready to inject into
   the LLM prompt.
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


def _cosine(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


class GraphRetriever:
    """
    Retrieves relevant text chunks from the knowledge graph.

    Parameters
    ----------
    graph : nx.Graph
        Built by GraphBuilder.
    chunks : list[dict]
        Chunk records from GraphBuilder (fields: id, text, embedding, doc_id).
    embedding_model : str
        Must match the model used during graph construction.
    top_k : int
        Number of seed chunks to retrieve by embedding similarity.
    graph_depth : int
        BFS depth for graph expansion from seed entities.
    max_context_chunks : int
        Hard cap on total chunks returned to the LLM.
    """

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        graph: nx.Graph,
        chunks: list[dict[str, Any]],
        embedding_model: str = EMBEDDING_MODEL,
        top_k: int = 3,
        graph_depth: int = 2,
        max_context_chunks: int = 8,
    ):
        self.graph  = graph
        self.chunks = {c["id"]: c for c in chunks}   # id → chunk
        self.top_k  = top_k
        self.graph_depth = graph_depth
        self.max_context_chunks = max_context_chunks

        self._embedder = None
        if _ST_AVAILABLE:
            self._embedder = SentenceTransformer(embedding_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """
        Return a ranked list of chunk dicts relevant to *query*.
        Each dict: {id, doc_id, text, score, source}
        """
        if self._embedder and any(c.get("embedding") for c in self.chunks.values()):
            seeds = self._embedding_search(query)
        else:
            seeds = self._keyword_search(query)

        # expand via graph
        expanded = self._graph_expand(seeds)

        # deduplicate, keep highest-score version of each chunk
        seen: dict[str, dict] = {}
        for chunk in seeds + expanded:
            cid = chunk["id"]
            if cid not in seen or chunk["score"] > seen[cid]["score"]:
                seen[cid] = chunk

        ranked = sorted(seen.values(), key=lambda c: c["score"], reverse=True)
        return ranked[: self.max_context_chunks]

    def format_context(self, query: str) -> str:
        """Return a formatted context string ready to embed into a prompt."""
        results = self.retrieve(query)
        if not results:
            return "(No relevant context found in the knowledge graph.)"
        lines = [f"[{i+1}] (doc={r['doc_id']}, score={r['score']:.2f})\n{r['text']}"
                 for i, r in enumerate(results)]
        return "\n\n---\n\n".join(lines)

    # ------------------------------------------------------------------
    # Internal: embedding search
    # ------------------------------------------------------------------

    def _embedding_search(self, query: str) -> list[dict[str, Any]]:
        q_emb = self._embedder.encode(query).tolist()
        scored = []
        for chunk in self.chunks.values():
            emb = chunk.get("embedding")
            if emb:
                score = _cosine(q_emb, emb)
                scored.append({**chunk, "score": score, "source": "embedding"})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: self.top_k]

    # ------------------------------------------------------------------
    # Internal: fallback keyword search (no embeddings)
    # ------------------------------------------------------------------

    def _keyword_search(self, query: str) -> list[dict[str, Any]]:
        query_words = set(query.lower().split())
        scored = []
        for chunk in self.chunks.values():
            text_words = set(chunk["text"].lower().split())
            overlap = len(query_words & text_words)
            if overlap:
                scored.append({**chunk, "score": overlap / len(query_words), "source": "keyword"})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: self.top_k]

    # ------------------------------------------------------------------
    # Internal: graph expansion (BFS from entity nodes)
    # ------------------------------------------------------------------

    def _graph_expand(self, seeds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        From each seed chunk, walk the graph up to `graph_depth` hops.
        Collect neighbouring chunk nodes and assign a decayed score.
        """
        expanded: list[dict[str, Any]] = []
        seed_ids = {s["id"] for s in seeds}

        for seed in seeds:
            if seed["id"] not in self.graph:
                continue
            # BFS
            visited = {seed["id"]}
            frontier = [(seed["id"], seed["score"], 0)]
            while frontier:
                node_id, score, depth = frontier.pop(0)
                if depth >= self.graph_depth:
                    continue
                for neighbour in self.graph.neighbors(node_id):
                    if neighbour in visited:
                        continue
                    visited.add(neighbour)
                    node_data = self.graph.nodes.get(neighbour, {})
                    if node_data.get("type") == "chunk" and neighbour not in seed_ids:
                        chunk = self.chunks.get(neighbour)
                        if chunk:
                            decay = 0.6 ** (depth + 1)
                            expanded.append({**chunk, "score": score * decay, "source": "graph"})
                    # keep expanding through entity nodes
                    edge_data = self.graph.edges.get((node_id, neighbour), {})
                    if node_data.get("type") == "entity":
                        frontier.append((neighbour, score * 0.8, depth + 1))

        return expanded
