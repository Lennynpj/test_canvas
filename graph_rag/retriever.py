"""
retriever.py
------------
Graph-aware retriever: given a user query, finds the relevant sub-graph
and returns ALL its relations (entity→relation→entity triplets) so the
LLM can reason over the full graph structure, not just isolated chunks.

Strategy
--------
1. Encode the query → find seed entities by embedding or keyword match.
2. From each seed entity, do a full BFS with NO depth cap on entity nodes
   → collect every entity + every edge in the reachable sub-graph.
3. For each entity node, collect the source text chunks that mention it.
4. Return:
     - A list of (subject, relation, object, metadata) triplets
     - The source passages for grounding
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import networkx as nx

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────

class GraphRetriever:
    """
    Retrieves the relevant sub-graph for a query and formats it as
    structured (subject, relation, object) triplets for the LLM.

    Parameters
    ----------
    graph : nx.Graph
    chunks : list[dict]
    top_k_seeds : int
        How many seed entities to start the graph walk from.
    max_entities : int
        Hard cap on entities included in the sub-graph sent to the LLM.
    max_triplets : int
        Hard cap on triplets included in the sub-graph sent to the LLM.
    max_source_chunks : int
        Max source text passages appended for grounding.
    """

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        graph: nx.Graph,
        chunks: list[dict[str, Any]],
        embedding_model: str = EMBEDDING_MODEL,
        top_k_seeds: int = 5,
        max_entities: int = 50,
        max_triplets: int = 100,
        max_source_chunks: int = 5,
    ):
        self.graph  = graph
        self.chunks = {c["id"]: c for c in chunks}
        self.top_k_seeds       = top_k_seeds
        self.max_entities      = max_entities
        self.max_triplets      = max_triplets
        self.max_source_chunks = max_source_chunks

        self._embedder = None
        if _ST_AVAILABLE:
            self._embedder = SentenceTransformer(embedding_model)

    # ── Public API ────────────────────────────────────────────────────────

    def retrieve_graph(self, query: str) -> dict[str, Any]:
        """
        Return the full sub-graph relevant to *query*.

        Returns
        -------
        dict with keys:
          "triplets"       : list of {subject, relation, object, weight}
          "entities"       : list of {name, mention_count}
          "source_chunks"  : list of {doc_id, text, score}
          "seed_entities"  : list[str]   (entry points of the walk)
        """
        seed_entities = self._find_seed_entities(query)
        entities, triplets = self._walk_full_subgraph(seed_entities)
        source_chunks = self._collect_source_chunks(query, entities)

        return {
            "seed_entities": seed_entities,
            "entities":      entities,
            "triplets":      triplets[:self.max_triplets],
            "source_chunks": source_chunks,
        }

    def format_graph_context(self, query: str) -> str:
        """
        Return a human-readable graph context string for the LLM prompt.

        Format:
          SEED ENTITIES: Alice, SUEZ, water treatment

          GRAPH RELATIONS (all paths found):
          • Alice  ──[co-occurs, w=3]──  SUEZ
          • SUEZ   ──[co-occurs, w=2]──  water treatment
          ...

          SOURCE PASSAGES:
          [1] (doc=water_treatment) "SUEZ is a global leader…"
          ...
        """
        data = self.retrieve_graph(query)

        if not data["entities"]:
            return "(No relevant entities found in the knowledge graph.)"

        lines: list[str] = []

        # ── Seed entities ────────────────────────────────────────────────
        lines.append(f"SEED ENTITIES: {', '.join(data['seed_entities']) or 'none'}")
        lines.append("")

        # ── All graph relations ──────────────────────────────────────────
        lines.append(f"GRAPH RELATIONS ({len(data['triplets'])} relations found):")
        if data["triplets"]:
            for t in data["triplets"]:
                weight_str = f", weight={t['weight']}" if t.get("weight") else ""
                lines.append(f"  • {t['subject']}  ──[{t['relation']}{weight_str}]──  {t['object']}")
        else:
            lines.append("  (no entity-entity relations found)")
        lines.append("")

        # ── Source passages ──────────────────────────────────────────────
        if data["source_chunks"]:
            lines.append(f"SOURCE PASSAGES ({len(data['source_chunks'])} passages):")
            for i, c in enumerate(data["source_chunks"], 1):
                lines.append(f"  [{i}] (doc={c['doc_id']}, score={c['score']:.2f})")
                lines.append(f"       {c['text'][:300]}")
                lines.append("")

        return "\n".join(lines)

    # Legacy method kept for compatibility with --retrieve-only CLI flag
    def retrieve(self, query: str) -> list[dict[str, Any]]:
        data = self.retrieve_graph(query)
        return data["source_chunks"]

    # ── Seed entity detection ─────────────────────────────────────────────

    def _find_seed_entities(self, query: str) -> list[str]:
        """
        Find the entity nodes in the graph most relevant to the query.
        Uses embedding similarity if available, otherwise keyword overlap.
        """
        entity_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("type") == "entity"
        ]
        if not entity_nodes:
            return []

        if self._embedder:
            return self._embedding_seed(query, entity_nodes)
        return self._keyword_seed(query, entity_nodes)

    def _embedding_seed(self, query: str, entity_nodes: list[str]) -> list[str]:
        """Score entities by embedding similarity to the query."""
        q_emb = self._embedder.encode(query).tolist()

        # Build entity → text map from their neighbouring chunk nodes
        scored: list[tuple[str, float]] = []
        for entity in entity_nodes:
            # Represent entity by the text of its connected chunks
            neighbour_texts = []
            for nb in self.graph.neighbors(entity):
                nb_data = self.graph.nodes.get(nb, {})
                if nb_data.get("type") == "chunk":
                    chunk = self.chunks.get(nb)
                    if chunk:
                        neighbour_texts.append(chunk["text"][:200])

            if neighbour_texts:
                combined = f"{entity}. {' '.join(neighbour_texts[:3])}"
                emb = self._embedder.encode(combined).tolist()
                score = _cosine(q_emb, emb)
            else:
                # Simple string overlap as fallback for isolated entities
                q_words = set(query.lower().split())
                e_words = set(entity.lower().split())
                score = len(q_words & e_words) / (len(q_words) + 1)

            scored.append((entity, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[: self.top_k_seeds]]

    def _keyword_seed(self, query: str, entity_nodes: list[str]) -> list[str]:
        """Score entities by keyword overlap with the query."""
        q_words = set(query.lower().split())
        scored: list[tuple[str, float]] = []
        for entity in entity_nodes:
            e_words = set(entity.lower().split())
            # Also look in connected chunk texts
            chunk_words: set[str] = set()
            for nb in self.graph.neighbors(entity):
                chunk = self.chunks.get(nb)
                if chunk:
                    chunk_words.update(chunk["text"].lower().split())
            score = len(q_words & (e_words | chunk_words)) / (len(q_words) + 1)
            scored.append((entity, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[: self.top_k_seeds] if _[1] > 0]

    # ── Full sub-graph walk ───────────────────────────────────────────────

    def _walk_full_subgraph(
        self, seed_entities: list[str]
    ) -> tuple[list[dict], list[dict]]:
        """
        BFS from seed entities through ALL entity-entity edges.
        No depth cap — we want the full connected sub-graph.

        Returns
        -------
        entities : list of {name, mention_count}
        triplets : list of {subject, relation, object, weight}
        """
        visited_entities: set[str] = set()
        triplets: list[dict[str, Any]] = []

        queue = deque(seed_entities)
        for e in seed_entities:
            visited_entities.add(e)

        while queue:
            current = queue.popleft()
            if len(visited_entities) >= self.max_entities:
                break

            for neighbour in self.graph.neighbors(current):
                nb_data = self.graph.nodes.get(neighbour, {})

                if nb_data.get("type") != "entity":
                    continue  # skip chunk nodes

                edge_data = self.graph.edges.get((current, neighbour), {})
                relation  = edge_data.get("relation", "related-to")
                weight    = edge_data.get("weight", None)

                # Record triplet (avoid duplicates in both directions)
                triplet_key = tuple(sorted([current, neighbour]))
                if not any(
                    tuple(sorted([t["subject"], t["object"]])) == triplet_key
                    for t in triplets
                ):
                    triplets.append({
                        "subject":  current,
                        "relation": relation,
                        "object":   neighbour,
                        "weight":   weight,
                    })

                if neighbour not in visited_entities:
                    visited_entities.add(neighbour)
                    queue.append(neighbour)

        entities = [
            {
                "name": e,
                "mention_count": self.graph.nodes[e].get("mention_count", 0),
            }
            for e in visited_entities
        ]
        # Sort by mention count (most central entities first)
        entities.sort(key=lambda x: x["mention_count"], reverse=True)

        return entities, triplets

    # ── Source passages ───────────────────────────────────────────────────

    def _collect_source_chunks(
        self, query: str, entities: list[dict]
    ) -> list[dict[str, Any]]:
        """
        For the discovered entities, find the chunks that mention them and
        rank by relevance to the query.
        """
        candidate_chunk_ids: set[str] = set()

        for ent in entities:
            entity_name = ent["name"]
            if entity_name not in self.graph:
                continue
            for nb in self.graph.neighbors(entity_name):
                nb_data = self.graph.nodes.get(nb, {})
                if nb_data.get("type") == "chunk":
                    candidate_chunk_ids.add(nb)

        if not candidate_chunk_ids:
            return []

        # Score candidates
        if self._embedder:
            q_emb = self._embedder.encode(query).tolist()
            scored = []
            for cid in candidate_chunk_ids:
                chunk = self.chunks.get(cid)
                if not chunk:
                    continue
                emb = chunk.get("embedding")
                score = _cosine(q_emb, emb) if emb else 0.0
                scored.append({**chunk, "score": score})
        else:
            q_words = set(query.lower().split())
            scored = []
            for cid in candidate_chunk_ids:
                chunk = self.chunks.get(cid)
                if not chunk:
                    continue
                overlap = len(q_words & set(chunk["text"].lower().split()))
                scored.append({**chunk, "score": overlap / (len(q_words) + 1)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: self.max_source_chunks]
