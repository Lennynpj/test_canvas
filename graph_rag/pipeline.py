"""
pipeline.py
-----------
Main Graph RAG pipeline: ties together GraphBuilder, GraphRetriever, LocalSLM.

Flow
----
  Documents ──► GraphBuilder ──► KnowledgeGraph
                                       │
  Query ─────────────────────► GraphRetriever ──► Context chunks
                                       │
  Context + Query ──────────► LocalSLM ──────────► Answer
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

from .graph_builder import GraphBuilder
from .retriever import GraphRetriever
from .slm import LocalSLM


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly
on the provided context. If the context does not contain enough information to
answer the question, say so honestly. Be concise and accurate."""

RAG_PROMPT_TEMPLATE = """Context extracted from the knowledge graph:
{context}

---

Question: {question}

Answer based on the context above:"""


class GraphRAGPipeline:
    """
    End-to-end Graph RAG pipeline.

    Parameters
    ----------
    slm_backend : str
        "auto" | "ollama" | "llama_cpp" | "transformers"
    slm_model : str | None
        Model name / path for the SLM.
    use_embeddings : bool
        Whether to use sentence-transformer embeddings for retrieval.
    top_k : int
        Number of seed chunks retrieved per query.
    graph_depth : int
        BFS depth for graph expansion.
    max_context_chunks : int
        Max chunks injected into the LLM prompt.
    graph_save_path : str | None
        If set, the graph is persisted here after indexing.
    """

    def __init__(
        self,
        slm_backend: str = "auto",
        slm_model: str | None = None,
        use_embeddings: bool = True,
        top_k: int = 3,
        graph_depth: int = 2,
        max_context_chunks: int = 6,
        graph_save_path: str | None = "graph_rag_index.json",
        **slm_kwargs,
    ):
        self._builder: GraphBuilder | None = None
        self._retriever: GraphRetriever | None = None
        self._slm: LocalSLM | None = None

        self._use_embeddings  = use_embeddings
        self._top_k           = top_k
        self._graph_depth     = graph_depth
        self._max_ctx         = max_context_chunks
        self._graph_save_path = graph_save_path
        self._slm_backend     = slm_backend
        self._slm_model       = slm_model
        self._slm_kwargs      = slm_kwargs

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_text(self, doc_id: str, text: str) -> "GraphRAGPipeline":
        """Add a single text document to the index."""
        self._ensure_builder()
        self._builder.add_document(doc_id, text)
        self._retriever = None   # invalidate retriever
        return self

    def index_directory(self, directory: str | Path, glob: str = "*.txt") -> "GraphRAGPipeline":
        """Index all matching files in a directory."""
        self._ensure_builder()
        self._builder.add_documents_from_dir(directory, glob)
        self._retriever = None
        return self

    def build_index(self) -> "GraphRAGPipeline":
        """Finalise the index (and optionally persist it)."""
        self._ensure_builder()
        self._builder.build()
        if self._graph_save_path:
            self._builder.save(self._graph_save_path)
        self._init_retriever()
        return self

    def load_index(self, path: str | Path) -> "GraphRAGPipeline":
        """Load a previously saved index (skips re-indexing)."""
        self._builder = GraphBuilder.load(path)
        self._init_retriever()
        return self

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str, stream: bool = False) -> str | Generator[str, None, None]:
        """
        Answer a question using the indexed knowledge graph.

        Parameters
        ----------
        question : str
        stream : bool
            If True, returns a generator that yields tokens.

        Returns
        -------
        str or Generator[str]
        """
        if self._retriever is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        self._ensure_slm()
        context = self._retriever.format_context(question)
        prompt  = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        if stream:
            return self._slm.stream(prompt, system=SYSTEM_PROMPT)
        return self._slm.generate(prompt, system=SYSTEM_PROMPT)

    def retrieve_only(self, question: str) -> list[dict[str, Any]]:
        """Return raw retrieved chunks without generating an answer."""
        if self._retriever is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")
        return self._retriever.retrieve(question)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_builder(self):
        if self._builder is None:
            self._builder = GraphBuilder(use_embeddings=self._use_embeddings)

    def _ensure_slm(self):
        if self._slm is None:
            self._slm = LocalSLM(
                backend=self._slm_backend,
                model=self._slm_model,
                **self._slm_kwargs,
            )

    def _init_retriever(self):
        graph  = self._builder.graph
        chunks = self._builder.chunks
        self._retriever = GraphRetriever(
            graph=graph,
            chunks=chunks,
            top_k=self._top_k,
            graph_depth=self._graph_depth,
            max_context_chunks=self._max_ctx,
        )
        print(f"[Pipeline] Retriever ready. Graph: "
              f"{graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges, "
              f"{len(chunks)} chunks")
