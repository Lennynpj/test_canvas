#!/usr/bin/env python3
"""
main.py  –  CLI entry-point for the Graph RAG system
-----------------------------------------------------

Usage examples
--------------

# Index a directory of .txt files then start an interactive chat:
python -m graph_rag.main --index-dir ./docs --chat

# Index a single file:
python -m graph_rag.main --index-file ./my_doc.txt --chat

# One-shot question:
python -m graph_rag.main --load-index graph_rag_index.json \
    --query "What does SUEZ do?"

# Use a specific Ollama model:
python -m graph_rag.main --index-dir ./docs --model llama3.2 --chat

# Debug: show retrieved chunks without generating an answer:
python -m graph_rag.main --load-index graph_rag_index.json \
    --retrieve-only "water treatment process"
"""

import argparse
import sys

from .pipeline import GraphRAGPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Local SLM + Graph RAG  –  question-answering on your documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── source ──────────────────────────────────────────────────────────
    src = p.add_mutually_exclusive_group()
    src.add_argument("--index-dir",  metavar="DIR",  help="Index all .txt files in this directory")
    src.add_argument("--index-file", metavar="FILE", help="Index a single text file")
    src.add_argument("--load-index", metavar="JSON", help="Load a previously saved index file")

    # ── query mode ──────────────────────────────────────────────────────
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--chat",          action="store_true", help="Interactive chat mode (default if source given)")
    mode.add_argument("--query",         metavar="Q",         help="Single one-shot question")
    mode.add_argument("--retrieve-only", metavar="Q",         help="Show retrieved chunks (no LLM)")

    # ── SLM options ──────────────────────────────────────────────────────
    p.add_argument("--backend", default="auto",  choices=["auto", "ollama", "llama_cpp", "transformers"])
    p.add_argument("--model",   default=None,    help="Model name/path (e.g. phi3, mistral, llama3.2)")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens",  type=int,   default=512)

    # ── retrieval options ────────────────────────────────────────────────
    p.add_argument("--top-k",        type=int, default=3,  help="Seed chunks from embedding search")
    p.add_argument("--graph-depth",  type=int, default=2,  help="BFS depth for graph expansion")
    p.add_argument("--max-context",  type=int, default=6,  help="Max chunks injected into prompt")
    p.add_argument("--no-embeddings", action="store_true", help="Disable sentence-transformer embeddings")

    # ── persistence ──────────────────────────────────────────────────────
    p.add_argument("--save-index", default="graph_rag_index.json", metavar="JSON",
                   help="Where to save the index after building (default: graph_rag_index.json)")

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args   = parser.parse_args(argv)

    # ── build pipeline ──────────────────────────────────────────────────
    rag = GraphRAGPipeline(
        slm_backend      = args.backend,
        slm_model        = args.model,
        temperature      = args.temperature,
        max_tokens       = args.max_tokens,
        use_embeddings   = not args.no_embeddings,
        top_k            = args.top_k,
        graph_depth      = args.graph_depth,
        max_context_chunks = args.max_context,
        graph_save_path  = args.save_index,
    )

    # ── load / build index ──────────────────────────────────────────────
    if args.load_index:
        rag.load_index(args.load_index)
    elif args.index_dir:
        rag.index_directory(args.index_dir).build_index()
    elif args.index_file:
        from pathlib import Path
        text = Path(args.index_file).read_text(encoding="utf-8")
        rag.index_text(Path(args.index_file).stem, text).build_index()
    else:
        # Demo mode: index a tiny built-in corpus
        print("[Demo] No source provided – indexing built-in demo corpus …\n")
        _load_demo(rag)

    # ── run query / chat ────────────────────────────────────────────────
    if args.retrieve_only:
        chunks = rag.retrieve_only(args.retrieve_only)
        print(f"\n[Retrieved {len(chunks)} chunk(s) for: '{args.retrieve_only}']\n")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i}] doc={c['doc_id']}  score={c['score']:.3f}  source={c['source']}")
            print(f"      {c['text'][:200]!r}\n")
        return

    if args.query:
        print(f"\nQuestion: {args.query}\n")
        print("Answer:", end=" ", flush=True)
        for token in rag.query(args.query, stream=True):
            print(token, end="", flush=True)
        print()
        return

    # Default: interactive chat
    _interactive_chat(rag)


def _interactive_chat(rag: GraphRAGPipeline) -> None:
    print("\n" + "="*60)
    print("  Local SLM + Graph RAG  –  Interactive Chat")
    print("  Type 'quit' or Ctrl-C to exit, '!debug <q>' to show chunks")
    print("="*60 + "\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break
        if user_input.startswith("!debug "):
            q = user_input[7:]
            chunks = rag.retrieve_only(q)
            for c in chunks:
                print(f"  [{c['source']}] {c['doc_id']} score={c['score']:.3f}: {c['text'][:120]!r}")
            continue

        print("\nAssistant: ", end="", flush=True)
        for token in rag.query(user_input, stream=True):
            print(token, end="", flush=True)
        print("\n")


def _load_demo(rag: GraphRAGPipeline) -> None:
    """Built-in demo corpus so the tool works out-of-the-box."""
    demo_docs = {
        "water_treatment": """
            Water treatment is the process of improving water quality to make it
            suitable for a specific end-use. SUEZ is a global leader in water and
            waste management. Treatment steps include coagulation, sedimentation,
            filtration, and disinfection with chlorine or UV. Advanced treatment
            uses reverse osmosis membranes for desalination. Wastewater treatment
            plants process sewage to recover clean water and biogas.
        """,
        "graph_rag": """
            Graph RAG (Retrieval-Augmented Generation) combines knowledge graphs
            with large language models. Entities are extracted from documents and
            linked in a graph. At query time, the most relevant entities and their
            neighbours are retrieved and injected as context into the LLM prompt.
            This approach improves factual accuracy and allows multi-hop reasoning
            across related documents.
        """,
        "slm_overview": """
            A Small Language Model (SLM) is a language model with fewer than
            10 billion parameters designed to run efficiently on consumer hardware.
            Examples include Microsoft Phi-3, Mistral 7B, LLaMA 3.2, and Gemma 2.
            SLMs can be run locally using frameworks such as Ollama, llama.cpp,
            or HuggingFace Transformers. Local inference ensures data privacy
            because no data leaves the user's machine.
        """,
    }
    for doc_id, text in demo_docs.items():
        rag.index_text(doc_id, text)
    rag.build_index()


if __name__ == "__main__":
    main()
