from setuptools import setup, find_packages

setup(
    name="graph-rag-slm",
    version="0.1.0",
    description="Local SLM + Graph RAG pipeline",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "networkx>=3.2",
        "sentence-transformers>=2.7",
        "numpy>=1.26",
        "ollama>=0.2",
    ],
    extras_require={
        "spacy":   ["spacy>=3.7"],
        "llama":   ["llama-cpp-python>=0.2.56"],
        "hf":      ["transformers>=4.40", "torch>=2.2", "accelerate>=0.30"],
    },
    entry_points={
        "console_scripts": [
            "graph-rag=graph_rag.main:main",
        ]
    },
)
