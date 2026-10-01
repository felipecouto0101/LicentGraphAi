"""RAG components, loaded only when explicitly requested."""

from importlib import import_module

_EXPORTS = {
    "create_licit_graph_workflow": "langgraph_workflow",
    "run_licit_graph_pipeline": "langgraph_workflow",
    "Node1ReaderChunker": "node_1_reader_chunker",
    "Node2EmbeddingGenerator": "node_2_embeddings",
    "Node3RequirementAnalyzer": "node_3_analyzer",
    "Node4DocumentGenerator": "node_4_document_generator",
    "PDFReader": "pdf_reader",
    "TextChunker": "text_chunker",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
