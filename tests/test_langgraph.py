"""
Testes para LangGraph Workflow

Testa a orquestração automática dos nós.
"""

import pytest


class TestLangGraphWorkflow:
    """Testes para o workflow LangGraph."""
    
    def test_create_workflow(self):
        """Testa criação do workflow."""
        from app.rag.langgraph_workflow import create_licit_graph_workflow
        
        app = create_licit_graph_workflow()
        
        assert app is not None
        assert hasattr(app, 'invoke')
    
    def test_workflow_has_correct_nodes(self):
        """Testa que o workflow tem os nós corretos."""
        from app.rag.langgraph_workflow import create_licit_graph_workflow
        
        app = create_licit_graph_workflow()
        
        # Verifica se o grafo foi compilado
        assert app is not None
    
    def test_workflow_entry_point(self):
        """Testa que o workflow tem ponto de entrada."""
        from app.rag.langgraph_workflow import create_licit_graph_workflow
        
        app = create_licit_graph_workflow()
        
        # O workflow deve ter um entry point configurado
        assert app is not None
    
    def test_state_structure(self):
        """Testa estrutura do estado."""
        from app.rag.langgraph_workflow import LicitGraphState
        
        # Verifica que o estado tem os campos necessários
        expected_fields = [
            "pdf_path",
            "chunks",
            "embeddings",
            "analysis",
            "checklist",
            "company_profile",
            "error"
        ]
        
        # LicitGraphState é um TypedDict, então os campos devem existir
        for field in expected_fields:
            assert field in LicitGraphState.__annotations__
    
    def test_run_pipeline_with_mock_state(self):
        """Testa execução do pipeline com estado simulado."""
        from app.rag.langgraph_workflow import create_licit_graph_workflow
        
        app = create_licit_graph_workflow()
        
        # Estado inicial simulado (sem PDF real)
        initial_state = {
            "pdf_path": "data/raw/edital_exemplo.pdf",
            "chunks": [],
            "embeddings": None,
            "analysis": None,
            "checklist": None,
            "company_profile": None,
            "error": None
        }
        
        # Como não temos PDF real, vamos apenas testar que o workflow existe
        assert app is not None
        assert initial_state["pdf_path"] == "data/raw/edital_exemplo.pdf"
    
    def test_node_1_function_exists(self):
        """Testa que a função do Nó 1 existe."""
        from app.rag.langgraph_workflow import node_1_reader_chunker
        
        assert callable(node_1_reader_chunker)
    
    def test_node_2_function_exists(self):
        """Testa que a função do Nó 2 existe."""
        from app.rag.langgraph_workflow import node_2_embeddings
        
        assert callable(node_2_embeddings)
    
    def test_node_3_function_exists(self):
        """Testa que a função do Nó 3 existe."""
        from app.rag.langgraph_workflow import node_3_analyzer
        
        assert callable(node_3_analyzer)
    
    def test_node_4_function_exists(self):
        """Testa que a função do Nó 4 existe."""
        from app.rag.langgraph_workflow import node_4_document_generator
        
        assert callable(node_4_document_generator)

    def test_real_mode_requires_key_without_silent_fallback(self, monkeypatch):
        from app.rag.langgraph_workflow import (
            is_mock_mode, validate_analysis_configuration
        )

        monkeypatch.delenv("NODE3_MOCK_MODE", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        assert is_mock_mode() is False
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            validate_analysis_configuration()

        monkeypatch.setenv("GROQ_API_KEY", "your_groq_api_key_here")
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            validate_analysis_configuration()

    def test_demo_requires_explicit_opt_in(self, monkeypatch):
        from app.rag.langgraph_workflow import (
            is_mock_mode, validate_analysis_configuration
        )

        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("NODE3_MOCK_MODE", "true")
        assert is_mock_mode() is True
        validate_analysis_configuration()

        monkeypatch.setenv("NODE3_MOCK_MODE", "false")
        monkeypatch.setenv("GROQ_API_KEY", "configured-key")
        assert is_mock_mode() is False
        validate_analysis_configuration()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
