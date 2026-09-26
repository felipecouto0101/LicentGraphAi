"""
Testes para Etapa 2 do Nó 3: Análise básica de requisitos

TDD Approach: Testes escritos antes da implementação
"""

import pytest
from typing import Dict, List


class TestNode3BasicAnalysis:
    """Testes para análise básica de requisitos."""
    
    def test_analyze_single_chunk(self):
        """Testa análise de um único chunk."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        chunk = {
            "content": "Processador Intel Core i5 ou superior, 8GB de RAM",
            "section": "exigencias_tecnicas"
        }
        
        result = node.analyze_chunk(chunk)
        
        assert "requirements" in result
        assert "analysis" in result
        assert "raw_response" in result
    
    def test_analyze_multiple_chunks(self):
        """Testa análise de múltiplos chunks."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        chunks = [
            {"content": "Processador Intel Core i5", "section": "exigencias_tecnicas"},
            {"content": "Prazo de entrega 30 dias", "section": "prazos"},
            {"content": "CNH e RG exigidos", "section": "documentacao"}
        ]
        
        result = node.analyze_chunks(chunks)
        
        assert "total_chunks" in result
        assert "analyses" in result
        assert len(result["analyses"]) == 3
    
    def test_analyze_empty_chunks(self):
        """Testa análise com chunks vazios."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        result = node.analyze_chunks([])
        
        assert result["total_chunks"] == 0
        assert result["analyses"] == []
    
    def test_analyze_with_system_prompt(self):
        """Testa análise com prompt de sistema customizado."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        system_prompt = "Você é um especialista em licitações."
        chunk = {"content": "Teste", "section": "test"}
        
        result = node.analyze_chunk(chunk, system_prompt=system_prompt)
        
        assert "system_prompt" in result
        assert result["system_prompt"] == system_prompt
    
    def test_analyze_preserves_metadata(self):
        """Testa se metadados são preservados na análise."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        chunk = {
            "content": "Processador Intel Core i5",
            "section": "exigencias_tecnicas",
            "metadata": {"chunk_id": 0, "page": 1}
        }
        
        result = node.analyze_chunk(chunk)
        
        assert "chunk_metadata" in result
        assert result["chunk_metadata"]["section"] == "exigencias_tecnicas"
        assert result["chunk_metadata"]["content_length"] == len(chunk["content"])
    
    def test_validate_analysis_output(self):
        """Testa validação da saída da análise."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        # Saída válida
        valid_output = {
            "requirements": ["CPU i5"],
            "analysis": "Requisito técnico claro",
            "raw_response": "response"
        }
        assert node.validate_analysis_output(valid_output) is True
        
        # Saída inválida (sem requirements)
        invalid_output = {"analysis": "teste"}
        assert node.validate_analysis_output(invalid_output) is False
    
    def test_mock_mode_response(self):
        """Testa resposta em modo mock."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        chunk = {"content": "Teste", "section": "test"}
        result = node.analyze_chunk(chunk)
        
        # Em mock mode, deve retornar resposta simulada
        assert "requirements" in result
        assert "analysis" in result
        assert "raw_response" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
