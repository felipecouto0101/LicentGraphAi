"""
Testes para Etapa 4 do Nó 3: Identificação de pontos críticos

TDD Approach: Testes escritos antes da implementação
"""

import pytest
from typing import Dict, List


class TestNode3CriticalPoints:
    """Testes para identificação de pontos críticos."""
    
    def test_identify_critical_points(self):
        """Testa identificação de pontos críticos."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        chunk = {
            "content": "Prazo de 5 dias para entrega, multa de 10% por atraso",
            "section": "prazos"
        }
        
        result = node.identify_critical_points(chunk)
        
        assert "critical_points" in result
        assert "risk_level" in result
        assert "recommendations" in result
    
    def test_risk_level_classification(self):
        """Testa classificação de nível de risco."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        # Alto risco
        high_risk_chunk = {
            "content": "Prazo impossível de 2 dias, multa severa",
            "section": "prazos"
        }
        result = node.identify_critical_points(high_risk_chunk)
        assert result["risk_level"] in ["ALTO", "MEDIO", "BAIXO"]
        
        # Baixo risco
        low_risk_chunk = {
            "content": "Prazo razoável de 60 dias, sem multas",
            "section": "prazos"
        }
        result = node.identify_critical_points(low_risk_chunk)
        assert result["risk_level"] in ["ALTO", "MEDIO", "BAIXO"]
    
    def test_recommendations_generation(self):
        """Testa geração de recomendações."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        chunk = {
            "content": "Certificação ISO 9001 obrigatória",
            "section": "exigencias_tecnicas"
        }
        
        result = node.identify_critical_points(chunk)
        
        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)
    
    def test_analyze_critical_ambiguities(self):
        """Testa análise de ambiguidades críticas."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        chunk = {
            "content": "Especificações técnicas a serem definidas posteriormente",
            "section": "exigencias_tecnicas"
        }
        
        result = node.identify_critical_points(chunk)
        
        assert "ambiguities" in result
        assert isinstance(result["ambiguities"], list)
    
    def test_critical_points_summary(self):
        """Testa resumo de pontos críticos."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        chunks = [
            {"content": "Prazo curto", "section": "prazos"},
            {"content": "Multa alta", "section": "outros"}
        ]
        
        result = node.summarize_critical_points(chunks)
        
        assert "total_critical_points" in result
        assert "overall_risk_level" in result
        assert "priority_actions" in result
    
    def test_validate_critical_analysis(self):
        """Testa validação da análise crítica."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        # Saída válida
        valid_output = {
            "critical_points": ["Prazo curto"],
            "risk_level": "ALTO",
            "recommendations": ["Negociar prazo"]
        }
        assert node.validate_critical_analysis(valid_output) is True
        
        # Saída inválida (sem risk_level)
        invalid_output = {
            "critical_points": ["Prazo curto"],
            "recommendations": ["Negociar prazo"]
        }
        assert node.validate_critical_analysis(invalid_output) is False
    
    def test_empty_chunk_critical_analysis(self):
        """Testa análise crítica com chunk vazio."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        chunk = {"content": "", "section": "test"}
        result = node.identify_critical_points(chunk)
        
        assert result["critical_points"] == []
        assert result["risk_level"] == "BAIXO"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
