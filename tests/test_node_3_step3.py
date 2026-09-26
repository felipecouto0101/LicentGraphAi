"""
Testes para Etapa 3 do Nó 3: Extração estruturada de informações

TDD Approach: Testes escritos antes da implementação
"""

import pytest
from typing import Dict, List


class TestNode3StructuredExtraction:
    """Testes para extração estruturada de informações."""
    
    def test_extract_technical_requirements(self):
        """Testa extração de requisitos técnicos."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        chunk = {
            "content": "Processador Intel Core i5 ou superior, 8GB de RAM mínimo, SSD 256GB",
            "section": "exigencias_tecnicas"
        }
        
        result = node.extract_structured_info(chunk)
        
        assert "technical_requirements" in result
        assert isinstance(result["technical_requirements"], list)
        assert len(result["technical_requirements"]) > 0
    
    def test_extract_deadlines(self):
        """Testa extração de prazos."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        chunk = {
            "content": "Prazo de entrega: 30 dias corridos após assinatura do contrato",
            "section": "prazos"
        }
        
        result = node.extract_structured_info(chunk)
        
        assert "deadlines" in result
        assert isinstance(result["deadlines"], list)
    
    def test_extract_documentation(self):
        """Testa extração de documentação exigida."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        chunk = {
            "content": "Documentação exigida: CNH válida, comprovante de endereço, certidão negativa",
            "section": "documentacao"
        }
        
        result = node.extract_structured_info(chunk)
        
        assert "documentation" in result
        assert isinstance(result["documentation"], list)
    
    def test_extract_object_info(self):
        """Testa extração de informações do objeto."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        chunk = {
            "content": "Objeto: Aquisição de 50 computadores desktop para uso administrativo",
            "section": "bens"
        }
        
        result = node.extract_structured_info(chunk)
        
        assert "object_info" in result
        assert "quantity" in result["object_info"] or "description" in result["object_info"]
    
    def test_extract_risk_factors(self):
        """Testa extração de fatores de risco."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        chunk = {
            "content": "Prazo muito curto de 5 dias para entrega complexa",
            "section": "outros"
        }
        
        result = node.extract_structured_info(chunk)
        
        assert "risk_factors" in result
        assert isinstance(result["risk_factors"], list)
    
    def test_structured_info_complete(self):
        """Testa extração completa de todas as categorias."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        chunk = {
            "content": "Processador i5, 8GB RAM, prazo 30 dias, CNH exigida",
            "section": "exigencias_tecnicas"
        }
        
        result = node.extract_structured_info(chunk)
        
        # Verifica se todas as categorias estão presentes
        expected_categories = [
            "technical_requirements",
            "deadlines", 
            "documentation",
            "object_info",
            "risk_factors"
        ]
        
        for category in expected_categories:
            assert category in result
    
    def test_validate_structured_output(self):
        """Testa validação da saída estruturada."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        # Saída válida
        valid_output = {
            "technical_requirements": ["CPU i5"],
            "deadlines": ["30 dias"],
            "documentation": ["CNH"],
            "object_info": {"description": "Computadores"},
            "risk_factors": []
        }
        assert node.validate_structured_output(valid_output) is True
        
        # Saída inválida (sem categoria obrigatória)
        invalid_output = {
            "technical_requirements": ["CPU i5"]
        }
        assert node.validate_structured_output(invalid_output) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
