"""
Testes para Etapa 5 do Nó 3: Integração completa com Nó 2

TDD Approach: Testes escritos antes da implementação
"""

import pytest
from typing import Dict, List


class TestNode3IntegrationWithNode2:
    """Testes para integração do Nó 3 com Nó 2."""
    
    def test_process_chunks_from_node2(self):
        """Testa processamento de chunks vindos do Nó 2."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node3 = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        # Simula chunks do Nó 2
        node2_chunks = [
            {
                "content": "Processador Intel Core i5 ou superior",
                "section": "exigencias_tecnicas",
                "metadata": {"chunk_id": 0}
            },
            {
                "content": "Prazo de entrega: 30 dias",
                "section": "prazos",
                "metadata": {"chunk_id": 1}
            }
        ]
        
        result = node3.process_from_node2(node2_chunks)
        
        assert "total_chunks_processed" in result
        assert "structured_analysis" in result
        assert "critical_analysis" in result
        assert result["total_chunks_processed"] == 2
    
    def test_real_mode_dispatches_to_unified_llm(self, monkeypatch):
        """O fluxo real usa a passagem única e preserva a saída para o nó 4."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        analyzer = object.__new__(Node3RequirementAnalyzer)
        analyzer.mock_mode = False
        chunks = [{"content": "Prazo: 30 dias", "section": "prazos"}]
        expected = {
            "total_chunks_processed": 1,
            "structured_analysis": {},
            "critical_analysis": {},
            "llm_analysis": {"prazos": ["30 dias"]},
        }
        monkeypatch.setattr(analyzer, "_process_with_rag_llm", lambda incoming: expected if incoming is chunks else None)

        assert analyzer.process_from_node2(chunks) is expected

    def test_complete_workflow_node2_to_node3(self):
        """Testa workflow completo Nó 2 → Nó 3."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node3 = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        # Chunks do Nó 2 com embeddings
        node2_output = {
            "chunks_with_embeddings": [
                {
                    "content": "CPU i5, 8GB RAM",
                    "section": "exigencias_tecnicas",
                    "embedding": [0.1, 0.2, 0.3]
                },
                {
                    "content": "Prazo 30 dias",
                    "section": "prazos",
                    "embedding": [0.4, 0.5, 0.6]
                }
            ],
            "total_embeddings": 2
        }
        
        result = node3.process_complete_analysis(node2_output)
        
        assert "summary" in result
        assert "technical_analysis" in result
        assert "risk_assessment" in result
        assert "recommendations" in result
    
    def test_combine_structured_and_critical_analysis(self):
        """Testa combinação de análise estruturada e crítica."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node3 = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        chunks = [
            {"content": "Processador i5", "section": "exigencias_tecnicas"},
            {"content": "Prazo curto", "section": "prazos"}
        ]
        
        result = node3.combine_analyses(chunks)
        
        assert "structured_info" in result
        assert "critical_info" in result
        assert "combined_risk_level" in result
        assert "priority_recommendations" in result
    
    def test_generate_executive_summary(self):
        """Testa geração de resumo executivo."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node3 = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        analysis_result = {
            "structured_info": [
                {
                    "technical_requirements": ["CPU i5"],
                    "deadlines": ["30 dias"]
                }
            ],
            "critical_info": [
                {
                    "risk_level": "MEDIO",
                    "critical_points": ["Prazo curto"]
                }
            ],
            "combined_risk_level": "MEDIO",
            "priority_recommendations": ["Verificar prazo"]
        }
        
        summary = node3.generate_executive_summary(analysis_result)
        
        assert "overview" in summary
        assert "key_findings" in summary
        assert "risk_summary" in summary
        assert "next_steps" in summary
    
    def test_validate_integration_output(self):
        """Testa validação da saída da integração."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node3 = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        # Saída válida
        valid_output = {
            "total_chunks_processed": 2,
            "structured_analysis": {"test": "data"},
            "critical_analysis": {"risk_level": "ALTO"}
        }
        assert node3.validate_integration_output(valid_output) is True
        
        # Saída inválida (sem campo obrigatório)
        invalid_output = {
            "structured_analysis": {"test": "data"}
        }
        assert node3.validate_integration_output(invalid_output) is False
    
    def test_empty_chunks_integration(self):
        """Testa integração com chunks vazios."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node3 = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        result = node3.process_from_node2([])
        
        assert result["total_chunks_processed"] == 0
        assert result["structured_analysis"] == {}
        assert result["critical_analysis"] == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
