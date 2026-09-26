"""
Testes para Etapa 1 do Nó 3: Configuração básica do Groq + Llama 3.1

TDD Approach: Testes escritos antes da implementação
"""

import pytest
from typing import Dict, Optional
import os


class TestNode3GroqConfig:
    """Testes para configuração do Groq + Llama 3.1."""
    
    def test_init_with_api_key(self):
        """Testa inicialização com chave da API."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        # Testa com chave de API fornecida e mock mode
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        assert node.api_key == "test_key"
        assert node.model_name == "openai/gpt-oss-120b"
        assert node.mock_mode is True
    
    def test_init_with_env_variable(self):
        """Testa inicialização usando variável de ambiente."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        # Salva variável de ambiente original
        original_key = os.environ.get("GROQ_API_KEY")
        
        # Define variável de ambiente temporária
        os.environ["GROQ_API_KEY"] = "env_test_key"
        
        try:
            node = Node3RequirementAnalyzer(mock_mode=True)
            assert node.api_key == "env_test_key"
        finally:
            # Restaura variável de ambiente original
            if original_key:
                os.environ["GROQ_API_KEY"] = original_key
            elif "GROQ_API_KEY" in os.environ:
                del os.environ["GROQ_API_KEY"]
    
    def test_init_custom_model(self):
        """Testa inicialização com modelo customizado."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(
            api_key="test_key",
            model_name="llama-3.1-70b-versatile",
            mock_mode=True
        )
        
        assert node.model_name == "llama-3.1-70b-versatile"
    
    def test_init_without_api_key_raises_error(self):
        """Testa erro quando não há chave da API."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        # Remove variável de ambiente se existir
        original_key = os.environ.get("GROQ_API_KEY")
        if "GROQ_API_KEY" in os.environ:
            del os.environ["GROQ_API_KEY"]
        
        try:
            with pytest.raises(ValueError, match="API key é obrigatória"):
                Node3RequirementAnalyzer(mock_mode=False)
        finally:
            # Restaura variável de ambiente original
            if original_key:
                os.environ["GROQ_API_KEY"] = original_key
    
    def test_default_model_config(self):
        """Testa configuração padrão do modelo."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        assert node.model_name == "openai/gpt-oss-120b"
        assert node.temperature == 0.7
        assert node.max_tokens == 2000
    
    def test_custom_temperature_config(self):
        """Testa configuração customizada de temperatura."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(
            api_key="test_key",
            temperature=0.3,
            mock_mode=True
        )
        
        assert node.temperature == 0.3
    
    def test_custom_max_tokens_config(self):
        """Testa configuração customizada de max tokens."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node = Node3RequirementAnalyzer(
            api_key="test_key",
            max_tokens=4000,
            mock_mode=True
        )
        
        assert node.max_tokens == 4000


class TestNode3BasicConnection:
    """Testes básicos de conexão com Groq."""
    
    def test_connection_test(self):
        """Testa se a conexão com Groq pode ser estabelecida."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        # Testa estrutura básica em mock mode
        node = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        assert hasattr(node, 'llm')
        assert hasattr(node, 'api_key')
        assert hasattr(node, 'model_name')
        assert node.mock_mode is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
