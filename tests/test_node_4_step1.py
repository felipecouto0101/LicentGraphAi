"""
Testes para Etapa 1 do Nó 4: Configuração básica

TDD Approach: Testes escritos antes da implementação
"""

import pytest
from typing import Dict, List


class TestNode4Config:
    """Testes para configuração do Nó 4."""
    
    def test_init_with_default_config(self):
        """Testa inicialização com configuração padrão."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        assert node.api_key == "test_key"
        assert node.model_name == "openai/gpt-oss-120b"
        assert node.mock_mode is True
    
    def test_init_with_custom_model(self):
        """Testa inicialização com modelo customizado."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(
            api_key="test_key",
            model_name="llama-3.3-70b-versatile",
            mock_mode=True
        )
        
        assert node.model_name == "llama-3.3-70b-versatile"
    
    def test_init_with_env_variable(self):
        """Testa inicialização usando variável de ambiente."""
        import os
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        os.environ["GROQ_API_KEY"] = "env_test_key"
        node = Node4DocumentGenerator(mock_mode=True)
        
        assert node.api_key == "env_test_key"
    
    def test_init_without_api_key_raises_error(self):
        """Testa que erro é levantado sem API key em modo real."""
        import os
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        # Remove temporariamente a env var
        original_key = os.environ.get("GROQ_API_KEY")
        if "GROQ_API_KEY" in os.environ:
            del os.environ["GROQ_API_KEY"]
        
        try:
            with pytest.raises(ValueError):
                Node4DocumentGenerator(api_key=None, mock_mode=False)
        finally:
            # Restaura a env var
            if original_key:
                os.environ["GROQ_API_KEY"] = original_key
    
    def test_init_custom_temperature_config(self):
        """Testa configuração customizada de temperatura."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", temperature=0.5, mock_mode=True)
        
        assert node.temperature == 0.5
    
    def test_init_custom_max_tokens_config(self):
        """Testa configuração customizada de max tokens."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", max_tokens=1000, mock_mode=True)
        
        assert node.max_tokens == 1000
    
    def test_connection_test(self):
        """Testa conexão básica do Nó 4."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        assert node is not None
        assert hasattr(node, 'api_key')
        assert hasattr(node, 'model_name')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
