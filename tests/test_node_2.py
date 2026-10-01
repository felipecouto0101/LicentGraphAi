"""
Testes para o Nó 2: Geração de Embeddings e Armazenamento no ChromaDB

TDD Approach: Testes escritos antes da implementação
"""

import pytest
import tempfile
import shutil

# These tests load real Hugging Face models and a persistent Chroma database.
pytestmark = pytest.mark.integration


class TestNode2EmbeddingGenerator:
    """Testes para o Node2EmbeddingGenerator."""
    
    def test_init_default_model(self):
        """Testa inicialização com modelo padrão."""
        from app.rag.node_2_embeddings import Node2EmbeddingGenerator
        
        node = Node2EmbeddingGenerator()
        
        assert node.embeddings is not None
        assert node.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert node.chroma_client is None
    
    def test_init_custom_model(self):
        """Testa inicialização com modelo customizado."""
        from app.rag.node_2_embeddings import Node2EmbeddingGenerator
        
        custom_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        node = Node2EmbeddingGenerator(model_name=custom_model)
        
        assert node.model_name == custom_model
        assert node.embeddings is not None
    
    def test_generate_embeddings_single_chunk(self):
        """Testa geração de embedding para um único chunk."""
        from app.rag.node_2_embeddings import Node2EmbeddingGenerator
        
        node = Node2EmbeddingGenerator()
        chunks = [{"content": "Processador Intel Core i5 ou superior"}]
        
        result = node.generate_embeddings(chunks)
        
        assert "embeddings" in result
        assert "chunks_with_embeddings" in result
        assert "processing_time" in result
        assert len(result["embeddings"]) == 1
        assert len(result["embeddings"][0]) == 384  # Dimensão do modelo MiniLM
        assert all(isinstance(x, float) for x in result["embeddings"][0])
    
    def test_generate_embeddings_multiple_chunks(self):
        """Testa geração de embeddings para múltiplos chunks."""
        from app.rag.node_2_embeddings import Node2EmbeddingGenerator
        
        node = Node2EmbeddingGenerator()
        chunks = [
            {"content": "Processador Intel Core i5"},
            {"content": "Memória RAM 8GB mínimo"},
            {"content": "Prazo de entrega 30 dias"}
        ]
        
        result = node.generate_embeddings(chunks)
        
        assert len(result["embeddings"]) == 3
        assert all(len(emb) == 384 for emb in result["embeddings"])
    
    def test_generate_embeddings_empty_chunks(self):
        """Testa geração de embeddings com chunks vazios."""
        from app.rag.node_2_embeddings import Node2EmbeddingGenerator
        
        node = Node2EmbeddingGenerator()
        chunks = []
        
        result = node.generate_embeddings(chunks)
        
        assert result["embeddings"] == []
        assert result["chunks_with_embeddings"] == []
        assert result["processing_time"] == 0.0
    
    def test_generate_embeddings_preserves_metadata(self):
        """Testa se metadados são preservados durante geração de embeddings."""
        from app.rag.node_2_embeddings import Node2EmbeddingGenerator
        
        node = Node2EmbeddingGenerator()
        chunks = [
            {
                "content": "Processador Intel Core i5",
                "section": "exigencias_tecnicas",
                "metadata": {"chunk_id": 0, "total_chunks": 3}
            }
        ]
        
        result = node.generate_embeddings(chunks)
        
        assert "chunks_with_embeddings" in result
        assert len(result["chunks_with_embeddings"]) == 1
        assert result["chunks_with_embeddings"][0]["section"] == "exigencias_tecnicas"
        assert result["chunks_with_embeddings"][0]["metadata"]["chunk_id"] == 0
    
    def test_store_in_chromadb_create_collection(self):
        """Testa criação de coleção no ChromaDB."""
        from app.rag.node_2_embeddings import Node2EmbeddingGenerator
        
        node = Node2EmbeddingGenerator()
        
        # Usa diretório temporário
        temp_dir = tempfile.mkdtemp()
        
        try:
            embedded_chunks = [
                {
                    "content": "Processador Intel Core i5",
                    "embedding": [0.1] * 384,
                    "metadata": {"section": "exigencias_tecnicas"}
                }
            ]
            
            collection_id = node.store_in_chromadb(embedded_chunks, persist_directory=temp_dir)
            
            assert collection_id is not None
            assert isinstance(collection_id, str)
            assert node.chroma_client is not None
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_store_in_chromadb_search(self):
        """Testa busca no ChromaDB após armazenamento."""
        from app.rag.node_2_embeddings import Node2EmbeddingGenerator
        
        node = Node2EmbeddingGenerator()
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Armazena chunks
            embedded_chunks = [
                {
                    "content": "Processador Intel Core i5 ou superior",
                    "embedding": node.embeddings.encode("Processador Intel Core i5", convert_to_numpy=True).tolist(),
                    "metadata": {"section": "exigencias_tecnicas"}
                }
            ]
            
            collection_id = node.store_in_chromadb(embedded_chunks, persist_directory=temp_dir)
            
            # Busca similar
            query_embedding = node.embeddings.encode("CPU necessária", convert_to_numpy=True).tolist()
            collection = node.chroma_client.get_collection(name=collection_id)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=1
            )
            
            assert len(results["documents"][0]) > 0
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_process_chunks_complete_workflow(self):
        """Testa o fluxo completo de processamento do Nó 2."""
        from app.rag.node_2_embeddings import Node2EmbeddingGenerator
        
        node = Node2EmbeddingGenerator()
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Simula chunks do Nó 1
            chunks = [
                {
                    "content": "OBJETO: Aquisição de computadores desktop",
                    "section": "bens",
                    "metadata": {"chunk_id": 0, "total_chunks": 2}
                },
                {
                    "content": "PRAZO: 30 dias para entrega",
                    "section": "prazos",
                    "metadata": {"chunk_id": 1, "total_chunks": 2}
                }
            ]
            
            result = node.process_chunks(chunks, persist_directory=temp_dir)
            
            assert "collection_id" in result
            assert "total_embeddings" in result
            assert "embedding_model" in result
            assert "vector_dimension" in result
            assert "chunks_with_embeddings" in result
            assert "processing_time" in result
            assert "embedding_time" in result
            
            assert result["total_embeddings"] == 2
            assert result["vector_dimension"] == 384
            assert len(result["chunks_with_embeddings"]) == 2
            assert result["processing_time"] > 0
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_validate_embeddings_output(self):
        """Testa validação da saída de embeddings."""
        from app.rag.node_2_embeddings import Node2EmbeddingGenerator
        
        node = Node2EmbeddingGenerator()
        
        # Saída válida (com dimensão correta de 384)
        valid_output = {
            "embeddings": [[0.1] * 384],  # Dimensão correta
            "chunks_with_embeddings": [{"content": "test"}]
        }
        assert node.validate_embeddings_output(valid_output) is True
        
        # Saída inválida (sem embeddings)
        invalid_output = {"chunks_with_embeddings": []}
        assert node.validate_embeddings_output(invalid_output) is False
        
        # Saída inválida (embedding com dimensão errada)
        invalid_output2 = {
            "embeddings": [[0.1, 0.2]],  # Dimensão errada (2 em vez de 384)
            "chunks_with_embeddings": []
        }
        assert node.validate_embeddings_output(invalid_output2, expected_dimension=384) is False
    
    def test_get_embedding_stats(self):
        """Testa geração de estatísticas dos embeddings."""
        from app.rag.node_2_embeddings import Node2EmbeddingGenerator
        
        node = Node2EmbeddingGenerator()
        chunks = [
            {"content": "Texto curto"},
            {"content": "Texto médio para teste"},
            {"content": "Texto mais longo para verificar estatísticas"}
        ]
        
        result = node.generate_embeddings(chunks)
        stats = node.get_embedding_stats(result)
        
        assert "total_embeddings" in stats
        assert "vector_dimension" in stats
        assert "avg_embedding_norm" in stats
        assert stats["total_embeddings"] == 3
        assert stats["vector_dimension"] == 384


class TestNode2Integration:
    """Testes de integração do Nó 2 com Nó 1."""
    
    def test_integration_with_node_1(self):
        """Testa integração completa Nó 1 → Nó 2."""
        from app.rag.node_2_embeddings import Node2EmbeddingGenerator
        
        # Simula output do Nó 1
        node1_result = {
            "chunks": [
                {"content": "OBJETO: Aquisição de computadores", "section": "bens"},
                {"content": "PRAZO: 30 dias", "section": "prazos"},
                {"content": "ESPECIFICAÇÕES: Processador i5", "section": "exigencias_tecnicas"}
            ]
        }
        
        # Nó 2: Processa chunks do Nó 1
        node2 = Node2EmbeddingGenerator()
        temp_dir = tempfile.mkdtemp()
        
        try:
            node2_result = node2.process_chunks(node1_result["chunks"], persist_directory=temp_dir)
            
            assert node2_result["total_embeddings"] == 3
            assert node2_result["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
