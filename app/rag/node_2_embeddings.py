import logging
from pathlib import Path
from typing import ClassVar

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Node2EmbeddingGenerator:
    """
    Nó 2: Geração de Embeddings e Armazenamento no ChromaDB

    Responsável por:
    - Gerar embeddings dos chunks usando modelo HuggingFace
    - Armazenar embeddings no ChromaDB
    - Preparar dados para busca semântica
    - Suportar busca vetorial eficiente

    Modelos suportados:
    - sentence-transformers/all-MiniLM-L6-v2 (padrão, rápido, 384 dims)
    - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (multilíngue)
    """

    # Cache de modelos para evitar re-carregamento
    _model_cache: ClassVar[dict[str, SentenceTransformer]] = {}

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        show_progress: bool = False,
    ):
        """
        Inicializa o Nó 2.

        Args:
            model_name: Nome do modelo de embedding
            device: Dispositivo para execução (cpu/cuda)
            batch_size: Tamanho do batch para processamento
            show_progress: Mostrar barra de progresso
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.chroma_client: chromadb.Client | None = None

        # Usa cache se disponível
        cache_key = f"{model_name}_{device}"
        if cache_key in self._model_cache:
            self.embeddings = self._model_cache[cache_key]
            logger.info(f"Modelo carregado do cache: {model_name}")
        else:
            self.embeddings = SentenceTransformer(model_name, device=device)
            self._model_cache[cache_key] = self.embeddings
            logger.info(f"Nó 2 inicializado: modelo={model_name}, device={device}")

        # Obtém dimensão do embedding
        self.embedding_dimension = self.embeddings.get_embedding_dimension()
        logger.info(f"Dimensão do embedding: {self.embedding_dimension}")

    def generate_embeddings(
        self, chunks: list[dict], batch_size: int | None = None
    ) -> dict:
        """
        Gera embeddings para os chunks com processamento em batch.

        Args:
            chunks: Lista de chunks com conteúdo e metadados
            batch_size: Tamanho do batch (usa padrão se não especificado)

        Returns:
            Dicionário com embeddings e chunks enriquecidos
        """
        logger.info(f"Gerando embeddings para {len(chunks)} chunks")

        if not chunks:
            return {
                "embeddings": [],
                "chunks_with_embeddings": [],
                "processing_time": 0.0,
            }

        import time

        start_time = time.time()

        # Extrai textos dos chunks
        texts = [chunk["content"] for chunk in chunks]

        # Usa batch_size específico ou padrão
        actual_batch_size = batch_size or self.batch_size

        # Gera embeddings em batch
        embeddings = self.embeddings.encode(
            texts, batch_size=actual_batch_size, convert_to_numpy=True
        )

        # Cria chunks com embeddings
        chunks_with_embeddings = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding["embedding"] = embedding.tolist()
            chunk_with_embedding["embedding_dimension"] = self.embedding_dimension
            chunks_with_embeddings.append(chunk_with_embedding)

        processing_time = time.time() - start_time

        result = {
            "embeddings": embeddings.tolist(),
            "chunks_with_embeddings": chunks_with_embeddings,
            "processing_time": processing_time,
            "chunks_per_second": (
                len(chunks) / processing_time if processing_time > 0 else 0
            ),
        }

        logger.info(
            f"Embeddings gerados: {len(embeddings)} vetores de {self.embedding_dimension} dims em {processing_time:.2f}s"
        )
        return result

    def store_in_chromadb(
        self,
        embedded_chunks: list[dict],
        collection_name: str = "licitacoes",
        persist_directory: str | None = None,
        upsert: bool = False,
    ) -> str:
        """
        Armazena embeddings no ChromaDB com opções avançadas.

        Args:
            embedded_chunks: Chunks com embeddings
            collection_name: Nome da coleção
            persist_directory: Diretório para persistência
            upsert: Se True, atualiza documentos existentes

        Returns:
            ID da coleção criada/atualizada
        """
        logger.info(f"Armazenando {len(embedded_chunks)} chunks no ChromaDB")

        # Inicializa cliente ChromaDB
        if persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.chroma_client = chromadb.Client()

        # Cria ou obtém coleção
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            logger.info(
                f"Coleção '{collection_name}' já existe com {collection.count()} documentos"
            )
        except Exception:
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "model": self.model_name,
                    "dimension": self.embedding_dimension,
                },
            )
            logger.info(f"Coleção '{collection_name}' criada")

        # Prepara dados para inserção
        ids = [
            f"chunk_{i}_{hash(chunk['content']) % 10000}"
            for i, chunk in enumerate(embedded_chunks)
        ]
        documents = [chunk["content"] for chunk in embedded_chunks]
        embeddings = [chunk["embedding"] for chunk in embedded_chunks]
        metadatas = []

        for i, chunk in enumerate(embedded_chunks):
            metadata = {
                "section": chunk.get("section", "unknown"),
                "chunk_id": chunk.get("metadata", {}).get("chunk_id", i),
                "embedding_dimension": chunk.get(
                    "embedding_dimension", self.embedding_dimension
                ),
            }
            # Adiciona metadados adicionais se existirem
            if "metadata" in chunk:
                metadata.update(chunk["metadata"])
            metadatas.append(metadata)

        # Adiciona ou atualiza na coleção
        if upsert:
            collection.upsert(
                ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
            )
            logger.info(f"Chunks atualizados na coleção '{collection_name}'")
        else:
            collection.add(
                ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
            )
            logger.info(f"Chunks adicionados à coleção '{collection_name}'")

        return collection_name

    def process_chunks(
        self,
        chunks: list[dict],
        collection_name: str = "licitacoes",
        persist_directory: str | None = None,
        batch_size: int | None = None,
    ) -> dict:
        """
        Processamento completo do Nó 2 com métricas de performance.

        Args:
            chunks: Chunks do Nó 1
            collection_name: Nome da coleção ChromaDB
            persist_directory: Diretório para persistência
            batch_size: Tamanho do batch para processamento

        Returns:
            Dicionário com resultado completo do processamento
        """
        import time

        total_start_time = time.time()

        logger.info("Iniciando processamento completo do Nó 2")

        # Gera embeddings
        embedding_result = self.generate_embeddings(chunks, batch_size=batch_size)

        # Armazena no ChromaDB
        collection_id = self.store_in_chromadb(
            embedding_result["chunks_with_embeddings"],
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

        total_processing_time = time.time() - total_start_time

        result = {
            "collection_id": collection_id,
            "total_embeddings": len(embedding_result["embeddings"]),
            "embedding_model": self.model_name,
            "vector_dimension": self.embedding_dimension,
            "chunks_with_embeddings": embedding_result["chunks_with_embeddings"],
            "processing_time": total_processing_time,
            "embedding_time": embedding_result.get("processing_time", 0.0),
            "chunks_per_second": embedding_result.get("chunks_per_second", 0.0),
        }

        logger.info(
            f"Processamento Nó 2 concluído: {result['total_embeddings']} embeddings em {total_processing_time:.2f}s"
        )
        return result

    def validate_embeddings_output(
        self, output: dict, expected_dimension: int | None = None
    ) -> bool:
        """
        Valida a saída do processamento de embeddings.

        Args:
            output: Dicionário de saída do processamento
            expected_dimension: Dimensão esperada do embedding (opcional, usa self.embedding_dimension se não fornecido)

        Returns:
            True se válido, False caso contrário
        """
        required_fields = ["embeddings", "chunks_with_embeddings"]

        for field in required_fields:
            if field not in output:
                logger.error(f"Campo obrigatório ausente: {field}")
                return False

        if not output["embeddings"]:
            logger.error("Nenhum embedding gerado")
            return False

        # Verifica dimensão consistente
        first_dim = len(output["embeddings"][0])
        for emb in output["embeddings"]:
            if len(emb) != first_dim:
                logger.error("Embeddings com dimensões inconsistentes")
                return False

        # Verifica dimensão esperada (usa self.embedding_dimension se não fornecido)
        actual_expected = expected_dimension or self.embedding_dimension
        if first_dim != actual_expected:
            logger.error(
                f"Dimensão do embedding incorreta: esperado {actual_expected}, obtido {first_dim}"
            )
            return False

        logger.info("Validação de embeddings concluída com sucesso")
        return True

    def get_embedding_stats(self, embedding_result: dict) -> dict:
        """
        Gera estatísticas dos embeddings.

        Args:
            embedding_result: Resultado da geração de embeddings

        Returns:
            Dicionário com estatísticas
        """
        embeddings = embedding_result["embeddings"]

        if not embeddings:
            return {
                "total_embeddings": 0,
                "vector_dimension": 0,
                "avg_embedding_norm": 0.0,
            }

        # Calcula norma média dos embeddings
        norms = [np.linalg.norm(emb) for emb in embeddings]
        avg_norm = np.mean(norms)

        return {
            "total_embeddings": len(embeddings),
            "vector_dimension": len(embeddings[0]),
            "avg_embedding_norm": float(avg_norm),
        }

    def search_similar(
        self,
        query: str,
        collection_name: str = "licitacoes",
        n_results: int = 3,
        persist_directory: str | None = None,
    ) -> dict:
        """
        Busca chunks similares semanticamente.

        Args:
            query: Texto da busca
            collection_name: Nome da coleção
            n_results: Número de resultados
            persist_directory: Diretório de persistência

        Returns:
            Dicionário com resultados da busca
        """
        logger.info(f"Buscando similaridade para: '{query}'")

        # Inicializa cliente se necessário
        if not self.chroma_client:
            if persist_directory:
                self.chroma_client = chromadb.PersistentClient(path=persist_directory)
            else:
                self.chroma_client = chromadb.Client()

        # Obtém coleção
        collection = self.chroma_client.get_collection(name=collection_name)

        # Gera embedding da query
        query_embedding = self.embeddings.encode([query], convert_to_numpy=True)

        # Busca
        results = collection.query(
            query_embeddings=query_embedding.tolist(), n_results=n_results
        )

        logger.info(f"Encontrados {len(results['documents'][0])} resultados")
        return results


# Função de conveniência para uso direto
def process_edital_embeddings(
    chunks: list[dict],
    collection_name: str = "licitacoes",
    persist_directory: str | None = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict:
    """
    Função de conveniência para processar embeddings de chunks.

    Args:
        chunks: Chunks do Nó 1
        collection_name: Nome da coleção ChromaDB
        persist_directory: Diretório para persistência
        model_name: Modelo de embedding

    Returns:
        Dicionário com resultado do processamento
    """
    node = Node2EmbeddingGenerator(model_name=model_name)
    result = node.process_chunks(chunks, collection_name, persist_directory)

    if not node.validate_embeddings_output(result):
        raise ValueError("Output de embeddings inválido")

    return result
