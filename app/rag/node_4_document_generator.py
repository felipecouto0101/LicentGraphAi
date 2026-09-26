"""
Nó 4: Gerador de Checklist de Documentos

Responsável por:
- Receber requisitos do edital (do Nó 3)
- Extrair documentos necessários
- Categorizar documentos (habilitação, técnica, fiscal, etc.)
- Gerar checklist estruturado pronto para uso
"""

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional, Dict, List
import os
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Node4DocumentGenerator:
    """
    Nó 4: Gerador de Checklist de Documentos
    
    Responsável por:
    - Extrair documentos necessários do edital
    - Categorizar documentos
    - Gerar checklist estruturado
    - Usar Groq + OpenAI GPT-OSS-120b para análise inteligente
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        mock_mode: bool = False
    ):
        """
        Inicializa o Nó 4 com configuração do Groq.
        
        Args:
            api_key: Chave da API Groq (opcional, usa env var se não fornecido)
            model_name: Nome do modelo LLM
            temperature: Temperatura para geração
            max_tokens: Máximo de tokens na resposta
            mock_mode: Se True, usa modo mock (sem API real)
        """
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
        
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mock_mode = mock_mode
        
        if mock_mode:
            logger.info("Nó 4 em modo mock (sem LLM real)")
            self.llm = None
        else:
            if not api_key:
                raise ValueError("API key não fornecida e mock_mode=False")
            
            logger.info(f"Nó 4 inicializado: modelo={model_name}, temperature={temperature}, mock_mode={mock_mode}")
            self.llm = ChatGroq(
                model_name=model_name,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens
            )
    
    def extract_documents(self, chunk: dict) -> dict:
        """
        Extrai documentos de um chunk do edital.
        
        Args:
            chunk: Chunk com conteúdo e metadados
            
        Returns:
            Dicionário com documentos extraídos
        """
        logger.info(f"Extraindo documentos da seção: {chunk.get('section', 'unknown')}")
        
        content = chunk["content"].lower()
        
        if not content.strip():
            return {"documents": [], "deadlines": [], "requirements": []}
        
        documents = self._extract_document_types(content)
        deadlines = self._extract_deadlines(content)
        requirements = self._extract_requirements(content)
        
        return {
            "documents": documents,
            "deadlines": deadlines,
            "requirements": requirements
        }
    
    def extract_documents_from_chunks(self, chunks: List[dict]) -> dict:
        """
        Extrai documentos de múltiplos chunks.
        
        Args:
            chunks: Lista de chunks
            
        Returns:
            Dicionário com todos os documentos extraídos
        """
        logger.info(f"Extraindo documentos de {len(chunks)} chunks")
        
        all_documents = []
        all_deadlines = []
        all_requirements = []
        
        for chunk in chunks:
            result = self.extract_documents(chunk)
            all_documents.extend(result["documents"])
            all_deadlines.extend(result["deadlines"])
            all_requirements.extend(result["requirements"])
        
        return {
            "documents": list(set(all_documents)),  # Remove duplicatas
            "deadlines": list(set(all_deadlines)),
            "requirements": list(set(all_requirements))
        }
    
    def _extract_document_types(self, content: str) -> List[str]:
        """Extrai tipos de documentos do conteúdo."""
        document_keywords = [
            "cnpj", "rg", "cpf", "cnh", "certidão", "atestado",
            "declaração", "contrato", "balanço", "demonstrativo",
            "faturamento", "imposto", "licença", "alvará"
        ]
        
        found_documents = []
        for keyword in document_keywords:
            if keyword in content:
                found_documents.append(keyword)
        
        return found_documents
    
    def _extract_deadlines(self, content: str) -> List[str]:
        """Extrai prazos do conteúdo."""
        deadline_patterns = [
            r"(\d+)\s*dias?",
            r"(\d+)\s*horas?",
            r"(\d+)/\d+/\d+",  # Data
            r"até\s+(\d+)"
        ]
        
        deadlines = []
        for pattern in deadline_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            deadlines.extend(matches)
        
        return deadlines
    
    def _extract_requirements(self, content: str) -> List[str]:
        """Extrai requisitos de documentos do conteúdo."""
        requirement_keywords = [
            "autenticado", "cartório", "firma", "reconhecida",
            "original", "cópia", "validade", "atualizado"
        ]
        
        found_requirements = []
        for keyword in requirement_keywords:
            if keyword in content:
                found_requirements.append(keyword)
        
        return found_requirements
    
    def validate_document_extraction(self, output: dict) -> bool:
        """
        Valida a extração de documentos.
        
        Args:
            output: Dicionário de saída da extração
            
        Returns:
            True se válido, False caso contrário
        """
        required_fields = ["documents", "deadlines", "requirements"]
        
        for field in required_fields:
            if field not in output:
                logger.error(f"Campo obrigatório ausente: {field}")
                return False
        
        logger.info("Validação de extração de documentos concluída")
        return True
