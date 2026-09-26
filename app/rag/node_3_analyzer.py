from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional, Dict, List
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Node3RequirementAnalyzer:
    """
    Nó 3: Análise de Requisitos com IA (Groq + Llama 3.1)
    
    Responsável por:
    - Configurar conexão com Groq API
    - Inicializar modelo Llama 3.1
    - Analisar requisitos de editais
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        mock_mode: bool = False
    ):
        """
        Inicializa o Nó 3 com configuração do Groq.
        
        Args:
            api_key: Chave da API Groq (ou usa variável de ambiente)
            model_name: Nome do modelo Llama
            temperature: Temperatura para geração
            max_tokens: Máximo de tokens na resposta
            mock_mode: Se True, não inicializa LLM real (para testes)
            
        Raises:
            ValueError: Se não houver chave da API e não estiver em mock_mode
        """
        # Obtém API key
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.mock_mode = mock_mode
        
        if not self.api_key and not mock_mode:
            raise ValueError("API key é obrigatória. Forneça api_key ou configure GROQ_API_KEY")
        
        # Configurações do modelo
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Inicializa o LLM (se não estiver em mock mode)
        if not mock_mode:
            self.llm = ChatGroq(
                model_name=self.model_name,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        else:
            self.llm = None
            logger.info("Nó 3 em modo mock (sem LLM real)")
        
        logger.info(f"Nó 3 inicializado: modelo={model_name}, temperature={temperature}, mock_mode={mock_mode}")
