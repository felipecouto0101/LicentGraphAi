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
