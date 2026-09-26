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
    
    def analyze_chunk(self, chunk: Dict, system_prompt: Optional[str] = None) -> Dict:
        """
        Analisa um único chunk de texto.
        
        Args:
            chunk: Chunk com conteúdo e metadados
            system_prompt: Prompt de sistema customizado
            
        Returns:
            Dicionário com análise do chunk
        """
        logger.info(f"Analisando chunk da seção: {chunk.get('section', 'unknown')}")
        
        if self.mock_mode:
            # Resposta simulada em modo mock
            mock_response = f"Análise mock do chunk: {chunk['content'][:50]}..."
            return {
                "requirements": self._extract_requirements_mock(chunk),
                "analysis": mock_response,
                "raw_response": mock_response,
                "chunk_metadata": {
                    "section": chunk.get("section"),
                    "content_length": len(chunk["content"])
                },
                "system_prompt": system_prompt
            }
        
        # Análise real com LLM
        prompt = self._build_analysis_prompt(chunk)
        
        if system_prompt:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        else:
            messages = [HumanMessage(content=prompt)]
        
        response = self.llm.invoke(messages)
        
        return {
            "requirements": self._extract_requirements_from_response(response.content),
            "analysis": response.content,
            "raw_response": response.content,
            "chunk_metadata": {
                "section": chunk.get("section"),
                "content_length": len(chunk["content"])
            },
            "system_prompt": system_prompt
        }
    
    def analyze_chunks(self, chunks: List[Dict], system_prompt: Optional[str] = None) -> Dict:
        """
        Analisa múltiplos chunks.
        
        Args:
            chunks: Lista de chunks
            system_prompt: Prompt de sistema customizado
            
        Returns:
            Dicionário com análises de todos os chunks
        """
        logger.info(f"Analisando {len(chunks)} chunks")
        
        if not chunks:
            return {
                "total_chunks": 0,
                "analyses": [],
                "system_prompt": system_prompt
            }
        
        analyses = []
        for chunk in chunks:
            analysis = self.analyze_chunk(chunk, system_prompt)
            analyses.append(analysis)
        
        return {
            "total_chunks": len(chunks),
            "analyses": analyses,
            "system_prompt": system_prompt
        }
    
    def validate_analysis_output(self, output: Dict) -> bool:
        """
        Valida a saída da análise.
        
        Args:
            output: Dicionário de saída da análise
            
        Returns:
            True se válido, False caso contrário
        """
        required_fields = ["requirements", "analysis", "raw_response"]
        
        for field in required_fields:
            if field not in output:
                logger.error(f"Campo obrigatório ausente: {field}")
                return False
        
        if not output["requirements"]:
            logger.warning("Nenhum requisito extraído")
        
        logger.info("Validação de análise concluída")
        return True
    
    def _build_analysis_prompt(self, chunk: Dict) -> str:
        """Constrói o prompt para análise do chunk."""
        section = chunk.get("section", "desconhecida")
        content = chunk["content"]
        
        prompt = f"""
Analise o seguinte chunk de um edital de licitação:

Seção: {section}
Conteúdo: {content}

Identifique:
1. Requisitos técnicos mencionados
2. Prazos ou cronogramas
3. Documentação exigida
4. Pontos críticos ou ambíguos

Responda de forma estruturada e concisa.
"""
        return prompt
    
    def _extract_requirements_mock(self, chunk: Dict) -> List[str]:
        """Extrai requisitos simulados em modo mock."""
        content = chunk["content"].lower()
        section = chunk.get("section", "")
        
        requirements = []
        
        # Simples extrações baseadas em palavras-chave
        if "processador" in content or "cpu" in content:
            requirements.append("Requisito de processador encontrado")
        if "memoria" in content or "ram" in content:
            requirements.append("Requisito de memória encontrado")
        if "prazo" in content:
            requirements.append("Prazo mencionado")
        if "document" in content:
            requirements.append("Documentação exigida")
        
        return requirements
    
    def _extract_requirements_from_response(self, response: str) -> List[str]:
        """Extrai requisitos da resposta do LLM."""
        # Implementação básica - pode ser melhorada com parsing mais sofisticado
        lines = response.split('\n')
        requirements = []
        
        for line in lines:
            line = line.strip()
            if line and ('requisito' in line.lower() or 'exigência' in line.lower() or 'deve' in line.lower()):
                requirements.append(line)
        
        return requirements
