from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional, Dict, List
import os
import logging
import re

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
        model_name: str = "openai/gpt-oss-120b",
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
    
    def extract_structured_info(self, chunk: Dict) -> Dict:
        """
        Extrai informações estruturadas de um chunk.
        
        Args:
            chunk: Chunk com conteúdo e metadados
            
        Returns:
            Dicionário com informações categorizadas
        """
        logger.info(f"Extraindo informações estruturadas da seção: {chunk.get('section', 'unknown')}")
        
        content = chunk["content"].lower()
        section = chunk.get("section", "")
        
        result = {
            "technical_requirements": self._extract_technical_requirements(content, section),
            "deadlines": self._extract_deadlines(content, section),
            "documentation": self._extract_documentation(content, section),
            "object_info": self._extract_object_info(content, section),
            "risk_factors": self._extract_risk_factors(content, section),
            "section": section
        }
        
        logger.info(f"Informações extraídas: {len(result['technical_requirements'])} técnicos, {len(result['deadlines'])} prazos")
        return result
    
    def validate_structured_output(self, output: Dict) -> bool:
        """
        Valida a saída estruturada.
        
        Args:
            output: Dicionário de saída estruturada
            
        Returns:
            True se válido, False caso contrário
        """
        required_categories = [
            "technical_requirements",
            "deadlines",
            "documentation",
            "object_info",
            "risk_factors"
        ]
        
        for category in required_categories:
            if category not in output:
                logger.error(f"Categoria obrigatória ausente: {category}")
                return False
        
        # Verifica tipos
        list_categories = ["technical_requirements", "deadlines", "documentation", "risk_factors"]
        for category in list_categories:
            if not isinstance(output[category], list):
                logger.error(f"Categoria {category} deve ser uma lista")
                return False
        
        # object_info pode ser dicionário
        if not isinstance(output["object_info"], dict):
            logger.error("object_info deve ser um dicionário")
            return False
        
        logger.info("Validação estruturada concluída")
        return True
    
    def _extract_technical_requirements(self, content: str, section: str) -> List[str]:
        """Extrai requisitos técnicos do conteúdo."""
        requirements = []
        
        # Palavras-chave técnicas comuns em editais
        tech_keywords = [
            "processador", "cpu", "memória", "ram", "armazenamento", "ssd", "hdd",
            "monitor", "placa de vídeo", "sistema operacional", "software",
            "especificação técnica", "norma", "abnt", "iso", "certificação"
        ]
        
        for keyword in tech_keywords:
            if keyword in content:
                # Encontra a frase completa contendo a palavra-chave
                sentences = content.split('.')
                for sentence in sentences:
                    if keyword in sentence:
                        requirements.append(sentence.strip())
        
        return requirements
    
    def _extract_deadlines(self, content: str, section: str) -> List[str]:
        """Extrai prazos e deadlines do conteúdo."""
        deadlines = []
        
        deadline_keywords = ["prazo", "deadline", "entrega", "vigência", "dia", "mês", "ano"]
        
        for keyword in deadline_keywords:
            if keyword in content:
                sentences = content.split('.')
                for sentence in sentences:
                    if keyword in sentence:
                        deadlines.append(sentence.strip())
        
        return deadlines
    
    def _extract_documentation(self, content: str, section: str) -> List[str]:
        """Extrai documentação exigida do conteúdo."""
        documentation = []
        
        doc_keywords = ["document", "certidão", "licença", "registro", "alvará", "cnh", "rg", "cpf", "cnpj"]
        
        for keyword in doc_keywords:
            if keyword in content:
                sentences = content.split('.')
                for sentence in sentences:
                    if keyword in sentence:
                        documentation.append(sentence.strip())
        
        return documentation
    
    def _extract_object_info(self, content: str, section: str) -> Dict:
        """Extrai informações do objeto/bem."""
        info = {
            "description": "",
            "quantity": None,
            "unit": None
        }
        
        # Tenta extrair quantidade
        quantity_pattern = r'(\d+)\s*(unidade|computadores?|itens?|equipamentos?|dispositivos?)'
        match = re.search(quantity_pattern, content)
        if match:
            info["quantity"] = int(match.group(1))
            info["unit"] = match.group(2)
        
        # Tenta extrair descrição do objeto
        if "objeto" in content or "aquisição" in content:
            sentences = content.split('.')
            for sentence in sentences:
                if "objeto" in sentence or "aquisição" in sentence:
                    info["description"] = sentence.strip()
                    break
        
        return info
    
    def _extract_risk_factors(self, content: str, section: str) -> List[str]:
        """Extrai fatores de risco do conteúdo."""
        risks = []
        
        risk_keywords = ["risco", "crítico", "urgente", "complexo", "difícil", "curto prazo", "multa"]
        
        for keyword in risk_keywords:
            if keyword in content:
                sentences = content.split('.')
                for sentence in sentences:
                    if keyword in sentence:
                        risks.append(sentence.strip())
        
        return risks
    
    def identify_critical_points(self, chunk: Dict) -> Dict:
        """
        Identifica pontos críticos no chunk.
        
        Args:
            chunk: Chunk com conteúdo e metadados
            
        Returns:
            Dicionário com pontos críticos e nível de risco
        """
        logger.info(f"Identificando pontos críticos na seção: {chunk.get('section', 'unknown')}")
        
        content = chunk["content"].lower()
        section = chunk.get("section", "")
        
        if not content.strip():
            return {
                "critical_points": [],
                "risk_level": "BAIXO",
                "recommendations": [],
                "ambiguities": []
            }
        
        # Identifica pontos críticos
        critical_points = self._extract_critical_keywords(content)
        
        # Classifica nível de risco
        risk_level = self._classify_risk_level(content, critical_points)
        
        # Gera recomendações
        recommendations = self._generate_recommendations(content, critical_points, risk_level)
        
        # Identifica ambiguidades
        ambiguities = self._identify_ambiguities(content)
        
        return {
            "critical_points": critical_points,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "ambiguities": ambiguities,
            "section": section
        }
    
    def summarize_critical_points(self, chunks: List[Dict]) -> Dict:
        """
        Resume pontos críticos de múltiplos chunks.
        
        Args:
            chunks: Lista de chunks
            
        Returns:
            Dicionário com resumo crítico
        """
        logger.info(f"Resumindo pontos críticos de {len(chunks)} chunks")
        
        all_critical_points = []
        all_risk_levels = []
        
        for chunk in chunks:
            analysis = self.identify_critical_points(chunk)
            all_critical_points.extend(analysis["critical_points"])
            all_risk_levels.append(analysis["risk_level"])
        
        # Determina nível de risco geral
        overall_risk = self._calculate_overall_risk(all_risk_levels)
        
        # Ações prioritárias
        priority_actions = self._generate_priority_actions(all_critical_points, overall_risk)
        
        return {
            "total_critical_points": len(all_critical_points),
            "overall_risk_level": overall_risk,
            "priority_actions": priority_actions,
            "critical_points_by_risk": self._group_by_risk(all_critical_points, all_risk_levels)
        }
    
    def validate_critical_analysis(self, output: Dict) -> bool:
        """
        Valida a análise crítica.
        
        Args:
            output: Dicionário de saída da análise crítica
            
        Returns:
            True se válido, False caso contrário
        """
        required_fields = ["critical_points", "risk_level", "recommendations"]
        
        for field in required_fields:
            if field not in output:
                logger.error(f"Campo obrigatório ausente: {field}")
                return False
        
        # Verifica se risk_level é válido
        valid_risk_levels = ["ALTO", "MEDIO", "BAIXO"]
        if output["risk_level"] not in valid_risk_levels:
            logger.error(f"Nível de risco inválido: {output['risk_level']}")
            return False
        
        logger.info("Validação crítica concluída")
        return True
    
    def _extract_critical_keywords(self, content: str) -> List[str]:
        """Extrai palavras-chave críticas do conteúdo."""
        critical_keywords = [
            "impossível", "inviável", "curto prazo", "multa", "penalidade",
            "severo", "obrigatório", "exclusivo", "único", "imediato",
            "urgente", "complexo", "difícil", "risco", "perigo"
        ]
        
        points = []
        for keyword in critical_keywords:
            if keyword in content:
                sentences = content.split('.')
                for sentence in sentences:
                    if keyword in sentence:
                        points.append(sentence.strip())
        
        return points
    
    def _classify_risk_level(self, content: str, critical_points: List[str]) -> str:
        """Classifica o nível de risco."""
        if len(critical_points) >= 3:
            return "ALTO"
        elif len(critical_points) >= 1:
            return "MEDIO"
        else:
            return "BAIXO"
    
    def _generate_recommendations(self, content: str, critical_points: List[str], risk_level: str) -> List[str]:
        """Gera recomendações baseadas nos pontos críticos."""
        recommendations = []
        
        if "prazo" in content:
            recommendations.append("Verificar viabilidade do prazo estabelecido")
        
        if "multa" in content or "penalidade" in content:
            recommendations.append("Avaliar impacto financeiro das penalidades")
        
        if "obrigatório" in content or "exclusivo" in content:
            recommendations.append("Confirmar capacidade de atender requisitos obrigatórios")
        
        if risk_level == "ALTO":
            recommendations.append("Recomendada revisão detalhada do edital")
        
        return recommendations
    
    def _identify_ambiguities(self, content: str) -> List[str]:
        """Identifica ambiguidades no conteúdo."""
        ambiguity_keywords = [
            "a ser definido", "posteriormente", "conforme", "apropriado",
            "adequado", "suficiente", "necessário", "de acordo com"
        ]
        
        ambiguities = []
        for keyword in ambiguity_keywords:
            if keyword in content:
                sentences = content.split('.')
                for sentence in sentences:
                    if keyword in sentence:
                        ambiguities.append(sentence.strip())
        
        return ambiguities
    
    def _calculate_overall_risk(self, risk_levels: List[str]) -> str:
        """Calcula nível de risco geral."""
        if "ALTO" in risk_levels:
            return "ALTO"
        elif "MEDIO" in risk_levels:
            return "MEDIO"
        else:
            return "BAIXO"
    
    def _generate_priority_actions(self, critical_points: List[str], overall_risk: str) -> List[str]:
        """Gera ações prioritárias."""
        actions = []
        
        if overall_risk == "ALTO":
            actions.append("Reunião imediata com equipe técnica")
            actions.append("Análise detalhada de capacidade")
        elif overall_risk == "MEDIO":
            actions.append("Revisão de requisitos")
            actions.append("Planejamento de contingência")
        else:
            actions.append("Acompanhamento padrão do processo")
        
        return actions
    
    def _group_by_risk(self, critical_points: List[str], risk_levels: List[str]) -> Dict:
        """Agrupa pontos críticos por nível de risco."""
        grouped = {
            "ALTO": [],
            "MEDIO": [],
            "BAIXO": []
        }
        
        for point, risk in zip(critical_points, risk_levels):
            grouped[risk].append(point)
        
        return grouped
    
    def process_from_node2(self, chunks: List[Dict]) -> Dict:
        """
        Processa chunks vindos do Nó 2.
        
        Args:
            chunks: Lista de chunks do Nó 2
            
        Returns:
            Dicionário com análise completa
        """
        logger.info(f"Processando {len(chunks)} chunks do Nó 2")
        
        if not chunks:
            return {
                "total_chunks_processed": 0,
                "structured_analysis": {},
                "critical_analysis": {}
            }
        
        # Análise estruturada
        structured_analysis = {
            chunk.get("metadata", {}).get("chunk_id", i): self.extract_structured_info(chunk)
            for i, chunk in enumerate(chunks)
        }
        
        # Análise crítica
        critical_analysis = {
            chunk.get("metadata", {}).get("chunk_id", i): self.identify_critical_points(chunk)
            for i, chunk in enumerate(chunks)
        }
        
        return {
            "total_chunks_processed": len(chunks),
            "structured_analysis": structured_analysis,
            "critical_analysis": critical_analysis
        }
    
    def process_complete_analysis(self, node2_output: Dict) -> Dict:
        """
        Processa análise completa usando saída do Nó 2.
        
        Args:
            node2_output: Saída do Nó 2 com embeddings
            
        Returns:
            Dicionário com análise completa integrada
        """
        logger.info("Iniciando análise completa integrada")
        
        chunks = node2_output.get("chunks_with_embeddings", [])
        
        # Processa análise
        analysis = self.process_from_node2(chunks)
        
        # Combina análises
        combined = self.combine_analyses(chunks)
        
        # Gera resumo executivo
        summary = self.generate_executive_summary(combined)
        
        return {
            "summary": summary,
            "technical_analysis": analysis["structured_analysis"],
            "risk_assessment": analysis["critical_analysis"],
            "recommendations": combined["priority_recommendations"],
            "total_chunks_processed": analysis["total_chunks_processed"]
        }
    
    def combine_analyses(self, chunks: List[Dict]) -> Dict:
        """
        Combina análise estruturada e crítica.
        
        Args:
            chunks: Lista de chunks
            
        Returns:
            Dicionário com análises combinadas
        """
        logger.info("Combinando análises estruturada e crítica")
        
        all_structured = []
        all_critical = []
        all_risk_levels = []
        
        for chunk in chunks:
            structured = self.extract_structured_info(chunk)
            critical = self.identify_critical_points(chunk)
            
            all_structured.append(structured)
            all_critical.append(critical)
            all_risk_levels.append(critical["risk_level"])
        
        # Determina nível de risco combinado
        combined_risk = self._calculate_overall_risk(all_risk_levels)
        
        # Recomendações prioritárias
        priority_recommendations = []
        for critical in all_critical:
            priority_recommendations.extend(critical["recommendations"])
        
        return {
            "structured_info": all_structured,
            "critical_info": all_critical,
            "combined_risk_level": combined_risk,
            "priority_recommendations": list(set(priority_recommendations))  # Remove duplicatas
        }
    
    def generate_executive_summary(self, analysis_result: Dict) -> Dict:
        """
        Gera resumo executivo da análise.
        
        Args:
            analysis_result: Resultado da análise combinada
            
        Returns:
            Dicionário com resumo executivo
        """
        logger.info("Gerando resumo executivo")
        
        structured = analysis_result.get("structured_info", [])
        critical = analysis_result.get("critical_info", [])
        
        # Extrai principais descobertas
        key_findings = []
        for struct in structured:
            if struct.get("technical_requirements"):
                key_findings.extend(struct["technical_requirements"][:2])
            if struct.get("deadlines"):
                key_findings.extend(struct["deadlines"][:1])
        
        # Resumo de risco
        risk_summary = f"Nível de risco geral: {analysis_result.get('combined_risk_level', 'BAIXO')}"
        
        # Próximos passos
        next_steps = analysis_result.get("priority_recommendations", [])
        
        return {
            "overview": f"Análise de {len(structured)} chunks processados",
            "key_findings": key_findings[:5],  # Limita a 5 descobertas principais
            "risk_summary": risk_summary,
            "next_steps": next_steps[:3]  # Limita a 3 próximos passos
        }
    
    def validate_integration_output(self, output: Dict) -> bool:
        """
        Valida a saída da integração.
        
        Args:
            output: Dicionário de saída da integração
            
        Returns:
            True se válido, False caso contrário
        """
        required_fields = ["total_chunks_processed", "structured_analysis", "critical_analysis"]
        
        for field in required_fields:
            if field not in output:
                logger.error(f"Campo obrigatório ausente: {field}")
                return False
        
        # Verifica se total_chunks_processed é inteiro
        if not isinstance(output["total_chunks_processed"], int):
            logger.error("total_chunks_processed deve ser um inteiro")
            return False
        
        logger.info("Validação de integração concluída")
        return True
