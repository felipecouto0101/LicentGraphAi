import logging
import os
import re
import time
import unicodedata

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Callback global de progresso — preenchido pela API quando há um job ativo
_progress_callback = None


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
        api_key: str | None = None,
        model_name: str = "qwen/qwen3.8-27b",
        temperature: float = 0.3,
        max_tokens: int = 1600,
        mock_mode: bool = False,
        persist_directory: str = "./data/vector_db",
        collection_name: str = "licitacoes",
    ):
        """
        Inicializa o Nó 3.

        Args:
            api_key: Chave da API Groq (ou usa variável de ambiente)
            model_name: Nome do modelo LLM
            temperature: Temperatura para geração
            max_tokens: Máximo de tokens na resposta
            mock_mode: Se True, não inicializa LLM real (para testes)
            persist_directory: Diretório do ChromaDB (para busca RAG)
            collection_name: Nome da coleção ChromaDB
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.mock_mode = mock_mode
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        if not self.api_key and not mock_mode:
            raise ValueError(
                "API key é obrigatória. Forneça api_key ou configure GROQ_API_KEY"
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not mock_mode:
            self._api_keys = self._load_api_keys()
            self._current_key_idx = 0
            self.llm = self._make_llm(self._api_keys[0])
            logger.info(f"Nó 3: {len(self._api_keys)} chave(s) API carregada(s)")
        else:
            self._api_keys = []
            self.llm = None
            logger.info("Nó 3 em modo mock (sem LLM real)")

        logger.info(
            f"Nó 3 inicializado: modelo={model_name}, temperature={temperature}, mock_mode={mock_mode}"
        )

    def _load_api_keys(self) -> list[str]:
        """Carrega todas as chaves API disponíveis do ambiente."""
        keys = []
        # GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, ...
        for suffix in ["", "_2", "_3", "_4", "_5"]:
            k = os.getenv(f"GROQ_API_KEY{suffix}", "").strip()
            if k and k != "your_groq_api_key_here":
                keys.append(k)
        if not keys and self.api_key:
            keys.append(self.api_key)
        return keys

    def _make_llm(self, api_key: str) -> ChatGroq:
        """Cria uma instância do LLM com a chave fornecida."""
        return ChatGroq(
            model_name=self.model_name,
            api_key=api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _rotate_key(self) -> bool:
        """
        Tenta rotacionar para a próxima chave API disponível.
        Retorna True se conseguiu, False se não há mais chaves.
        """
        next_idx = self._current_key_idx + 1
        if next_idx >= len(self._api_keys):
            logger.error("Todas as chaves API esgotaram o limite. Sem mais chaves disponíveis.")
            return False
        self._current_key_idx = next_idx
        self.llm = self._make_llm(self._api_keys[next_idx])
        logger.warning(f"Rotacionando para chave {next_idx + 1}/{len(self._api_keys)}")
        return True

    def _invoke_with_rotation(self, messages: list, sleep_between: float = 8.0):
        """
        Invoca o LLM com rotação automática de chave em caso de 429.
        Tenta todas as chaves disponíveis antes de desistir.
        """
        keys_tried = 0
        while keys_tried <= len(self._api_keys):
            try:
                return self.llm.invoke(messages)
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
                is_daily_limit = "tokens per day" in err_str.lower() or "TPD" in err_str

                if is_rate_limit:
                    if is_daily_limit:
                        # Limite diário esgotado — tenta próxima chave
                        logger.warning(f"Chave {self._current_key_idx + 1} esgotou limite diário.")
                        if not self._rotate_key():
                            raise
                        keys_tried += 1
                    else:
                        # Rate limit por minuto — aguarda e tenta de novo
                        import re as _re
                        wait_match = _re.search(r"try again in (\d+)m([\d.]+)s", err_str)
                        wait_secs = 60.0
                        if wait_match:
                            wait_secs = int(wait_match.group(1)) * 60 + float(wait_match.group(2))
                            wait_secs = min(wait_secs + 5, 120)  # máx 2 min de espera
                        logger.warning(f"Rate limit por minuto. Aguardando {wait_secs:.0f}s...")
                        time.sleep(wait_secs)
                        # Tenta a mesma chave de novo
                        keys_tried += 1
                        continue
                else:
                    raise
        raise RuntimeError("Todas as chaves API falharam.")

    def analyze_chunk(self, chunk: dict, system_prompt: str | None = None) -> dict:
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
                    "content_length": len(chunk["content"]),
                },
                "system_prompt": system_prompt,
            }

        # Análise real com LLM
        prompt = self._build_analysis_prompt(chunk)

        if system_prompt:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ]
        else:
            messages = [HumanMessage(content=prompt)]

        response = self.llm.invoke(messages)

        return {
            "requirements": self._extract_requirements_from_response(response.content),
            "analysis": response.content,
            "raw_response": response.content,
            "chunk_metadata": {
                "section": chunk.get("section"),
                "content_length": len(chunk["content"]),
            },
            "system_prompt": system_prompt,
        }

    def analyze_chunks(
        self, chunks: list[dict], system_prompt: str | None = None
    ) -> dict:
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
            return {"total_chunks": 0, "analyses": [], "system_prompt": system_prompt}

        analyses = []
        for chunk in chunks:
            analysis = self.analyze_chunk(chunk, system_prompt)
            analyses.append(analysis)

        return {
            "total_chunks": len(chunks),
            "analyses": analyses,
            "system_prompt": system_prompt,
        }

    def validate_analysis_output(self, output: dict) -> bool:
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

    def _build_analysis_prompt(self, chunk: dict) -> str:
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

    def _extract_requirements_mock(self, chunk: dict) -> list[str]:
        """Extrai requisitos simulados em modo mock."""
        content = chunk["content"].lower()

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

    def _extract_requirements_from_response(self, response: str) -> list[str]:
        """Extrai requisitos da resposta do LLM."""
        # Implementação básica - pode ser melhorada com parsing mais sofisticado
        lines = response.split("\n")
        requirements = []

        for line in lines:
            line = line.strip()
            if line and (
                "requisito" in line.lower()
                or "exigência" in line.lower()
                or "deve" in line.lower()
            ):
                requirements.append(line)

        return requirements

    def extract_structured_info(self, chunk: dict) -> dict:
        """
        Extrai informações estruturadas de um chunk.

        Args:
            chunk: Chunk com conteúdo e metadados

        Returns:
            Dicionário com informações categorizadas
        """
        logger.info(
            f"Extraindo informações estruturadas da seção: {chunk.get('section', 'unknown')}"
        )

        content = chunk["content"].lower()
        section = chunk.get("section", "")

        result = {
            "technical_requirements": self._extract_technical_requirements(
                content, section
            ),
            "deadlines": self._extract_deadlines(content, section),
            "documentation": self._extract_documentation(content, section),
            "object_info": self._extract_object_info(content, section),
            "risk_factors": self._extract_risk_factors(content, section),
            "section": section,
        }

        logger.info(
            f"Informações extraídas: {len(result['technical_requirements'])} técnicos, {len(result['deadlines'])} prazos"
        )
        return result

    def validate_structured_output(self, output: dict) -> bool:
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
            "risk_factors",
        ]

        for category in required_categories:
            if category not in output:
                logger.error(f"Categoria obrigatória ausente: {category}")
                return False

        # Verifica tipos
        list_categories = [
            "technical_requirements",
            "deadlines",
            "documentation",
            "risk_factors",
        ]
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

    # ------------------------------------------------------------------
    # Helpers de limpeza de texto
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_sentences(sentences: list[str], min_words: int = 4, max_chars: int = 180) -> list[str]:
        """
        Remove frases muito curtas, muito longas ou claramente inúteis.

        Regras:
        - Pelo menos `min_words` palavras
        - No máximo `max_chars` caracteres (trunca com reticências)
        - Sem linhas que sejam só números/pontuação
        """
        seen: set[str] = set()
        result: list[str] = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            words = s.split()
            if len(words) < min_words:
                continue
            # Remove frases que são basicamente ruído (só dígitos/pontuação)
            if sum(c.isalpha() for c in s) < 10:
                continue
            # Trunca frases longas demais
            if len(s) > max_chars:
                s = s[:max_chars].rsplit(" ", 1)[0] + "..."
            # Dedup por conteúdo normalizado
            key = " ".join(words[:8])  # primeiras 8 palavras como chave
            if key in seen:
                continue
            seen.add(key)
            result.append(s)
        return result

    def _extract_technical_requirements(self, content: str, section: str) -> list[str]:
        """Extrai requisitos técnicos do conteúdo."""
        # Palavras-chave específicas o suficiente para não gerar ruído
        tech_keywords = [
            "processador", "cpu", "memória ram", "armazenamento",
            "ssd", "hdd", "monitor", "placa de vídeo",
            "sistema operacional", "especificação técnica",
            "norma abnt", "certificação iso", "certificação abnt",
        ]

        found: list[str] = []
        sentences = content.split(".")
        for keyword in tech_keywords:
            for sentence in sentences:
                if keyword in sentence:
                    found.append(sentence)

        return self._clean_sentences(found, min_words=5, max_chars=160)

    def _extract_deadlines(self, content: str, section: str) -> list[str]:
        """
        Extrai prazos usando regex para capturar só o trecho relevante
        (ex.: '30 dias', '10 dias úteis', '15/03/2026'), não a frase inteira.
        """
        import re

        patterns = [
            # "prazo de X [por extenso] dias [úteis/corridos] ..."
            r"prazo\s+de\s+(?:\d+|\w+)\s*(?:\([^)]+\)\s*)?\s*dias?[^,;\n.]{0,60}",
            # "X (por extenso) dias após/contado/a contar"
            r"\d+\s*(?:\([^)]+\)\s*)?dias?\s+[a-záàâãéêíóôõúç ]{0,15}(?:após|contado|a contar)[^,;\n.]{0,50}",
            # "até DD/MM/AAAA"
            r"até\s+\d{1,2}/\d{1,2}/\d{2,4}",
            # "vigência de X meses/anos"
            r"vigência\s+de\s+\d+\s*(?:meses?|anos?)[^,;\n.]{0,40}",
            # "entrega em/no prazo de X dias"
            r"entrega\s+(?:em|no\s+prazo\s+de)\s+\d+\s*dias?[^,;\n.]{0,40}",
        ]

        found: list[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                text = match.group(0).strip().rstrip(",;:")
                if len(text.split()) >= 3:  # mínimo 3 palavras
                    found.append(text)

        # Dedup por prefixo (primeiros 35 chars)
        seen: set[str] = set()
        result: list[str] = []
        for item in found:
            key = item[:35].lower()
            if key not in seen:
                seen.add(key)
                result.append(item.capitalize())

        return result[:20]

    def _extract_documentation(self, content: str, section: str) -> list[str]:
        """
        Extrai nomes de documentos, não frases inteiras.
        Prioriza a keyword em si + palavras adjacentes para nomear o documento.
        """
        import re

        # Documentos com nome fixo — retorna o nome canônico direto
        fixed_docs = {
            "certidão negativa de débitos": "Certidão Negativa de Débitos (CND)",
            "certidão negativa": "Certidão Negativa",
            "certidão positiva": "Certidão Positiva com Efeito de Negativa",
            "balanço patrimonial": "Balanço Patrimonial",
            "demonstrativo de resultados": "Demonstrativo de Resultados",
            "registro no cnpj": "Comprovante de Inscrição no CNPJ",
            "cnpj": "Cartão CNPJ",
            "contrato social": "Contrato Social",
            "estatuto social": "Estatuto Social",
            "ata de eleição": "Ata de Eleição da Diretoria",
            "procuração": "Procuração",
            "atestado de capacidade técnica": "Atestado de Capacidade Técnica",
            "atestado técnico": "Atestado Técnico",
            "licença de funcionamento": "Licença de Funcionamento",
            "alvará": "Alvará de Funcionamento",
            "cnh": "CNH",
            "registro geral": "RG",
            r"\brg\b": "RG",
            r"\bcpf\b": "CPF",
            "inscrição estadual": "Inscrição Estadual",
            "inscrição municipal": "Inscrição Municipal",
            "certidão trabalhista": "Certidão de Regularidade Trabalhista (CNDT)",
            "prova de regularidade": "Prova de Regularidade Fiscal",
            "comprovante de regularidade": "Comprovante de Regularidade",
            "declaração de idoneidade": "Declaração de Idoneidade",
            "declaração de inexistência": "Declaração de Inexistência de Fatos Impeditivos",
        }

        found: list[str] = []
        seen: set[str] = set()

        for pattern, canonical_name in fixed_docs.items():
            if re.search(pattern, content, re.IGNORECASE):
                if canonical_name not in seen:
                    seen.add(canonical_name)
                    found.append(canonical_name)

        return found

    def _extract_object_info(self, content: str, section: str) -> dict:
        """Extrai informações do objeto/bem."""
        info = {"description": "", "quantity": None, "unit": None}

        # Tenta extrair quantidade
        quantity_pattern = (
            r"(\d+)\s*(unidade|computadores?|itens?|equipamentos?|dispositivos?)"
        )
        match = re.search(quantity_pattern, content)
        if match:
            info["quantity"] = int(match.group(1))
            info["unit"] = match.group(2)

        # Tenta extrair descrição do objeto
        if "objeto" in content or "aquisição" in content:
            sentences = content.split(".")
            for sentence in sentences:
                if "objeto" in sentence or "aquisição" in sentence:
                    info["description"] = sentence.strip()
                    break

        return info

    def _extract_risk_factors(self, content: str, section: str) -> list[str]:
        """Extrai fatores de risco do conteúdo."""
        risks = []

        risk_keywords = [
            "risco",
            "crítico",
            "urgente",
            "complexo",
            "difícil",
            "curto prazo",
            "multa",
        ]

        for keyword in risk_keywords:
            if keyword in content:
                sentences = content.split(".")
                for sentence in sentences:
                    if keyword in sentence:
                        risks.append(sentence.strip())

        return risks

    def identify_critical_points(self, chunk: dict) -> dict:
        """
        Identifica pontos críticos no chunk.

        Args:
            chunk: Chunk com conteúdo e metadados

        Returns:
            Dicionário com pontos críticos e nível de risco
        """
        logger.info(
            f"Identificando pontos críticos na seção: {chunk.get('section', 'unknown')}"
        )

        content = chunk["content"].lower()
        section = chunk.get("section", "")

        if not content.strip():
            return {
                "critical_points": [],
                "risk_level": "BAIXO",
                "recommendations": [],
                "ambiguities": [],
            }

        # Identifica pontos críticos
        critical_points = self._extract_critical_keywords(content)

        # Classifica nível de risco
        risk_level = self._classify_risk_level(content, critical_points)

        # Gera recomendações
        recommendations = self._generate_recommendations(
            content, critical_points, risk_level
        )

        # Identifica ambiguidades
        ambiguities = self._identify_ambiguities(content)

        return {
            "critical_points": critical_points,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "ambiguities": ambiguities,
            "section": section,
        }

    def summarize_critical_points(self, chunks: list[dict]) -> dict:
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
        priority_actions = self._generate_priority_actions(
            all_critical_points, overall_risk
        )

        return {
            "total_critical_points": len(all_critical_points),
            "overall_risk_level": overall_risk,
            "priority_actions": priority_actions,
            "critical_points_by_risk": self._group_by_risk(
                all_critical_points, all_risk_levels
            ),
        }

    def validate_critical_analysis(self, output: dict) -> bool:
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

    def _extract_critical_keywords(self, content: str) -> list[str]:
        """
        Extrai pontos críticos do conteúdo.

        Filtra frases que claramente vieram corrompidas do PDF:
        - Muitas palavras com ≤2 letras (abreviações/ruído)
        - Menos de 6 palavras reais
        - Referências ao nome do edital (repetitivas, sem valor)
        """
        critical_keywords = [
            "impossível", "inviável", "curto prazo", "multa", "penalidade",
            "severo", "obrigatório", "exclusivo", "único", "imediato",
            "urgente", "complexo", "difícil", "risco", "perigo",
            "impedimento", "sanção", "rescisão", "inadimplência",
        ]

        # Padrões de ruído que indicam texto corrompido
        noise_patterns = [
            r"\bedital\s+concorrência\s+eletrônica\b",  # referência repetitiva ao edital
            r"\bprocesso\s+administrativo\s+n[°º]",
            r"[a-z]\)\s+[a-z]\)\s+[a-z]\)",             # sequências tipo "a) b) c)"
        ]

        candidates: list[str] = []
        sentences = content.split(".")
        for keyword in critical_keywords:
            for sentence in sentences:
                if keyword in sentence:
                    candidates.append(sentence)

        result: list[str] = []
        seen: set[str] = set()

        for sent in candidates:
            sent = sent.strip()
            if not sent:
                continue

            # Descarta se contém padrões de ruído
            is_noisy = any(re.search(p, sent, re.IGNORECASE) for p in noise_patterns)
            if is_noisy:
                continue

            words = sent.split()
            # Descarta frases muito curtas
            if len(words) < 6:
                continue

            # Descarta se >40% das palavras têm ≤2 letras (texto corrompido)
            short_words = sum(1 for w in words if len(re.sub(r"[^a-záàâãéêíóôõúç]", "", w.lower())) <= 2)
            if short_words / len(words) > 0.40:
                continue

            # Trunca e dedup
            if len(sent) > 200:
                sent = sent[:200].rsplit(" ", 1)[0] + "..."
            key = " ".join(words[:7])
            if key in seen:
                continue
            seen.add(key)
            result.append(sent.capitalize())

        return result[:15]  # máximo 15 pontos críticos por chunk

    def _classify_risk_level(self, content: str, critical_points: list[str]) -> str:
        """Classifica o nível de risco."""
        if len(critical_points) >= 3:
            return "ALTO"
        elif len(critical_points) >= 1:
            return "MEDIO"
        else:
            return "BAIXO"

    def _generate_recommendations(
        self, content: str, critical_points: list[str], risk_level: str
    ) -> list[str]:
        """Gera recomendações baseadas nos pontos críticos."""
        recommendations = []

        if "prazo" in content:
            recommendations.append("Verificar viabilidade do prazo estabelecido")

        if "multa" in content or "penalidade" in content:
            recommendations.append("Avaliar impacto financeiro das penalidades")

        if "obrigatório" in content or "exclusivo" in content:
            recommendations.append(
                "Confirmar capacidade de atender requisitos obrigatórios"
            )

        if risk_level == "ALTO":
            recommendations.append("Recomendada revisão detalhada do edital")

        return recommendations

    def _identify_ambiguities(self, content: str) -> list[str]:
        """Identifica ambiguidades no conteúdo."""
        ambiguity_keywords = [
            "a ser definido",
            "posteriormente",
            "conforme",
            "apropriado",
            "adequado",
            "suficiente",
            "necessário",
            "de acordo com",
        ]

        ambiguities = []
        for keyword in ambiguity_keywords:
            if keyword in content:
                sentences = content.split(".")
                for sentence in sentences:
                    if keyword in sentence:
                        ambiguities.append(sentence.strip())

        return ambiguities

    def _calculate_overall_risk(self, risk_levels: list[str]) -> str:
        """Calcula nível de risco geral."""
        if "ALTO" in risk_levels:
            return "ALTO"
        elif "MEDIO" in risk_levels:
            return "MEDIO"
        else:
            return "BAIXO"

    def _generate_priority_actions(
        self, critical_points: list[str], overall_risk: str
    ) -> list[str]:
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

    def _group_by_risk(
        self, critical_points: list[str], risk_levels: list[str]
    ) -> dict:
        """Agrupa pontos críticos por nível de risco."""
        grouped = {"ALTO": [], "MEDIO": [], "BAIXO": []}

        for point, risk in zip(critical_points, risk_levels):
            grouped[risk].append(point)

        return grouped

    def _build_structured_prompt(self, chunks: list[dict]) -> str:
        """
        Monta prompt para análise estruturada de múltiplos chunks via LLM.
        Retorna JSON com campos padronizados.
        """
        # Concatena conteúdo dos chunks com identificação de seção
        sections_text = ""
        for chunk in chunks:
            section = chunk.get("section", "geral")
            content = chunk.get("content", "").strip()
            if content:
                sections_text += f"\n--- SEÇÃO: {section.upper()} ---\n{content}\n"

        prompt = f"""Você é um especialista em licitações públicas brasileiras (Lei 14.133/2021).
Analise os trechos abaixo de um edital de licitação e extraia as informações solicitadas.

TRECHOS DO EDITAL:
{sections_text}

Responda SOMENTE com um JSON válido no seguinte formato (sem markdown, sem explicações fora do JSON):
{{
  "objeto": "descrição resumida do objeto da licitação em 1-2 frases",
  "requisitos_tecnicos": ["lista de requisitos técnicos objetivos, cada item em 1 frase clara"],
  "documentos_exigidos": ["nome canônico de cada documento exigido, ex: Certidão Negativa de Débitos, CNPJ, Atestado Técnico"],
  "prazos": ["cada prazo em formato legível, ex: 30 dias úteis para entrega"],
  "pontos_criticos": ["cada ponto crítico ou risco em 1 frase objetiva"],
  "nivel_risco": "ALTO|MEDIO|BAIXO",
  "recomendacoes": ["até 3 recomendações objetivas para a empresa participante"]
}}

Regras:
- Seja objetivo e conciso. Cada item deve ser uma frase completa e compreensível.
- Não repita informações entre campos.
- Se não houver informação para um campo, use lista vazia [].
- Para nivel_risco: ALTO se houver multas severas/prazos curtos/requisitos complexos, MEDIO se moderado, BAIXO se simples.
"""
        return prompt

    def _parse_llm_json(self, response: str) -> dict:
        """
        Parseia o JSON retornado pelo LLM de forma robusta.
        Tenta extrair o JSON mesmo se o LLM incluir texto extra.
        """
        import json

        # Tenta parsear direto
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Tenta extrair bloco JSON com regex
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: retorna estrutura vazia
        logger.warning("Não foi possível parsear JSON do LLM, usando fallback")
        return {
            "objeto": "",
            "requisitos_tecnicos": [],
            "documentos_exigidos": [],
            "prazos": [],
            "pontos_criticos": [],
            "nivel_risco": "BAIXO",
            "recomendacoes": [],
        }

    # ──────────────────────────────────────────────────────────────
    # Queries RAG temáticas
    # ──────────────────────────────────────────────────────────────

    # Cada entrada: (chave_interna, label_display, texto_da_query_para_embedding)
    RAG_QUERIES: list[tuple[str, str, str]] = [
        (
            "objeto_licitacao",
            "Objeto da Licitação",
            "objeto da licitação contratação aquisição serviço bem fornecimento "
            "descrição do objeto finalidade do contrato o que será contratado",
        ),
        (
            "posso_participar",
            "Posso participar?",
            "requisitos habilitação participação licitante empresa qualificação "
            "escolaridade idade registro profissional documentos específicos "
            "impedimento inabilitação condições de participação",
        ),
        (
            "prazos",
            "Quais são os prazos?",
            "prazo entrega vigência contrato cronograma data limite "
            "dias úteis corridos início execução encerramento",
        ),
        (
            "custos_pagamento",
            "Quanto custa e como pago?",
            "valor estimado preço pagamento faturamento nota fiscal "
            "reajuste reequilíbrio financeiro garantia caução depósito",
        ),
        (
            "selecao_processo",
            "Como será a seleção ou avaliação?",
            "critério julgamento menor preço técnica proposta avaliação "
            "pontuação classificação habilitação etapas processo seleção",
        ),
        (
            "objeto_escopo",
            "O que devo entregar ou produzir?",
            "objeto escopo serviço produto bem especificação técnica "
            "entregável quantidade unidade memorial descritivo planilha",
        ),
        (
            "eliminacao_penalidades",
            "Quais são as regras de eliminação?",
            "eliminação desclassificação inabilitação penalidade multa "
            "sanção rescisão impedimento declaração inidoneidade",
        ),
        (
            "documentos",
            "Quais documentos são exigidos?",
            "documentação exigida certidão comprovante declaração atestado "
            "registro alvará licença CNPJ balanço contrato social habilitação "
            "regularidade fiscal trabalhista jurídica técnica",
        ),
        (
            "pontos_criticos",
            "Pontos críticos e riscos",
            "risco crítico urgente multa penalidade obrigatório prazo curto "
            "complexo impossível severo rescisão inadimplência garantia",
        ),
    ]

    def _retrieve_rag_chunks(
        self,
        top_k: int = 5,
        keys_override: list[str] | None = None,
    ) -> dict[str, list[dict]]:
        """
        Para cada query RAG temática, busca os top_k chunks mais relevantes
        no ChromaDB usando similaridade vetorial.

        Args:
            top_k: Número de chunks a recuperar por query
            keys_override: Se fornecido, busca apenas as queries com essas chaves

        Returns:
            Dict: {chave_query: [chunks_relevantes]}
        """
        from .node_2_embeddings import Node2EmbeddingGenerator

        logger.info(f"RAG: iniciando busca vetorial em '{self.persist_directory}'")

        node2 = Node2EmbeddingGenerator()
        results_by_query: dict[str, list[dict]] = {}

        queries_to_run = [
            (key, label, qt) for key, label, qt in self.RAG_QUERIES
            if keys_override is None or key in keys_override
        ]

        for key, label, query_text in queries_to_run:
            try:
                raw = node2.search_similar(
                    query=query_text,
                    collection_name=self.collection_name,
                    n_results=top_k,
                    persist_directory=self.persist_directory,
                )
                docs = raw.get("documents", [[]])[0]
                metas = raw.get("metadatas", [[]])[0]

                chunks = [
                    {
                        "content": doc,
                        "section": meta.get("section", "geral"),
                        "metadata": meta,
                        "rag_query": label,
                    }
                    for doc, meta in zip(docs, metas)
                    if doc and doc.strip()
                ]
                results_by_query[key] = chunks
                logger.info(f"RAG '{label}': {len(chunks)} chunks recuperados")

            except Exception as e:
                raise RuntimeError(f"Falha na busca vetorial RAG para '{label}': {e}") from e

        return results_by_query


    def _build_unified_prompt(self, chunks: list[dict]) -> str:
        """
        Prompt unificado: extrai TODAS as informacoes relevantes de um batch
        em uma unica chamada ao LLM.
        """
        context = "\n\n".join(
            f"[Trecho {i+1}]\n{c['content'].strip()}"
            for i, c in enumerate(chunks)
            if c.get("content", "").strip()
        )
        return f"""Especialista em licitacoes publicas brasileiras (Lei 14.133/2021).

Analise os trechos abaixo e extraia TODAS as informacoes encontradas.

TRECHOS:
{context}

Responda SOMENTE com JSON valido (sem markdown):
{{
  "objeto": "descricao do objeto se encontrado, senao string vazia",
  "documentos": ["documentos que o licitante deve apresentar na proposta ou habilitacao, ex: certidao, atestado ou planilha"],
  "documentos_execucao": ["documentos a produzir ou entregar somente apos a contratacao, se a etapa estiver explicita"],
  "anexos_referencia": ["anexos e documentos fornecidos pelo orgao para consulta, sem pedido de entrega pelo licitante"],
  "pendencias_documentais": ["documentos citados cuja etapa ou exigencia de entrega nao ficou clara nos trechos"],
  "requisitos_participacao": ["cada requisito objetivo para participar, ex: CNPJ ativo, Registro no CREA"],
  "prazos": ["cada prazo com valor, ex: 90 dias corridos para execucao, 30 dias para pagamento"],
  "custos": ["valores e formas de pagamento encontrados"],
  "selecao": ["etapas e criterios de selecao encontrados"],
  "entregas": ["o que deve ser entregue ou executado"],
  "eliminacao": ["causas de eliminacao ou penalidades com valores"],
  "riscos": ["riscos e pontos criticos objetivos"],
  "nivel_risco": "ALTO|MEDIO|BAIXO"
}}

Regras:
- Inclua apenas o que esta EXPLICITAMENTE nos trechos.
- Classifique cada documento em UMA das quatro listas: entrega na proposta/habilitacao,
  entrega apos contratacao, anexo fornecido pelo orgao ou etapa incerta.
- Projeto Basico, minutas e anexos para consulta nao sao documentos do checklist.
- Condicoes, proibicoes e atividades vao em requisitos_participacao, eliminacao ou entregas.
- Nao crie um documento a partir de uma obrigacao que nao pede comprovante explicito.
- Em selecao, inclua somente etapas da disputa, julgamento e avaliacao de propostas.
- Em requisitos_participacao, inclua condicoes de participacao e habilitacao.
- Nao repita um mesmo fato com palavras diferentes no mesmo campo.
- Se trechos parecerem contraditorios, preserve as formulacoes sem escolher uma como correta.
- Se nao ha informacao para um campo, use lista vazia ou string vazia.
- Seja conciso: cada item em no maximo 15 palavras.
"""

    @staticmethod
    def _report_context(agg: dict, limit_per_topic: int = 12) -> tuple[str, dict]:
        """Amostra distribuída dos itens extraídos, com cobertura explícita."""
        import json

        topics = {
            "participacao": agg["requisitos_participacao"],
            "criterios_tecnicos_e_selecao": agg["selecao"],
            "prazos": agg["prazos"],
            "custos": agg["custos"],
            "entregas": agg["entregas"],
            "eliminacao": agg["eliminacao"],
            "riscos": agg["riscos"],
        }
        coverage = {}
        excerpts = {}
        for topic, items in topics.items():
            count = min(len(items), limit_per_topic)
            if count == len(items):
                selected = items
            elif count == 1:
                selected = [items[0]]
            else:
                positions = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
                selected = [items[pos] for pos in positions]
            excerpts[topic] = selected
            coverage[topic] = {"included": count, "total": len(items)}

        context = json.dumps(
            {"objeto": agg["objeto"], "nivel_risco": agg["nivel_risco"], "itens": excerpts},
            ensure_ascii=False,
        )
        return context, coverage

    def _synthesize_explanatory_report(
        self, agg: dict, source_refs: dict | None = None
    ) -> dict:
        """Gera explicação separada dos itens brutos, sem alterar o checklist."""
        context, coverage = self._report_context(agg)
        import json
        sources = source_refs or {}
        # Limita o contexto de evidências sem esconder as fontes completas,
        # que permanecem na resposta para conferência na interface.
        evidence = {
            topic: [
                {"page": item["page"], "excerpt": item["excerpt"][:350]}
                for item in items[:2]
            ]
            for topic, items in sources.items()
        }
        prompt = (
            "Escreva em português um relatório claro para quem está lendo um edital. "
            "Use SOMENTE os itens extraídos abaixo; não invente obrigações, valores, "
            "prazos, conclusões jurídicas nem aptidão da empresa. "
            "Se não houver informação, diga que não foi identificada na amostra. "
            "A amostra não representa necessariamente o edital inteiro. "
            "Responda APENAS em JSON com as chaves resumo e requisitos. "
            "Em resumo, escreva 2 parágrafos explicando o objeto, o que se espera "
            "do participante, prazos e principais pontos de atenção. "
            "Cada parágrafo deve ter no máximo 65 palavras. "
            "Em requisitos, escreva 2 ou 3 parágrafos conectando condições de participação, "
            "critérios técnicos e de seleção com suas implicações práticas, conforme os itens. "
            "Não faça lista e não inclua documentos obrigatórios: eles têm checklist próprio. "
            "Não repita contagens nem trate risco como decisão definitiva. "
            "Quando mencionar fato apoiado pelos trechos de evidência, cite [p. N] "
            "com a página física indicada; não crie páginas não fornecidas. "
            "Não atribua ao PDF fatos que apareçam apenas nos itens extraídos.\n\n"
            f"ITENS EXTRAÍDOS:\n{context}\n\n"
            f"TRECHOS RECUPERADOS DO MESMO EDITAL:\n{json.dumps(evidence, ensure_ascii=False)}"
        )
        response = self._invoke_with_rotation([
            SystemMessage(content="Redija apenas JSON válido, sem markdown."),
            HumanMessage(content=prompt),
        ])
        parsed = self._parse_llm_json(response.content)
        resumo = parsed.get("resumo")
        requisitos = parsed.get("requisitos")
        if not isinstance(resumo, str) or not resumo.strip():
            raise ValueError("Síntese do resumo ausente ou inválida")
        if not isinstance(requisitos, str) or not requisitos.strip():
            raise ValueError("Síntese dos requisitos ausente ou inválida")
        pages = {
            str(source["page"])
            for items in evidence.values() for source in items
            if source.get("page") is not None
        }
        for cited_page in re.findall(r"\[p\.\s*(\d+)\]", resumo + " " + requisitos):
            if cited_page not in pages:
                raise ValueError(f"A síntese citou uma página não recuperada: {cited_page}")
        return {
            "resumo": resumo.strip(),
            "requisitos": requisitos.strip(),
            "coverage": coverage,
        }

    def _explain_all_requirements(self, agg: dict) -> dict[str, list[dict]]:
        """Explica cada item extraído; recusa respostas com itens faltando ou trocados."""
        import json

        topics = {
            "participacao": agg["requisitos_participacao"],
            "selecao": agg["selecao"],
        }
        batch_size = 5
        tasks = [
            (topic, items, offset)
            for topic, items in topics.items()
            for offset in range(0, len(items), batch_size)
        ]
        explained: dict[str, list[dict]] = {topic: [] for topic in topics}

        for task_index, (topic, items, offset) in enumerate(tasks, 1):
            batch = [
                {"id": i + 1, "item": item}
                for i, item in enumerate(items[offset:offset + batch_size], offset)
            ]
            if _progress_callback:
                _progress_callback(
                    stage="Explicando todos os requisitos",
                    query_atual=f"{topic}: itens {offset + 1}–{offset + len(batch)}",
                    query_num=task_index,
                    query_total=len(tasks),
                    batch_atual=task_index,
                    batch_total=len(tasks),
                )
            prompt = (
                "Explique TODOS os itens abaixo, um por um, em português simples para quem "
                "nunca participou de uma licitação. Para cada ID, escreva 2 ou 3 frases "
                "sobre o que o enunciado significa e o que a pessoa precisa conferir ou fazer. "
                "Use exclusivamente o texto do item: não acrescente prazos, percentuais, "
                "documentos, procedimentos, conclusões jurídicas ou páginas ausentes. "
                "Se o item não disser como algo funciona, deixe claro que é preciso conferir "
                "os detalhes no edital. Se houver ambiguidade, explique-a sem decidir qual "
                "regra se aplica. Não agrupe, omita ou renumere IDs. "
                "Responda somente JSON no formato "
                '{"explicacoes":[{"id":1,"explicacao":"Texto claro."}]}.'
                f"\nTEMA: {topic}\nITENS: {json.dumps(batch, ensure_ascii=False)}"
            )
            response = self._invoke_with_rotation([
                SystemMessage(content="Responda apenas JSON válido com todos os IDs recebidos."),
                HumanMessage(content=prompt),
            ])
            parsed = self._parse_llm_json(response.content)
            rows = parsed.get("explicacoes") if isinstance(parsed, dict) else None
            expected_ids = {entry["id"] for entry in batch}
            if (not isinstance(rows, list) or len(rows) != len(batch)
                    or any(not isinstance(row, dict) or type(row.get("id")) is not int
                           or not isinstance(row.get("explicacao"), str)
                           or len(row["explicacao"].split()) < 8
                           for row in rows)
                    or {row["id"] for row in rows} != expected_ids):
                raise RuntimeError(
                    f"Explicação incompleta em {topic}, itens {offset + 1}–"
                    f"{offset + len(batch)}; relatório não publicado."
                )
            explanations_by_id = {row["id"]: row["explicacao"].strip() for row in rows}
            explained[topic].extend([
                {**entry, "explicacao": explanations_by_id[entry["id"]]}
                for entry in batch
            ])
            if task_index < len(tasks):
                time.sleep(8)

        for topic, items in topics.items():
            if len(explained[topic]) != len(items):
                raise RuntimeError(f"Explicação incompleta em {topic}; relatório não publicado.")
        return explained

    def _process_with_rag_llm(self, chunks: list[dict]) -> dict:
        """
        Passagem unica: 1 chamada ao LLM por batch, extraindo TODAS as informacoes.
        31 batches em vez de 9 queries x 31 = 279 chamadas.
        Reducao de 90pct no consumo de tokens.
        """
        BATCH_SIZE    = 5
        SLEEP_BETWEEN = 8

        system_msg = SystemMessage(content=(
            "Voce e um especialista em licitacoes publicas brasileiras (Lei 14.133/2021). "
            "Responda sempre com JSON valido, sem markdown, sem texto fora do JSON."
        ))

        risk_order = {"ALTO": 2, "MEDIO": 1, "BAIXO": 0}

        # O TextChunker já limita os trechos. Preserve o conteúdo completo.
        all_batches = [chunks[i:i+BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
        total = len(all_batches)
        logger.info(f"Passagem unica: {len(chunks)} chunks -> {total} batches")

        # Acumuladores por campo
        agg = {
            "objeto": "",
            "documentos": [],
            "documentos_execucao": [],
            "anexos_referencia": [],
            "pendencias_documentais": [],
            "requisitos_participacao": [],
            "prazos": [],
            "custos": [],
            "selecao": [],
            "entregas": [],
            "eliminacao": [],
            "riscos": [],
            "nivel_risco": "BAIXO",
        }
        max_risk = "BAIXO"
        failed_batches = []

        def normalized(item):
            text = unicodedata.normalize("NFKD", item.casefold())
            text = "".join(c for c in text if not unicodedata.combining(c))
            return re.sub(r"[^\w]+", " ", text).strip()

        def add_unique(lst, items):
            seen = {normalized(existing) for existing in lst}
            for item in items:
                item = item.strip()
                key = normalized(item)
                if key and key not in seen:
                    lst.append(item)
                    seen.add(key)

        for b_idx, batch in enumerate(all_batches):
            if _progress_callback:
                total_itens = sum(len(v) for v in agg.values() if isinstance(v, list))
                _progress_callback(
                    stage="Analisando edital com IA",
                    query_atual=f"Processando trechos do edital",
                    query_num=b_idx + 1,
                    query_total=total,
                    batch_atual=b_idx + 1,
                    batch_total=total,
                    itens_encontrados=total_itens,
                )
            try:
                prompt = self._build_unified_prompt(batch)
                response = self._invoke_with_rotation(
                    [system_msg, HumanMessage(content=prompt)], SLEEP_BETWEEN
                )
                parsed = self._parse_llm_json(response.content)
                fields = ("documentos", "documentos_execucao", "anexos_referencia",
                          "pendencias_documentais", "requisitos_participacao", "prazos",
                          "custos", "selecao", "entregas", "eliminacao", "riscos")
                if any(not isinstance(parsed.get(field), list) or
                       any(not isinstance(item, str) for item in parsed[field])
                       for field in fields):
                    raise ValueError("JSON de extração incompleto ou inválido")

                if not agg["objeto"] and parsed.get("objeto"):
                    agg["objeto"] = parsed["objeto"]

                for field in ["documentos", "documentos_execucao", "anexos_referencia",
                               "pendencias_documentais", "requisitos_participacao", "prazos",
                               "custos", "selecao", "entregas", "eliminacao", "riscos"]:
                    add_unique(agg[field], parsed.get(field, []))

                batch_risk = parsed.get("nivel_risco", "BAIXO").upper()
                if risk_order.get(batch_risk, 0) > risk_order.get(max_risk, 0):
                    max_risk = batch_risk

                gained = sum(len(parsed.get(f, [])) for f in
                             ["documentos","requisitos_participacao","prazos","eliminacao","riscos"])
                if gained:
                    total_acc = sum(len(v) for v in agg.values() if isinstance(v, list))
                    logger.info(f"  batch {b_idx+1}/{total}: +{gained} itens (total={total_acc})")

            except Exception as e:
                logger.error(f"  batch {b_idx+1} erro: {e}")
                failed_batches.append(b_idx + 1)

            if b_idx + 1 < total:
                time.sleep(SLEEP_BETWEEN)

        if failed_batches:
            raise RuntimeError(
                f"Análise incompleta: falha nos lotes {failed_batches} de {total}. "
                "Nenhum relatório final foi publicado."
            )

        agg["nivel_risco"] = max_risk

        # O mesmo nome pode surgir em lotes diferentes com etapas conflitantes.
        # Deixe-o para conferência, em vez de publicá-lo como entrega do licitante.
        document_fields = ("documentos", "documentos_execucao", "anexos_referencia",
                           "pendencias_documentais")
        owners = {}
        originals = {}
        for field in document_fields:
            for item in agg[field]:
                key = normalized(item)
                owners.setdefault(key, set()).add(field)
                originals.setdefault(key, item)
        conflicts = {key for key, fields in owners.items() if len(fields) > 1}
        if conflicts:
            for field in document_fields:
                agg[field] = [item for item in agg[field] if normalized(item) not in conflicts]
            add_unique(agg["pendencias_documentais"],
                       [originals[key] for key in originals if key in conflicts])

        # A recuperação temática consulta somente a coleção deste edital.
        # Os trechos e páginas acompanham o relatório para conferência.
        rag_sources = self._retrieve_rag_chunks(top_k=3)
        if not any(rag_sources.values()):
            raise RuntimeError("A busca vetorial não retornou trechos deste edital")
        source_refs = {
            topic: [
                {
                    "page": chunk.get("metadata", {}).get("page"),
                    "chunk_id": chunk.get("metadata", {}).get("chunk_id"),
                    "section": chunk.get("section", "geral"),
                    "excerpt": chunk["content"][:450],
                }
                for chunk in retrieved
            ]
            for topic, retrieved in rag_sources.items()
        }

        # Uma chamada adicional produz explicações após a extração. Falhas nesta
        # etapa ficam visíveis; os itens extraídos continuam disponíveis.
        explanatory_report = None
        explanatory_error = None
        if any(agg[field] for field in ("requisitos_participacao", "selecao", "prazos", "riscos")):
            try:
                if _progress_callback:
                    _progress_callback(stage="Redigindo resumo e requisitos")
                explanatory_report = self._synthesize_explanatory_report(agg, source_refs)
            except Exception as exc:
                logger.exception("Falha ao redigir o relatório explicativo")
                explanatory_error = str(exc)

        # A síntese acima é uma amostra; esta etapa cobre TODOS os itens das duas
        # seções mostradas na interface, inclusive quando há muitos itens.
        detailed_explanations = self._explain_all_requirements(agg)

        # Mapeia para o formato compativel com display e Node 4
        # rag_answers simula as respostas por pergunta para a tab "Perguntas Respondidas"
        rag_answers = {
            "objeto_licitacao": {
                "label": "Objeto da Licitacao",
                "resposta": agg["objeto"],
                "detalhes": agg["entregas"],
                "nivel_risco": "BAIXO",
                "observacao": "",
            },
            "posso_participar": {
                "label": "Posso participar?",
                "resposta": f"{len(agg['requisitos_participacao'])} requisitos identificados." if agg["requisitos_participacao"] else "Sem restricoes especificas identificadas.",
                "detalhes": agg["requisitos_participacao"],
                "nivel_risco": "BAIXO",
                "observacao": "",
            },
            "prazos": {
                "label": "Quais sao os prazos?",
                "resposta": f"{len(agg['prazos'])} prazos identificados." if agg["prazos"] else "Nenhum prazo especifico identificado.",
                "detalhes": agg["prazos"],
                "nivel_risco": "BAIXO",
                "observacao": "",
            },
            "custos_pagamento": {
                "label": "Quanto custa e como pago?",
                "resposta": f"{len(agg['custos'])} informacoes de custo/pagamento." if agg["custos"] else "Sem informacoes de custo identificadas.",
                "detalhes": agg["custos"],
                "nivel_risco": "BAIXO",
                "observacao": "",
            },
            "selecao_processo": {
                "label": "Como sera a selecao ou avaliacao?",
                "resposta": f"{len(agg['selecao'])} criterios identificados." if agg["selecao"] else "Criterios nao identificados.",
                "detalhes": agg["selecao"],
                "nivel_risco": "BAIXO",
                "observacao": "",
            },
            "objeto_escopo": {
                "label": "O que devo entregar ou produzir?",
                "resposta": f"{len(agg['entregas'])} entregaveis identificados." if agg["entregas"] else "Entregaveis nao identificados.",
                "detalhes": agg["entregas"],
                "nivel_risco": "BAIXO",
                "observacao": "",
            },
            "eliminacao_penalidades": {
                "label": "Quais sao as regras de eliminacao?",
                "resposta": f"{len(agg['eliminacao'])} regras identificadas." if agg["eliminacao"] else "Sem regras de eliminacao identificadas.",
                "detalhes": agg["eliminacao"],
                "nivel_risco": max_risk,
                "observacao": "",
            },
            "documentos": {
                "label": "Documentos para proposta ou habilitação",
                "resposta": f"{len(agg['documentos'])} documentos candidatos; confirme as exigências no edital." if agg["documentos"] else "Nenhum documento de proposta ou habilitação identificado.",
                "detalhes": agg["documentos"],
                "nivel_risco": "BAIXO",
                "observacao": "",
            },
            "pontos_criticos": {
                "label": "Pontos criticos e riscos",
                "resposta": f"{len(agg['riscos'])} riscos identificados. Nivel: {max_risk}." if agg["riscos"] else "Nenhum risco critico identificado.",
                "detalhes": agg["riscos"],
                "nivel_risco": max_risk,
                "observacao": "",
            },
        }

        llm_analysis = {
            "objeto": agg["objeto"],
            "requisitos_tecnicos": agg["requisitos_participacao"],
            "documentos_exigidos": agg["documentos"],
            "prazos": agg["prazos"],
            "pontos_criticos": list(dict.fromkeys(agg["eliminacao"] + agg["riscos"])),
            "nivel_risco": max_risk,
            "recomendacoes": agg["requisitos_participacao"][:5],
        }

        structured_analysis = {
            0: {
                "technical_requirements": agg["requisitos_participacao"],
                "documentation": llm_analysis["documentos_exigidos"],
                "deadlines": agg["prazos"],
                "object_info": {"description": agg["objeto"], "quantity": None, "unit": None},
                "risk_factors": llm_analysis["pontos_criticos"],
                "section": "full_scan",
            }
        }
        critical_analysis = {
            0: {
                "critical_points": llm_analysis["pontos_criticos"],
                "risk_level": max_risk,
                "recommendations": agg["requisitos_participacao"][:5],
                "ambiguities": [],
                "section": "full_scan",
            }
        }

        return {
            "total_chunks_processed": len(chunks),
            "structured_analysis": structured_analysis,
            "critical_analysis": critical_analysis,
            "llm_analysis": llm_analysis,
            "rag_answers": rag_answers,
            "selection_process": agg["selecao"],
            "execution_items": agg["entregas"],
            "documentos_execucao": agg["documentos_execucao"],
            "anexos_referencia": agg["anexos_referencia"],
            "pendencias_documentais": agg["pendencias_documentais"],
            "detailed_explanations": detailed_explanations,
            "rag_sources": source_refs,
            "explanatory_report": explanatory_report,
            "explanatory_error": explanatory_error,
        }

    def process_from_node2(self, chunks: list[dict]) -> dict:
        """Processa todos os chunks usando LLM real ou análise heurística explícita."""
        logger.info(f"Processando {len(chunks)} chunks do Nó 2")

        if not chunks:
            return {
                "total_chunks_processed": 0,
                "structured_analysis": {},
                "critical_analysis": {},
            }

        if not self.mock_mode:
            return self._process_with_rag_llm(chunks)

        # Modo de demonstração: mantém a estrutura esperada pelo workflow
        # e pelo checklist, sem chamar a API externa.
        structured_analysis = {
            chunk.get("metadata", {}).get("chunk_id", i): self.extract_structured_info(chunk)
            for i, chunk in enumerate(chunks)
        }
        critical_analysis = {
            chunk.get("metadata", {}).get("chunk_id", i): self.identify_critical_points(chunk)
            for i, chunk in enumerate(chunks)
        }
        return {
            "total_chunks_processed": len(chunks),
            "structured_analysis": structured_analysis,
            "critical_analysis": critical_analysis,
        }

    def process_complete_analysis(self, node2_output: dict) -> dict:
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
            "total_chunks_processed": analysis["total_chunks_processed"],
        }

    def combine_analyses(self, chunks: list[dict]) -> dict:
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
            "priority_recommendations": list(
                set(priority_recommendations)
            ),  # Remove duplicatas
        }

    def generate_executive_summary(self, analysis_result: dict) -> dict:
        """
        Gera resumo executivo da análise.

        Args:
            analysis_result: Resultado da análise combinada

        Returns:
            Dicionário com resumo executivo
        """
        logger.info("Gerando resumo executivo")

        structured = analysis_result.get("structured_info", [])

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
            "next_steps": next_steps[:3],  # Limita a 3 próximos passos
        }

    def validate_integration_output(self, output: dict) -> bool:
        """
        Valida a saída da integração.

        Args:
            output: Dicionário de saída da integração

        Returns:
            True se válido, False caso contrário
        """
        required_fields = [
            "total_chunks_processed",
            "structured_analysis",
            "critical_analysis",
        ]

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
