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
    
    def categorize_document(self, document: str) -> str:
        """
        Categoriza um documento em uma categoria específica.
        
        Args:
            document: Nome do documento
            
        Returns:
            Categoria do documento
        """
        document_lower = document.lower()
        
        # Categorias de documentos
        category_keywords = {
            "habilitacao": ["cnpj", "cpf", "rg", "cnh", "inscrição estadual"],
            "tecnica": ["iso", "atestado", "certificação", "técnica", "qualidade"],
            "fiscal": ["fiscal", "tributária", "imposto", "balanço", "demonstrativo"],
            "juridica": ["contrato social", "estatuto", "ata", "procuração"],
            "trabalhista": ["clt", "fgts", "inss", "trabalhista"]
        }
        
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in document_lower:
                    return category
        
        return "outros"
    
    def categorize_documents(self, documents: List[str]) -> dict:
        """
        Categoriza uma lista de documentos.
        
        Args:
            documents: Lista de documentos
            
        Returns:
            Dicionário com documentos organizados por categoria
        """
        categories = {
            "habilitacao": [],
            "tecnica": [],
            "fiscal": [],
            "juridica": [],
            "trabalhista": [],
            "outros": []
        }
        
        for document in documents:
            category = self.categorize_document(document)
            categories[category].append(document)
        
        return categories
    
    def validate_categorization(self, output: dict) -> bool:
        """
        Valida a categorização de documentos.
        
        Args:
            output: Dicionário de saída da categorização
            
        Returns:
            True se válido, False caso contrário
        """
        required_categories = ["habilitacao", "tecnica", "fiscal", "juridica", "trabalhista", "outros"]
        
        for category in required_categories:
            if category not in output:
                logger.error(f"Categoria obrigatória ausente: {category}")
                return False
        
        logger.info("Validação de categorização concluída")
        return True
    
    def generate_checklist_item(self, document: str, requirements: List[str] = None) -> dict:
        """
        Gera um item de checklist.
        
        Args:
            document: Nome do documento
            requirements: Lista de requisitos
            
        Returns:
            Dicionário com item do checklist
        """
        return {
            "documento": document,
            "obrigatorio": True,
            "status": "pendente",
            "observacoes": ", ".join(requirements) if requirements else "",
            "prazo": None
        }
    
    def generate_checklist_category(self, documents: List[str], category: str) -> dict:
        """
        Gera checklist para uma categoria.
        
        Args:
            documents: Lista de documentos
            category: Nome da categoria
            
        Returns:
            Dicionário com checklist da categoria
        """
        items = [self.generate_checklist_item(doc) for doc in documents]
        
        return {
            "categoria": category,
            "itens": items,
            "total": len(items)
        }
    
    def generate_complete_checklist(self, categorized_docs: dict) -> dict:
        """
        Gera checklist completo estruturado.
        
        Args:
            categorized_docs: Dicionário com documentos categorizados
            
        Returns:
            Dicionário com checklist completo
        """
        logger.info("Gerando checklist completo")
        
        checklist = {}
        
        for category, documents in categorized_docs.items():
            if documents:  # Só adiciona categorias com documentos
                checklist[category] = self.generate_checklist_category(documents, category)
        
        summary = self.generate_checklist_summary(checklist)
        
        return {
            "checklist": checklist,
            "resumo": summary
        }
    
    def add_deadline_to_item(self, item: dict, deadline: str) -> dict:
        """
        Adiciona prazo a um item do checklist.
        
        Args:
            item: Item do checklist
            deadline: Prazo
            
        Returns:
            Item atualizado com prazo
        """
        item["prazo"] = deadline
        return item
    
    def generate_checklist_summary(self, checklist: dict) -> dict:
        """
        Gera resumo do checklist.
        
        Args:
            checklist: Dicionário com checklist
            
        Returns:
            Dicionário com resumo
        """
        total_documentos = 0
        obrigatorios = 0
        opcionais = 0
        
        for category, data in checklist.items():
            if isinstance(data, dict) and "itens" in data:
                total_documentos += len(data["itens"])
                for item in data["itens"]:
                    if item.get("obrigatorio", False):
                        obrigatorios += 1
                    else:
                        opcionais += 1
        
        return {
            "total_documentos": total_documentos,
            "obrigatorios": obrigatorios,
            "opcionais": opcionais,
            "categorias": len(checklist)
        }
    
    def validate_checklist(self, output: dict) -> bool:
        """
        Valida o checklist.
        
        Args:
            output: Dicionário de saída do checklist
            
        Returns:
            True se válido, False caso contrário
        """
        required_fields = ["checklist", "resumo"]
        
        for field in required_fields:
            if field not in output:
                logger.error(f"Campo obrigatório ausente: {field}")
                return False
        
        if "total_documentos" not in output["resumo"]:
            logger.error("Campo total_documentos ausente no resumo")
            return False
        
        logger.info("Validação de checklist concluída")
        return True
    
    def process_from_node3(self, node3_output: dict) -> dict:
        """
        Processa saída do Nó 3 para gerar checklist.
        
        Args:
            node3_output: Saída do Nó 3
            
        Returns:
            Dicionário com checklist gerado
        """
        logger.info("Processando saída do Nó 3")
        
        structured_info = node3_output.get("structured_info", {})
        documentation = structured_info.get("documentation", [])
        
        if not documentation:
            return {
                "checklist": {},
                "resumo": {
                    "total_documentos": 0,
                    "obrigatorios": 0,
                    "opcionais": 0,
                    "categorias": 0
                }
            }
        
        # Categoriza documentos
        categorized = self.categorize_documents(documentation)
        
        # Gera checklist
        checklist = self.generate_complete_checklist(categorized)
        
        return checklist
    
    def process_complete_analysis(self, node3_analysis: dict) -> dict:
        """
        Processa análise completa do Nó 3.
        
        Args:
            node3_analysis: Análise completa do Nó 3
            
        Returns:
            Dicionário com análise completa e checklist
        """
        logger.info("Processando análise completa do Nó 3")
        
        # Extrai estrutura do Nó 3
        extracted = self._extract_from_node3_structure(node3_analysis)
        
        # Categoriza documentos
        categorized = self.categorize_documents(extracted["documents"])
        
        # Gera checklist
        checklist = self.generate_complete_checklist(categorized)
        
        # Mescla prazos
        if extracted["deadlines"]:
            checklist = self.merge_with_deadlines(checklist, extracted["deadlines"])
        
        # Gera resumo final
        summary = self.generate_checklist_summary(checklist["checklist"])
        
        return {
            "checklist": checklist["checklist"],
            "categorized": categorized,
            "summary": summary,
            "deadlines": extracted["deadlines"],
            "requirements": extracted["requirements"]
        }
    
    def _extract_from_node3_structure(self, node3_structure: dict) -> dict:
        """
        Extrai documentos da estrutura do Nó 3.
        
        Args:
            node3_structure: Estrutura do Nó 3
            
        Returns:
            Dicionário com documentos extraídos
        """
        documentation = node3_structure.get("documentation", [])
        deadlines = node3_structure.get("deadlines", [])
        requirements = node3_structure.get("requirements", [])
        
        return {
            "documents": documentation,
            "deadlines": deadlines,
            "requirements": requirements
        }
    
    def merge_with_deadlines(self, checklist: dict, deadlines: List[str]) -> dict:
        """
        Mescla prazos ao checklist.
        
        Args:
            checklist: Checklist sem prazos
            deadlines: Lista de prazos
            
        Returns:
            Checklist com prazos mesclados
        """
        # Converte lista de prazos para dicionário documento -> prazo
        deadline_map = {}
        for deadline in deadlines:
            deadline_map[f"Prazo {deadline}"] = deadline
        
        # Aplica prazos aos itens
        for category, data in checklist["checklist"].items():
            if isinstance(data, dict) and "itens" in data:
                for item in data["itens"]:
                    for doc_name, deadline in deadline_map.items():
                        if doc_name.lower() in item["documento"].lower():
                            item["prazo"] = deadline
        
        return checklist
    
    def validate_integration_output(self, output: dict) -> bool:
        """
        Valida a saída da integração.
        
        Args:
            output: Dicionário de saída da integração
            
        Returns:
            True se válido, False caso contrário
        """
        required_fields = ["checklist", "categorized", "summary"]
        
        for field in required_fields:
            if field not in output:
                logger.error(f"Campo obrigatório ausente: {field}")
                return False
        
        if "total_documentos" not in output["summary"]:
            logger.error("Campo total_documentos ausente no resumo")
            return False
        
        logger.info("Validação de integração concluída")
        return True
    
    def generate_final_report(self, checklist: dict) -> dict:
        """
        Gera relatório final do checklist.
        
        Args:
            checklist: Checklist gerado
            
        Returns:
            Dicionário com relatório final
        """
        logger.info("Gerando relatório final do checklist")
        
        total_docs = checklist["resumo"]["total_documentos"]
        
        recommendations = []
        if total_docs > 0:
            recommendations.append("Verificar todos os documentos obrigatórios")
            recommendations.append("Organizar documentos por categoria")
        
        if total_docs > 5:
            recommendations.append("Considerar priorizar documentos com prazos mais curtos")
        
        priority_actions = []
        if total_docs > 0:
            priority_actions.append("Iniciar coleta de documentos imediatamente")
            priority_actions.append("Verificar validade dos documentos")
        
        return {
            "checklist": checklist["checklist"],
            "summary": checklist["resumo"],
            "recommendations": recommendations,
            "priority_actions": priority_actions
        }
