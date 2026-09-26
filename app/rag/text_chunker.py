from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional
import re


class TextChunker:
    """Fragmenta texto de editais em partes menores usando LangChain."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Inicializa o fragmentador de texto.
        
        Args:
            chunk_size: Tamanho máximo de cada chunk em caracteres
            chunk_overlap: Sobreposição entre chunks para manter contexto
            separators: Separadores personalizados para divisão inteligente
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Separadores otimizados para documentos de licitação
        default_separators = [
            "\n\n\n",  # Parágrafos múltiplos
            "\n\n",    # Parágrafos
            "\n",      # Linhas
            ". ",      # Final de sentenças
            ", ",      # Vírgulas
            " ",       # Espaços
            ""         # Caracteres individuais
        ]
        
        self.separators = separators or default_separators
        
        # Inicializa o splitter do LangChain
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Divide o texto em chunks menores.
        
        Args:
            text: Texto completo do edital
            
        Returns:
            Lista de chunks de texto
        """
        if not text or not text.strip():
            return []
        
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def chunk_by_sections(self, text: str) -> Dict[str, List[str]]:
        """
        Divide o texto em chunks organizados por seções típicas de editais.
        
        Args:
            text: Texto completo do edital
            
        Returns:
            Dicionário com chunks organizados por seção
        """
        sections = {
            "bens": [],
            "prazos": [],
            "exigencias_tecnicas": [],
            "documentacao": [],
            "outros": []
        }
        
        # Padrões para identificar seções em editais
        patterns = {
            "bens": [
                r"(?i)(objeto|bens|serviços|itens|especificações).*?(?=\n\n|\n[A-Z]|$)",
                r"(?i)(quadro|tabela).*?(?=\n\n|\n[A-Z]|$)"
            ],
            "prazos": [
                r"(?i)(prazo|vigência|duração|entrega).*?(?=\n\n|\n[A-Z]|$)",
                r"(?i)(cronograma|calendar).*?(?=\n\n|\n[A-Z]|$)"
            ],
            "exigencias_tecnicas": [
                r"(?i)(exigência|técnica|especificação técnica|requisito).*?(?=\n\n|\n[A-Z]|$)",
                r"(?i)(norma|abnt|iso).*?(?=\n\n|\n[A-Z]|$)"
            ],
            "documentacao": [
                r"(?i)(habilitação|documentação|certidão|licença).*?(?=\n\n|\n[A-Z]|$)",
                r"(?i)(comprovação|declaração).*?(?=\n\n|\n[A-Z]|$)"
            ]
        }
        
        # Primeiro, tenta extrair seções específicas
        for section_name, section_patterns in patterns.items():
            for pattern in section_patterns:
                matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
                for match in matches:
                    section_text = match.group(0)
                    if len(section_text) > 50:  # Ignora matches muito curtos
                        chunks = self.chunk_text(section_text)
                        sections[section_name].extend(chunks)
        
        # Divide o restante do texto em "outros"
        all_section_text = " ".join([
            " ".join(sections[key]) for key in sections if key != "outros"
        ])
        
        remaining_text = text
        for section_text in all_section_text.split():
            remaining_text = remaining_text.replace(section_text, "", 1)
        
        if remaining_text.strip():
            sections["outros"] = self.chunk_text(remaining_text.strip())
        
        return sections
    
    def get_chunk_metadata(self, chunk: str, chunk_index: int, total_chunks: int) -> Dict:
        """
        Gera metadados para um chunk.
        
        Args:
            chunk: Texto do chunk
            chunk_index: Índice do chunk
            total_chunks: Total de chunks
            
        Returns:
            Dicionário com metadados do chunk
        """
        return {
            "chunk_id": chunk_index,
            "total_chunks": total_chunks,
            "char_count": len(chunk),
            "word_count": len(chunk.split()),
            "line_count": len(chunk.split("\n"))
        }
    
    def create_document_chunks(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Cria chunks no formato de documentos LangChain.
        
        Args:
            text: Texto completo
            metadata: Metadados adicionais do documento
            
        Returns:
            Lista de documentos com metadados
        """
        chunks = self.chunk_text(text)
        documents = []
        
        base_metadata = metadata or {}
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = self.get_chunk_metadata(chunk, i, len(chunks))
            chunk_metadata.update(base_metadata)
            
            documents.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
        
        return documents
