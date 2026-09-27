import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """Fragmenta texto de editais em partes menores usando LangChain."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        default_separators = ["\n\n", "\n", ". ", ", ", " ", ""]
        self.separators = separators or default_separators

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def chunk_text(self, text: str) -> list[str]:
        """Divide o texto em chunks menores."""
        if not text or not text.strip():
            return []
        return self.text_splitter.split_text(text)

    def chunk_by_sections(self, text: str) -> dict[str, list[str]]:
        """
        Divide o texto em chunks organizados por seções típicas de editais.

        Estratégia:
        1. Divide o texto em parágrafos (blocos separados por linha em branco).
        2. Classifica cada parágrafo na seção mais provável pelo título/conteúdo.
        3. Chunka o conteúdo de cada seção individualmente.

        Isso evita os problemas do regex greedy anterior que gerava texto
        corrompido ao capturar com `.*?` sem limite de linha.
        """
        sections: dict[str, list[str]] = {
            "bens": [],
            "prazos": [],
            "exigencias_tecnicas": [],
            "documentacao": [],
            "outros": [],
        }

        # Heurísticas de classificação por palavras no início do parágrafo
        section_keywords: dict[str, list[str]] = {
            "bens": [
                "objeto", "bem", "serviço", "item", "especificação", "tabela",
                "quadro", "descrição", "produto", "material", "equipamento",
                "aquisição", "fornecimento",
            ],
            "prazos": [
                "prazo", "vigência", "duração", "entrega", "cronograma",
                "calendário", "data", "vencimento", "início", "fim",
            ],
            "exigencias_tecnicas": [
                "exigência", "requisito", "técnica", "especificação técnica",
                "norma", "abnt", "iso", "certificação", "padrão", "capacidade",
                "qualificação técnica",
            ],
            "documentacao": [
                "habilitação", "documentação", "certidão", "licença",
                "comprovação", "declaração", "atestado", "registro",
                "regularidade", "cnpj", "cpf",
            ],
        }

        # Divide em parágrafos (2+ quebras de linha)
        paragraphs = re.split(r"\n{2,}", text)

        section_texts: dict[str, list[str]] = {k: [] for k in sections}

        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 30:
                continue

            para_lower = para.lower()
            assigned = False

            for section, keywords in section_keywords.items():
                for kw in keywords:
                    # Palavra-chave nos primeiros 120 chars (título/início)
                    if kw in para_lower[:120]:
                        section_texts[section].append(para)
                        assigned = True
                        break
                if assigned:
                    break

            if not assigned:
                section_texts["outros"].append(para)

        # Chunka cada seção
        for section, paras in section_texts.items():
            combined = "\n\n".join(paras)
            if combined.strip():
                sections[section] = self.chunk_text(combined)

        return sections

    def get_chunk_metadata(self, chunk: str, chunk_index: int, total_chunks: int) -> dict:
        """Gera metadados para um chunk."""
        return {
            "chunk_id": chunk_index,
            "total_chunks": total_chunks,
            "char_count": len(chunk),
            "word_count": len(chunk.split()),
            "line_count": len(chunk.split("\n")),
        }

    def create_document_chunks(self, text: str, metadata: dict | None = None) -> list[dict]:
        """Cria chunks no formato de documentos LangChain."""
        chunks = self.chunk_text(text)
        base_metadata = metadata or {}
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = self.get_chunk_metadata(chunk, i, len(chunks))
            chunk_metadata.update(base_metadata)
            documents.append({"content": chunk, "metadata": chunk_metadata})
        return documents
