from pathlib import Path
import re

import pdfplumber


class PDFReadError(Exception):
    """Exceção customizada para erros de leitura de PDF."""


class PDFReader:
    """Lê e extrai texto de arquivos PDF de editais de licitação."""

    def __init__(self):
        self.pdf_path: Path | None = None
        self.text: str = ""
        self.pages_text: list[str] = []

    def load_pdf(self, pdf_path: str, preserve_lines: bool = False) -> str:
        """
        Carrega um arquivo PDF e extrai todo o texto.

        Args:
            pdf_path: Caminho para o arquivo PDF

        Returns:
            Texto extraído e limpo do PDF

        Raises:
            FileNotFoundError: Se o arquivo não existir
            PDFReadError: Se houver erro na leitura do PDF
        """
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")

        try:
            # Preserva o índice original, inclusive páginas sem texto extraível.
            with pdfplumber.open(self.pdf_path) as pdf:
                self.pages_text = [
                    self._clean_text(page.extract_text() or "", preserve_lines=preserve_lines)
                    for page in pdf.pages
                ]

            self.text = "\n\n".join(text for text in self.pages_text if text)
            return self.text

        except Exception as e:
            raise PDFReadError(f"Erro ao ler PDF: {e!s}")

    @staticmethod
    def _clean_text(text: str, preserve_lines: bool = False) -> str:
        """
        Limpa o texto extraído do PDF.

        Problemas comuns em PDFs de editais:
        - Linhas com só números de página, cabeçalhos repetidos
        - Palavras hifenizadas ao fim da linha (cor-\nrupção → corrupção)
        - Múltiplos espaços e tabs
        - Linhas de ruído (só dígitos, só pontuação, muito curtas)
        """
        # 1. Juntar palavras hifenizadas quebradas no fim de linha
        text = re.sub(r"-\s*\n\s*", "", text)

        # 2. Normalizar quebras de linha: 2+ quebras viram parágrafo, 1 vira espaço
        text = re.sub(r"\n{3,}", "\n\n", text)          # 3+ \n → parágrafo
        if not preserve_lines:
            text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        # 3. Normalizar espaços múltiplos
        text = re.sub(r"[ \t]{2,}", " ", text)

        # 4. Remover linhas que são claramente ruído:
        #    - Só dígitos (números de página)
        #    - Menos de 15 caracteres úteis
        #    - Só pontuação/símbolos
        clean_lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                clean_lines.append("")
                continue
            # Ignora linha se for só números (número de página)
            if re.fullmatch(r"\d+", stripped):
                continue
            # Ignora linhas muito curtas que não são títulos (sem letra suficiente)
            if len(stripped) < 8 and not re.search(r"[A-Za-zÀ-ú]{3,}", stripped):
                continue
            clean_lines.append(line)

        text = "\n".join(clean_lines)

        # 5. Remover espaços no início/fim
        return text.strip()

    def get_text(self) -> str:
        """Retorna o texto extraído do PDF."""
        return self.text

    def get_page_count(self) -> int:
        """Retorna o número de páginas do PDF."""
        if not self.pdf_path or not self.pdf_path.exists():
            return 0

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                return len(pdf.pages)
        except Exception:
            return 0

    def validate_pdf(self, pdf_path: str | None = None) -> bool:
        """Valida se o arquivo é um PDF válido."""
        if pdf_path:
            self.pdf_path = Path(pdf_path)

        if not self.pdf_path or not self.pdf_path.exists():
            return False

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                return len(pdf.pages) > 0
        except Exception:
            return False
