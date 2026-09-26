import pdfplumber
from pathlib import Path
from typing import Optional


class PDFReader:
    """Lê e extrai texto de arquivos PDF de editais de licitação."""
    
    def __init__(self):
        self.pdf_path: Optional[Path] = None
        self.text: str = ""
    
    def load_pdf(self, pdf_path: str) -> str:
        """
        Carrega um arquivo PDF e extrai todo o texto.
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            
        Returns:
            Texto extraído do PDF
            
        Raises:
            FileNotFoundError: Se o arquivo não existir
            Exception: Se houver erro na leitura do PDF
        """
        self.pdf_path = Path(pdf_path)
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")
        
        try:
            text_parts = []
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            
            self.text = "\n".join(text_parts)
            return self.text
            
        except Exception as e:
            raise Exception(f"Erro ao ler PDF: {str(e)}")
    
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
    
    def validate_pdf(self, pdf_path: Optional[str] = None) -> bool:
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
