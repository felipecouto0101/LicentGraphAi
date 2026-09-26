"""
Teste simples da API FastAPI

Verifica se a API pode ser iniciada.
"""

import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.api.main import app

if __name__ == "__main__":
    print("App FastAPI criada com sucesso!")
    print(f"App: {app}")
    print(f"Title: {app.title}")
    print(f"Version: {app.version}")
