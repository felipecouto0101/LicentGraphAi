"""
Script para iniciar FastAPI e Streamlit simultaneamente

Inicia o servidor FastAPI em background e abre o Streamlit.
"""

import subprocess
import sys
import time
from pathlib import Path


def start_fastapi():
    """Inicia o servidor FastAPI."""
    print("Iniciando FastAPI...")
    uvicorn_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"],
        cwd=Path(__file__).parent.parent
    )
    return uvicorn_process


def start_streamlit():
    """Inicia o Streamlit."""
    print("Iniciando Streamlit...")
    time.sleep(2)  # Aguarda FastAPI iniciar
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"],
        cwd=Path(__file__).parent.parent
    )
    return streamlit_process


if __name__ == "__main__":
    print("=" * 60)
    print("LicitGraphAi - Iniciando Aplicação")
    print("=" * 60)
    print()
    
    # Iniciar FastAPI
    fastapi_process = start_fastapi()
    print("FastAPI rodando em: http://127.0.0.1:8000")
    print("Swagger UI: http://127.0.0.1:8000/docs")
    print()
    
    # Iniciar Streamlit
    streamlit_process = start_streamlit()
    print("Streamlit rodando em: http://localhost:8501")
    print()
    
    print("=" * 60)
    print("Aplicação iniciada com sucesso!")
    print("Pressione Ctrl+C para parar")
    print("=" * 60)
    
    try:
        # Aguardar processos
        fastapi_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\nParando processos...")
        fastapi_process.terminate()
        streamlit_process.terminate()
        print("Processos parados.")
