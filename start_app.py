"""
Script para iniciar FastAPI e Streamlit simultaneamente

Inicia o servidor FastAPI em background e abre o Streamlit.
"""

import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE_DIR = Path(__file__).parent


def wait_for_api(url: str, timeout: int = 60, interval: float = 2.0) -> bool:
    """Aguarda a API ficar disponível, verificando o endpoint /status."""
    print(f"Aguardando API em {url} (timeout: {timeout}s)...", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{url}/status", timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(interval)
        elapsed = int(time.time() - start)
        print(f"  Aguardando FastAPI iniciar... ({elapsed}s)", flush=True)
    return False


def start_fastapi():
    """Inicia o servidor FastAPI."""
    print("Iniciando FastAPI...", flush=True)
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api.main:app",
         "--host", "127.0.0.1", "--port", "8002"],
        cwd=BASE_DIR
    )
    return process


def start_streamlit():
    """Inicia o Streamlit."""
    print("Iniciando Streamlit...", flush=True)
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"],
        cwd=BASE_DIR
    )
    return process


if __name__ == "__main__":
    print("=" * 60)
    print("LicitGraphAi - Iniciando Aplicação")
    print("=" * 60)
    print()

    # Iniciar FastAPI
    fastapi_process = start_fastapi()

    # Aguardar API ficar pronta (dependências pesadas levam ~25s no primeiro load)
    api_ready = wait_for_api("http://127.0.0.1:8002", timeout=90)

    if not api_ready:
        print()
        print("ERRO: FastAPI não respondeu em 90 segundos.")
        print("Verifique se todas as dependências estão instaladas:")
        print("  pip install -r requirements.txt")
        fastapi_process.terminate()
        sys.exit(1)

    print()
    print("FastAPI rodando em: http://127.0.0.1:8002")
    print("Swagger UI:         http://127.0.0.1:8002/docs")
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
        fastapi_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\nParando processos...")
        fastapi_process.terminate()
        streamlit_process.terminate()
        print("Processos parados.")
