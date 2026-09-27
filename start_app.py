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


def wait_for_api(url: str, process: subprocess.Popen, timeout: int = 180, interval: float = 2.0) -> bool:
    """Aguarda a API ficar disponível, verificando o endpoint /status."""
    print(f"Aguardando API em {url} (timeout: {timeout}s)...", flush=True)
    start = time.monotonic()
    last_error = None
    while time.monotonic() - start < timeout:
        exit_code = process.poll()
        if exit_code is not None:
            print(f"ERRO: processo FastAPI terminou com código {exit_code}.", flush=True)
            return False
        try:
            with urllib.request.urlopen(f"{url}/status", timeout=3) as resp:
                if resp.status == 200:
                    return True
                last_error = f"/status respondeu HTTP {resp.status}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(interval)
        elapsed = int(time.monotonic() - start)
        print(f"  Aguardando FastAPI iniciar... ({elapsed}s)", flush=True)
    print(f"ERRO: /status não respondeu em {timeout}s. Última tentativa: {last_error}", flush=True)
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

    # Aguardar a API e detectar se o subprocesso encerra durante a inicialização.
    api_ready = wait_for_api("http://127.0.0.1:8002", fastapi_process)

    if not api_ready:
        print()
        print("Veja o erro do Uvicorn acima. Para executá-lo isoladamente:")
        print("  python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8002 --log-level debug")
        if fastapi_process.poll() is None:
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
