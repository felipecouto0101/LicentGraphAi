"""
Script para iniciar FastAPI e Streamlit simultaneamente

Inicia o servidor FastAPI em background e abre o Streamlit.
"""

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent


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


def _stop_process(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main():
    print("LicitGraphAi - Iniciando API e interface", flush=True)
    processes = []
    try:
        # The interface can render while the backend is still importing modules.
        processes.append(("FastAPI", start_fastapi()))
        processes.append(("Streamlit", start_streamlit()))
        print("Interface: http://localhost:8501", flush=True)
        print("API: http://127.0.0.1:8002/docs", flush=True)
        print("A interface informa a disponibilidade da API. Ctrl+C para parar.", flush=True)
        while True:
            for name, process in processes:
                code = process.poll()
                if code is not None:
                    print(f"{name} terminou com código {code}. Veja os logs acima.", flush=True)
                    return code
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Parando processos...", flush=True)
        return 0
    except OSError as exc:
        print(f"Não foi possível iniciar os processos: {exc}", flush=True)
        return 1
    finally:
        for _, process in processes:
            _stop_process(process)


if __name__ == "__main__":
    sys.exit(main())
