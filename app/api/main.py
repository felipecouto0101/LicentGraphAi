"""
API FastAPI para LicitGraphAi
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
import threading
import time
from typing import Optional
import logging
from dotenv import load_dotenv

load_dotenv()

from app.rag.langgraph_workflow import run_licit_graph_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LicitGraphAi API",
    description="API para análise automática de editais de licitação",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data/raw/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Armazenamento de jobs em memória ────────────────────────────────────────
# { job_id: { "status": "running"|"done"|"error", "progress": {...}, "result": {...} } }
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _update_progress(job_id: str, **kwargs):
    """Atualiza o progresso de um job. Chamado pelo Node 3 via callback."""
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["progress"].update(kwargs)
            _jobs[job_id]["progress"]["updated_at"] = time.time()


def _run_pipeline_job(job_id: str, file_path: str, company_profile: Optional[dict]):
    """Executa o pipeline em background, publicando progresso."""
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["progress"]["stage"] = "Iniciando pipeline..."

        # Injeta callback de progresso no estado global acessível pelo Node 3
        import app.rag.node_3_analyzer as n3_module
        n3_module._progress_callback = lambda **kw: _update_progress(job_id, **kw)

        result = run_licit_graph_pipeline(file_path, company_profile)

        with _jobs_lock:
            if result.get("error"):
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = result["error"]
            else:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["result"] = {
                    "chunks_count": len(result.get("chunks", [])),
                    "embeddings": result.get("embeddings"),
                    "analysis": result.get("analysis"),
                    "checklist": result.get("checklist"),
                }

    except Exception as e:
        logger.error(f"Job {job_id} falhou: {e}")
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)
    finally:
        import app.rag.node_3_analyzer as n3_module
        n3_module._progress_callback = None


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "LicitGraphAi API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload",
            "analyze": "/analyze/upload",
            "job_status": "/job/{job_id}",
            "status": "/status",
        }
    }


@app.get("/status")
async def get_status():
    return {
        "status": "online",
        "service": "LicitGraphAi API",
        "version": "1.0.0",
        "nodes": ["node_1", "node_2", "node_3", "node_4"],
        "orchestration": "LangGraph"
    }


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Retorna o status e progresso de um job de análise."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    return job


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são aceitos")

    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {
            "message": "Upload realizado com sucesso",
            "file_id": file_id,
            "filename": file.filename,
            "file_path": str(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar arquivo: {str(e)}")


@app.post("/analyze/upload")
async def analyze_with_upload(file: UploadFile = File(...), company_profile: Optional[dict] = None):
    """
    Inicia análise em background e retorna job_id imediatamente.
    Use GET /job/{job_id} para acompanhar o progresso.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são aceitos")

    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar arquivo: {str(e)}")

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "file": file.filename,
            "progress": {
                "stage": "Na fila...",
                "query_atual": "",
                "query_num": 0,
                "query_total": 0,
                "batch_atual": 0,
                "batch_total": 0,
                "itens_encontrados": 0,
                "updated_at": time.time(),
            },
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_run_pipeline_job,
        args=(job_id, str(file_path), company_profile),
        daemon=True,
    )
    thread.start()

    return {
        "message": "Análise iniciada",
        "job_id": job_id,
        "poll_url": f"/job/{job_id}",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
