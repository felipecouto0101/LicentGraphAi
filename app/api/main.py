"""
API FastAPI para LicitGraphAi
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
import threading
import time
from typing import Optional
import logging
import json
from dotenv import load_dotenv
from app.rag.gemini_client import provider_name, validate_topic_configuration, GeminiTopicClient

load_dotenv()

from app.rag.langgraph_workflow import run_licit_graph_pipeline, is_mock_mode, validate_analysis_configuration

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
_documents: dict[str, dict] = {}


def _run_organization_job(job_id: str):
    """Organiza nomes e agrupamentos; retém o índice e lotes válidos após falhas."""
    from app.rag.topic_organizer import organize_topic_map, inspect_annex_references
    from app.rag.topic_map import all_chunk_ids
    from app.rag.source_recovery import SourceRecovery
    with _jobs_lock:
        session = _documents[job_id]
    try:
        if is_mock_mode():
            raise ValueError("A organização exige NODE3_MOCK_MODE=false e uma chave válida do provedor selecionado.")
        provider = provider_name()
        if provider == "gemini":
            validate_topic_configuration()
        else:
            validate_analysis_configuration()
        with session["lock"]:
            recovery = SourceRecovery(session.get("pages_text", []), session["chunks"], session.get("pages_tables", []))
            if session["llm"] is None:
                if provider == "gemini":
                    session["llm"] = GeminiTopicClient(progress=lambda **kw: _update_progress(job_id, **kw))
                else:
                    from app.rag.node_3_analyzer import Node3RequirementAnalyzer
                    session["llm"] = Node3RequirementAnalyzer(mock_mode=False)
            def invoke(system, payload):
                if provider == "gemini":
                    return session["llm"].invoke(system, payload)
                from langchain_core.messages import SystemMessage, HumanMessage
                response = session["llm"]._invoke_with_rotation([
                    SystemMessage(content=system),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False))])
                return session["llm"]._parse_llm_json(response.content)
            def publish_partial(topic_map):
                topic_map = recovery.enrich(topic_map)
                with _jobs_lock:
                    _jobs[job_id]["result"].update(topic_map=topic_map, partial_map=True)
            organized = organize_topic_map(
                session["structural_topics"], session["chunks"], invoke,
                cache=session["organization_cache"],
                progress=lambda **kw: _update_progress(job_id, **kw), on_partial=publish_partial,
                batch_max_chars=16000 if provider == "gemini" else 8000,
                batch_max_entries=12 if provider == "gemini" else 8)
            if all_chunk_ids(organized) != all_chunk_ids(session["structural_topics"]):
                raise ValueError("A organização perdeu referências do PDF.")
            organized = recovery.enrich(organized)
            session["topics"] = organized
        with _jobs_lock:
            _jobs[job_id]["result"].update(topic_map=organized, organization_status="done", organization_error=None, partial_map=False,
                                           annex_references=inspect_annex_references(session["chunks"]))
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["error"] = None
        logger.info("Mapa organizado com IA: %s temas, %s subtemas", len(organized),
                    sum(len(t["subtopics"]) for t in organized))
    except Exception as exc:
        logger.exception("Organização do mapa falhou")
        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            result = _jobs[job_id]["result"]
            result.update(organization_status="partial" if result.get("partial_map") else "error", organization_error=str(exc))
            _jobs[job_id]["progress"]["stage"] = "Organização interrompida; subtemas validados e índice original preservados"


def _run_topic_job(job_id: str, file_path: str):
    """Lê o índice estrutural e organiza assuntos com IA, sem explicações."""
    try:
        from app.rag.node_1_reader_chunker import Node1ReaderChunker
        from app.rag.topic_map import build_topic_map
        from app.rag.topic_organizer import inspect_annex_references
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["progress"]["stage"] = "Lendo páginas e identificando seções"
        parsed = Node1ReaderChunker(chunk_by_sections=False,
                                   preserve_document_structure=True).process_pdf(file_path)
        chunks = parsed["chunks"]
        if not chunks:
            raise ValueError("O PDF não contém texto extraível.")
        topic_map = build_topic_map(chunks)
        session = {"chunks": chunks, "pages_text": parsed.get("pages_text", []),
                   "pages_tables": parsed.get("pages_tables", []),
                   "topics": topic_map, "structural_topics": topic_map,
                   "organization_cache": {}, "llm": None, "lock": threading.Lock()}
        with _jobs_lock:
            _documents[job_id] = session
            _jobs[job_id]["result"] = {"topic_map": topic_map, "map_version": 4,
                "chunks_count": len(chunks), "page_count": parsed.get("page_count"),
                "organization_status": "running", "organization_error": None,
                "annex_references": inspect_annex_references(chunks)}
        _run_organization_job(job_id)
    except Exception as exc:
        logger.exception("Falha na preparação do mapa de assuntos")
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)


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
                    "analysis_mode": "demo" if is_mock_mode() else "real",
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
            "full_analysis": "/analyze/full",
            "job_status": "/job/{job_id}",
            "organize_map": "/job/{job_id}/organize-map",
            "status": "/status",
        }
    }


@app.get("/status")
async def get_status():
    try:
        if provider_name() == "gemini":
            validate_topic_configuration()
        else:
            validate_analysis_configuration()
        configuration_error = None
    except ValueError as exc:
        configuration_error = str(exc)
    return {
        "status": "online",
        "ready": configuration_error is None,
        "configuration_error": configuration_error,
        "service": "LicitGraphAi API",
        "topic_provider": provider_name() if configuration_error is None else None,
        "version": "1.0.0",
        "nodes": ["PDF", "índice estrutural", "organização de assuntos com IA"],
        "orchestration": "Mapa de assuntos; LangGraph disponível no fluxo completo",
        "analysis_mode": "demo" if is_mock_mode() else "real"
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
async def analyze_with_upload(file: UploadFile = File(...)):
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
            "analysis_mode": "demo" if is_mock_mode() else "real",
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

    thread = threading.Thread(target=_run_topic_job, args=(job_id, str(file_path)), daemon=True)
    thread.start()

    return {
        "message": "Preparação do mapa iniciada",
        "job_id": job_id,
        "poll_url": f"/job/{job_id}",
    }


@app.post("/analyze/full")
async def analyze_full_with_upload(file: UploadFile = File(...)):
    """Mantém o pipeline LangGraph anterior para consumidores da API."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são aceitos")
    try:
        validate_analysis_configuration()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    try:
        with path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro no upload: {exc}") from exc
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued", "file": file.filename,
            "analysis_mode": "demo" if is_mock_mode() else "real",
            "progress": {"stage": "Na fila...", "updated_at": time.time()},
            "result": None, "error": None,
        }
    threading.Thread(target=_run_pipeline_job, args=(job_id, str(path), None), daemon=True).start()
    return {"job_id": job_id, "poll_url": f"/job/{job_id}"}


@app.post("/job/{job_id}/organize-map")
def retry_map_organization(job_id: str):
    """Retoma a organização sem reenviar ou reler o PDF."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job or job_id not in _documents:
            raise HTTPException(status_code=404, detail="Mapa não encontrado; envie novamente o PDF.")
        if job["status"] in ("running", "queued"):
            raise HTTPException(status_code=409, detail="A organização já está em andamento.")
        if job["result"].get("organization_status") == "done":
            return {"job_id": job_id, "status": "done"}
        job["status"] = "running"
        job["result"]["organization_status"] = "running"
        job["result"]["organization_error"] = None
        job["progress"]["stage"] = "Retomando organização dos assuntos"
    threading.Thread(target=_run_organization_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "status": "running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
