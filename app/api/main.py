"""
API FastAPI para LicitGraphAi
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
import threading
import time
from typing import Optional
import logging
import json
import os
import re
from dotenv import load_dotenv

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


class TopicQuestion(BaseModel):
    topic_id: str | None = None
    question: str = Field(min_length=3, max_length=1000)
    history: list[dict[str, str]] = Field(default_factory=list)


class TopicRename(BaseModel):
    theme_id: str


def _run_topic_job(job_id: str, file_path: str):
    """Prepara o mapa sem rodar os 33 lotes ou as explicações individuais."""
    try:
        from app.rag.node_1_reader_chunker import Node1ReaderChunker
        from app.rag.node_2_embeddings import Node2EmbeddingGenerator
        from app.rag.topic_map import build_topic_map, merge_similar_themes
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["progress"]["stage"] = "Lendo páginas e identificando seções"
        chunks = Node1ReaderChunker().process_pdf(file_path)["chunks"]
        if not chunks:
            raise ValueError("O PDF não contém texto extraível.")
        topic_map = build_topic_map(chunks)
        with _jobs_lock:
            _jobs[job_id]["progress"]["stage"] = "Indexando trechos para perguntas"
        indexer = Node2EmbeddingGenerator()
        indexed = indexer.process_chunks(
            chunks, collection_name=f"licitacoes_{uuid.uuid4().hex}",
            persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/vector_db"),
        )
        topic_map = merge_similar_themes(topic_map, indexer.embeddings)
        session = {"chunks": chunks, "topics": topic_map,
                   "collection": indexed["collection_id"], "llm": None,
                   "lock": threading.Lock()}
        with _jobs_lock:
            _documents[job_id] = session
            _jobs[job_id]["result"] = {"topic_map": topic_map,
                                       "chunks_count": len(chunks)}
            _jobs[job_id]["status"] = "done"
    except Exception as exc:
        logger.exception("Falha na preparação do mapa de assuntos")
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)


def _topic_sources(session: dict, topic_id: str | None, question: str) -> list[dict]:
    """Recupera contexto e preserva a página física para cada citação."""
    topic = next((sub for theme in session["topics"] for sub in theme["subtopics"]
                  if sub["id"] == topic_id), None) if topic_id else None
    if topic_id and topic is None:
        raise HTTPException(status_code=404, detail="Subtema não encontrado neste edital")
    selected = set(topic["chunk_ids"]) if topic else None
    tokens = set(re.findall(r"\w{4,}", question.casefold()))
    candidates = [chunk for chunk in session["chunks"]
                  if selected is None or chunk["metadata"]["chunk_id"] in selected]
    # A busca vetorial descobre referências fora da seção quando a pergunta é livre.
    if not topic:
        from app.rag.node_2_embeddings import Node2EmbeddingGenerator
        node2 = Node2EmbeddingGenerator()
        found = node2.search_similar(question, session["collection"], n_results=min(6, len(session["chunks"])),
                                    persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/vector_db"))
        ids = {meta.get("chunk_id") for meta in found.get("metadatas", [[]])[0]}
        candidates = [chunk for chunk in candidates if chunk["metadata"]["chunk_id"] in ids]
    ranked = sorted(candidates, key=lambda chunk: (
        len(tokens & set(re.findall(r"\w{4,}", chunk["content"].casefold()))),
        -session["chunks"].index(chunk)), reverse=True)
    return [{"page": chunk["metadata"].get("page"),
             "chunk_id": chunk["metadata"].get("chunk_id"),
             "text": chunk["content"][:1050]}
            for chunk in ranked[:4]]


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
            "ask": "/job/{job_id}/ask",
            "status": "/status",
        }
    }


@app.get("/status")
async def get_status():
    try:
        validate_analysis_configuration()
        configuration_error = None
    except ValueError as exc:
        configuration_error = str(exc)
    return {
        "status": "online",
        "ready": configuration_error is None,
        "configuration_error": configuration_error,
        "service": "LicitGraphAi API",
        "version": "1.0.0",
        "nodes": ["PDF", "embeddings", "mapa", "chat sob demanda"],
        "orchestration": "Mapa rápido; LangGraph disponível no fluxo completo",
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


@app.post("/job/{job_id}/ask")
def ask_about_edital(job_id: str, query: TopicQuestion):
    """Explica apenas o tema escolhido ou a pergunta feita no chat."""
    from app.rag.node_3_analyzer import Node3RequirementAnalyzer
    from langchain_core.messages import HumanMessage, SystemMessage

    if is_mock_mode():
        raise HTTPException(status_code=503, detail="O chat exige NODE3_MOCK_MODE=false e uma chave Groq válida.")
    try:
        validate_analysis_configuration()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    with _jobs_lock:
        session = _documents.get(job_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Mapa não encontrado; envie o PDF novamente.")
    sources = _topic_sources(session, query.topic_id, query.question)
    if not sources:
        return {"answer": "Não encontrei trechos para responder a esta pergunta no PDF.",
                "citations": [], "found": False}

    topic = next((sub for theme in session["topics"] for sub in theme["subtopics"]
                  if sub["id"] == query.topic_id), None)
    with session["lock"]:
        if session["llm"] is None:
            session["llm"] = Node3RequirementAnalyzer(
                mock_mode=False, collection_name=session["collection"])
        history = [item for item in query.history[-4:]
                   if item.get("role") in ("user", "assistant")
                   and isinstance(item.get("content"), str)]
        prompt = {
            "topic": topic["title"] if topic else "Pergunta livre",
            "question": query.question,
            "recent_context": [{"role": item["role"], "text": item["content"][:500]}
                               for item in history],
            "sources": [{"page": item["page"], "chunk_id": item["chunk_id"],
                         "text": item["text"]} for item in sources],
        }
        try:
            response = session["llm"]._invoke_with_rotation([
                SystemMessage(content=(
                    "Explique em português claro apenas o que as fontes do PDF sustentam. "
                    "Preserve condições, exceções e valores. Se não houver apoio, responda "
                    "que não foi possível identificar. Responda apenas JSON com campos "
                    "found (boolean), answer (texto), citations (lista de objetos com "
                    "page e quote literal, entre 15 e 240 caracteres). Para found=true "
                    "inclua ao menos uma citação. Não invente páginas nem exigências."
                )),
                HumanMessage(content=json.dumps(prompt, ensure_ascii=False)),
            ])
            parsed = session["llm"]._parse_llm_json(response.content)
        except Exception as exc:
            logger.warning("Pergunta ao edital falhou: %s", exc)
            raise HTTPException(status_code=503, detail=f"Não foi possível responder agora: {exc}") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("answer"), str) or not parsed["answer"].strip():
        raise HTTPException(status_code=502, detail="Resposta incompleta da IA; tente novamente.")
    if parsed.get("found") is not True:
        return {"answer": "Não foi possível confirmar essa informação nos trechos encontrados.",
                "citations": [], "found": False}
    citations = parsed.get("citations")
    if not isinstance(citations, list) or not citations:
        raise HTTPException(status_code=502, detail="A resposta veio sem fonte verificável.")
    verified = []
    for citation in citations:
        if not isinstance(citation, dict):
            raise HTTPException(status_code=502, detail="Formato de citação inválido.")
        quote, page = citation.get("quote"), citation.get("page")
        if not isinstance(quote, str) or not 15 <= len(quote.strip()) <= 240 or not any(
            page == source["page"] and " ".join(quote.split()).casefold()
            in " ".join(source["text"].split()).casefold() for source in sources
        ):
            raise HTTPException(status_code=502, detail="Citação não localizada na página informada.")
        verified.append({"page": page, "quote": quote.strip()})
    return {"answer": parsed["answer"].strip(), "citations": verified, "found": True}


@app.post("/job/{job_id}/organize")
def improve_topic_names(job_id: str, request: TopicRename):
    """Melhora nomes sob demanda; nunca altera IDs, fontes ou o título do PDF."""
    from app.rag.node_3_analyzer import Node3RequirementAnalyzer
    from langchain_core.messages import HumanMessage, SystemMessage

    if is_mock_mode():
        raise HTTPException(status_code=503, detail="A organização com IA exige NODE3_MOCK_MODE=false.")
    try:
        validate_analysis_configuration()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    with _jobs_lock:
        session = _documents.get(job_id)
    if not session:
        raise HTTPException(status_code=404, detail="Mapa não encontrado; envie o PDF novamente.")
    theme = next((item for item in session["topics"] if item["id"] == request.theme_id), None)
    if theme is None:
        raise HTTPException(status_code=404, detail="Tema não encontrado.")
    with session["lock"]:
        if session["llm"] is None:
            session["llm"] = Node3RequirementAnalyzer(mock_mode=False,
                                                       collection_name=session["collection"])
        entries = [{"id": sub["id"], "source_title": sub["title"]}
                   for sub in theme["subtopics"][:12]]
        try:
            response = session["llm"]._invoke_with_rotation([
                SystemMessage(content=(
                    "Organize SOMENTE os títulos dados em nomes claros para um leitor leigo. "
                    "Não explique regras nem crie assuntos. Retorne JSON com theme_label "
                    "(até 7 palavras) e subtopics (lista de id e label, até 9 palavras cada). "
                    "Mantenha cada id e o sentido do título original."
                )),
                HumanMessage(content=json.dumps({"theme": theme["title"],
                                                 "subtopics": entries}, ensure_ascii=False)),
            ])
            parsed = session["llm"]._parse_llm_json(response.content)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Não foi possível organizar os nomes: {exc}") from exc
        if not isinstance(parsed, dict) or not isinstance(parsed.get("theme_label"), str):
            raise HTTPException(status_code=502, detail="A IA não retornou nomes válidos.")
        labels = parsed.get("subtopics")
        if not isinstance(labels, list) or len(labels) != len(entries):
            raise HTTPException(status_code=502, detail="A IA não cobriu todos os subtemas enviados.")
        mapped = {row.get("id"): row.get("label") for row in labels if isinstance(row, dict)}
        if set(mapped) != {entry["id"] for entry in entries} or not all(
            isinstance(label, str) and 3 <= len(label.strip()) <= 90 for label in mapped.values()
        ):
            raise HTTPException(status_code=502, detail="A IA trocou IDs ou omitiu subtemas.")
        if not 3 <= len(parsed["theme_label"].strip()) <= 90:
            raise HTTPException(status_code=502, detail="Nome de tema inválido.")
        theme["display_title"] = parsed["theme_label"].strip()
        for sub in theme["subtopics"]:
            if sub["id"] in mapped:
                sub["display_title"] = mapped[sub["id"]].strip()
        with _jobs_lock:
            _jobs[job_id]["result"]["topic_map"] = session["topics"]
    return {"topic_map": session["topics"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
