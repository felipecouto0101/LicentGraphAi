"""
API FastAPI para LicitGraphAi

Endpoints para upload de PDFs e análise de editais.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
from typing import Optional
import logging

# Importar workflow LangGraph
from app.rag.langgraph_workflow import run_licit_graph_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Criar aplicação FastAPI
app = FastAPI(
    title="LicitGraphAi API",
    description="API para análise automática de editais de licitação",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar origens
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Diretórios
UPLOAD_DIR = Path("data/raw/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def root():
    """Endpoint raiz."""
    return {
        "message": "LicitGraphAi API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload - Upload de PDF",
            "analyze": "/analyze - Análise de edital",
            "status": "/status - Status do sistema"
        }
    }


@app.get("/status")
async def get_status():
    """Status do sistema."""
    return {
        "status": "online",
        "service": "LicitGraphAi API",
        "version": "1.0.0",
        "nodes": ["node_1", "node_2", "node_3", "node_4"],
        "orchestration": "LangGraph"
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload de arquivo PDF.
    
    Args:
        file: Arquivo PDF enviado
        
    Returns:
        Caminho do arquivo salvo
    """
    logger.info(f"Recebendo upload: {file.filename}")
    
    # Validar tipo de arquivo
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são aceitos")
    
    # Gerar nome único
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    
    # Salvar arquivo
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Arquivo salvo: {file_path}")
        
        return {
            "message": "Upload realizado com sucesso",
            "file_id": file_id,
            "filename": file.filename,
            "file_path": str(file_path)
        }
        
    except Exception as e:
        logger.error(f"Erro ao salvar arquivo: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao salvar arquivo: {str(e)}")


@app.post("/analyze")
async def analyze_edital(file_path: str, company_profile: Optional[dict] = None):
    """
    Analisa edital usando LangGraph workflow.
    
    Args:
        file_path: Caminho do arquivo PDF
        company_profile: Perfil da empresa (opcional)
        
    Returns:
        Resultado da análise completa
    """
    logger.info(f"Iniciando análise: {file_path}")
    
    # Validar se arquivo existe
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    
    try:
        # Executar workflow LangGraph
        result = run_licit_graph_pipeline(str(pdf_path), company_profile)
        
        # Verificar se houve erro
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        
        logger.info("Análise concluída com sucesso")
        
        return {
            "message": "Análise concluída com sucesso",
            "result": {
                "chunks_count": len(result.get("chunks", [])),
                "embeddings": result.get("embeddings"),
                "analysis": result.get("analysis"),
                "checklist": result.get("checklist")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na análise: {str(e)}")


@app.post("/analyze/upload")
async def analyze_with_upload(file: UploadFile = File(...), company_profile: Optional[dict] = None):
    """
    Upload e análise em um único endpoint.
    
    Args:
        file: Arquivo PDF enviado
        company_profile: Perfil da empresa (opcional)
        
    Returns:
        Resultado da análise completa
    """
    logger.info(f"Upload e análise: {file.filename}")
    
    # Validar tipo de arquivo
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são aceitos")
    
    # Gerar nome único
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    
    # Salvar arquivo
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Arquivo salvo: {file_path}")
        
        # Executar análise
        result = run_licit_graph_pipeline(str(file_path), company_profile)
        
        # Verificar se houve erro
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        
        logger.info("Análise concluída com sucesso")
        
        return {
            "message": "Upload e análise concluídos com sucesso",
            "file_id": file_id,
            "result": {
                "chunks_count": len(result.get("chunks", [])),
                "embeddings": result.get("embeddings"),
                "analysis": result.get("analysis"),
                "checklist": result.get("checklist")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no upload e análise: {e}")
        raise HTTPException(status_code=500, detail=f"Erro no upload e análise: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
