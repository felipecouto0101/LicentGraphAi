"""
Exemplo de uso da API FastAPI

Demonstra como fazer requisições para a API.
"""

import requests
import json


def test_root():
    """Testa endpoint raiz."""
    response = requests.get("http://localhost:8000/")
    print("Root Endpoint:")
    print(json.dumps(response.json(), indent=2))
    print()


def test_status():
    """Testa endpoint de status."""
    response = requests.get("http://localhost:8000/status")
    print("Status Endpoint:")
    print(json.dumps(response.json(), indent=2))
    print()


def test_upload():
    """Testa endpoint de upload."""
    # Substitua por um PDF real
    pdf_path = "data/raw/edital_exemplo.pdf"
    
    if not Path(pdf_path).exists():
        print(f"Arquivo não encontrado: {pdf_path}")
        print("Crie um PDF em data/raw/edital_exemplo.pdf para testar")
        return
    
    with open(pdf_path, "rb") as f:
        files = {"file": f}
        response = requests.post("http://localhost:8000/upload", files=files)
    
    print("Upload Endpoint:")
    print(json.dumps(response.json(), indent=2))
    print()


def test_analyze():
    """Testa endpoint de análise."""
    # Primeiro precisa fazer upload e pegar o caminho
    pdf_path = "data/raw/uploads/seu_arquivo.pdf"
    
    if not Path(pdf_path).exists():
        print(f"Arquivo não encontrado: {pdf_path}")
        print("Primeiro faça o upload")
        return
    
    payload = {
        "file_path": pdf_path,
        "company_profile": {
            "name": "Empresa Exemplo Ltda",
            "cnpj": "00.000.000/0001-00"
        }
    }
    
    response = requests.post("http://localhost:8000/analyze", json=payload)
    
    print("Analyze Endpoint:")
    print(json.dumps(response.json(), indent=2))
    print()


def test_analyze_upload():
    """Testa endpoint de upload + análise em um só."""
    pdf_path = "data/raw/edital_exemplo.pdf"
    
    if not Path(pdf_path).exists():
        print(f"Arquivo não encontrado: {pdf_path}")
        print("Crie um PDF em data/raw/edital_exemplo.pdf para testar")
        return
    
    with open(pdf_path, "rb") as f:
        files = {"file": f}
        company_profile = {
            "name": "Empresa Exemplo Ltda",
            "cnpj": "00.000.000/0001-00"
        }
        response = requests.post(
            "http://localhost:8000/analyze/upload",
            files=files,
            data={"company_profile": json.dumps(company_profile)}
        )
    
    print("Analyze + Upload Endpoint:")
    print(json.dumps(response.json(), indent=2))
    print()


if __name__ == "__main__":
    from pathlib import Path
    
    print("=" * 60)
    print("Testando API FastAPI")
    print("=" * 60)
    print()
    
    # Testar endpoints
    test_root()
    test_status()
    
    # Upload e análise precisam de PDF real
    print("Para testar upload e análise, você precisa de um PDF real")
    print("Descomente as linhas abaixo após ter um PDF:")
    # test_upload()
    # test_analyze()
    # test_analyze_upload()
    
    print()
    print("=" * 60)
    print("Documentação Swagger: http://localhost:8000/docs")
    print("=" * 60)
